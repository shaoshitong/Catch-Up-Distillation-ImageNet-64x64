"""
Train a diffusion model on images.
"""

import argparse
import copy

import torch.distributed as dist
from cm import dist_util, logger
from cm.image_datasets import load_data
from cm.resample import create_named_schedule_sampler
from cm.script_util import (add_dict_to_argparser, args_to_dict,
                            cm_train_defaults, create_ema_and_scales_fn,
                            create_forward_model, create_model_and_diffusion,
                            forward_model_defaults,
                            model_and_diffusion_defaults)
from cm.train_util import CMTrainLoop

"""
mpiexec -n 2 python cm_train.py --training_mode consistency_distillation --target_ema_mode fixed \
    --start_ema 0.95 --scale_mode fixed --start_scales 40 --total_training_steps 600000 \
    --loss_norm lpips --lr_anneal_steps 0 --teacher_model_path /path/to/edm_imagenet64_ema.pt \
    --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.0 \
    --teacher_dropout 0.1 --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 2048 \
    --image_size 64 --lr 0.000008 --num_channels 192 --num_head_channels 64 --num_res_blocks 3 \
    --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 --weight_schedule uniform --data_dir /path/to/data \
    --predstep 1 --adapt-cu uniform --TN 16

mpiexec --allow-run-as-root -n 8 python cm_train.py --training_mode catchingup_distillation \
    --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 40 \
    --total_training_steps 1200000 --loss_norm l2 --lr_anneal_steps 0 \
    --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 \
    --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 1024 --image_size 64 --lr 0.00015 --num_channels 192 \
    --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 \
    --weight_schedule uniform --data_dir /home/imagenet/train \
    --predstep 1 --adapt_cu uniform --TN 16 --resume_checkpoint /tmp/checkpoint/model064000.pt
"""
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    ema_scale_fn = create_ema_and_scales_fn(
        target_ema_mode=args.target_ema_mode,
        start_ema=args.start_ema,
        scale_mode=args.scale_mode,
        start_scales=args.start_scales,
        end_scales=args.end_scales,
        total_steps=args.total_training_steps,
        distill_steps_per_iter=args.distill_steps_per_iter,
    )
    catchingup = False
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    elif "catchingup" in args.training_mode:
        catchingup = True
        distillation = True
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")

    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model_and_diffusion_kwargs["catchingup"] = catchingup
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    print("After Compling ......")
    if catchingup:
        forward_model_kwargs = forward_model_defaults()
        forward_model_kwargs["image_size"] = args.image_size
        forward_model = create_forward_model(**forward_model_kwargs)
        forward_model.to(dist_util.dev())
        forward_model.train()
        if args.use_fp16:
            forward_model.encoder.convert_to_fp16()
    else:
        forward_model = None
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.train()

    if args.use_fp16:
        model.convert_to_fp16()

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    data = load_data(
        data_dir=args.data_dir,
        batch_size=batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    if len(args.teacher_model_path) > 0:  # path to the teacher score model.
        logger.log(f"loading the teacher model from {args.teacher_model_path}")
        teacher_model_and_diffusion_kwargs = copy.deepcopy(model_and_diffusion_kwargs)
        teacher_model_and_diffusion_kwargs["dropout"] = args.teacher_dropout
        teacher_model_and_diffusion_kwargs["distillation"] = False
        teacher_model, teacher_diffusion = create_model_and_diffusion(
            **teacher_model_and_diffusion_kwargs,
        )

        teacher_model.load_state_dict(
            dist_util.load_state_dict(args.teacher_model_path, map_location="cpu"),
        )

        teacher_model.to(dist_util.dev())
        teacher_model.eval()

        for dst, src in zip(model.parameters(), teacher_model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            teacher_model.convert_to_fp16()

    else:
        teacher_model = None
        teacher_diffusion = None

    # load the target model for distillation, if path specified.

    if "consistency" in args.training_mode:
        logger.log("creating the target model")
        target_model, _ = create_model_and_diffusion(
            **model_and_diffusion_kwargs,
        )

        target_model.to(dist_util.dev())
        target_model.train()

        dist_util.sync_params(target_model.parameters())
        dist_util.sync_params(target_model.buffers())

        for dst, src in zip(target_model.parameters(), model.parameters()):
            dst.data.copy_(src.data)

        if args.use_fp16:
            target_model.convert_to_fp16()
    else:
        target_model = None

    logger.log("training...")
    CMTrainLoop(
        model=model,
        target_model=target_model,
        teacher_model=teacher_model,
        forward_model = forward_model,
        teacher_diffusion=teacher_diffusion,
        training_mode=args.training_mode,
        ema_scale_fn=ema_scale_fn,
        total_training_steps=args.total_training_steps,
        diffusion=diffusion,
        data=data,
        batch_size=batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        independent = args.independent,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        global_batch_size=2048,
        batch_size=-1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=2000,
        resume_checkpoint="",
        model_path = "",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        catchingup = False,
        predstep = 1,
        num_steps = 16,
        adapt_cu = "uniform",
        independent = False,
        prior_shakedrop = True,
        phi = 0.75,
        TN = 16,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(cm_train_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
