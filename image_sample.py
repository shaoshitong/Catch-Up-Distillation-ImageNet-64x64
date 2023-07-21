"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist
from cm import dist_util, logger
from cm.karras_diffusion import karras_sample, rectified_sample
from cm.random_util import get_generator
from cm.script_util import (NUM_CLASSES, add_dict_to_argparser, args_to_dict,
                            create_model_and_diffusion,
                            model_and_diffusion_defaults)
from torchvision.utils import save_image

"""
CUDA_VISIBLE_DEVICES=0 mpiexec --allow-run-as-root -n 1 python image_sample.py \
 --batch_size 256 --training_mode catchingup_distillation --sampler euler \
 --model_path /tmp/openai-2023-06-07-13-33-42-685241/ema_0.9999432189950708_500000.pt  --attention_resolutions 32,16,8 \
 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --save_z True --prior_shakedrop True \
 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000  --resblock_updown True --use_fp16 True --weight_schedule uniform --steps 16

CUDA_VISIBLE_DEVICES=0 mpiexec --allow-run-as-root -n 1 python image_sample.py \
 --batch_size 256 --training_mode catchingup_distillation --sampler dpm_solver_2 \
 --model_path /tmp/openai-2023-06-07-13-33-42-685241/ema_0.9999432189950708_500000.pt  --attention_resolutions 32,16,8 \
 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --save_z True --prior_shakedrop True \
 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000  --resblock_updown True --use_fp16 True --weight_schedule uniform --steps 6

 
CUDA_VISIBLE_DEVICES=0 mpiexec --allow-run-as-root -n 1 python image_sample.py \
 --batch_size 256 --training_mode catchingup_distillation --sampler euler \
 --model_path /tmp/openai-2023-06-07-13-33-42-685241/ema_0.9999432189950708_500000.pt  --attention_resolutions 32,16,8 \
 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --save_z True --prior_shakedrop True \
 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000  --resblock_updown True --use_fp16 True --weight_schedule uniform --steps 8

CUDA_VISIBLE_DEVICES=0 mpiexec --allow-run-as-root -n 1 python image_sample.py \
 --batch_size 256 --training_mode catchingup_distillation --sampler euler \
 --model_path /tmp/openai-2023-06-07-13-33-42-685241/ema_0.9999432189950708_500000.pt  --attention_resolutions 32,16,8 \
 --class_cond True --use_scale_shift_norm True --dropout 0.0 --image_size 64 --num_channels 192 --save_z True --prior_shakedrop True \
 --num_head_channels 64 --num_res_blocks 3 --num_samples 50000  --resblock_updown True --use_fp16 True --weight_schedule uniform --steps 4
"""
def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    catchingup = False
    karra = False
    if args.training_mode == "progdist":
        distillation = False
    elif "consistency" in args.training_mode:
        distillation = True
    elif "catchingup" in args.training_mode:
        catchingup = True
        distillation = True
        if "karra" in args.training_mode:
            karra = True
        else:
            karra = False
    else:
        raise ValueError(f"unknown training mode {args.training_mode}")
    
    logger.log("creating model and diffusion...")
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = distillation
    model_and_diffusion_kwargs["catchingup"] = catchingup
    model_and_diffusion_kwargs["karra"] = karra
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    if args.sampler == "multistep":
        assert len(args.ts) > 0
        ts = tuple(int(x) for x in args.ts.split(","))
    else:
        ts = None

    all_images = []
    all_labels = []
    all_noises = []
    if args.sampler == "heun":
        all_x0hat_images = [[] for i in range((args.steps+1)//2)]
    else:
        all_x0hat_images = [[] for i in range(args.steps)]
    generator = get_generator(args.generator, args.num_samples, args.seed)

    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
            )
            model_kwargs["y"] = classes

        if catchingup and not karra:
            print("clip_denoised",args.clip_denoised)
            sample, noise, x0hat_list = rectified_sample(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                steps=args.steps,
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                sampler=args.sampler,
                generator_id=args.generator_id,
                generator=generator,
            )
            save_image(sample[:64]*0.5+0.5,"test.png",nrow=8)
            # for _ii,x0hat in enumerate(x0hat_list):
            #     save_image(x0hat[:64]*0.5+0.5,f"test_4_x0hat_{_ii}.png",nrow=8)

        else:
            print(args)
            sample = karras_sample(
                diffusion,
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                steps=args.steps,
                model_kwargs=model_kwargs,
                device=dist_util.dev(),
                clip_denoised=args.clip_denoised,
                sampler=args.sampler,
                sigma_min=args.sigma_min,
                sigma_max=args.sigma_max,
                s_churn=args.s_churn,
                s_tmin=args.s_tmin,
                s_tmax=args.s_tmax,
                s_noise=args.s_noise,
                generator=generator,
                ts=ts,
                estimate_velocity=args.estimate_velocity
            )
            save_image(sample[:64]*0.5+0.5,"test.png",nrow=8)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()
        if catchingup and args.save_z:
            noise = noise.contiguous()
            gathered_noise = [th.zeros_like(noise) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_noise, noise)  # gather not supported with NCCL
            all_noises.extend([noise.cpu().numpy() for noise in gathered_noise])
            print(len(x0hat_list))
            for _ in range(len(x0hat_list)):
                x0hat_list[_] = x0hat_list[_].contiguous()
                gather_x0hat = [th.zeros_like(x0hat_list[_]) for _ in range(dist.get_world_size())]
                dist.all_gather(gather_x0hat, x0hat_list[_])  # gather not supported with NCCL
                all_x0hat_images[_].extend([x0hat.cpu().numpy() for x0hat in gather_x0hat])

        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            gathered_labels = [
                th.zeros_like(classes) for _ in range(dist.get_world_size())
            ]
            dist.all_gather(gathered_labels, classes)
            all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]

    if catchingup and args.save_z:
        arr_noise = np.concatenate(all_noises, axis=0)
        arr_noise = arr_noise[:args.num_samples]
        arr_x0hat_list = []
        for _ in range(len(all_x0hat_images)):
            arr_x0hat = np.concatenate(all_x0hat_images[_], axis=0)[:args.num_samples]
            arr_x0hat_list.append(arr_x0hat)
        arr_x0hat_list = np.stack(arr_x0hat_list, axis=0)
        print(arr_x0hat_list.shape)
        arr_x0hat_list = np.stack([arr_x0hat_list[-4],arr_x0hat_list[-3],arr_x0hat_list[-2],arr_x0hat_list[-1]],axis=0)

    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        if catchingup and args.save_z:
            if args.class_cond:
                np.savez(out_path, arr, label_arr, arr_noise, arr_x0hat_list)
            else:
                np.savez(out_path, arr, arr_noise, arr_x0hat_list)
        else:
            if args.class_cond:
                np.savez(out_path, arr, label_arr)
            else:
                np.savez(out_path, arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        training_mode="edm",
        generator="determ",
        clip_denoised=False,
        num_samples=10000,
        batch_size=16,
        sampler="heun",
        s_churn=0.0,
        s_tmin=0.0,
        s_tmax=float("inf"),
        s_noise=1.0,
        generator_id=1,
        steps=40,
        model_path="",
        save_z=False,
        prior_shakedrop=False,
        seed=42,
        ts="",
        estimate_velocity=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
