import argparse

import cm.dist_util as dist_util
import numpy as np
import torch as th

from .forward_unet import UNetEncoder
from .karras_diffusion import KarrasDenoiser, RectifiedDenoiser
from .unet import MetaGenerator, UNetModel

NUM_CLASSES = 1000


def cm_train_defaults():
    return dict(
        teacher_model_path="",
        teacher_dropout=0.1,
        training_mode="consistency_distillation",
        target_ema_mode="fixed",
        scale_mode="fixed",
        total_training_steps=600000,
        start_ema=0.0,
        start_scales=40,
        end_scales=40,
        distill_steps_per_iter=50000,
        loss_norm="lpips",
    )


def model_and_diffusion_defaults():
    """
    Defaults for image training.
    """
    res = dict(
        save_z = False,
        sigma_min=0.002,
        sigma_max=80.0,
        image_size=64,
        total_number = 16,
        num_channels=128,
        num_res_blocks=2,
        num_heads=4,
        num_heads_upsample=-1,
        num_head_channels=-1,
        attention_resolutions="32,16,8",
        channel_mult="",
        dropout=0.0,
        class_cond=False,
        use_checkpoint=False,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=False,
        use_new_attention_order=False,
        learn_sigma=False,
        weight_schedule="karras",
        prior_shakedrop = False,
        phi = 0.75,
    )
    return res

def forward_model_defaults():
    res = dict(
        image_size=64,
        f_num_channels=32,
        f_num_res_blocks=2,
        f_num_heads=4,
        f_num_heads_upsample=-1,
        f_num_head_channels=-1,
        f_attention_resolutions="16",
        f_channel_mult="2,2,2",
        f_dropout=0.13,
        f_class_cond=False,
        f_use_checkpoint=False,
        f_use_scale_shift_norm=True,
        f_resblock_updown=False,
        f_use_fp16=True,
        f_use_new_attention_order=False,
        f_learn_sigma=False,
    )
    return res

def create_meta_generator(
    in_channel,
    out_channel,
    dim,
):
    assert out_channel == 3 or out_channel == 1,"out_channel must be 1 or 3"
    return MetaGenerator(in_channel=in_channel,out_channel=out_channel,dim=dim)
    

def create_forward_model(    
    image_size,
    f_num_channels,
    f_num_res_blocks,
    f_channel_mult="",
    f_learn_sigma=False,
    f_class_cond=False,
    f_use_checkpoint=False,
    f_attention_resolutions="16",
    f_num_heads=1,
    f_num_head_channels=-1,
    f_num_heads_upsample=-1,
    f_use_scale_shift_norm=False,
    f_dropout=0,
    f_resblock_updown=False,
    f_use_fp16=False,
    f_use_new_attention_order=False,
):
    if f_channel_mult == "":
        if image_size == 512:
            f_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            f_channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            f_channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            f_channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        f_channel_mult = tuple(int(ch_mult) for ch_mult in f_channel_mult.split(","))

    attention_ds = []
    for res in f_attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    encoder =  UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=f_num_channels,
        out_channels=6,
        num_res_blocks=f_num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=f_dropout,
        channel_mult=f_channel_mult,
        num_classes=(NUM_CLASSES if f_class_cond else None),
        use_checkpoint=f_use_checkpoint,
        use_fp16=f_use_fp16,
        num_heads=f_num_heads,
        num_head_channels=f_num_head_channels,
        num_heads_upsample=f_num_heads_upsample,
        use_scale_shift_norm=f_use_scale_shift_norm,
        resblock_updown=f_resblock_updown,
        use_new_attention_order=f_use_new_attention_order,
    )
    return UNetEncoder(encoder=encoder,input_nc=3)
    

def create_model_and_diffusion(
    image_size,
    class_cond,
    learn_sigma,
    num_channels,
    num_res_blocks,
    channel_mult,
    num_heads,
    num_head_channels,
    num_heads_upsample,
    attention_resolutions,
    dropout,
    use_checkpoint,
    use_scale_shift_norm,
    resblock_updown,
    use_fp16,
    use_new_attention_order,
    weight_schedule,
    sigma_min=0.002,
    save_z = False,
    total_number = 16,
    karra = True,
    sigma_max=80.0,
    catchingup = False,
    predstep = 1,
    num_steps = 16,
    adapt_cu = "uniform",
    TN = 16,
    distillation=False,
    prior_shakedrop = False,
    phi = 0.75,
):
    model = create_model(
        image_size,
        num_channels,
        num_res_blocks,
        channel_mult=channel_mult,
        learn_sigma=learn_sigma,
        class_cond=class_cond,
        use_checkpoint=use_checkpoint,
        attention_resolutions=attention_resolutions,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        dropout=dropout,
        resblock_updown=resblock_updown,
        use_fp16=use_fp16,
        use_new_attention_order=use_new_attention_order,
        predstep = predstep,
        prior_shakedrop = prior_shakedrop,
        phi = phi,
    )
        
    if not karra:
        diffusion = RectifiedDenoiser(
            device=dist_util.dev(),
            num_steps=num_steps,
            TN = TN,
            adapt_cu= adapt_cu,
            predstep=predstep,
        )
    else:
        diffusion = KarrasDenoiser(
            sigma_data=0.5,
            sigma_max=sigma_max,
            sigma_min=sigma_min,
            distillation=distillation,
            weight_schedule=weight_schedule,
            predstep = predstep,
            TN = TN,
        )
    return model, diffusion

def create_model(
    image_size,
    num_channels,
    num_res_blocks,
    channel_mult="",
    learn_sigma=False,
    class_cond=False,
    use_checkpoint=False,
    attention_resolutions="16",
    num_heads=1,
    num_head_channels=-1,
    num_heads_upsample=-1,
    use_scale_shift_norm=False,
    dropout=0,
    resblock_updown=False,
    use_fp16=False,
    use_new_attention_order=False,
    predstep = 1,
    prior_shakedrop = False,
    phi = 0.75,
):
    if channel_mult == "":
        if image_size == 512:
            channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
        elif image_size == 256:
            channel_mult = (1, 1, 2, 2, 4, 4)
        elif image_size == 128:
            channel_mult = (1, 1, 2, 3, 4)
        elif image_size == 64:
            channel_mult = (1, 2, 3, 4)
        elif image_size == 32:
            channel_mult = (1, 2, 3, 4)
        else:
            raise ValueError(f"unsupported image size: {image_size}")
    else:
        channel_mult = tuple(int(ch_mult) for ch_mult in channel_mult.split(","))

    attention_ds = []
    for res in attention_resolutions.split(","):
        attention_ds.append(image_size // int(res))

    return UNetModel(
        image_size=image_size,
        in_channels=3,
        model_channels=num_channels,
        out_channels=(3 if not learn_sigma else 6),
        num_res_blocks=num_res_blocks,
        attention_resolutions=tuple(attention_ds),
        dropout=dropout,
        channel_mult=channel_mult,
        num_classes=(NUM_CLASSES if class_cond else None),
        use_checkpoint=use_checkpoint,
        use_fp16=use_fp16,
        num_heads=num_heads,
        num_head_channels=num_head_channels,
        num_heads_upsample=num_heads_upsample,
        use_scale_shift_norm=use_scale_shift_norm,
        resblock_updown=resblock_updown,
        use_new_attention_order=use_new_attention_order,
        predstep=predstep,
        prior_shakedrop = prior_shakedrop,
        phi = phi,
    )


def create_ema_and_scales_fn(
    target_ema_mode,
    start_ema,
    scale_mode,
    start_scales,
    end_scales,
    total_steps,
    distill_steps_per_iter,
):
    def ema_and_scales_fn(step):
        if target_ema_mode == "fixed" and scale_mode == "fixed":
            target_ema = start_ema
            scales = start_scales
        elif target_ema_mode == "fixed" and scale_mode == "progressive":
            target_ema = start_ema
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            scales = scales + 1

        elif target_ema_mode == "adaptive" and scale_mode == "progressive":
            scales = np.ceil(
                np.sqrt(
                    (step / total_steps) * ((end_scales + 1) ** 2 - start_scales**2)
                    + start_scales**2
                )
                - 1
            ).astype(np.int32)
            scales = np.maximum(scales, 1)
            c = -np.log(start_ema) * start_scales
            target_ema = np.exp(-c / scales)
            scales = scales + 1
        elif target_ema_mode == "fixed" and scale_mode == "progdist":
            distill_stage = step // distill_steps_per_iter
            scales = start_scales // (2**distill_stage)
            scales = np.maximum(scales, 2)

            sub_stage = np.maximum(
                step - distill_steps_per_iter * (np.log2(start_scales) - 1),
                0,
            )
            sub_stage = sub_stage // (distill_steps_per_iter * 2)
            sub_scales = 2 // (2**sub_stage)
            sub_scales = np.maximum(sub_scales, 1)

            scales = np.where(scales == 2, sub_scales, scales)

            target_ema = 1.0
        else:
            raise NotImplementedError

        return float(target_ema), int(scales)

    return ema_and_scales_fn


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")
