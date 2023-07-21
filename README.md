# Catching Up Distillation

This repository contains the codebase for [Catch-Up-Distillation](https://arxiv.org/abs/2305.10769), implemented using PyTorch for conducting large-scale experiments on ImageNet-64. We have based our repository on [Consistency Model](https://github.com/openai/consistency_models).

The repository for CIFAR-10 and MNIST experiments is in [Catch-Up-Distillation-CIFAT10-MNIST](https://github.com/shaoshitong/Catch-Up-Distillation/tree/main).

# Pre-trained models

We have released the checkpoint for ImageNet-64x64 in the paper.
Here are the download links for each model checkpoint:

 * CUD on ImageNet-64: [edm_imagenet64_ema.pt](https://pan.baidu.com/s/1C9u8k1lfi9wraN9tLRJztg), password is `cuds`.

# Dependencies

To install all packages in this codebase along with their dependencies, run

```sh
pip install -e .
```

# Model training and sampling

We provide examples of Catch-Up-Distillation training, inference:

```sh
# Training:
mpiexec --allow-run-as-root -n 8 python cud_train.py --training_mode catchingup_distillation \
    --target_ema_mode fixed --start_ema 0.95 --scale_mode fixed --start_scales 80 \
    --total_training_steps 1200000 --loss_norm l2 --lr_anneal_steps 0 \
    --attention_resolutions 32,16,8 --class_cond True --use_scale_shift_norm True --dropout 0.1 \
    --ema_rate 0.999,0.9999,0.9999432189950708 --global_batch_size 1024 --image_size 64 --lr 0.0004 --num_channels 192 \
    --num_head_channels 64 --num_res_blocks 3 --resblock_updown True --schedule_sampler uniform --use_fp16 True --weight_decay 0.0 \
    --weight_schedule uniform --data_dir /home/imagenet/train \
    --predstep 1 --adapt_cu uniform --TN 16 --prior_shakedrop True
  
# Inference:

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
```

# Evaluations

We follow Consistency Model to compare different generative models, we use FID, Precision, Recall, and Inception Score. These metrics can all be calculated using batches of samples stored in `.npz` (numpy) files. One can evaluate samples with [cm/evaluations/evaluator.py](evaluations/evaluator.py) in the same way as described in [openai/guided-diffusion](https://github.com/openai/guided-diffusion), with reference dataset batches provided therein.