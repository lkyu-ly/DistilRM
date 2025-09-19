#!/bin/bash
#SBATCH --job-name=Llama-2-13b-chat_lr_2e-5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:4  # 使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus # 分区默认即可
WANDB_DISABLED=true CUDA_VISIBLE_DEVICES="4,5,6,7" accelerate launch \
    --config_file ../accelerate/fsdp_config.yaml \
    --main_process_port=12542 \
    src/train.py "configs/train/qwen2.5_full_sft.yaml"
