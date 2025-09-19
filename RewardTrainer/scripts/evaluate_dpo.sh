#!/bin/bash
#SBATCH --job-name=yolov5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  #使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可
export CUDA_VISIBLE_DEVICES="4" 
python -m eval.rewardbench.rewardbench \
    --model="../models/Qwen3-4B" \
    --ref_model="../models/Qwen3-4B-Base" \
    --dataset="data/rewardbench.json" \
    --output_dir="result/rewardbench" \
    --batch_size=1 \
    --load_json \
    --save_all \
