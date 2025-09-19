#!/bin/bash
#SBATCH --job-name=yolov5  # 任务名字
#SBATCH --nodes=1  # 节点
#SBATCH --ntasks=4  # 需要多少个 cpu
#SBATCH --gres=gpu:1  #使用 gpu 数量，如果 2 个 gpu 写成 gpu:2
#SBATCH --partition=gpus  # 分区默认即可
# export PYTHONPATH=/home/DistilRM/RewardTrainer:$PYTHONPATH
export CUDA_VISIBLE_DEVICES="4" 
python -m eval.rewardbench.rewardbench \
    --model="/H1/hanlvyuan/DistilRM/RewardTrainer/output/sft_distill_Qwen2.5-3B-Instruct_Qwen3-14B_len1024_fulltrain_1e-06_dataskywork_10k_rm.json_64" \
    --dataset="data/rewardbench.json" \
    --output_dir="result/rewardbench" \
    --batch_size=64 \
    --load_json \
    --save_all
