#!/usr/bin/env bash
# Joint Training Script: Reward Model + Distillation
# This script combines reward modeling with knowledge distillation from a teacher model

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="0,1"

main_process_port=12542

# 基础配置
STUDENT_MODEL=/H1/zhouhongli/models/Qwen2.5-3B-Instruct
TEACHER_MODEL=/H1/zhouhongli/models/Qwen3-14B
RM_DATASET='data/skywork_10k_rm.json'
DISTILL_DATASET='../responses/responses_qwen3-14b.jsonl'
OUTPUT_DIR="./joint_reward_models_train"

# 训练超参数
n_gpu=2
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=1e-6
num_train_epochs=1
max_length=1024

# 损失权重配置 (联合训练的核心)
reward_weight=0.5    # 奖励损失权重
sft_weight=0.25       # SFT损失权重
kl_weight=0.25        # KL蒸馏损失权重
temperature=1.0      # 蒸馏温度

# 评估和保存配置
eval_steps=5000      # 评估步数
save_steps=5000      # 保存步数

# GRM模型架构配置 (只影响value head结构，不影响损失计算)
LAYER_TYPE='mlp'  # value head类型: mlp/linear
# weight_ratio/reference_free/sft_only等GRM框架参数已不再需要
# 因为JointDistillRewardTrainer重写了compute_loss，使用自己的权重系统

echo "Starting Joint Distillation + Reward Model Training..."
echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Teacher Model: $TEACHER_MODEL"
echo "  RM Dataset: $RM_DATASET"
echo "  Distill Dataset: $DISTILL_DATASET"
echo "  Loss Weights - Reward: $reward_weight, SFT: $sft_weight, KL: $kl_weight"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $CUDA_VISIBLE_DEVICES (using $n_gpu GPUs)"

# 检查依赖
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 使用accelerate启动训练，兼容参考脚本的格式
echo "Executing training command..."
accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file ../accelerate/fsdp_config.yaml \
    train/reward_models/run_joint_distill_reward_train.py \
    --base_model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --dataset ${RM_DATASET} \
    --distill_dataset ${DISTILL_DATASET} \
    --kl_weight ${kl_weight} \
    --sft_weight ${sft_weight} \
    --reward_weight ${reward_weight} \
    --temperature ${temperature} \
    --max_length ${max_length} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --learning_rate ${learning_rate} \
    --num_train_epochs ${num_train_epochs} \
    --layer_type ${LAYER_TYPE} \
    --evaluation_strategy steps \
    --eval_steps ${eval_steps} \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --log_dir ${OUTPUT_DIR} \
    --gradient_checkpointing True \
    --bf16 True \

echo "Training completed successfully!"