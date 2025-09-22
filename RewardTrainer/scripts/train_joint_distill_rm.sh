#!/usr/bin/env bash
# Joint Training Script: Reward Model + Distillation
# This script combines reward modeling with knowledge distillation from a teacher model

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES="0,1,2,3"

main_process_port=12542

# 基础配置
STUDENT_MODEL=/H1/zhouhongli/models/Qwen2.5-3B-Instruct
TEACHER_MODEL=/H1/zhouhongli/models/Qwen3-14B
DATASET='data/skywork_10k_rm.json'
OUTPUT_DIR="./reward_models_train"

# 模型架构配置 - 使用GRM的默认配置
LAYER_TYPE='mlp'  # Value head类型: mlp, linear

# 训练超参数
n_gpu=4
per_device_train_batch_size=1
gradient_accumulation_steps=16
learning_rate=1e-5
num_train_epochs=2
max_length=512

# 损失权重配置 (联合训练的核心)
reward_weight=1.0    # 奖励损失权重
sft_weight=0.1       # SFT损失权重
kl_weight=0.5        # KL蒸馏损失权重
temperature=1.0      # 蒸馏温度

# 评估和保存配置
eval_steps=5000      # 评估步数
save_steps=5000      # 保存步数

# 其他配置
weight_ratio=0.01    # GRM权重比例，来自train_grm.sh
reference_free=True  # 参考自由模式
sft_only=True        # SFT-only模式
use_wandb=True       # 是否使用WandB
wandb_name="joint_distill_rm_baseline"  # WandB实验名称

echo "Starting Joint Distillation + Reward Model Training..."
echo "Configuration:"
echo "  Student Model: $STUDENT_MODEL"
echo "  Teacher Model: $TEACHER_MODEL"
echo "  Dataset: $DATASET"
echo "  Loss Weights - Reward: $reward_weight, SFT: $sft_weight, KL: $kl_weight"
echo "  Output: $OUTPUT_DIR"
echo "  GPUs: $CUDA_VISIBLE_DEVICES (using $n_gpu GPUs)"

# 检查依赖
echo "Checking dependencies..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"

# 确保在正确的目录下运行
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../train/reward_models"

# 构建WandB选项
WANDB_ARGS=""
if [ "$use_wandb" = "True" ]; then
    WANDB_ARGS="--report_to wandb --wandb_name ${wandb_name}"
fi

# 使用accelerate启动训练，兼容参考脚本的格式
echo "Executing training command..."
accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file ../accelerate/fsdp_config.yaml \
    run_joint_distill_reward_train.py \
    --base_model ${STUDENT_MODEL} \
    --teacher_model ${TEACHER_MODEL} \
    --dataset ${DATASET} \
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
    --reference_free ${reference_free} \
    --sft_only ${sft_only} \
    --weight_ratio ${weight_ratio} \
    ${WANDB_ARGS}

echo "Training completed successfully!"