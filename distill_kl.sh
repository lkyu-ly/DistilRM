#!/usr/bin/env bash
# usage: bash run_kl.sh

# export TRANSFORMERS_VERBOSITY=debug
# export TOKENIZERS_PARALLELISM=false
# export CUDA_LAUNCH_BLOCKING=1
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES="2,3" 

TEACHER=/H1/zhouhongli/models/Qwen3-14B
STUDENT=/H1/zhouhongli/models/Qwen2.5-3B-Instruct
DATA=responses/responses_qwen3-14b.jsonl
OUT=../DistillOutput/sft_distill_Qwen2.5-3B-Instruct_Qwen3-14B

GPUS=2
GRAD_ACC=4

torchrun --nproc_per_node=$GPUS distill_kl.py \
  --student_model $STUDENT \
  --teacher_model $TEACHER \
  --data_path $DATA \
  --output_dir $OUT \
  --ds_config accelerate/ds_config.json \
  --num_epochs 1 \
  --batch_size 8 \
  --gradient_accumulation_steps $GRAD_ACC \
  --lm_weight 0 \
  --kl_weight 1 \
  --max_length 1024

# 控制 GPUS * GRAD_ACC * batch_size = 64