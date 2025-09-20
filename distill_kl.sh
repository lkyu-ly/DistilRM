#!/usr/bin/env bash
# usage: bash run_kl.sh
main_process_port=12540
export CUDA_VISIBLE_DEVICES="0,1" 

TEACHER=/H1/zhouhongli/models/Qwen3-14B
STUDENT=/H1/zhouhongli/models/Qwen2.5-3B-Instruct
DATA=responses/responses_qwen3-14b.jsonl
OUT=../DistillOutput/sft_distill_Qwen2.5-3B-Instruct_Qwen3-14B

n_gpu=2
gradient_accumulation_steps=16
learning_rate=1e-6

accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file accelerate/fsdp_config.yaml \
    distill_kl.py \
    --student_model $STUDENT \
    --teacher_model $TEACHER \
    --data_path $DATA \
    --output_dir $OUT \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --learning_rate $learning_rate \
    --num_epochs 1 \
    --batch_size 1 \
    --lm_weight 0.5 \
    --kl_weight 0.5 \
    --max_length 1024
