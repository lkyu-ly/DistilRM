#!/bin/bash

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
export CUDA_VISIBLE_DEVICES=0,1

dataset_name='data/skywork_10k_joint.json'
base_model='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
log_dir='/root/autodl-tmp/output'
main_process_port=12541

n_gpu=2
learning_rate=2e-6
max_length=1024
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=128

# GRM parameters
weight_ratio=0.01
layer_type='mlp'
sft_only=True
reference_free=True

accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    train/reward_models/run_grm_reward_train.py \
    --base_model ${base_model} \
    --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_length ${max_length} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --bf16 True \
    --weight_ratio ${weight_ratio} \
    --layer_type ${layer_type} \
    --reference_free ${reference_free} \
    --sft_only ${sft_only} \
    --gradient_checkpointing True