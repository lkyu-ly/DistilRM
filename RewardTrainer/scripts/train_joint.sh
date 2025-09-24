#!/bin/bash

export NCCL_P2P_DISABLE=1
export WANDB_MODE=offline
# export CUDA_VISIBLE_DEVICES="0,1"

dataset_name='data/skywork_10k_joint.json'
base_model='/root/autodl-tmp/models/Qwen2.5-3B-Instruct'
teacher_model='/root/autodl-tmp/models/Qwen3-14B'
log_dir='/root/autodl-tmp/output'
main_process_port=12542

n_gpu=2
learning_rate=1e-6
max_length=1024
num_train_epochs=1
per_device_train_batch_size=1
gradient_accumulation_steps=8

# Joint training parameters
reward_weight=0.98
sft_weight=0.01
kl_weight=0.01
temperature=1.0

# Value head parameters
layer_type='mlp'
num_layers=1
num_neurons=1024

accelerate launch --num_processes ${n_gpu} \
    --main_process_port ${main_process_port} \
    --config_file ../accelerate/fsdp_config.yaml \
    train/reward_models/run_joint_reward_train.py \
    --base_model ${base_model} \
    --teacher_model ${teacher_model} \
    --log_dir ${log_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${per_device_train_batch_size} \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --max_length ${max_length} \
    --learning_rate ${learning_rate} \
    --dataset ${dataset_name} \
    --bf16 True \
    --gradient_checkpointing True \
    --reward_weight ${reward_weight} \
    --sft_weight ${sft_weight} \
    --kl_weight ${kl_weight} \
    --temperature ${temperature} \
    --layer_type ${layer_type} \
    --num_layers ${num_layers} \
    --num_neurons ${num_neurons}