from dataclasses import dataclass, field
from typing import Optional
import os
import torch
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    AutoModelForCausalLM,
)
from joint_reward_trainer import (
    JointDistillRewardTrainer,
    JointDataCollatorWithPadding,
)
from load_datasets import load_train_eval_dataset
from utils import print_trainable_parameters
from grm_utils import AutoModelForCausalLMWithValueHead

from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


@dataclass
class ScriptArguments:
    # training args
    per_device_train_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=16)
    learning_rate: Optional[float] = field(default=1e-5)
    num_train_epochs: Optional[int] = field(
        default=2,
        metadata={"help": "The number of training epochs for the reward model."},
    )
    optim: Optional[str] = field(
        default="adamw_hf", metadata={"help": "The optimizer to use."}
    )
    lr_scheduler_type: Optional[str] = field(
        default="cosine",
        metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=1024)
    gradient_checkpointing: Optional[bool] = field(default=True)
    bf16: Optional[bool] = field(default=True)
    attn_implementation: Optional[str] = field(default="")
    # data
    dataset: Optional[str] = field(default="llm-blender/Unified-Feedback",
        metadata={"help": "Path to preference dataset (RM training) in JSON format"}
    )
    distill_dataset: Optional[str] = field(default=None,
        metadata={"help": "Path to distillation dataset (KL/SFT training) in JSONL format"}
    )
    dataset_mode: Optional[str] = field(
        default="",
        metadata={"help": "use from '', '40k', and '400k' for the paper's experiments"},
    )
    # eval
    per_device_eval_batch_size: Optional[int] = field(default=1)
    evaluation_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=10000)
    # model and loss
    base_model: Optional[str] = field(default="google/gemma-2b-it")
    # Joint Training specific args
    teacher_model: Optional[str] = field(
        default="", metadata={"help": "Path to the teacher model for distillation"}
    )
    kl_weight: Optional[float] = field(
        default=0.5, metadata={"help": "Weight for KL distillation loss"}
    )
    sft_weight: Optional[float] = field(
        default=0.1, metadata={"help": "Weight for SFT loss"}
    )
    reward_weight: Optional[float] = field(
        default=1.0, metadata={"help": "Weight for reward loss"}
    )
    temperature: Optional[float] = field(
        default=1.0, metadata={"help": "Temperature for distillation"}
    )
    # log
    report_to: Optional[str] = field(
        default="none", metadata={"help": "use 'none', 'wandb'. "}
    )
    log_dir: Optional[str] = field(default="./reward_models_train")
    wandb_name: Optional[str] = field(
        default="joint_distill_rm",
    )
    save_strategy: Optional[str] = field(default="steps")
    save_steps: Optional[int] = field(default=10000)
    debug: Optional[bool] = field(
        default=False, metadata={"help": "if debug=True, only train with 100 samples"}
    )
    # Joint training核心参数 - value head相关配置 (用于GRM模型架构)
    layer_type: Optional[str] = field(default="mlp")  # mlp, linear
    num_layers: Optional[int] = field(default=1)
    num_neurons: Optional[int] = field(default=1024)
    no_logsigmoid_sft: Optional[bool] = field(default=False)


def main():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    model_name_split = script_args.base_model.split("/")[-1]
    teacher_name_split = (
        script_args.teacher_model.split("/")[-1]
        if script_args.teacher_model
        else "no_teacher"
    )
    output_name = f"{script_args.log_dir}/{model_name_split}_{teacher_name_split}_joint_distill_len{script_args.max_length}_kl{script_args.kl_weight}_sft{script_args.sft_weight}_rw{script_args.reward_weight}"

    # 创建训练数据目录
    os.makedirs(output_name, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=os.path.join(output_name, "logs"),
        learning_rate=script_args.learning_rate,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        evaluation_strategy=script_args.evaluation_strategy,
        eval_steps=script_args.eval_steps,
        save_strategy=script_args.save_strategy,
        save_steps=script_args.save_steps,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        bf16=script_args.bf16,
        logging_strategy="steps",
        logging_steps=1,
        warmup_ratio=0.03,
        optim=script_args.optim,
        lr_scheduler_type=script_args.lr_scheduler_type,
        run_name=script_args.wandb_name,
        max_grad_norm=5.0,
        report_to=script_args.report_to,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.base_model, use_fast=False)
    tokenizer.max_length = script_args.max_length
    if tokenizer.pad_token is None:
        if "Llama" in script_args.base_model:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        else:
            tokenizer.pad_token = tokenizer.eos_token

    # 按需加载教师模型
    teacher_tokenizer = None
    if script_args.teacher_model and script_args.kl_weight > 0:
        print(f"Loading teacher model tokenizer from: {script_args.teacher_model}")
        teacher_tokenizer = AutoTokenizer.from_pretrained(
            script_args.teacher_model, use_fast=False
        )
        if teacher_tokenizer.pad_token is None:
            if "Llama" in script_args.teacher_model:
                teacher_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            else:
                teacher_tokenizer.pad_token = teacher_tokenizer.eos_token
        teacher_tokenizer.max_length = script_args.max_length

    # Load datasets
    print(f"Loading preference dataset: {script_args.dataset}")
    if script_args.distill_dataset:
        print(f"Loading distillation dataset: {script_args.distill_dataset}")
    else:
        print("No distillation dataset provided, using preference dataset for all training")

    # 使用特殊的预处理来包含消息结构
    def custom_load_train_eval_dataset(
        data_path, tokenizer, size=None, mode="", max_length=512, is_distill_data=False
    ):
        """修改版本的数据加载函数，包含消息结构信息"""
        from datasets import load_dataset
        import logging

        logging.basicConfig(level=logging.INFO)

        try:
            if data_path.endswith('.jsonl'):
                # 加载JSONL格式的蒸馏数据
                ds = load_dataset("json", data_files=data_path, split="train")
                if 'question' in ds.features and 'response' in ds.features:
                    # 如果是蒸馏数据格式（问答对），转换为偏好数据格式
                    def convert_distill_to_preferred_format(example):
                        return {
                            "prompt": example["question"],
                            "chosen": example["response"],
                            "rejected": ""  # 为蒸馏数据创建空rejected作为占位符
                        }
                    ds = ds.map(convert_distill_to_preferred_format)
            else:
                # 加载JSON格式的偏好数据
                ds = load_dataset("json", data_files=data_path, split="train")
            logging.info(f"Loaded dataset from {data_path} with {len(ds)} examples")
        except Exception as e:
            logging.error(f"Failed to load dataset from {data_path}: {e}")
            raise

        if size is not None:
            ds = ds.select(range(min(size, len(ds))))
            logging.info(f"Reduced dataset to {len(ds)} examples")

        def convert_to_chat_with_messages(example):
            """转换数据并保留消息结构"""
            prompt = example.get("prompt", "").strip()
            chosen = example.get("chosen", "").strip() or " "
            rejected = example.get("rejected", "").strip() or " "

            if not prompt:
                raise ValueError(f"Empty prompt in example: {example}")

            # 处理蒸馏数据（rejected为空的情况
            is_distill_data = (not rejected or rejected.strip() == "" or rejected.strip() == " ") and example.get("rejected") == ""

            chosen_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": chosen},
            ]

            # 如果是蒸馏数据，为rejected创建一个虚拟响应
            if is_distill_data:
                rejected_messages = chosen_messages  # 使用相同的内容，后面DataCollator会处理
            else:
                rejected_messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": rejected},
                ]

            result = {
                "chosen": chosen_messages,
                "rejected": rejected_messages,
                "chosen_messages": chosen_messages,  # 新增：保留消息结构
                "rejected_messages": rejected_messages,  # 新增：保留消息结构
                "chosen_user_messages": [
                    {"role": "user", "content": prompt}
                ],  # 用户部分单独保存
                "rejected_user_messages": [{"role": "user", "content": prompt}],
                "is_distill_data": is_distill_data,  # 标记是否是蒸馏数据
            }

            # 添加评分信息（如果存在）
            chosen_score, rejected_score = example.get("chosen_score"), example.get(
                "rejected_score"
            )
            if chosen_score is not None:
                result["chosen_score"] = chosen_score
            if rejected_score is not None:
                result["rejected_score"] = rejected_score

            return result

        def formatting_func_with_messages(example):
            """tokenization并保留消息结构"""
            try:
                example = convert_to_chat_with_messages(example)
                kwargs = {
                    "padding": "max_length",
                    "truncation": True,
                    "max_length": max_length,
                    "return_tensors": "pt",
                }

                prompt_plus_chosen_response = tokenizer.apply_chat_template(
                    example["chosen"], tokenize=False
                )
                prompt_plus_rejected_response = tokenizer.apply_chat_template(
                    example["rejected"], tokenize=False
                )

                if (
                    not prompt_plus_chosen_response.strip()
                    or not prompt_plus_rejected_response.strip()
                ):
                    logging.warning(f"Empty tokenized response in example: {example}")
                    return None

                tokens_chosen = tokenizer.encode_plus(
                    prompt_plus_chosen_response, **kwargs
                )
                tokens_rejected = tokenizer.encode_plus(
                    prompt_plus_rejected_response, **kwargs
                )

                for key, tokens in [
                    ("chosen", tokens_chosen),
                    ("rejected", tokens_rejected),
                ]:
                    if (
                        tokens["input_ids"].shape[1] != max_length
                        or tokens["attention_mask"].shape[1] != max_length
                    ):
                        logging.warning(
                            f"Shape mismatch in {key} for example: {example}"
                        )
                        return None

                # GRM模式：计算labels
                prompt = example["chosen"][:-1]
                prompt_template = tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )
                tokens_prompt = tokenizer.encode_plus(prompt_template, **kwargs)[
                    "input_ids"
                ][0]

                label_chosen = tokens_chosen["input_ids"][0].clone()
                label_rejected = tokens_rejected["input_ids"][0].clone()
                label_chosen[: len(tokens_prompt)] = -100
                label_rejected[: len(tokens_prompt)] = -100

                result = {
                    "input_ids_chosen": tokens_chosen["input_ids"][0],
                    "attention_mask_chosen": tokens_chosen["attention_mask"][0],
                    "input_ids_rejected": tokens_rejected["input_ids"][0],
                    "attention_mask_rejected": tokens_rejected["attention_mask"][0],
                    "label_chosen": label_chosen,
                    "label_rejected": label_rejected,
                    # 保留消息结构
                    "chosen_messages": example["chosen_messages"],
                    "rejected_messages": example["rejected_messages"],
                    "chosen_user_messages": example["chosen_user_messages"],
                    "rejected_user_messages": example["rejected_user_messages"],
                }

                # 添加评分信息（如果存在）
                if "chosen_score" in example:
                    result["chosen_score"] = example["chosen_score"]
                if "rejected_score" in example:
                    result["rejected_score"] = example["rejected_score"]

                return result
            except Exception as e:
                logging.error(f"Error processing example {example}: {e}")
                raise

        # 应用数据处理
        ds = ds.map(formatting_func_with_messages, batched=False, num_proc=16)
        ds = ds.filter(lambda x: x is not None)
        logging.info(f"Dataset processed with messages, columns: {ds.column_names}")

        # 验证token长度
        def validate_tokens(example):
            keys = [
                ("input_ids_chosen", "attention_mask_chosen"),
                ("input_ids_rejected", "attention_mask_rejected"),
            ]
            for input_key, mask_key in keys:
                input_ids = example[input_key]
                attention_mask = example[mask_key]
                if len(input_ids) != max_length or len(attention_mask) != max_length:
                    logging.warning(f"Length mismatch for {input_key}")
                    return False
                valid_token_count = sum(attention_mask)
                if valid_token_count <= 1:
                    logging.warning(
                        f"Too few valid tokens in {input_key}, count: {valid_token_count}"
                    )
                    return False
            return True

        ds = ds.filter(validate_tokens)
        logging.info(f"Dataset after validation: {len(ds)} examples")

        # 保留messages相关列
        remove_columns = [
            col
            for col in ds.column_names
            if not (
                col.startswith("input")
                or col.startswith("attention")
                or col.startswith("label")
                or "messages" in col
                or "_score" in col
            )
        ]
        ds = ds.remove_columns(remove_columns)
        ds.set_format(type="torch")
        return ds

    # 如果提供了蒸馏数据集，准备复合数据集
    if script_args.distill_dataset:
        # 加载偏好数据集用于RM训练
        print("Loading preference dataset for reward modeling...")
        rm_train_dataset = custom_load_train_eval_dataset(
            script_args.dataset,
            tokenizer,
            split="train",
            size=100 if script_args.debug else None,
            max_length=script_args.max_length,
        )

        # 加载蒸馏数据集用于KL/SFT训练
        print("Loading distillation dataset for KL/SFT training...")
        distill_train_dataset = custom_load_train_eval_dataset(
            script_args.distill_dataset,
            tokenizer,
            split="train",
            size=100 if script_args.debug else None,
            max_length=script_args.max_length,
            is_distill_data=True
        )

        # 创建复合数据集（交替使用RM和蒸馏样本）
        class CombinedDataset:
            def __init__(self, rm_dataset, distill_dataset):
                self.rm_dataset = rm_dataset
                self.distill_dataset = distill_dataset
                self.rm_length = len(rm_dataset)
                self.distill_length = len(distill_dataset)
                self.length = max(self.rm_length, self.distill_length)

            def __len__(self):
                return self.length

            def __getitem__(self, index):
                # 交替返回RM和蒸馏样本
                if index % 2 == 0:  # 偶数索引返回RM样本
                    rm_idx = index % self.rm_length
                    return self.rm_dataset[rm_idx]
                else:  # 奇数索引返回蒸馏样本
                    distill_idx = index % self.distill_length
                    return self.distill_dataset[distill_idx]

        train_dataset = CombinedDataset(rm_train_dataset, distill_train_dataset)
        print(f"Combined training dataset size: {len(train_dataset)}")

    else:
        # 如果未提供蒸馏数据集，使用单一偏好数据集
        train_dataset = custom_load_train_eval_dataset(
            script_args.dataset,
            tokenizer,
            split="train",
            size=100 if script_args.debug else None,
            max_length=script_args.max_length,
        )

    eval_dataset = custom_load_train_eval_dataset(
        script_args.dataset,
        tokenizer,
        split="test" if "test" in script_args.dataset else "train",
        size=min(1000, len(train_dataset) // 20) if not script_args.debug else 100,
        max_length=script_args.max_length,
    )

    print(
        "Training dataset size: {}, validation dataset size: {}".format(
            len(train_dataset), len(eval_dataset)
        )
    )

    # Model parameters
    model_params = {
        "vhead_layer_type": script_args.layer_type,
        "vhead_num_neurons": script_args.num_neurons,
        "vhead_num_layers": script_args.num_layers,
    }
    if script_args.attn_implementation:
        model_params["attn_implementation"] = script_args.attn_implementation

    # 按需加载教师模型
    teacher_model = None
    if script_args.teacher_model and script_args.kl_weight > 0:
        print(f"Loading teacher model from: {script_args.teacher_model}")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            script_args.teacher_model,
            torch_dtype=torch.bfloat16,
            attn_implementation=(
                "flash_attention_2" if script_args.attn_implementation else None
            ),
        )
        teacher_model.resize_token_embeddings(len(tokenizer))
        teacher_model.config.pad_token_id = tokenizer.pad_token_id
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded and frozen")

    # Load student model (GRM model)
    # reference_model不再使用，因为JointDistillRewardTrainer不调用GRM框架的reference相关逻辑
    reference_model = None

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        script_args.base_model, torch_dtype=torch.bfloat16, **model_params
    )

    model.pretrained_model.resize_token_embeddings(len(tokenizer))
    print_trainable_parameters(model)
    model.config.pad_token_id = tokenizer.pad_token_id

    # Let value head be trainable
    if hasattr(model, "v_head"):
        for parameter in model.v_head.parameters():
            parameter.requires_grad = True
    print_trainable_parameters(model)

    # Create data collator
    data_collator = JointDataCollatorWithPadding(
        tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        max_length=script_args.max_length,
        padding="max_length",
    )

    # Define trainer parameters
    trainer_params = {
        "model": model,
        "reference_model": reference_model,
        "teacher_model": teacher_model,
        "args": training_args,
        "tokenizer": tokenizer,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": data_collator,
        # Loss weights
        "reward_weight": script_args.reward_weight,
        "sft_weight": script_args.sft_weight,
        "kl_weight": script_args.kl_weight,
        "temperature": script_args.temperature,
        # GRM框架参数已被JointDistillRewardTrainer重写的compute_loss绕过，不再使用
        "weight_ratio": 0,  # 设为0以确保不启用GRM框架的加权逻辑
        "reference_free": True,  # 设为True确保不启用参考模型
        "sft_only": True,  # 设为True确保使用SFT逻辑而非DPO
        "no_logsigmoid_sft": script_args.no_logsigmoid_sft,
        "info_to_save": {
            "base_model": script_args.base_model,
            "teacher_model": script_args.teacher_model,
            "layer_type": script_args.layer_type,
            "num_neurons": script_args.num_neurons,
            "num_layers": script_args.num_layers,
            "reward_weight": script_args.reward_weight,
            "sft_weight": script_args.sft_weight,
            "kl_weight": script_args.kl_weight,
            "temperature": script_args.temperature,
        },
    }

    # Train the model
    print("Starting joint training with distillation and reward modeling...")
    trainer = JointDistillRewardTrainer(**trainer_params)

    print("Training start")
    trainer.train()

    # Save model
    print("Saving model...")
    model.config.use_cache = True

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()

    print(f"Model saved to: {output_name}")


if __name__ == "__main__":
    main()
