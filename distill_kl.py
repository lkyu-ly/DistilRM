# -*- coding: utf-8 -*-
import os
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from tqdm import tqdm
from torch.distributed.fsdp import StateDictType, FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP


def safe_item(x):
    return x.item() if isinstance(x, torch.Tensor) else float(x)


# ====================== 数据集 ======================
class JsonlDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_length=512, teacher_tokenizer=None):
        self.examples = []
        self.tokenizer = tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        # 首先统计文件行数以显示进度条
        with open(file_path, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)

        # 处理数据并显示进度条
        with open(file_path, "r", encoding="utf-8") as f:
            for line in tqdm(f, total=line_count, desc="处理训练数据"):
                data = json.loads(line)
                messages = [
                    {"role": "user", "content": data["question"]},
                    {"role": "assistant", "content": data["response"]},
                ]

                # 学生模型输入
                student_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                student_tokenized = tokenizer(
                    student_text,
                    truncation=True,
                    max_length=max_length,
                    padding="max_length",
                )

                # 找到学生模型的 assistant 开始位置
                user_only_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": data["question"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                user_only_ids = tokenizer(
                    user_only_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length,
                )["input_ids"]
                student_prefix_len = len(user_only_ids)

                # 教师模型输入
                if teacher_tokenizer is not None:
                    teacher_text = teacher_tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False,
                        enable_thinking=False,
                    )
                    teacher_tokenized = teacher_tokenizer(
                        teacher_text,
                        truncation=True,
                        max_length=max_length,
                        padding="max_length",
                    )

                    # 找到教师模型的 assistant 开始位置
                    teacher_user_only_text = teacher_tokenizer.apply_chat_template(
                        [{"role": "user", "content": data["question"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    teacher_user_only_ids = teacher_tokenizer(
                        teacher_user_only_text,
                        add_special_tokens=False,
                        truncation=True,
                        max_length=max_length,
                    )["input_ids"]
                    teacher_prefix_len = len(teacher_user_only_ids)
                else:
                    teacher_tokenized = student_tokenized
                    teacher_prefix_len = student_prefix_len

                # 构造 labels（基于学生模型）
                labels = student_tokenized["input_ids"][:]
                labels[:student_prefix_len] = [-100] * student_prefix_len

                # 保存所有字段，包括前缀长度
                example = {
                    "input_ids": student_tokenized["input_ids"],
                    "attention_mask": student_tokenized["attention_mask"],
                    "labels": labels,
                    "teacher_input_ids": teacher_tokenized["input_ids"],
                    "teacher_attention_mask": teacher_tokenized["attention_mask"],
                    "student_prefix_len": student_prefix_len,
                    "teacher_prefix_len": teacher_prefix_len,
                }
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "input_ids": ex["input_ids"],
            "attention_mask": ex["attention_mask"],
            "labels": ex["labels"],
            "teacher_input_ids": ex["teacher_input_ids"],
            "teacher_attention_mask": ex["teacher_attention_mask"],
            "student_prefix_len": ex["student_prefix_len"],
            "teacher_prefix_len": ex["teacher_prefix_len"],
        }


# ====================== Trainer ======================
class DistillTrainer(Trainer):
    def __init__(
        self,
        teacher_model,
        student_vocab_size,
        temperature=1.0,
        lm_weight=1.0,
        kl_weight=0.0,
        teacher_tokenizer=None,
        *args,
        **kwargs,
    ):
        self.teacher_tokenizer = teacher_tokenizer
        super().__init__(*args, **kwargs)

        self.teacher = teacher_model
        if self.teacher is not None:
            self.teacher.eval()
            for p in self.teacher.parameters():
                p.requires_grad = False

        self.student_vocab_size = student_vocab_size
        self.temperature = temperature
        self.eps = 1e-9
        self.lm_weight = lm_weight
        self.kl_weight = kl_weight

        # 保存 tokenizer 和 dataset，用于生成测试
        self.tokenizer = kwargs.get("tokenizer", None)
        self.train_dataset = kwargs.get("train_dataset", None)
        if self.tokenizer is None:
            raise ValueError("请传入 tokenizer 到 Trainer！")

    # ---------- 设备同步 ----------
    def _sync_teacher_device(self, model):
        """
        检查教师模型是否与学生模型分片在同一设备上。
        在 FSDP 中，我们假设教师模型在 Trainer 初始化时已经被正确放置。
        """
        if self.teacher is None:
            return

        try:
            # 获取当前输入数据所在的设备 (即当前 FSDP 分片所在的设备)
            model_device = next(model.parameters()).device
        except StopIteration:
            # 模型可能没有参数（例如在某些初始化阶段）
            return

        # 检查教师模型参数是否在正确的设备上
        teacher_device = next(self.teacher.parameters()).device

        if teacher_device != model_device:
            # 仅发出警告，不再尝试移动模型，因为那会导致 FSDP 死锁
            print(
                f"警告: 教师模型设备 ({teacher_device}) 与学生模型分片设备 ({model_device}) 不一致。"
                "在 FSDP 模式下，请确保在训练前将教师模型移至正确的 local_rank。"
            )

    # ---------- 计算 loss ----------

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        device = next(model.parameters()).device
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs.get("attention_mask", None)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        # 先获取 labels
        labels = inputs["labels"].to(device)

        batch_size = input_ids.shape[0]

        # 获取预计算的前缀长度
        student_prefix_lens = inputs["student_prefix_len"]
        teacher_prefix_lens = inputs["teacher_prefix_len"]

        # ----------- LM-SFT loss ----------
        s_out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if self.lm_weight > 0 else None,
        )

        student_logits = s_out.logits
        lm_loss = s_out.loss if self.lm_weight > 0 else torch.tensor(0.0, device=device)

        # ----------- KL(teacher||student) ----------
        kl_loss = 0.0
        if self.kl_weight > 0 and self.teacher is not None:
            self._sync_teacher_device(model)

            teacher_input_ids = inputs["teacher_input_ids"].to(device)
            teacher_attention_mask = inputs.get("teacher_attention_mask", None)
            if teacher_attention_mask is not None:
                teacher_attention_mask = teacher_attention_mask.to(device)

            with torch.no_grad():
                t_out = self.teacher(
                    input_ids=teacher_input_ids, attention_mask=teacher_attention_mask
                )
                teacher_logits = t_out.logits

            T = float(self.temperature)

            # 对每个样本进行前缀掩码和对齐
            batch_kl_losses = []

            for b in range(batch_size):
                teacher_start = teacher_prefix_lens[b]
                student_start = student_prefix_lens[b]

                # 截取可对齐长度
                teacher_available_len = teacher_logits.shape[1] - teacher_start - 1
                student_available_len = student_logits.shape[1] - student_start - 1
                min_len = min(teacher_available_len, student_available_len)

                if min_len <= 0:
                    batch_kl_losses.append(torch.tensor(0.0, device=device))
                    continue

                teacher_slice = teacher_logits[
                    b : b + 1, teacher_start : teacher_start + min_len, :
                ]
                student_slice = student_logits[
                    b : b + 1, student_start : student_start + min_len, :
                ]

                t_log_prob = F.log_softmax(teacher_slice / T, dim=-1)
                s_log_prob = F.log_softmax(student_slice / T, dim=-1)
                t_prob = torch.exp(t_log_prob)

                per_elem = t_prob * (t_log_prob - s_log_prob)
                per_token_kl = per_elem.sum(dim=-1)

                # 使用 labels 进行 mask
                shift_labels = labels[
                    b : b + 1, student_start + 1 : student_start + 1 + min_len
                ]
                kl_mask = (shift_labels != -100).to(dtype=per_token_kl.dtype)

                total_nonpad = kl_mask.sum()

                if total_nonpad.item() == 0:
                    sample_kl_loss = torch.tensor(0.0, device=device)
                else:
                    sample_kl_loss = (per_token_kl * kl_mask).sum() / (
                        total_nonpad + self.eps
                    )

                batch_kl_losses.append(sample_kl_loss * (T * T))

            if batch_kl_losses:
                kl_loss = torch.stack(batch_kl_losses).mean()
            else:
                kl_loss = torch.tensor(0.0, device=device)

        # ----------- 加权 ----------
        loss = self.lm_weight * lm_loss + self.kl_weight * kl_loss

        return (loss, s_out.logits) if return_outputs else loss


# ====================== main ======================
def main(
    student_name,
    teacher_name,
    data_path,
    output_dir,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    num_epochs=1,
    batch_size=2,
    max_length=512,
    lm_weight=1.0,
    kl_weight=0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16
    )
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is not None:
        local_rank = int(local_rank_str)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("未检测到 LOCAL_RANK，假定单卡或非 FSDP 启动。")

    # ----------- 按需加载教师模型 ----------
    teacher = None
    teacher_tokenizer = None
    if kl_weight > 0:
        print(f"kl_weight={kl_weight}，正在加载教师模型：{teacher_name}")

        # 加载教师模型
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name,
            torch_dtype=torch.float16,
        )
        teacher.eval()

        print(f"将教师模型移动到设备: {device}")
        teacher.to(device)

        for p in teacher.parameters():
            p.requires_grad = False
    else:
        print("kl_weight=0，跳过教师模型加载，仅做 SFT")

    dataset = JsonlDataset(
        tokenizer, data_path, max_length=max_length, teacher_tokenizer=teacher_tokenizer
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=1,
        save_strategy="no",
        save_total_limit=0,
        save_only_model=True,
        bf16=True,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        gradient_accumulation_steps=gradient_accumulation_steps,
    )

    vocab_size = len(tokenizer.get_vocab())
    trainer = DistillTrainer(
        model=student,
        teacher_model=teacher,
        student_vocab_size=vocab_size,
        lm_weight=lm_weight,
        kl_weight=kl_weight,
        temperature=1.0,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        teacher_tokenizer=teacher_tokenizer,
    )

    trainer.train()

    # Save model
    student.config.use_cache = True
    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


# ====================== 启动 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./distilled_student")
    parser.add_argument(
        "--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lm_weight", type=float, default=0.0, help="SFT loss 权重")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL loss 权重")
    args = parser.parse_args()

    main(
        args.student_model,
        args.teacher_model,
        args.data_path,
        args.output_dir,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        max_length=args.max_length,
        lm_weight=args.lm_weight,
        kl_weight=args.kl_weight,
    )
