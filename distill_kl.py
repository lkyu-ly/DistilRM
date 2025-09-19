# -*- coding: utf-8 -*-
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
        if self.teacher is None:
            return
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hasattr(self.teacher, "device") and self.teacher.device != model_device:
            self.teacher.to(model_device)
        else:
            for p in self.teacher.parameters():
                if p.device != model_device:
                    self.teacher.to(model_device)
                    break
            for b in self.teacher.buffers():
                if b.device != model_device:
                    self.teacher.to(model_device)
                    break

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

        # 只 forward 一次学生模型
        s_out = model(input_ids=input_ids, attention_mask=attention_mask)
        student_logits = s_out.logits

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

            # 添加调试信息
            if self.args.local_rank in (-1, 0) and self.state.global_step % 5 == 0:
                print("\nKL 散度计算调试信息:")

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

                # 调试：检查切片的统计信息
                if (
                    self.args.local_rank in (-1, 0)
                    and self.state.global_step % 5 == 0
                    and b == 0
                ):
                    print(
                        f"  样本{b}: teacher_slice shape={teacher_slice.shape}, student_slice shape={student_slice.shape}"
                    )
                    print(
                        f"  teacher logits 统计: mean={teacher_slice.mean().item():.4f}, std={teacher_slice.std().item():.4f}"
                    )
                    print(
                        f"  student logits 统计: mean={student_slice.mean().item():.4f}, std={student_slice.std().item():.4f}"
                    )

                t_log_prob = F.log_softmax(teacher_slice / T, dim=-1)
                s_log_prob = F.log_softmax(student_slice / T, dim=-1)
                t_prob = torch.exp(t_log_prob)

                per_elem = t_prob * (t_log_prob - s_log_prob)
                per_token_kl = per_elem.sum(dim=-1)

                # 使用 labels 进行 mask
                shift_labels = labels[
                    b : b + 1, student_start + 1 : student_start + 1 + min_len
                ]
                kl_mask = (shift_labels != -100).to(per_token_kl.dtype)

                total_nonpad = kl_mask.sum()

                # 调试：检查 KL 散度值
                if (
                    self.args.local_rank in (-1, 0)
                    and self.state.global_step % 5 == 0
                    and b == 0
                ):
                    print(f"  per_token_kl 前 10 个值: {per_token_kl[0, :10].tolist()}")
                    print(f"  kl_mask 前 10 个值: {kl_mask[0, :10].tolist()}")
                    print(f"  有效 token 数: {total_nonpad.item()}/{min_len}")

                    # 检查是否有异常大的 KL 值
                    max_kl = per_token_kl.max().item()
                    mean_kl = per_token_kl.mean().item()
                    print(f"  KL 散度统计: max={max_kl:.4f}, mean={mean_kl:.4f}")

                    # 检查教师和学生的预测是否过于不同
                    t_argmax = teacher_slice[0, :10].argmax(-1)
                    s_argmax = student_slice[0, :10].argmax(-1)
                    print(f"  教师前 10 个 token 预测: {t_argmax.tolist()}")
                    print(f"  学生前 10 个 token 预测: {s_argmax.tolist()}")

                    # 检查概率分布的熵
                    t_entropy = -(t_prob[0, :10] * t_log_prob[0, :10]).sum(-1)
                    s_entropy = -(
                        torch.exp(s_log_prob[0, :10]) * s_log_prob[0, :10]
                    ).sum(-1)
                    print(f"  教师熵: {t_entropy.mean().item():.4f}")
                    print(f"  学生熵: {s_entropy.mean().item():.4f}")

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

        # ----------- LM-SFT loss ----------
        lm_loss = 0.0
        if self.lm_weight > 0:
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                labels=labels,
            )
            lm_loss = outputs.loss

        # ----------- 加权 ----------
        loss = self.lm_weight * lm_loss + self.kl_weight * kl_loss
        # 在调试部分添加更多检查
        if self.args.local_rank in (-1, 0):
            print(f"Loss requires_grad: {loss.requires_grad}")
            b = 0
            # 找到 answer 开始位置
            ans_start = (labels[b] != -100).nonzero(as_tuple=True)[0]
            if ans_start.numel() > 0:
                ans_start = ans_start[0].item()
                L = 15
                print("\n" + "-" * 80)
                print(f">>> STEP {self.state.global_step}  ANSWER 区段  <<<")

                # 1. 检查模型设备和数据类型
                print(f"Model device: {next(model.parameters()).device}")
                print(f"Model dtype: {next(model.parameters()).dtype}")
                print(f"Input device: {input_ids.device}")
                print(f"Logits shape: {student_logits.shape}")
                print(f"Logits dtype: {student_logits.dtype}")

                # 2. 检查 logits 的统计信息
                print(
                    f"Logits stats - mean: {student_logits.mean().item():.4f}, std: {student_logits.std().item():.4f}"
                )
                print(
                    f"Logits min: {student_logits.min().item():.4f}, max: {student_logits.max().item():.4f}"
                )

                # 3. 正确处理 shift
                if ans_start > 0:  # 确保不会越界
                    # logits[i] 预测 input_ids[i+1]
                    print(
                        "input_ids (被预测):",
                        input_ids[b, ans_start : ans_start + L].tolist(),
                    )
                    print(
                        "labels           :",
                        labels[b, ans_start : ans_start + L].tolist(),
                    )

                    # 使用 ans_start-1 的 logits 来预测 ans_start 的 token
                    logits_slice = student_logits[b, ans_start - 1 : ans_start + L - 1]
                    probs = F.softmax(logits_slice, dim=-1)

                    # 检查 softmax 后的概率
                    print(f"Probs shape: {probs.shape}")
                    print(f"Probs sum (should be ~1.0): {probs[0].sum().item():.4f}")

                    # 模型预测的 token
                    pred_ids = logits_slice.argmax(-1)

                    # 获取真实标签的概率
                    target_ids = input_ids[b, ans_start : ans_start + L]
                    label_probs = []
                    for i in range(min(L, len(target_ids))):
                        if i < probs.shape[0]:
                            prob = probs[i, target_ids[i]].item()
                            label_probs.append(prob)

                    print("模型预测     :", pred_ids.tolist()[: len(label_probs)])
                    print("真实标签概率 :", [f"{p:.3f}" for p in label_probs])

                    # 4. 检查前几个最高概率的预测
                    top_k = 5
                    top_probs, top_indices = torch.topk(probs[0], top_k)
                    print(f"Top {top_k} predictions for first position:")
                    print(f"  Indices: {top_indices.tolist()}")
                    print(f"  Probs: {[f'{p:.3f}' for p in top_probs.tolist()]}")
                    print(f"  Target token: {target_ids[0].item()}")

                print("-" * 80 + "\n")
        # ---------- 调试信息 ----------
        if self.args.local_rank in (-1, 0):
            from datetime import datetime

            ts = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{ts}] step={self.state.global_step}  "
                f"lm_loss={safe_item(lm_loss):.6f}  kl_loss={safe_item(kl_loss):.6f}  "
                f"total_loss={safe_item(loss):.6f}"
            )

            # 每 10 步打印详细调试信息
            if self.state.global_step % 10 == 0:
                # 检查 labels 中非 -100 的数量
                valid_labels = (inputs["labels"] != -100).sum().item()
                total_labels = inputs["labels"].numel()
                print(
                    f"Valid labels: {valid_labels}/{total_labels} ({valid_labels/total_labels*100:.1f}%)"
                )

                # 检查前缀长度差异
                prefix_diff = [
                    t - s
                    for t, s in zip(
                        teacher_prefix_lens.tolist(), student_prefix_lens.tolist()
                    )
                ]
                print(f"前缀长度差异: {prefix_diff[:5]}...")

        if (
            self.kl_weight > 0
            and self.teacher is not None
            and self.state.global_step % 10 == 0
        ):
            print(
                "🧑‍🏫 Teacher 最后 5 个位置 argmax:",
                teacher_logits[0, -5:].argmax(-1).tolist(),
            )
            print(
                "🧑‍🎓 Student 最后 5 个位置 argmax:",
                student_logits[0, -5:].argmax(-1).tolist(),
            )

        # ======  对齐文本核对  ======
        if (
            self.args.local_rank in (-1, 0)
            and self.state.global_step % 10 == 0
            and self.kl_weight > 0
        ):
            b = 0  # 打印第一个样本
            teacher_start = teacher_prefix_lens[b]
            student_start = student_prefix_lens[b]

            teacher_available_len = teacher_logits.shape[1] - teacher_start - 1
            student_available_len = student_logits.shape[1] - student_start - 1
            min_len = min(teacher_available_len, student_available_len)

            print("\n" + "=" * 80)
            print(
                f"step={self.state.global_step}  teacher_start={teacher_start}, student_start={student_start}, min_len={min_len}"
            )

            if min_len > 0:
                t_text = self.teacher_tokenizer.decode(
                    teacher_input_ids[
                        b, teacher_start : teacher_start + min(20, min_len)
                    ],
                    skip_special_tokens=False,
                )
                s_text = self.tokenizer.decode(
                    input_ids[b, student_start : student_start + min(20, min_len)],
                    skip_special_tokens=False,
                )
                print("教师片段（前 20 个 token）：", repr(t_text))
                print("学生片段（前 20 个 token）：", repr(s_text))
            print("=" * 80 + "\n")

        return (loss, s_out.logits) if return_outputs else loss


# ====================== main ======================
def main(
    student_name,
    teacher_name,
    data_path,
    output_dir,
    ds_config,
    num_epochs=1,
    batch_size=2,
    gradient_accumulation_steps=1,
    max_length=512,
    lm_weight=1.0,
    kl_weight=0.0,
):
    tokenizer = AutoTokenizer.from_pretrained(student_name, use_fast=True)
    student = AutoModelForCausalLM.from_pretrained(
        student_name, torch_dtype=torch.float16
    )

    # ----------- 按需加载教师模型 ----------
    teacher = None
    teacher_tokenizer = None
    if kl_weight > 0:
        print(f"kl_weight={kl_weight}，正在加载教师模型：{teacher_name}")
        teacher = AutoModelForCausalLM.from_pretrained(
            teacher_name, torch_dtype=torch.float16
        ).eval()
        teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_name, use_fast=True)
    else:
        print("kl_weight=0，跳过教师模型加载，仅做 SFT")

    dataset = JsonlDataset(
        tokenizer, data_path, max_length=max_length, teacher_tokenizer=teacher_tokenizer
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=1.0e-6,
        logging_steps=1,
        save_strategy="steps",
        save_total_limit=1,
        bf16=True,
        deepspeed=ds_config,
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


# ====================== 启动 ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_model", type=str, required=True)
    parser.add_argument("--teacher_model", type=str, default="")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./distilled_student")
    parser.add_argument(
        "--ds_config", type=str, required=True, help="DeepSpeed 配置 JSON"
    )
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--lm_weight", type=float, default=0.0, help="SFT loss 权重")
    parser.add_argument("--kl_weight", type=float, default=1.0, help="KL loss 权重")
    args = parser.parse_args()

    main(
        args.student_model,
        args.teacher_model,
        args.data_path,
        args.output_dir,
        args.ds_config,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        lm_weight=args.lm_weight,
        kl_weight=args.kl_weight,
    )
