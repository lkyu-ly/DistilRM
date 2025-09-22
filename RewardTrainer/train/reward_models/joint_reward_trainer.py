"""
Fixed Joint Distill Reward Trainer
联合奖励模型和知识蒸馏训练器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from grm_reward_trainer import GRMDataCollatorWithPadding, GRMRewardTrainer

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer


@dataclass
class JointDataCollatorWithPadding(GRMDataCollatorWithPadding):
    """简化版本的联合数据collator，只处理必要的功能"""

    teacher_tokenizer: Optional[AutoTokenizer] = None

    def get_user_prefix_length(self, tokenizer, messages):
        """计算用户前缀长度"""
        if not messages or len(messages) == 0:
            return 0

        # 获取用户对话部分
        user_messages = []
        for msg in messages[:-1]:  # 除了最后一条assistant消息
            user_messages.append(msg)

        if not user_messages:
            return 0

        user_text = tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        user_tokens = tokenizer(
            user_text,
            add_special_tokens=False,
            truncation=True,
            max_length=tokenizer.model_max_length or 1024,
            return_tensors=None,
        )
        return len(user_tokens["input_ids"])

    def process_teacher_input(self, chosen_messages):
        """处理教师模型输入"""
        if not self.teacher_tokenizer or not chosen_messages:
            return None, 0

        if len(chosen_messages) < 2:
            return None, 0

        try:
            # 生成完整对话的tokens
            teacher_text = self.teacher_tokenizer.apply_chat_template(
                chosen_messages,
                tokenize=False,
                add_generation_prompt=False,
                enable_thinking=False,
            )

            teacher_tokens = self.teacher_tokenizer(
                teacher_text,
                truncation=True,
                max_length=self.max_length or 1024,
                padding="max_length",
                return_tensors="pt",
            )

            # 计算用户前缀长度
            teacher_prefix_len = self.get_user_prefix_length(
                self.teacher_tokenizer, chosen_messages
            )

            return {
                "input_ids": teacher_tokens["input_ids"][0],
                "attention_mask": teacher_tokens["attention_mask"][0],
            }, teacher_prefix_len

        except Exception as e:
            print(f"Error processing teacher input: {e}")
            # 出错时返回默认值
            fake_ids = torch.zeros(self.max_length or 512, dtype=torch.long)
            fake_mask = torch.zeros(self.max_length or 512, dtype=torch.long)
            return {"input_ids": fake_ids, "attention_mask": fake_mask}, 0

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        """主要的collator函数"""
        # 先调用父类方法获取基本的批次数据
        batch = super().__call__(features)

        if self.teacher_tokenizer is None:
            return batch

        device = batch["input_ids"].device

        # 处理chosen样本（用于蒸馏的只有chosen样本）
        teacher_features = []
        student_prefix_lens = []
        teacher_prefix_lens = []

        # features是成对出现的：0=chosen, 1=rejected, 2=chosen, 3=rejected...
        for i, feature in enumerate(features):
            if i % 2 == 0:  # chosen样本
                # 获取chosen的对话消息
                if "chosen_messages" in feature:
                    messages = feature["chosen_messages"]
                else:
                    messages = []

                # 计算学生模型前缀长度
                s_prefix_len = 0
                if messages:
                    s_prefix_len = self.get_user_prefix_length(self.tokenizer, messages)

                student_prefix_lens.append(s_prefix_len)

                # 检查是否是蒸馏数据（rejected为空的情况）
                is_distill_data = feature.get("is_distill_data", False)

                # 处理教师模型输入
                teacher_result, t_prefix_len = self.process_teacher_input(messages)

                if teacher_result:
                    teacher_features.append(teacher_result)
                    teacher_prefix_lens.append(t_prefix_len)
                else:
                    # 如果没有有效的教师输入，使用学生输入作为占位符
                    teacher_features.append(
                        {
                            "input_ids": feature["input_ids_chosen"],
                            "attention_mask": feature["attention_mask_chosen"],
                        }
                    )
                    teacher_prefix_lens.append(s_prefix_len)

                # 标记蒸馏数据以便训练器处理
                feature["is_distill_data"] = is_distill_data

        # 对教师输入进行padding
        if teacher_features:
            teacher_batch = self.teacher_tokenizer.pad(
                teacher_features,
                padding=self.padding,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_tensors="pt",
            )

            # 添加到批次
            batch.update(
                {
                    "teacher_input_ids": teacher_batch["input_ids"].to(device),
                    "teacher_attention_mask": teacher_batch["attention_mask"].to(
                        device
                    ),
                    "student_prefix_len": torch.tensor(
                        student_prefix_lens, dtype=torch.long
                    ).to(device),
                    "teacher_prefix_len": torch.tensor(
                        teacher_prefix_lens, dtype=torch.long
                    ).to(device),
                }
            )
        else:
            # 如果没有有效的教师特征，使用默认值
            chosen_count = len([i for i in range(len(features)) if i % 2 == 0])
            fake_ids = torch.zeros(
                (chosen_count, self.max_length or 512), dtype=torch.long
            ).to(device)
            fake_mask = torch.zeros(
                (chosen_count, self.max_length or 512), dtype=torch.long
            ).to(device)

            batch.update(
                {
                    "teacher_input_ids": fake_ids,
                    "teacher_attention_mask": fake_mask,
                    "student_prefix_len": torch.zeros(
                        chosen_count, dtype=torch.long
                    ).to(device),
                    "teacher_prefix_len": torch.zeros(
                        chosen_count, dtype=torch.long
                    ).to(device),
                }
            )

        return batch


class JointDistillRewardTrainer(GRMRewardTrainer):
    """
    联合训练奖励模型和知识蒸馏的新训练器
    同时优化：奖励损失 + SFT损失 + KL蒸馏损失
    """

    def __init__(self, **kwargs):
        # 提取联合训练相关的参数
        self.teacher_model = kwargs.pop("teacher_model", None)
        self.reward_weight = kwargs.pop("reward_weight", 1.0)
        self.sft_weight = kwargs.pop("sft_weight", 0.1)
        self.kl_weight = kwargs.pop("kl_weight", 0.1)
        self.temperature = kwargs.pop("temperature", 1.0)

        # 调用父类构造函数
        super().__init__(**kwargs)

        # 设置教师模型
        if self.teacher_model is not None:
            self.teacher_model.eval()
            for p in self.teacher_model.parameters():
                p.requires_grad = False

        self.eps = 1e-9

    def compute_kl_loss(
        self,
        student_logits,
        teacher_logits,
        student_prefix_len,
        teacher_prefix_len,
        labels,
    ):
        """
        计算KL散度蒸馏损失，基于distill_kl.py的实现
        """
        device = student_logits.device
        batch_size = student_logits.shape[0]

        kl_losses = []

        for b in range(batch_size):
            s_prefix = (
                student_prefix_len[b].item()
                if hasattr(student_prefix_len[b], "item")
                else student_prefix_len[b]
            )
            t_prefix = (
                teacher_prefix_len[b].item()
                if hasattr(teacher_prefix_len[b], "item")
                else teacher_prefix_len[b]
            )

            # 计算可对齐长度
            teacher_available_len = teacher_logits.shape[1] - t_prefix - 1
            student_available_len = student_logits.shape[1] - s_prefix - 1
            min_len = min(teacher_available_len, student_available_len)

            if min_len <= 0:
                kl_losses.append(torch.tensor(0.0, device=device))
                continue

            # 截取assistant响应部分进行对齐
            teacher_slice = teacher_logits[b : b + 1, t_prefix : t_prefix + min_len, :]
            student_slice = student_logits[b : b + 1, s_prefix : s_prefix + min_len, :]

            T = float(self.temperature)

            # 计算softmax和log_softmax
            t_log_prob = F.log_softmax(teacher_slice / T, dim=-1)
            s_log_prob = F.log_softmax(student_slice / T, dim=-1)
            t_prob = torch.exp(t_log_prob)

            # 计算KL散度
            per_elem = t_prob * (t_log_prob - s_log_prob)
            per_token_kl = per_elem.sum(dim=-1)

            # 应用mask：只计算非padding部分的损失
            if labels.dim() == 2 and labels.shape[0] == batch_size:
                shift_labels = labels[b : b + 1, s_prefix + 1 : s_prefix + 1 + min_len]
                kl_mask = (shift_labels != -100).to(dtype=per_token_kl.dtype)
                total_nonpad = kl_mask.sum()

                if total_nonpad.item() == 0:
                    kl_loss = torch.tensor(0.0, device=device)
                else:
                    kl_loss = (per_token_kl * kl_mask).sum() / (total_nonpad + self.eps)
            else:
                # 如果没有labels，使用简单平均
                kl_loss = per_token_kl.mean()

            # 乘以温度平方（蒸馏中的标准做法）
            kl_losses.append(kl_loss * (T * T))

        if kl_losses:
            return torch.stack(kl_losses).mean()
        else:
            return torch.tensor(0.0, device=device)

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        联合训练的核心损失计算
        同时计算：奖励损失 + SFT损失 + KL蒸馏损失
        """
        device = next(model.parameters()).device

        # 获取学生模型输出 (使用GRM模型)(lm_logits, _, value)
        lm_logits, _, rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )

        batch_size = rewards.size(0)
        jidx = torch.arange(0, batch_size, 2)  # chosen样本索引
        kidx = jidx + 1  # rejected样本索引

        # 检测蒸馏样本（rejected样本与chosen样本相同的情况）
        distill_indices = []
        rm_indices = []

        for i in range(len(jidx)):
            chosen_idx = jidx[i]
            rejected_idx = kidx[i]

            # 检查rejected样本是否与chosen样本相同（蒸馏数据特征）
            is_distill = torch.equal(inputs["input_ids"][chosen_idx], inputs["input_ids"][rejected_idx])
            if is_distill:
                distill_indices.append(i)
            else:
                rm_indices.append(i)

        # 1. 计算奖励损失 (Reward Loss) - 只对非蒸馏样本
        if rm_indices:
            rm_jidx = jidx[rm_indices]
            rm_kidx = kidx[rm_indices]
            reward_loss = -nn.functional.logsigmoid(rewards[rm_jidx] - rewards[rm_kidx]).mean()
        else:
            # 如果没有非蒸馏样本，奖励损失为0
            reward_loss = torch.tensor(0.0, device=device)

        # 2. 计算SFT损失 (对所有chosen样本，包括蒸馏样本)
        if self.sft_weight > 0:
            logps = self.get_batch_logps(lm_logits, inputs["label"])
            sft_loss = -logps[jidx].mean()  # 使用所有chosen样本的log概率
        else:
            sft_loss = torch.tensor(0.0, device=device)

        # 3. 计算KL蒸馏损失 (仅对chosen样本)
        kl_loss = torch.tensor(0.0, device=device)
        if self.kl_weight > 0 and self.teacher_model is not None:
            with torch.no_grad():
                # 获取教师模型输出
                teacher_output = self.teacher_model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"],
                )
                teacher_lm_logits = teacher_output.logits

            # 只对学生模型中的chosen样本计算KL损失
            chosen_student_logits = lm_logits[jidx]  # 只取chosen样本
            kl_loss = self.compute_kl_loss(
                chosen_student_logits,
                teacher_lm_logits,
                inputs["student_prefix_len"],
                inputs["teacher_prefix_len"],
                inputs["label"][jidx],
            )

        # 4. 组合总损失
        total_loss = (
            self.reward_weight * reward_loss
            + self.sft_weight * sft_loss
            + self.kl_weight * kl_loss
        )

        # 记录详细的损失信息用于监控
        if hasattr(self, "state") and hasattr(self.state, "global_step"):
            if self.state.global_step % max(1, self.args.logging_steps) == 0:
                logs = {
                    "loss_reward": reward_loss.item(),
                    "loss_sft": sft_loss.item(),
                    "loss_kl": (
                        kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
                    ),
                    "loss_total": total_loss.item(),
                    "num_distill_samples": len(distill_indices),
                    "num_rm_samples": len(rm_indices),
                }
                self.log(logs)

        if return_outputs:
            return total_loss, {
                "rewards": rewards,
                "lm_logits": lm_logits,
                "reward_loss": reward_loss,
                "sft_loss": sft_loss,
                "kl_loss": kl_loss,
            }

        return total_loss
