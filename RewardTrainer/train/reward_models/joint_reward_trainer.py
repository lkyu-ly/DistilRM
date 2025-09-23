from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
from accelerate import Accelerator
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
from utils import get_trainable_weights


@dataclass
class JointDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    label_pad_token_id = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 处理 RM 训练部分：chosen 和 rejected
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        # pad labels for RM part
        paded_length = batch["input_ids"].shape[1]
        label_paded = []
        for feature in features:
            label_chosen_paded = torch.tensor(feature["label_chosen"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_chosen"])) , dtype=torch.int64)
            label_rejected_paded = torch.tensor(feature["label_rejected"].tolist() + [self.label_pad_token_id] * (paded_length - len(feature["label_rejected"])) , dtype=torch.int64)
            label_paded.extend([label_chosen_paded.view(1, -1), label_rejected_paded.view(1, -1)])
        label_paded = torch.concatenate(label_paded, dim=0)

        # 处理蒸馏部分：teacher response
        teacher_features = []
        teacher_attention_masks = []
        teacher_prefix_lengths = []
        student_prefix_lengths = []

        for feature in features:
            # teacher response for distillation
            teacher_features.append({
                "input_ids": feature["teacher_input_ids"],
                "attention_mask": feature["teacher_attention_mask"]
            })
            teacher_prefix_lengths.append(feature["teacher_prefix_len"])
            student_prefix_lengths.append(feature["student_prefix_len"])

        teacher_batch = self.tokenizer.pad(
            teacher_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )

        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "label": label_paded,
            # 蒸馏相关字段
            "teacher_input_ids": teacher_batch["input_ids"],
            "teacher_attention_mask": teacher_batch["attention_mask"],
            "teacher_prefix_len": teacher_prefix_lengths,
            "student_prefix_len": student_prefix_lengths,
        }
        return batch


class JointRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        # RM 训练相关参数
        self.use_lora = kwargs.pop('use_lora', True)
        self.info_to_save = kwargs.pop('info_to_save', {})

        # 蒸馏相关参数
        self.teacher_model = kwargs.pop('teacher_model', None)
        self.temperature = kwargs.pop('temperature', 1.0)
        self.eps = 1e-9

        # 损失权重
        self.reward_weight = kwargs.pop('reward_weight', 1.0)
        self.sft_weight = kwargs.pop('sft_weight', 1.0)
        self.kl_weight = kwargs.pop('kl_weight', 1.0)

        self.label_pad_token_id = -100
        # 初始化全局步数计数器
        self.global_step = 0
        super(JointRewardTrainer, self).__init__(**kwargs)


    def get_batch_logps(
        self,
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
    ) -> torch.FloatTensor:
        """Compute the log probabilities of the given labels under the given logits."""
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        labels = labels[:, 1:].clone()
        logits = logits[:, :-1, :]
        loss_mask = labels != self.label_pad_token_id

        labels[labels == self.label_pad_token_id] = 0
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
        return (per_token_logps * loss_mask).sum(-1)


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 获取模型输出: logits, last_hidden_state, rewards
        logits, last_hidden_state, rewards = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"]
        )

        # ========== 1. RM Loss ==========
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)  # chosen_ids
        kidx = jidx + 1                 # rejected_ids
        reward_loss = -nn.functional.logsigmoid(rewards[jidx] - rewards[kidx]).mean()

        # ========== 2. SFT Loss ==========
        sft_loss = torch.tensor(0.0, device=rewards.device)
        if self.sft_weight > 0 and self.teacher_model is not None:
            # SFT损失应该学习teacher_response，我们需要基于teacher_input_ids计算学生模型的logits
            batch_size = logits.shape[0] // 2
            batch_sft_losses = []

            for b in range(batch_size):
                # 获取teacher_response的输入序列
                teacher_input_ids_batch = inputs["teacher_input_ids"][b:b+1]
                teacher_attention_mask_batch = inputs["teacher_attention_mask"][b:b+1]
                teacher_prefix_len = inputs["teacher_prefix_len"][b]

                # 计算学生模型在teacher_response上的logits
                with torch.no_grad():
                    # 获取学生模型在teacher_input_ids上的输出
                    sft_logits, _, _ = model(
                        input_ids=teacher_input_ids_batch,
                        attention_mask=teacher_attention_mask_batch
                    )

                # 构建labels：基于teacher_response，prefix部分mask为-100
                labels = teacher_input_ids_batch.clone()
                labels[0, :teacher_prefix_len] = -100  # mask prefix部分

                # 计算SFT损失
                sft_logps = self.get_batch_logps(sft_logits, labels)
                sft_loss_sample = -F.logsigmoid(sft_logps).mean()
                batch_sft_losses.append(sft_loss_sample)

            if batch_sft_losses:
                sft_loss = torch.stack(batch_sft_losses).mean()
            else:
                sft_loss = torch.tensor(0.0, device=logits.device)

        # ========== 3. KL Loss ==========
        kl_loss = torch.tensor(0.0, device=rewards.device)
        if self.kl_weight > 0 and self.teacher_model is not None:
            # KL损失应该基于teacher_response计算教师和学生模型的对齐

            with torch.no_grad():
                # 教师模型前向传播
                teacher_output = self.teacher_model(
                    input_ids=inputs["teacher_input_ids"],
                    attention_mask=inputs["teacher_attention_mask"]
                )
                teacher_logits = teacher_output.logits

            T = float(self.temperature)
            batch_size = logits.shape[0] // 2
            batch_kl_losses = []

            for b in range(batch_size):
                # 获取teacher_response的输入序列
                teacher_input_ids_batch = inputs["teacher_input_ids"][b:b+1]
                teacher_attention_mask_batch = inputs["teacher_attention_mask"][b:b+1]
                teacher_prefix_len = inputs["teacher_prefix_len"][b]
                student_prefix_len = inputs["student_prefix_len"][b]

                # 计算学生模型在teacher_response上的logits
                with torch.no_grad():
                    kl_logits, _, _ = model(
                        input_ids=teacher_input_ids_batch,
                        attention_mask=teacher_attention_mask_batch
                    )

                # 计算可对齐长度 (去除prefix部分，并且考虑logits的shift)
                teacher_available_len = teacher_logits.shape[1] - teacher_prefix_len - 1
                student_available_len = kl_logits.shape[1] - student_prefix_len - 1
                min_len = min(teacher_available_len, student_available_len)

                if min_len <= 0:
                    batch_kl_losses.append(torch.tensor(0.0, device=logits.device))
                    continue

                # 截取对应片段 (去除prefix，并且考虑logits的shift)
                teacher_slice = teacher_logits[b:b+1, teacher_prefix_len:teacher_prefix_len+min_len, :]
                student_slice = kl_logits[b:b+1, student_prefix_len:student_prefix_len+min_len, :]

                # 计算 KL 散度
                t_log_prob = F.log_softmax(teacher_slice / T, dim=-1)
                s_log_prob = F.log_softmax(student_slice / T, dim=-1)
                t_prob = torch.exp(t_log_prob)

                per_elem = t_prob * (t_log_prob - s_log_prob)
                per_token_kl = per_elem.sum(dim=-1)

                # 使用 labels 进行 mask (参照distill_kl.py的做法)
                teacher_response_tokens = teacher_input_ids_batch[0, teacher_prefix_len:teacher_prefix_len+min_len]
                shift_labels = teacher_response_tokens.unsqueeze(0)
                kl_mask = (shift_labels != self.tokenizer.pad_token_id).to(dtype=per_token_kl.dtype)

                total_nonpad = kl_mask.sum()

                if total_nonpad.item() == 0:
                    sample_kl_loss = torch.tensor(0.0, device=logits.device)
                else:
                    sample_kl_loss = (per_token_kl * kl_mask).sum() / (total_nonpad + self.eps)

                batch_kl_losses.append(sample_kl_loss * (T * T))

            if batch_kl_losses:
                kl_loss = torch.stack(batch_kl_losses).mean()
            else:
                kl_loss = torch.tensor(0.0, device=logits.device)

        # ========== 4. 总损失 ==========
        total_loss = (
            self.reward_weight * reward_loss +
            self.sft_weight * sft_loss +
            self.kl_weight * kl_loss
        )

        # 打印损失信息（仅在主进程打印）
        if hasattr(self, 'global_step'):
            self.global_step += 1
        else:
            self.global_step = 1

        if self.accelerator.is_main_process:
            # 转换为CPU标量以便打印
            reward_loss_val = reward_loss.detach().cpu().item()
            sft_loss_val = sft_loss.detach().cpu().item()
            kl_loss_val = kl_loss.detach().cpu().item()
            total_loss_val = total_loss.detach().cpu().item()

            # 计算加权后的各个损失分量
            weighted_reward = self.reward_weight * reward_loss_val
            weighted_sft = self.sft_weight * sft_loss_val
            weighted_kl = self.kl_weight * kl_loss_val

            print(f"\n[Step {self.global_step}] 损失详细信息:")
            print(f"  - RM Loss (raw):     {reward_loss_val:.6f} | Weighted: {weighted_reward:.6f} (weight={self.reward_weight})")
            print(f"  - SFT Loss (raw):    {sft_loss_val:.6f} | Weighted: {weighted_sft:.6f} (weight={self.sft_weight})")
            print(f"  - KL Loss (raw):     {kl_loss_val:.6f} | Weighted: {weighted_kl:.6f} (weight={self.kl_weight})")
            print(f"  - Total Loss:        {total_loss_val:.6f}")
            print(f"  - Loss Components:   RM={weighted_reward/total_loss_val*100:.1f}% | SFT={weighted_sft/total_loss_val*100:.1f}% | KL={weighted_kl/total_loss_val*100:.1f}%")
            print("-" * 80)

        if return_outputs:
            return total_loss, {
                "reward_loss": reward_loss,
                "sft_loss": sft_loss,
                "kl_loss": kl_loss,
                "total_loss": total_loss
            }
        return total_loss


    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            logits, _, rewards = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            logps = self.get_batch_logps(logits, inputs['label'])

        return (None, logps.reshape(-1, 2), rewards.reshape(-1, 2))


    def save_model(self, output_dir=None, _internal_call=False):
        if self.args.should_save and self.accelerator.is_main_process:
            os.makedirs(output_dir, exist_ok=True)
            model = self.accelerator.unwrap_model(self.model)
            ## add config
            model.config.vhead_layer_type = self.info_to_save['layer_type']
            model.config.vhead_num_neurons = self.info_to_save['num_neurons']
            model.config.vhead_num_layers = self.info_to_save['num_layers']

            state_dict = get_trainable_weights(model)
            model.save_pretrained(output_dir, state_dict=state_dict, safe_serialization=False)
            self.tokenizer.save_pretrained(output_dir)