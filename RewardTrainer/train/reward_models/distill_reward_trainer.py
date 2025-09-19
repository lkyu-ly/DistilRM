from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import torch
from base_trainer import RewardTrainer
from transformers.utils import PaddingStrategy
from transformers import AutoTokenizer
import torch.nn as nn

import torch


@dataclass
class DistilRewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = "max_length"
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_chosen"],
                    "attention_mask": feature["attention_mask_chosen"],
                    "score": feature['chosen_score'] if 'chosen_score' in feature.keys() else 0,
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_rejected"],
                    "attention_mask": feature["attention_mask_rejected"],
                    "score": feature['rejected_score'] if 'rejected_score' in feature.keys() else 0,
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
            "score": batch['score'],
        }
        return batch


class DistilRewardTrainer(RewardTrainer):
    def __init__(self, **kwargs):
        super(DistilRewardTrainer, self).__init__(**kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])["logits"]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]

        scores = inputs["score"]
        chosen_scores = scores[jidx]
        rejected_scores = scores[kidx]

        loss = (- nn.functional.logsigmoid(rewards_j - rewards_k) + 0.01 * (rewards_j - rewards_k - 0.5 * (torch.tensor(
            chosen_scores, device=chosen_scores.device).view(-1, 1) - torch.tensor(
                rejected_scores, device=rejected_scores.device).view(-1, 1))) ** 2).mean()

        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss
