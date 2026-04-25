from __future__ import annotations

from typing import Optional

import torch
from transformers import DataCollatorForTokenClassification


class DataCollatorForNERWithPOS:
    def __init__(
        self,
        tokenizer,
        pos_feature_type: str,
        pos_feature_dim: Optional[int] = None,
    ):
        self.base_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
        self.pos_feature_type = pos_feature_type
        self.pos_feature_dim = pos_feature_dim

    def __call__(self, features):
        if self.pos_feature_type in {"one_hot", "logits"}:
            pos_features = [feature.pop("pos_features") for feature in features]
            batch = self.base_collator(features)

            max_len = batch["input_ids"].shape[1]
            padded_pos_features = []

            for seq in pos_features:
                pad_len = max_len - len(seq)
                padded_seq = seq + [
                    [0.0] * self.pos_feature_dim for _ in range(pad_len)
                ]
                padded_pos_features.append(padded_seq)

            batch["pos_features"] = torch.tensor(padded_pos_features, dtype=torch.float)
            return batch

        if self.pos_feature_type == "trainable_embed":
            pos_ids = [feature.pop("pos_ids") for feature in features]
            batch = self.base_collator(features)

            max_len = batch["input_ids"].shape[1]
            padded_pos_ids = []

            for seq in pos_ids:
                pad_len = max_len - len(seq)
                padded_seq = seq + [-100] * pad_len
                padded_pos_ids.append(padded_seq)

            batch["pos_ids"] = torch.tensor(padded_pos_ids, dtype=torch.long)
            return batch

        raise ValueError(f"Unsupported pos_feature_type: {self.pos_feature_type}")
