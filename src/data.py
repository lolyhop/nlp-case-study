from __future__ import annotations

from typing import Optional

from datasets import load_dataset, load_from_disk

from src.constants import (
    GOLD_POS_COLUMN,
    PREDICTED_POS_COLUMN,
    PREDICTED_POS_LOGITS_COLUMN,
)


def build_pos_id_mapping(dataset, pos_column: str) -> dict[int, int]:
    unique_pos_ids = sorted(
        {
            int(pos_id)
            for split in dataset.keys()
            for example in dataset[split]
            for pos_id in example[pos_column]
        }
    )
    return {raw_id: new_id for new_id, raw_id in enumerate(unique_pos_ids)}


def load_ner_pos_dataset(
    *,
    dataset_name: str,
    predicted_pos_dataset_path: str,
    pos_source: str,
    pos_feature_type: Optional[str],
    mid_layer: bool,
):
    if pos_source == "gold":
        dataset = load_dataset(dataset_name)
        return dataset, GOLD_POS_COLUMN

    if pos_source == "predicted":
        dataset = load_from_disk(predicted_pos_dataset_path)
        if mid_layer:
            return dataset, PREDICTED_POS_COLUMN
        if pos_feature_type in {"one_hot", "trainable_embed"}:
            return dataset, PREDICTED_POS_COLUMN
        if pos_feature_type == "logits":
            return dataset, PREDICTED_POS_LOGITS_COLUMN
        raise ValueError(f"Unsupported pos_feature_type: {pos_feature_type}")

    raise ValueError(f"Unsupported pos_source: {pos_source}")
