from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from datasets import Dataset


def load_pos_error_transition_matrix(
    confusion_dir: Path,
) -> tuple[np.ndarray, dict[int, int]]:
    cm = np.load(confusion_dir / "test_confusion_matrix.npy")

    with open(confusion_dir / "confusion_summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    newid_to_oldid = {
        int(k): int(v) for k, v in summary["label_mapping"]["newid_to_oldid"].items()
    }

    error_only = cm.astype(np.float64).copy()
    np.fill_diagonal(error_only, 0.0)

    row_sums = error_only.sum(axis=1, keepdims=True)
    error_transition = np.divide(
        error_only,
        np.clip(row_sums, 1e-12, None),
        where=row_sums > 0,
    )

    n = error_transition.shape[0]
    for i in range(n):
        if row_sums[i, 0] == 0:
            probs = np.ones(n, dtype=np.float64)
            probs[i] = 0.0
            probs /= probs.sum()
            error_transition[i] = probs

    return error_transition, newid_to_oldid


def build_rawid_to_newid_from_dataset(dataset, pos_labels_column: str) -> dict[int, int]:
    unique_pos_ids = sorted(
        {
            int(pos_id)
            for split in dataset.keys()
            for example in dataset[split]
            for pos_id in example[pos_labels_column]
        }
    )
    return {raw_id: new_id for new_id, raw_id in enumerate(unique_pos_ids)}


def corrupt_pos_tags_for_example(
    raw_pos_tags: list[int],
    corruption_rate: float,
    rawid_to_newid: dict[int, int],
    newid_to_rawid: dict[int, int],
    error_transition: np.ndarray,
    rng: np.random.Generator,
) -> list[int]:
    corrupted = []

    for raw_pos in raw_pos_tags:
        raw_pos = int(raw_pos)

        if rng.random() >= corruption_rate:
            corrupted.append(raw_pos)
            continue

        src_newid = rawid_to_newid[raw_pos]
        probs = error_transition[src_newid]
        dst_newid = rng.choice(len(probs), p=probs)
        dst_rawid = newid_to_rawid[int(dst_newid)]
        corrupted.append(int(dst_rawid))

    return corrupted


def build_corrupted_test_dataset(
    test_dataset: Dataset,
    corruption_rate: float,
    rawid_to_newid: dict[int, int],
    newid_to_rawid: dict[int, int],
    error_transition: np.ndarray,
    seed: int,
    pos_labels_column: str,
) -> Dataset:
    rng = np.random.default_rng(seed)

    corrupted_pos_all = []
    for example in test_dataset:
        corrupted_pos = corrupt_pos_tags_for_example(
            raw_pos_tags=example[pos_labels_column],
            corruption_rate=corruption_rate,
            rawid_to_newid=rawid_to_newid,
            newid_to_rawid=newid_to_rawid,
            error_transition=error_transition,
            rng=rng,
        )
        corrupted_pos_all.append(corrupted_pos)

    corrupted_dataset = test_dataset.remove_columns([pos_labels_column])
    corrupted_dataset = corrupted_dataset.add_column(
        pos_labels_column, corrupted_pos_all
    )
    return corrupted_dataset
