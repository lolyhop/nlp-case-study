from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.collators import DataCollatorForNERWithPOS
from src.constants import GOLD_POS_COLUMN, NER_LABELS, NER_LABELS_COLUMN
from src.evaluation import evaluate_model
from src.metrics import build_metrics_fn
from src.models import (
    BertForTokenClassificationWithMidPOSInjection,
    BertForTokenClassificationWithPOSFeatures,
)
from src.pos_corruption import (
    build_corrupted_test_dataset,
    build_rawid_to_newid_from_dataset,
    load_pos_error_transition_matrix,
)
from src.seed import seed_everything
from src.tokenization import tokenize_and_align_features


def parse_args():
    p = argparse.ArgumentParser(description="Controlled POS tag corruption on test set evaluation.")
    p.add_argument(
        "--ner_model_dir",
        type=Path,
        default=Path(
            "/root/nlp-case-study/scripts/nlp_case_study_outputs/ner_with_mid_pos_injection"
            "/gold_mid_layer_trainable_embed_d32/best_checkpoint"
        ),
    )
    p.add_argument(
        "--pos_confusion_dir",
        type=Path,
        default=Path(
            "/root/nlp-case-study/scripts/case_study_outputs/pos_full_train_confusion"
            "/best_final_checkpoint"
        ),
    )
    p.add_argument(
        "--fusion_type",
        default="encoder_level",
        choices=("encoder_independent", "encoder_level"),
    )
    p.add_argument("--inject_layer_offset_from_end", type=int, default=2)
    p.add_argument("--dataset_name", default="/root/nlp-case-study/artifacts/dataset/conll2003")
    p.add_argument(
        "--tokenizer_model_name",
        default=None,
        help="If None, uses --ner_model_dir (checkpoint contains tokenizer).",
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument(
        "--device",
        default=None,
        help="e.g. cuda:0, cuda, cpu. Default: cuda:0 if available else cpu.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--pos_feature_type",
        default="trainable_embed",
        choices=("one_hot", "trainable_embed"),
    )
    p.add_argument("--pos_embed_dim", type=int, default=32)
    p.add_argument(
        "--corruption_rates",
        type=float,
        nargs="*",
        default=[
            0.0,
            0.05,
            0.10,
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            1.0,
        ],
    )
    p.add_argument(
        "--output_path",
        type=Path,
        default=None,
        help="Default: <ner_model_dir>/controlled_corruption_results_<fusion>_<suffix>.json",
    )
    return p.parse_args()


def main():
    a = parse_args()
    if a.pos_feature_type not in {"one_hot", "trainable_embed"}:
        raise ValueError(
            "Raw corruption script supports only one_hot or trainable_embed"
        )

    ner_model_dir = a.ner_model_dir
    model_name_for_load = a.tokenizer_model_name or str(ner_model_dir)

    if a.device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(a.device)

    seed_everything(a.seed)

    print("Loading dataset...")
    dataset = load_dataset(a.dataset_name)
    test_raw = dataset["test"]

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name_for_load)

    print("Loading POS confusion matrix...")
    error_transition, newid_to_oldid = load_pos_error_transition_matrix(
        a.pos_confusion_dir
    )

    rawid_to_newid = build_rawid_to_newid_from_dataset(dataset, GOLD_POS_COLUMN)
    num_pos_tags = len(rawid_to_newid)
    pos_feature_dim = (
        num_pos_tags if a.pos_feature_type == "one_hot" else a.pos_embed_dim
    )

    if set(rawid_to_newid.keys()) != set(newid_to_oldid.values()):
        raise ValueError("Mismatch between dataset POS ids and POS confusion mapping")

    id2label = {i: label for i, label in enumerate(NER_LABELS)}
    label2id = {label: i for i, label in enumerate(NER_LABELS)}

    print("Loading NER model...")
    hf_config = AutoConfig.from_pretrained(
        model_name_for_load,
        num_labels=len(NER_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    if a.fusion_type == "encoder_independent":
        model = BertForTokenClassificationWithPOSFeatures.from_pretrained(
            str(ner_model_dir),
            config=hf_config,
            pos_feature_type=a.pos_feature_type,
            pos_feature_dim=num_pos_tags if a.pos_feature_type == "one_hot" else None,
            num_pos_tags=num_pos_tags if a.pos_feature_type == "trainable_embed" else None,
            pos_embed_dim=(
                a.pos_embed_dim if a.pos_feature_type == "trainable_embed" else None
            ),
        ).to(device)

    elif a.fusion_type == "encoder_level":
        if a.pos_feature_type != "trainable_embed":
            raise ValueError(
                "encoder_level corruption currently supports only trainable_embed POS input"
            )

        inject_layer_idx = (
            hf_config.num_hidden_layers - a.inject_layer_offset_from_end
        )

        model = BertForTokenClassificationWithMidPOSInjection.from_pretrained(
            str(ner_model_dir),
            config=hf_config,
            num_pos_tags=num_pos_tags,
            pos_embed_dim=a.pos_embed_dim,
            inject_layer_idx=inject_layer_idx,
        ).to(device)

    else:
        raise ValueError(f"Unsupported fusion_type: {a.fusion_type}")

    compute_metrics = build_metrics_fn(id2label)
    data_collator = DataCollatorForNERWithPOS(
        tokenizer=tokenizer,
        pos_feature_type=a.pos_feature_type,
        pos_feature_dim=num_pos_tags if a.pos_feature_type == "one_hot" else None,
    )

    all_results = []

    for corruption_rate in a.corruption_rates:
        print(f"\nEvaluating corruption_rate = {corruption_rate:.2f}")

        corrupted_test = build_corrupted_test_dataset(
            test_dataset=test_raw,
            corruption_rate=corruption_rate,
            rawid_to_newid=rawid_to_newid,
            newid_to_rawid=newid_to_oldid,
            error_transition=error_transition,
            seed=a.seed,
            pos_labels_column=GOLD_POS_COLUMN,
        )

        num_total = 0
        num_changed = 0
        for gold_ex, corr_ex in zip(test_raw, corrupted_test):
            for g, c in zip(gold_ex[GOLD_POS_COLUMN], corr_ex[GOLD_POS_COLUMN]):
                num_total += 1
                if g != c:
                    num_changed += 1

        print("requested corruption_rate =", corruption_rate)
        print("actual changed fraction =", num_changed / num_total)

        tokenized_test = corrupted_test.map(
            lambda examples: tokenize_and_align_features(
                examples=examples,
                tokenizer=tokenizer,
                max_length=a.max_length,
                pos_feature_type=a.pos_feature_type,
                pos_column=GOLD_POS_COLUMN,
                pos_id_map=rawid_to_newid,
                pos_feature_dim=num_pos_tags if a.pos_feature_type == "one_hot" else None,
                ner_labels_column=NER_LABELS_COLUMN,
            ),
            batched=True,
            remove_columns=corrupted_test.column_names,
            desc=f"Tokenizing corrupted test set (rate={corruption_rate:.2f})",
        )

        test_loader = DataLoader(
            tokenized_test,
            batch_size=a.batch_size,
            shuffle=False,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
        )

        metrics = evaluate_model(model, test_loader, compute_metrics, device)
        metrics["corruption_rate"] = corruption_rate
        metrics["actual_changed_fraction"] = num_changed / num_total
        all_results.append(metrics)

        print(metrics)

    suffix = (
        a.pos_feature_type
        if a.pos_feature_type != "trainable_embed"
        else f"trainable_embed_d{a.pos_embed_dim}"
    )

    if a.output_path is not None:
        output_path = a.output_path
    else:
        output_path = (
            ner_model_dir
            / f"controlled_corruption_results_{a.fusion_type}_{suffix}.json"
        )
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\nSaved results to: {output_path}")


if __name__ == "__main__":
    main()
