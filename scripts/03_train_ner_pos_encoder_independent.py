from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoConfig, AutoTokenizer

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.collators import DataCollatorForNERWithPOS
from src.constants import NER_LABELS, NER_LABELS_COLUMN
from src.data import build_pos_id_mapping, load_ner_pos_dataset
from src.metrics import build_metrics_fn, summarize_manual_runs
from src.models import BertForTokenClassificationWithPOSFeatures
from src.tokenization import tokenize_and_align_features
from src.train_manual import (
    ManualTrainConfig,
    create_pos_dataloaders,
    run_manual_training_for_seed,
)


@dataclass(frozen=True)
class Config:
    model_name: str
    dataset_name: str
    predicted_pos_dataset_path: str
    output_dir: Path
    max_length: int
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    grad_clip_norm: float
    fp16: bool
    pos_source: str
    pos_feature_type: str
    pos_embed_dim: int
    run_seeds: tuple[int, ...]


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train NER with POS (encoder-side fusion, independent).")
    p.add_argument("--model_name", default="/root/bert-base-uncased")
    p.add_argument("--dataset_name", default="/root/nlp-case-study/artifacts/dataset/conll2003")
    p.add_argument(
        "--predicted_pos_dataset_path",
        default="/root/case_study_outputs/dataset_with_predicted_pos",
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("./nlp_case_study_outputs/ner_with_predicted_pos_embd")
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--grad_clip_norm", type=float, default=1.0)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--pos_source", default="gold", choices=("gold", "predicted"))
    p.add_argument(
        "--pos_feature_type",
        default="trainable_embed",
        choices=("one_hot", "logits", "trainable_embed"),
    )
    p.add_argument("--pos_embed_dim", type=int, default=32)
    a = p.parse_args()
    return Config(
        model_name=a.model_name,
        dataset_name=a.dataset_name,
        predicted_pos_dataset_path=a.predicted_pos_dataset_path,
        output_dir=a.output_dir,
        max_length=a.max_length,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,
        num_train_epochs=a.num_train_epochs,
        warmup_ratio=a.warmup_ratio,
        logging_steps=a.logging_steps,
        grad_clip_norm=a.grad_clip_norm,
        fp16=a.fp16,
        pos_source=a.pos_source,
        pos_feature_type=a.pos_feature_type,
        pos_embed_dim=a.pos_embed_dim,
        run_seeds=tuple(a.seeds),
    )


def main() -> None:
    cfg = parse_args()
    run_name = f"{cfg.pos_source}_{cfg.pos_feature_type}"
    if cfg.pos_feature_type == "trainable_embed":
        run_name += f"_d{cfg.pos_embed_dim}"
    output_dir = cfg.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset, pos_column = load_ner_pos_dataset(
        dataset_name=cfg.dataset_name,
        predicted_pos_dataset_path=cfg.predicted_pos_dataset_path,
        pos_source=cfg.pos_source,
        pos_feature_type=cfg.pos_feature_type,
        mid_layer=False,
    )

    print("Building label mappings...")
    id2label = {i: label for i, label in enumerate(NER_LABELS)}
    label2id = {label: i for i, label in enumerate(NER_LABELS)}

    pos_id_map = None
    num_pos_tags = None
    pos_feature_dim = None

    if cfg.pos_feature_type in {"one_hot", "trainable_embed"}:
        pos_id_map = build_pos_id_mapping(dataset, pos_column=pos_column)
        num_pos_tags = len(pos_id_map)

        if cfg.pos_feature_type == "one_hot":
            pos_feature_dim = num_pos_tags
        else:
            pos_feature_dim = cfg.pos_embed_dim

    elif cfg.pos_feature_type == "logits":
        sample_logits = dataset["train"][0][pos_column][0]
        pos_feature_dim = len(sample_logits)

    else:
        raise ValueError(f"Unsupported pos_feature_type: {cfg.pos_feature_type}")

    print(f"POS source: {cfg.pos_source}")
    print(f"POS feature type: {cfg.pos_feature_type}")
    print(f"POS feature dim: {pos_feature_dim}")
    if num_pos_tags is not None:
        print(f"Number of POS tags: {num_pos_tags}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_features(
            examples=examples,
            tokenizer=tokenizer,
            max_length=cfg.max_length,
            pos_feature_type=cfg.pos_feature_type,
            pos_column=pos_column,
            pos_id_map=pos_id_map,
            pos_feature_dim=(
                pos_feature_dim if cfg.pos_feature_type != "trainable_embed" else None
            ),
            ner_labels_column=NER_LABELS_COLUMN,
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc=f"Tokenizing and aligning NER labels with {cfg.pos_source} {cfg.pos_feature_type} POS features",
    )

    data_collator = DataCollatorForNERWithPOS(
        tokenizer=tokenizer,
        pos_feature_type=cfg.pos_feature_type,
        pos_feature_dim=(
            pos_feature_dim if cfg.pos_feature_type in {"one_hot", "logits"} else None
        ),
    )
    compute_metrics = build_metrics_fn(id2label)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = create_pos_dataloaders(
        tokenized_dataset,
        cfg.train_batch_size,
        cfg.eval_batch_size,
        data_collator,
    )

    hf_config = AutoConfig.from_pretrained(
        cfg.model_name,
        num_labels=len(NER_LABELS),
        id2label=id2label,
        label2id=label2id,
    )

    all_runs = []
    best_dir = output_dir / "best_checkpoint"
    global_best_f1 = [-1.0]
    global_best_seed = [None]

    train_cfg = ManualTrainConfig(
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        num_train_epochs=cfg.num_train_epochs,
        warmup_ratio=cfg.warmup_ratio,
        grad_clip_norm=cfg.grad_clip_norm,
        logging_steps=cfg.logging_steps,
    )

    for seed in cfg.run_seeds:
        print(f"\n{'=' * 80}")
        print(f"RUN WITH SEED = {seed}")
        print(f"{'=' * 80}")

        print("Loading model...")
        model = BertForTokenClassificationWithPOSFeatures.from_pretrained(
            cfg.model_name,
            config=hf_config,
            pos_feature_type=cfg.pos_feature_type,
            pos_feature_dim=(
                pos_feature_dim
                if cfg.pos_feature_type in {"one_hot", "logits"}
                else None
            ),
            num_pos_tags=(
                num_pos_tags if cfg.pos_feature_type == "trainable_embed" else None
            ),
            pos_embed_dim=(
                cfg.pos_embed_dim if cfg.pos_feature_type == "trainable_embed" else None
            ),
        ).to(device)

        run_result = run_manual_training_for_seed(
            model=model,
            tokenizer=tokenizer,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            device=device,
            cfg=train_cfg,
            seed=seed,
            compute_metrics=compute_metrics,
            best_dir=best_dir,
            global_best_f1=global_best_f1,
            global_best_seed=global_best_seed,
        )
        all_runs.append(run_result)

    summary = {
        "config": {
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "pos_source": cfg.pos_source,
            "pos_feature_type": cfg.pos_feature_type,
            "pos_feature_dim": pos_feature_dim,
            "num_pos_tags": num_pos_tags,
            "pos_embed_dim": (
                cfg.pos_embed_dim if cfg.pos_feature_type == "trainable_embed" else None
            ),
            "run_seeds": list(cfg.run_seeds),
            "num_train_epochs": cfg.num_train_epochs,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
        },
        "runs": all_runs,
        "validation_summary": summarize_manual_runs(all_runs, "final_validation"),
        "test_summary": summarize_manual_runs(all_runs, "test"),
        "best_seed": global_best_seed[0],
        "best_model_checkpoint": str(best_dir),
    }

    metrics_path = output_dir / "metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Best seed: {global_best_seed[0]}")
    print(f"Best model saved to: {best_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
