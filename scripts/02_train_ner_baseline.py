from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.constants import LABELS_COLUMN, NER_LABELS
from src.metrics import build_metrics_fn, summarize_trainer_runs
from src.seed import seed_everything
from src.tokenization import tokenize_and_align_labels


@dataclass(frozen=True)
class Config:
    model_name: str
    dataset_name: str
    output_dir: Path
    max_length: int
    learning_rate: float
    weight_decay: float
    train_batch_size: int
    eval_batch_size: int
    num_train_epochs: int
    warmup_ratio: float
    logging_steps: int
    save_total_limit: int
    fp16: bool
    run_seeds: tuple[int, ...]
    eval_steps: int


def parse_args() -> Config:
    p = argparse.ArgumentParser(description="Train NER baseline (HF Trainer).")
    p.add_argument("--model_name", default="/root/bert-base-uncased")
    p.add_argument("--dataset_name", default="/root/nlp-case-study/artifacts/dataset/conll2003")
    p.add_argument("--output_dir", type=Path, default=Path("./case_study_outputs/ner_baseline"))
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--fp16", action="store_true", default=False)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
    )
    p.add_argument("--eval_steps", type=int, default=100)
    a = p.parse_args()
    return Config(
        model_name=a.model_name,
        dataset_name=a.dataset_name,
        output_dir=a.output_dir,
        max_length=a.max_length,
        learning_rate=a.learning_rate,
        weight_decay=a.weight_decay,
        train_batch_size=a.train_batch_size,
        eval_batch_size=a.eval_batch_size,
        num_train_epochs=a.num_train_epochs,
        warmup_ratio=a.warmup_ratio,
        logging_steps=a.logging_steps,
        save_total_limit=a.save_total_limit,
        fp16=a.fp16,
        run_seeds=tuple(a.seeds),
        eval_steps=a.eval_steps,
    )


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available, but this script is configured for cuda:0"
        )

    id2label = {i: label for i, label in enumerate(NER_LABELS)}
    label2id = {label: i for i, label in enumerate(NER_LABELS)}

    print("Loading dataset...")
    dataset = load_dataset(cfg.dataset_name)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(
            examples, tokenizer, cfg.max_length, labels_column=LABELS_COLUMN
        ),
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing and aligning NER labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_metrics_fn(id2label, from_logits=True)

    all_run_metrics = []
    best_test_f1 = -1.0
    best_seed = None
    best_val_metrics = None
    best_test_metrics = None

    best_dir = cfg.output_dir / "best_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)

    for seed in cfg.run_seeds:
        print(f"\n{'=' * 80}")
        print(f"RUN WITH SEED = {seed}")
        print(f"{'=' * 80}")

        seed_everything(seed)

        run_dir = cfg.output_dir / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        print("Loading model...")
        model = AutoModelForTokenClassification.from_pretrained(
            cfg.model_name,
            num_labels=len(NER_LABELS),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir=str(run_dir),
            eval_strategy="steps",
            eval_steps=cfg.eval_steps,
            save_strategy="no",
            logging_strategy="steps",
            logging_steps=cfg.logging_steps,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.train_batch_size,
            per_device_eval_batch_size=cfg.eval_batch_size,
            num_train_epochs=cfg.num_train_epochs,
            weight_decay=cfg.weight_decay,
            warmup_ratio=cfg.warmup_ratio,
            load_best_model_at_end=False,
            report_to="none",
            fp16=cfg.fp16,
            seed=seed,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        print("Training...")
        trainer.train()

        print("\nValidation metrics:")
        val_metrics = trainer.evaluate(
            tokenized_dataset["validation"], metric_key_prefix="validation"
        )
        print(val_metrics)

        print("\nTest metrics:")
        test_metrics = trainer.evaluate(
            tokenized_dataset["test"], metric_key_prefix="test"
        )
        print(test_metrics)

        all_run_metrics.append(
            {
                "seed": seed,
                **val_metrics,
                **test_metrics,
            }
        )

        if test_metrics["test_f1"] > best_test_f1:
            best_test_f1 = test_metrics["test_f1"]
            best_seed = seed
            best_val_metrics = val_metrics
            best_test_metrics = test_metrics

            trainer.save_model(str(best_dir))
            tokenizer.save_pretrained(str(best_dir))

            print(f"\nNew best model saved from seed {seed} to: {best_dir}")

    summary = {
        "config": {
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "run_seeds": list(cfg.run_seeds),
            "num_train_epochs": cfg.num_train_epochs,
            "train_batch_size": cfg.train_batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "learning_rate": cfg.learning_rate,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
        },
        "runs": all_run_metrics,
        "validation_summary": summarize_trainer_runs(all_run_metrics, "validation"),
        "test_summary": summarize_trainer_runs(all_run_metrics, "test"),
        "best_seed": best_seed,
        "best_validation": best_val_metrics,
        "best_test": best_test_metrics,
        "best_model_checkpoint": str(best_dir),
    }

    metrics_path = best_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Best seed: {best_seed}")
    print(f"Best model saved to: {best_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
