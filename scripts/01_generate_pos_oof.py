from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

TOKENS_COLUMN = "tokens"
POS_LABELS_COLUMN = "pos_tags"
PREDICTED_POS_COLUMN = "predicted_pos_tags"
PREDICTED_POS_LOGITS_COLUMN = "predicted_pos_logits"


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
    eval_steps: int
    save_steps: int
    save_total_limit: int
    num_folds: int
    seed: int
    fp16: bool


def parse_args() -> Config:
    p = argparse.ArgumentParser(
        description="Generate OOF / final POS tagger predictions and save dataset on disk.",
        epilog="GPU: pass --cuda_visible_devices <id> in argv; it is applied before library imports (see file header).",
    )
    p.add_argument("--model_name", default="/root/bert-base-uncased")
    p.add_argument(
        "--dataset_name", default="/root/nlp-case-study/artifacts/dataset/conll2003"
    )
    p.add_argument(
        "--output_dir", type=Path, default=Path("./case_study_outputs/pos_oof")
    )
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--train_batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=16)
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=500)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fp16", action="store_true", default=False)
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
        eval_steps=a.eval_steps,
        save_steps=a.save_steps,
        save_total_limit=a.save_total_limit,
        num_folds=a.num_folds,
        seed=a.seed,
        fp16=a.fp16,
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    set_seed(seed)


def build_pos_mappings(
    dataset: DatasetDict,
) -> tuple[list[int], dict[int, str], dict[str, int], dict[int, int]]:
    unique_pos_ids = sorted(
        {
            int(pos_id)
            for split in dataset.keys()
            for example in dataset[split]
            for pos_id in example[POS_LABELS_COLUMN]
        }
    )

    oldid_to_newid = {old_id: new_id for new_id, old_id in enumerate(unique_pos_ids)}
    id2label = {new_id: f"POS_{old_id}" for old_id, new_id in oldid_to_newid.items()}
    label2id = {label: new_id for new_id, label in id2label.items()}

    return unique_pos_ids, id2label, label2id, oldid_to_newid


def tokenize_and_align_pos_labels(
    examples, tokenizer, max_length: int, oldid_to_newid: dict[int, int]
):
    tokenized = tokenizer(
        examples[TOKENS_COLUMN],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_labels = []
    for batch_index, word_labels in enumerate(examples[POS_LABELS_COLUMN]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                raw_label = int(word_labels[word_id])
                label_ids.append(oldid_to_newid[raw_label])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def build_metrics_fn():
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                true_predictions.append(int(pred_id))
                true_labels.append(int(label_id))

        acc = accuracy_score(true_labels, true_predictions)

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                true_labels,
                true_predictions,
                average="weighted",
                zero_division=0,
            )
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            true_labels,
            true_predictions,
            average="macro",
            zero_division=0,
        )

        return {
            "accuracy": acc,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "f1_weighted": f1_weighted,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
        }

    return compute_metrics


def make_training_args(
    output_dir: Path,
    cfg: Config,
    load_best_model_at_end: bool = True,
) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        learning_rate=cfg.learning_rate,
        per_device_train_batch_size=cfg.train_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        num_train_epochs=cfg.num_train_epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        load_best_model_at_end=load_best_model_at_end,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        fp16=cfg.fp16,
        seed=cfg.seed,
    )


def tokenize_and_align_pos_labels(
    examples, tokenizer, max_length: int, oldid_to_newid: dict[int, int]
):
    tokenized = tokenizer(
        examples[TOKENS_COLUMN],
        truncation=True,
        is_split_into_words=True,
        max_length=max_length,
    )

    aligned_labels = []
    for batch_index, word_labels in enumerate(examples[POS_LABELS_COLUMN]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                raw_label = int(word_labels[word_id])
                label_ids.append(oldid_to_newid[raw_label])
            else:
                label_ids.append(-100)

            previous_word_id = word_id

        aligned_labels.append(label_ids)

    tokenized["labels"] = aligned_labels
    return tokenized


def extract_word_level_predictions(predictions, labels) -> list[list[int]]:
    word_level_predictions = []

    for pred_seq, label_seq in zip(predictions, labels):
        curr_preds = []
        for pred_id, label_id in zip(pred_seq, label_seq):
            if label_id == -100:
                continue
            curr_preds.append(int(pred_id))
        word_level_predictions.append(curr_preds)

    return word_level_predictions


def extract_word_level_logits(logits, labels) -> list[list[list[float]]]:
    word_level_logits = []

    for logit_seq, label_seq in zip(logits, labels):
        curr_logits = []
        for token_logits, label_id in zip(logit_seq, label_seq):
            if label_id == -100:
                continue
            curr_logits.append([float(x) for x in token_logits])
        word_level_logits.append(curr_logits)

    return word_level_logits


def predict_word_level_pos(
    trainer: Trainer,
    tokenized_dataset: Dataset,
) -> tuple[list[list[int]], list[list[list[float]]]]:
    output = trainer.predict(tokenized_dataset)
    logits = output.predictions
    labels = output.label_ids
    predictions = np.argmax(logits, axis=-1)

    word_level_predictions = extract_word_level_predictions(predictions, labels)
    word_level_logits = extract_word_level_logits(logits, labels)

    return word_level_predictions, word_level_logits


def build_kfold_indices(n_examples: int, num_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_examples)
    rng.shuffle(indices)
    return [fold.astype(int) for fold in np.array_split(indices, num_folds)]


def build_tokenized_dataset(
    raw_dataset: Dataset, tokenizer, cfg: Config, oldid_to_newid: dict[int, int]
) -> Dataset:
    tokenized = raw_dataset.map(
        lambda examples: tokenize_and_align_pos_labels(
            examples,
            tokenizer,
            cfg.max_length,
            oldid_to_newid,
        ),
        batched=True,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing and aligning POS labels",
    )
    return tokenized


def train_pos_model(
    train_dataset: Dataset,
    eval_dataset: Dataset,
    tokenizer,
    cfg: Config,
    output_dir: Path,
    id2label: dict[int, str],
    label2id: dict[str, int],
    oldid_to_newid: dict[int, int],
) -> tuple[Trainer, dict]:
    tokenized_train = build_tokenized_dataset(
        train_dataset, tokenizer, cfg, oldid_to_newid
    )
    tokenized_eval = build_tokenized_dataset(
        eval_dataset, tokenizer, cfg, oldid_to_newid
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_metrics_fn()

    model = AutoModelForTokenClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )

    training_args = make_training_args(output_dir=output_dir, cfg=cfg)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_metrics = trainer.evaluate(tokenized_eval)
    return trainer, eval_metrics


def main() -> None:
    cfg = parse_args()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)

    print("Loading dataset...")
    dataset = load_dataset(cfg.dataset_name)

    print("Building POS label mappings...")
    unique_pos_ids, id2label, label2id, oldid_to_newid = build_pos_mappings(dataset)
    num_pos_labels = len(unique_pos_ids)

    print(f"Number of POS labels: {num_pos_labels}")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_raw = dataset["train"]
    val_raw = dataset["validation"]
    test_raw = dataset["test"]

    n_train = len(train_raw)
    fold_indices = build_kfold_indices(n_train, cfg.num_folds, cfg.seed)

    oof_predictions: list[list[int] | None] = [None] * n_train
    oof_logits: list[list[list[float]] | None] = [None] * n_train
    fold_metrics = []

    print("Generating out-of-fold POS predictions for train...")
    for fold_id in range(cfg.num_folds):
        val_idx = fold_indices[fold_id]
        train_idx = np.concatenate(
            [fold_indices[i] for i in range(cfg.num_folds) if i != fold_id]
        )

        fold_train_raw = train_raw.select(train_idx.tolist())
        fold_val_raw = train_raw.select(val_idx.tolist())

        fold_dir = cfg.output_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nFold {fold_id + 1}/{cfg.num_folds}")
        print(f"Train size: {len(fold_train_raw)}")
        print(f"Holdout size: {len(fold_val_raw)}")

        trainer, eval_metrics = train_pos_model(
            train_dataset=fold_train_raw,
            eval_dataset=fold_val_raw,
            tokenizer=tokenizer,
            cfg=cfg,
            output_dir=fold_dir,
            id2label=id2label,
            label2id=label2id,
            oldid_to_newid=oldid_to_newid,
        )

        fold_val_tokenized = build_tokenized_dataset(
            fold_val_raw, tokenizer, cfg, oldid_to_newid
        )
        fold_predicted_pos, fold_predicted_logits = predict_word_level_pos(
            trainer, fold_val_tokenized
        )

        for dataset_index, predicted_tags, predicted_logits, example in zip(
            val_idx.tolist(),
            fold_predicted_pos,
            fold_predicted_logits,
            fold_val_raw,
        ):
            if len(predicted_tags) != len(example[TOKENS_COLUMN]):
                raise ValueError(
                    f"Fold {fold_id}: predicted POS length {len(predicted_tags)} "
                    f"does not match token length {len(example[TOKENS_COLUMN])} "
                    f"for dataset index {dataset_index}"
                )

            if len(predicted_logits) != len(example[TOKENS_COLUMN]):
                raise ValueError(
                    f"Fold {fold_id}: predicted POS logits length {len(predicted_logits)} "
                    f"does not match token length {len(example[TOKENS_COLUMN])} "
                    f"for dataset index {dataset_index}"
                )

            oof_predictions[dataset_index] = predicted_tags
            oof_logits[dataset_index] = predicted_logits

        fold_metrics.append(
            {
                "fold": fold_id,
                "num_train_examples": len(fold_train_raw),
                "num_holdout_examples": len(fold_val_raw),
                **eval_metrics,
                "best_model_checkpoint": trainer.state.best_model_checkpoint,
            }
        )

    if any(logits is None for logits in oof_logits):
        missing = [i for i, logits in enumerate(oof_logits) if logits is None]
        raise RuntimeError(f"Missing OOF logits for indices: {missing[:20]}")

    print("\nTraining final POS model on full train...")
    final_model_dir = cfg.output_dir / "final_model"
    final_model_dir.mkdir(parents=True, exist_ok=True)

    final_trainer, final_val_metrics = train_pos_model(
        train_dataset=train_raw,
        eval_dataset=val_raw,
        tokenizer=tokenizer,
        cfg=cfg,
        output_dir=final_model_dir,
        id2label=id2label,
        label2id=label2id,
        oldid_to_newid=oldid_to_newid,
    )

    print("Generating POS predictions for validation and test...")
    val_tokenized = build_tokenized_dataset(val_raw, tokenizer, cfg, oldid_to_newid)
    test_tokenized = build_tokenized_dataset(test_raw, tokenizer, cfg, oldid_to_newid)

    val_predictions, val_logits = predict_word_level_pos(final_trainer, val_tokenized)
    test_predictions, test_logits = predict_word_level_pos(
        final_trainer, test_tokenized
    )

    for split_name, raw_split, preds, logits in [
        ("validation", val_raw, val_predictions, val_logits),
        ("test", test_raw, test_predictions, test_logits),
    ]:
        for idx, (example, predicted_tags, predicted_logits) in enumerate(
            zip(raw_split, preds, logits)
        ):
            if len(predicted_tags) != len(example[TOKENS_COLUMN]):
                raise ValueError(
                    f"{split_name} split: predicted POS length {len(predicted_tags)} "
                    f"does not match token length {len(example[TOKENS_COLUMN])} "
                    f"for example index {idx}"
                )

            if len(predicted_logits) != len(example[TOKENS_COLUMN]):
                raise ValueError(
                    f"{split_name} split: predicted POS logits length {len(predicted_logits)} "
                    f"does not match token length {len(example[TOKENS_COLUMN])} "
                    f"for example index {idx}"
                )

    print("Building dataset with predicted POS tags and logits...")
    train_with_predicted_pos = train_raw.add_column(
        PREDICTED_POS_COLUMN, oof_predictions
    )
    train_with_predicted_pos = train_with_predicted_pos.add_column(
        PREDICTED_POS_LOGITS_COLUMN, oof_logits
    )

    val_with_predicted_pos = val_raw.add_column(PREDICTED_POS_COLUMN, val_predictions)
    val_with_predicted_pos = val_with_predicted_pos.add_column(
        PREDICTED_POS_LOGITS_COLUMN, val_logits
    )

    test_with_predicted_pos = test_raw.add_column(
        PREDICTED_POS_COLUMN, test_predictions
    )
    test_with_predicted_pos = test_with_predicted_pos.add_column(
        PREDICTED_POS_LOGITS_COLUMN, test_logits
    )

    predicted_pos_dataset = DatasetDict(
        {
            "train": train_with_predicted_pos,
            "validation": val_with_predicted_pos,
            "test": test_with_predicted_pos,
        }
    )

    dataset_save_dir = cfg.output_dir / "dataset_with_predicted_pos"
    predicted_pos_dataset.save_to_disk(str(dataset_save_dir))

    best_dir = cfg.output_dir / "best_final_checkpoint"
    best_dir.mkdir(parents=True, exist_ok=True)
    final_trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    final_test_metrics = final_trainer.evaluate(test_tokenized)

    metrics = {
        "config": {
            "model_name": cfg.model_name,
            "dataset_name": cfg.dataset_name,
            "num_folds": cfg.num_folds,
            "num_pos_labels": num_pos_labels,
            "seed": cfg.seed,
        },
        "fold_metrics": fold_metrics,
        "final_validation_metrics": final_val_metrics,
        "final_test_metrics": final_test_metrics,
        "best_final_model_checkpoint": final_trainer.state.best_model_checkpoint,
        "dataset_with_predicted_pos_path": str(dataset_save_dir),
    }

    metrics_path = cfg.output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Dataset with predicted POS saved to: {dataset_save_dir}")
    print(f"Best final model saved to: {best_dir}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
