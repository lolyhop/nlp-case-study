from __future__ import annotations

import numpy as np


def build_metrics_fn(id2label: dict[int, str], *, from_logits: bool = False):
    def compute_metrics(eval_pred):
        if from_logits:
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions, labels = eval_pred

        true_predictions = []
        true_labels = []

        for pred_seq, label_seq in zip(predictions, labels):
            pred_tags = []
            gold_tags = []

            for pred_id, label_id in zip(pred_seq, label_seq):
                if label_id == -100:
                    continue
                pred_tags.append(id2label[int(pred_id)])
                gold_tags.append(id2label[int(label_id)])

            true_predictions.append(pred_tags)
            true_labels.append(gold_tags)

        flat_preds = [p for seq in true_predictions for p in seq]
        flat_labels = [g for seq in true_labels for g in seq]

        correct = sum(p == g for p, g in zip(flat_preds, flat_labels))
        total = len(flat_labels)
        accuracy = correct / total if total > 0 else 0.0

        labels_set = sorted(set(flat_labels) - {"O"})
        precision_scores = []
        recall_scores = []
        f1_scores = []

        for tag in labels_set:
            tp = sum((p == tag and g == tag) for p, g in zip(flat_preds, flat_labels))
            fp = sum((p == tag and g != tag) for p, g in zip(flat_preds, flat_labels))
            fn = sum((p != tag and g == tag) for p, g in zip(flat_preds, flat_labels))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if (precision + recall) > 0
                else 0.0
            )

            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

        macro_precision = np.mean(precision_scores) if precision_scores else 0.0
        macro_recall = np.mean(recall_scores) if recall_scores else 0.0
        macro_f1 = np.mean(f1_scores) if f1_scores else 0.0

        def extract_entities(seq):
            entities = []
            entity = None
            entity_type = None

            for idx, tag in enumerate(seq):
                if tag.startswith("B-"):
                    if entity is not None:
                        entities.append((entity_type, entity))
                    entity_type = tag[2:]
                    entity = [idx]
                elif (
                    tag.startswith("I-")
                    and entity is not None
                    and tag[2:] == entity_type
                ):
                    entity.append(idx)
                else:
                    if entity is not None:
                        entities.append((entity_type, entity))
                        entity = None
                        entity_type = None

            if entity is not None:
                entities.append((entity_type, entity))

            return set((etype, tuple(span)) for etype, span in entities)

        pred_entities = [extract_entities(seq) for seq in true_predictions]
        gold_entities = [extract_entities(seq) for seq in true_labels]

        tp_e, fp_e, fn_e = 0, 0, 0
        for pred_ents, gold_ents in zip(pred_entities, gold_entities):
            tp_e += len(pred_ents & gold_ents)
            fp_e += len(pred_ents - gold_ents)
            fn_e += len(gold_ents - pred_ents)

        entity_precision = tp_e / (tp_e + fp_e) if (tp_e + fp_e) > 0 else 0.0
        entity_recall = tp_e / (tp_e + fn_e) if (tp_e + fn_e) > 0 else 0.0
        entity_f1 = (
            2
            * entity_precision
            * entity_recall
            / (entity_precision + entity_recall)
            if (entity_precision + entity_recall) > 0
            else 0.0
        )

        return {
            "precision": entity_precision,
            "recall": entity_recall,
            "f1": entity_f1,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
        }

    return compute_metrics


def summarize_trainer_runs(run_metrics: list[dict], split_name: str) -> dict:
    metric_names = [
        f"{split_name}_precision",
        f"{split_name}_recall",
        f"{split_name}_f1",
        f"{split_name}_accuracy",
        f"{split_name}_macro_precision",
        f"{split_name}_macro_recall",
        f"{split_name}_macro_f1",
        f"{split_name}_loss",
    ]

    summary = {}
    for name in metric_names:
        values = [m[name] for m in run_metrics]
        summary[name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def summarize_manual_runs(runs: list[dict], split_name: str) -> dict:
    metric_names = [
        "precision",
        "recall",
        "f1",
        "accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "loss",
    ]

    summary = {}
    for metric_name in metric_names:
        values = [run[split_name][metric_name] for run in runs]
        summary[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        }

    return summary
