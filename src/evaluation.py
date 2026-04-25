from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def move_batch_to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate_model(model, dataloader, compute_metrics, device: torch.device):
    model.eval()

    all_predictions = []
    all_labels = []
    losses = []

    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        outputs = model(**batch)

        if outputs.loss is not None:
            losses.append(outputs.loss.item())

        predictions = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
        labels = batch["labels"].detach().cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels)

    metrics = compute_metrics((all_predictions, all_labels))
    metrics["loss"] = float(np.mean(losses)) if losses else 0.0
    return metrics


def save_model_checkpoint(model, tokenizer, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))
