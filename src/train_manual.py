from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.evaluation import evaluate_model, move_batch_to_device, save_model_checkpoint
from src.seed import seed_everything


@dataclass(frozen=True)
class ManualTrainConfig:
    learning_rate: float
    weight_decay: float
    num_train_epochs: int
    warmup_ratio: float
    grad_clip_norm: float
    logging_steps: int


def create_pos_dataloaders(
    tokenized_dataset,
    train_batch_size: int,
    eval_batch_size: int,
    data_collator,
):
    train_loader = DataLoader(
        tokenized_dataset["train"],
        batch_size=train_batch_size,
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(
        tokenized_dataset["validation"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False,
    )
    test_loader = DataLoader(
        tokenized_dataset["test"],
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=False,
    )
    return train_loader, val_loader, test_loader


def run_manual_training_for_seed(
    *,
    model: torch.nn.Module,
    tokenizer,
    train_loader,
    val_loader,
    test_loader,
    device: torch.device,
    cfg: ManualTrainConfig,
    seed: int,
    compute_metrics,
    best_dir: Path,
    global_best_f1: list,
    global_best_seed: list,
) -> dict:
    seed_everything(seed)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_training_steps = cfg.num_train_epochs * len(train_loader)
    warmup_steps = int(cfg.warmup_ratio * total_training_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_run_val_metrics = None
    best_run_val_f1 = -1.0
    best_val_f1 = global_best_f1[0]

    for epoch in range(cfg.num_train_epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % cfg.logging_steps == 0:
                avg_loss = running_loss / step
                print(
                    f"seed={seed} "
                    f"epoch={epoch + 1}/{cfg.num_train_epochs} "
                    f"step={step} "
                    f"train_loss={avg_loss:.4f}"
                )

        val_metrics = evaluate_model(model, val_loader, compute_metrics, device)
        print(
            f"seed={seed} "
            f"epoch={epoch + 1}/{cfg.num_train_epochs} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"val_macro_f1={val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["f1"] > best_run_val_f1:
            best_run_val_f1 = val_metrics["f1"]
            best_run_val_metrics = val_metrics

            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                global_best_f1[0] = val_metrics["f1"]
                global_best_seed[0] = seed
                save_model_checkpoint(model, tokenizer, best_dir)

    print("\nFinal evaluation for this seed...")
    final_val_metrics = evaluate_model(model, val_loader, compute_metrics, device)
    final_test_metrics = evaluate_model(model, test_loader, compute_metrics, device)

    print("Validation:", final_val_metrics)
    print("Test:", final_test_metrics)

    return {
        "seed": seed,
        "best_validation": best_run_val_metrics,
        "final_validation": final_val_metrics,
        "test": final_test_metrics,
    }
