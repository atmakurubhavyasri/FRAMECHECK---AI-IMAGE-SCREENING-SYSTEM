"""Complete, reproducible training and validation pipeline for Hugging Face image models.

Implements:
1. Seed setting across random, numpy, torch for exact reproducibility.
2. Training loop with AdamW, learning rate scheduler, and gradient clipping.
3. Validation loop tracking validation loss, accuracy, and macro F1.
4. Early stopping and best model checkpoint saving (safetensors + config).
5. Automatic post-training validation-only confidence calibration.
6. Recording complete run metadata (dataset version, model name, seed, hyperparams).
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Sequence

# Set torch environment
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoModelForImageClassification, get_cosine_schedule_with_warmup

from src.calibration.calibrator import calibrate_uncertainty_thresholds, save_calibration_config
from src.data.dataset import FrameCheckDataset
from src.evaluation.metrics import compute_binary_metrics


def set_seed(seed: int = 42) -> None:
    """Fix random seeds for exact reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"Random seed set to: {seed}")


def train_model(
    train_records: Sequence[dict[str, Any]],
    val_records: Sequence[dict[str, Any]],
    base_model_name: str = "google/vit-base-patch16-224",
    output_dir: str = "models/framecheck-ai-detector",
    batch_size: int = 16,
    epochs: int = 3,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    seed: int = 42,
    patience: int = 2,
    device: str | None = None,
    dataset_version: str = "2.0.0",
) -> dict[str, Any]:
    """Train and validate an image classification model with checkpoint saving and calibration."""
    set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Executing training on: {dev}")

    # Canonical labels: 0 = FAKE, 1 = REAL
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    print(f"Loading base processor and model: {base_model_name}...")
    processor = AutoImageProcessor.from_pretrained(base_model_name)
    model = AutoModelForImageClassification.from_pretrained(
        base_model_name,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(dev)

    # PyTorch Datasets and DataLoaders
    train_dataset = FrameCheckDataset(train_records, processor=processor)
    val_dataset = FrameCheckDataset(val_records, processor=processor)

    # Note: On Windows CPU, num_workers=0 avoids multiprocessing spawn overhead
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    total_training_steps = len(train_loader) * epochs
    warmup_steps = int(0.1 * total_training_steps)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps
    )
    criterion = nn.CrossEntropyLoss()

    checkpoint_path = Path(output_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, Any]] = []

    print(f"\nStarting training: {epochs} epochs | {len(train_records)} train | {len(val_records)} val")
    print(f"Batch size: {batch_size} | LR: {learning_rate} | Warmup steps: {warmup_steps}")

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ── Training Loop ─────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch["pixel_values"].to(dev)
            labels = batch["label"].to(dev)

            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * len(labels)
            preds = outputs.logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        avg_train_loss = train_loss / train_total if train_total > 0 else 0.0
        avg_train_acc = train_correct / train_total if train_total > 0 else 0.0

        # ── Validation Loop ───────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds: list[int] = []
        val_targets: list[int] = []
        val_real_probs: list[float] = []

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(dev)
                labels = batch["label"].to(dev)

                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)

                val_loss += loss.item() * len(labels)
                probs = torch.softmax(outputs.logits, dim=-1)
                real_probs = probs[:, 1].cpu().numpy()

                val_preds.extend(outputs.logits.argmax(dim=-1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                val_real_probs.extend(real_probs)

        avg_val_loss = val_loss / len(val_targets) if val_targets else 0.0
        val_metrics = compute_binary_metrics(val_targets, val_real_probs)

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(avg_train_loss, 4),
            "train_accuracy": round(avg_train_acc, 4),
            "val_loss": round(avg_val_loss, 4),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_fpr": val_metrics["false_positive_rate"],
            "val_fnr": val_metrics["false_negative_rate"],
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(epoch_stats)

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"Macro F1: {val_metrics['macro_f1']:.4f}"
        )

        # ── Checkpoint Saving ─────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            patience_counter = 0

            print(f"  -> Validation loss improved to {avg_val_loss:.4f}. Saving best checkpoint...")
            model.save_pretrained(str(checkpoint_path))
            processor.save_pretrained(str(checkpoint_path))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without validation loss improvement.")
                break

    training_time = round(time.time() - t_start, 2)
    print(f"\nTraining completed in {training_time} s. Best epoch: {best_epoch} with Val Loss: {best_val_loss:.4f}")

    # ── Post-Training Calibration (Strictly on Validation Data) ──────────────
    print("\nExecuting confidence calibration strictly on validation data...")
    calib_config = calibrate_uncertainty_thresholds(val_targets, val_real_probs)
    save_calibration_config(calib_config, checkpoint_path.parent / "calibration_config.json")

    # ── Save Training Run Metadata ───────────────────────────────────────────
    run_metadata = {
        "model": {
            "name": "FrameCheck AI Detector",
            "architecture": "ViTForImageClassification",
            "base_backbone": base_model_name,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_macro_f1": round(best_val_f1, 4),
            "total_parameters": sum(p.numel() for p in model.parameters()),
            "id2label": id2label,
        },
        "dataset": {
            "version": dataset_version,
            "train_samples": len(train_records),
            "val_samples": len(val_records),
        },
        "configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "seed": seed,
            "optimizer": "AdamW",
            "scheduler": "cosine_with_warmup",
            "device": device,
        },
        "training_time_seconds": training_time,
        "calibration": {
            "calibrated_on": "validation_data_only",
            "ece_score": calib_config["expected_calibration_error"],
            "uncertain_low": calib_config["uncertain_low"],
            "uncertain_high": calib_config["uncertain_high"],
        },
        "epoch_history": history,
    }

    metadata_path = checkpoint_path.parent / "training_run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    print(f"Training run metadata saved to: {metadata_path}")
    return run_metadata
