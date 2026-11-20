"""Model evaluation engine across untouched test sets and camera splits with fast batched inference.

Executes batched forward inference, aggregates probability scores, and computes
all required metrics:
- Accuracy, precision, recall, macro F1, ROC-AUC
- False positive rate (FPR), false negative rate (FNR)
- 2x2 confusion matrix
- Per-source and per-generator slice breakdowns
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

# Set torch environment
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoImageProcessor, AutoModelForImageClassification

from src.evaluation.metrics import compute_binary_metrics, evaluate_slices_by_source_and_generator


class DetectorEvaluator:
    """Evaluates an image classifier against test splits and reports comprehensive metrics."""

    def __init__(self, model_path_or_id: str, device: str = "cpu"):
        self.device = torch.device(device)
        print(f"Initializing DetectorEvaluator with model: {model_path_or_id} on {self.device}", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(model_path_or_id)
        self.model = AutoModelForImageClassification.from_pretrained(model_path_or_id).to(self.device)
        self.model.eval()

        # Determine label mapping
        id2label = self.model.config.id2label
        self.label_to_id = {str(v).upper(): int(k) for k, v in id2label.items()}
        self.real_idx = self.label_to_id.get("REAL", self.label_to_id.get("HUMAN", 1))
        self.fake_idx = self.label_to_id.get("FAKE", self.label_to_id.get("ARTIFICIAL", 0))

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> tuple[float, float]:
        """Return (p_real, p_fake) for a single image."""
        rgb_img = ImageOps.exif_transpose(image).convert("RGB")
        inputs = self.processor(images=rgb_img, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        p_real = float(probs[self.real_idx])
        p_fake = float(probs[self.fake_idx])
        total = p_real + p_fake
        if total > 0:
            p_real /= total
            p_fake /= total
        return p_real, p_fake

    @torch.no_grad()
    def predict_batch(self, images: list[Image.Image]) -> list[tuple[float, float]]:
        """Return list of (p_real, p_fake) for a batch of images using vectorization."""
        rgb_imgs = [ImageOps.exif_transpose(im).convert("RGB") for im in images]
        inputs = self.processor(images=rgb_imgs, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)

        logits = self.model(pixel_values=pixel_values).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()

        results = []
        for i in range(len(images)):
            p_real = float(probs[i, self.real_idx])
            p_fake = float(probs[i, self.fake_idx])
            total = p_real + p_fake
            if total > 0:
                p_real /= total
                p_fake /= total
            results.append((p_real, p_fake))
        return results

    def predict_records_probabilities(
        self,
        records: Sequence[dict[str, Any]],
        batch_size: int = 32,
    ) -> list[float]:
        """Compute p(REAL) for all records efficiently in batches."""
        p_real_list: list[float] = []
        total = len(records)

        for start_idx in range(0, total, batch_size):
            batch_recs = records[start_idx : start_idx + batch_size]
            batch_imgs = []
            for rec in batch_recs:
                fp = Path(rec["path"])
                try:
                    with Image.open(fp) as opened_img:
                        batch_imgs.append(opened_img.copy())
                except Exception:
                    batch_imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

            batch_scores = self.predict_batch(batch_imgs)
            p_real_list.extend([p_r for p_r, _ in batch_scores])

        return p_real_list

    def evaluate_records(
        self,
        records: Sequence[dict[str, Any]],
        uncertain_band: tuple[float, float] = (0.45, 0.55),
        batch_size: int = 32,
    ) -> dict[str, Any]:
        """Run batched predictions on records and compute slice-based metrics."""
        y_true = []
        p_real_list = []
        valid_records = []

        total = len(records)
        print(f"Evaluating {total} records with batch_size={batch_size}...", flush=True)

        for start_idx in range(0, total, batch_size):
            batch_recs = records[start_idx : start_idx + batch_size]
            batch_imgs = []
            active_recs = []

            for rec in batch_recs:
                fp = Path(rec["path"])
                if not fp.is_file():
                    continue
                try:
                    with Image.open(fp) as opened_img:
                        batch_imgs.append(opened_img.copy())
                    active_recs.append(rec)
                except Exception as exc:
                    print(f"Warning: Failed reading {fp.name}: {exc}", flush=True)

            if not batch_imgs:
                continue

            batch_scores = self.predict_batch(batch_imgs)
            for rec, (p_real, _) in zip(active_recs, batch_scores):
                y_true.append(rec["label"])
                p_real_list.append(p_real)
                valid_records.append(rec)

            processed = min(start_idx + batch_size, total)
            if processed % 160 == 0 or processed == total:
                print(f"  Processed {processed}/{total} images", flush=True)

        metrics = evaluate_slices_by_source_and_generator(
            valid_records, p_real_list, threshold=0.5, uncertain_band=uncertain_band
        )
        return metrics


def run_full_evaluation(
    model_path: str = "models/framecheck-ai-detector",
    manifest_path: str = "data/dataset_manifest.json",
    camera_manifest_path: str = "imagesdataset/camera_test/manifest.json",
    output_path: str = "models/evaluation_results.json",
    device: str = "cpu",
    uncertain_band: tuple[float, float] = (0.45, 0.55),
    batch_size: int = 32,
) -> dict[str, Any]:
    """Execute complete evaluation across held-out untouched test set and camera test set."""
    evaluator = DetectorEvaluator(model_path, device=device)
    results: dict[str, Any] = {
        "model_path": model_path,
        "device": device,
        "uncertain_band": list(uncertain_band),
        "evaluations": {},
    }

    # 1. Evaluate on Dataset Held-out Test Split (if manifest exists)
    if Path(manifest_path).is_file():
        with open(manifest_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        held_out_records = data.get("splits", {}).get("held_out_test", [])
        if held_out_records:
            print("\n--- Evaluating Held-Out Untouched Test Split ---", flush=True)
            held_out_metrics = evaluator.evaluate_records(held_out_records, uncertain_band, batch_size=batch_size)
            results["evaluations"]["held_out_test_split"] = held_out_metrics
            ov = held_out_metrics["overall"]
            print(
                f"Held-out Test -> Acc: {ov['accuracy']:.4f} | Macro F1: {ov['macro_f1']:.4f} | "
                f"ROC-AUC: {ov['roc_auc']} | FPR: {ov['false_positive_rate']:.4f} | FNR: {ov['false_negative_rate']:.4f}",
                flush=True,
            )

    # 2. Evaluate on User Camera Test Set (if camera manifest exists)
    if Path(camera_manifest_path).is_file():
        with open(camera_manifest_path, "r", encoding="utf-8") as f:
            cam_data = json.load(f)
        cam_records = cam_data.get("records", [])
        if cam_records:
            print("\n--- Evaluating User Camera Test Set ---", flush=True)
            cam_metrics = evaluator.evaluate_records(cam_records, uncertain_band, batch_size=batch_size)
            results["evaluations"]["user_camera_test_set"] = cam_metrics
            cov = cam_metrics["overall"]
            print(
                f"Camera Test Set -> Total: {cov['total_samples']} | Acc (Real-Recall): {cov['recall_real']:.4f} | "
                f"False Fake Alarms (FNR): {cov['false_negative_rate']:.4f} | Uncertain: {cov['uncertain_rate']:.2%}",
                flush=True,
            )

    # Save output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull evaluation metrics saved to: {out}", flush=True)
    return results


if __name__ == "__main__":
    run_full_evaluation()
