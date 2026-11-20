"""Reproducible, source-separated evaluation comparing FrameCheck ViT with a broad-generator detector.

Models compared:
1. FrameCheck ViT (models/framecheck-ai-detector): ViT-Base fine-tuned on TheKernel01/Tiny-GenImage.
   License: CC BY-NC-SA 4.0 (non-commercial).
2. Broad-generator detector (dima806/ai_vs_real_image_detection): ViT-Base fine-tuned on multi-source datasets.
   License: Apache 2.0.

Metrics reported per source:
- Confusion Matrix (TN, FP, FN, TP)
- False Positive Rate (FPR) & False Negative Rate (FNR)
- Precision, Recall, F1 for REAL and FAKE, Macro F1
- ROC-AUC score
- Uncertainty / borderline rate within the designated margin [0.45, 0.55]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

# Ensure PyTorch CPU backend is used for transformers
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

import numpy as np
from PIL import Image, ImageOps
import torch
from transformers import pipeline


def calculate_dhash(image: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) to identify duplicates/near-duplicates."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized)
    # Difference between consecutive columns
    diff = pixels[:, 1:] > pixels[:, :-1]
    return hex(int("".join(["1" if b else "0" for b in diff.flatten()]), 2))[2:].zfill(hash_size * hash_size // 4)


def compute_metrics(
    y_true: list[int] | np.ndarray,
    y_prob_real: list[float] | np.ndarray,
    threshold: float = 0.5,
    uncertain_band: tuple[float, float] = (0.45, 0.55),
) -> dict[str, Any]:
    """Compute evaluation metrics including confusion matrix, FPR, FNR, F1, and ROC-AUC."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob_real = np.asarray(y_prob_real, dtype=float)

    # Predictions at specified decision threshold (1=REAL, 0=FAKE)
    y_pred = (y_prob_real >= threshold).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    total = len(y_true)
    total_real = tp + fn
    total_fake = tn + fp

    fpr = float(fp / total_fake) if total_fake > 0 else 0.0
    fnr = float(fn / total_real) if total_real > 0 else 0.0

    precision_real = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_real = float(tp / total_real) if total_real > 0 else 0.0
    f1_real = (
        float(2 * precision_real * recall_real / (precision_real + recall_real))
        if (precision_real + recall_real) > 0
        else 0.0
    )

    precision_fake = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    recall_fake = float(tn / total_fake) if total_fake > 0 else 0.0
    f1_fake = (
        float(2 * precision_fake * recall_fake / (precision_fake + recall_fake))
        if (precision_fake + recall_fake) > 0
        else 0.0
    )

    macro_f1 = float((f1_real + f1_fake) / 2.0)
    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # ROC-AUC computation via rank/trapezoidal integration
    if len(np.unique(y_true)) > 1:
        try:
            from sklearn.metrics import roc_auc_score
            roc_auc = float(roc_auc_score(y_true, y_prob_real))
        except Exception:
            # Fallback pure-NumPy ROC-AUC
            desc_indices = np.argsort(y_prob_real)[::-1]
            y_prob_sorted = y_prob_real[desc_indices]
            y_true_sorted = y_true[desc_indices]

            distinct_val_indices = np.where(np.diff(y_prob_sorted))[0]
            threshold_idxs = np.r_[distinct_val_indices, y_true.size - 1]

            tps = np.cumsum(y_true_sorted)[threshold_idxs]
            fps = 1 + threshold_idxs - tps
            tps = np.r_[0, tps]
            fps = np.r_[0, fps]

            fpr_curve = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps)
            tpr_curve = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps)

            trapz_func = getattr(np, "trapezoid", getattr(np, "trapz", None))
            roc_auc = float(trapz_func(tpr_curve, fpr_curve))
    else:
        roc_auc = None

    # Borderline uncertainty
    low_b, high_b = uncertain_band
    uncertain_mask = (y_prob_real >= low_b) & (y_prob_real <= high_b)
    uncertain_count = int(np.sum(uncertain_mask))
    uncertain_rate = float(uncertain_count / total) if total > 0 else 0.0

    return {
        "total_samples": total,
        "true_real_count": total_real,
        "true_fake_count": total_fake,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],  # rows: [True FAKE, True REAL], cols: [Pred FAKE, Pred REAL]
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "precision_real": round(precision_real, 4),
        "recall_real": round(recall_real, 4),
        "f1_real": round(f1_real, 4),
        "precision_fake": round(precision_fake, 4),
        "recall_fake": round(recall_fake, 4),
        "f1_fake": round(f1_fake, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy": round(accuracy, 4),
        "roc_auc": round(roc_auc, 4),
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
    }


def load_model_pipeline(model_id_or_path: str) -> Any:
    """Load a Hugging Face image-classification pipeline."""
    print(f"Loading pipeline: {model_id_or_path}...")
    pipe = pipeline("image-classification", model=model_id_or_path, device=-1, top_k=2)
    return pipe


def get_real_probability(pipe: Any, image: Image.Image) -> float:
    """Execute forward inference and extract normalized P(REAL)."""
    raw_results = pipe(image)
    scores = {r["label"].upper(): float(r["score"]) for r in raw_results}
    real_p = scores.get("REAL", scores.get("HUMAN", 0.0))
    fake_p = scores.get("FAKE", scores.get("ARTIFICIAL", 0.0))
    total = real_p + fake_p
    return real_p / total if total > 0 else 0.5


def collect_images_from_source(
    source_dir: Path,
    sample_per_class: int | None = 250,
    seed: int = 42,
) -> list[tuple[Path, int, str]]:
    """Scan root/FAKE/ and root/REAL/, deduplicate by dHash, and sample balanced items."""
    rng = random.Random(seed)
    items: list[tuple[Path, int, str]] = []  # (path, label, dhash)

    seen_hashes: dict[str, Path] = {}

    for label_str, label_int in [("FAKE", 0), ("REAL", 1)]:
        class_folder = source_dir / label_str
        if not class_folder.is_dir():
            continue

        files = sorted(class_folder.glob("*.*"))
        valid_files = [f for f in files if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".gif"}]
        rng.shuffle(valid_files)

        class_items = []
        for fp in valid_files:
            try:
                with Image.open(fp) as img:
                    img_rgb = ImageOps.exif_transpose(img).convert("RGB")
                    h = calculate_dhash(img_rgb)
                    if h in seen_hashes:
                        # Near-duplicate detected; keep only one instance in test split
                        continue
                    seen_hashes[h] = fp
                    class_items.append((fp, label_int, h))
            except Exception:
                continue

            if sample_per_class and len(class_items) >= sample_per_class:
                break

        items.extend(class_items)

    return items


def run_evaluation(
    model_paths: dict[str, str],
    sources: dict[str, Path],
    sample_per_class: int | None = 250,
    seed: int = 42,
    output_json: Path | None = None,
) -> dict[str, Any]:
    """Execute full evaluation comparing all models across all sources."""
    results: dict[str, Any] = {
        "metadata": {
            "sample_per_class": sample_per_class,
            "seed": seed,
            "sources": {k: str(v) for k, v in sources.items()},
            "models": model_paths,
        },
        "per_source_evaluation": {},
    }

    # Load pipelines
    pipelines = {}
    for model_name, path_str in model_paths.items():
        pipelines[model_name] = load_model_pipeline(path_str)

    for source_name, source_path in sources.items():
        print(f"\nProcessing source: {source_name} ({source_path})...")
        items = collect_images_from_source(source_path, sample_per_class=sample_per_class, seed=seed)
        if not items:
            print(f"  No valid images found in {source_path}; skipping.")
            continue

        print(f"  Loaded {len(items)} images (after deduplication).")
        y_true = [it[1] for it in items]

        source_results: dict[str, Any] = {"total_evaluated": len(items)}

        for model_name, pipe in pipelines.items():
            print(f"  Evaluating {model_name}...")
            probs_real = []
            for fp, label_int, _ in items:
                try:
                    with Image.open(fp) as img:
                        img_rgb = ImageOps.exif_transpose(img).convert("RGB")
                        p = get_real_probability(pipe, img_rgb)
                        probs_real.append(p)
                except Exception as exc:
                    print(f"    Failed reading {fp.name}: {exc}")
                    probs_real.append(0.5)

            metrics = compute_metrics(y_true, probs_real)
            source_results[model_name] = metrics
            print(
                f"    {model_name} -> Acc: {metrics['accuracy']:.4f} | "
                f"Macro F1: {metrics['macro_f1']:.4f} | "
                f"ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"FPR: {metrics['false_positive_rate']:.4f} | "
                f"FNR: {metrics['false_negative_rate']:.4f} | "
                f"Uncertain: {metrics['uncertain_rate']:.2%}"
            )

        results["per_source_evaluation"][source_name] = source_results

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull evaluation results written to: {output_json}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible evaluation of FrameCheck ViT vs candidate detectors.")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=250,
        help="Number of samples per class per source (default 250 = 500 images per source; use 0 for all)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument(
        "--output",
        type=str,
        default="models/evaluation_comparison_metrics.json",
        help="Path to save output JSON metrics",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cifake_test_dir = project_root / "imagesdataset" / "test"

    sources: dict[str, Path] = {}
    if cifake_test_dir.is_dir():
        sources["CIFAKE_Test_Split"] = cifake_test_dir

    camera_dir = project_root / "imagesdataset" / "camera_test"
    if camera_dir.is_dir():
        sources["User_Camera_Photos_Test"] = camera_dir

    # Also check if any custom test dir exists
    custom_dir = project_root / "imagesdataset" / "custom_test"
    if custom_dir.is_dir():
        sources["Custom_Test"] = custom_dir

    models = {
        "FrameCheck_ViT_TinyGenImage": str(project_root / "models" / "framecheck-ai-detector"),
        "Broad_Candidate_ViT_dima806": "dima806/ai_vs_real_image_detection",
    }

    sample_cap = args.samples_per_class if args.samples_per_class > 0 else None
    out_path = project_root / args.output
    run_evaluation(models, sources, sample_per_class=sample_cap, seed=args.seed, output_json=out_path)


if __name__ == "__main__":
    main()
