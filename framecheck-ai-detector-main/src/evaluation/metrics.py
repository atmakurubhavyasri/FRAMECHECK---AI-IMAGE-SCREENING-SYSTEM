"""Evaluation metrics module for Real-vs-AI classification.

Reports:
- Accuracy, Precision (REAL/FAKE), Recall (REAL/FAKE), Macro F1
- ROC-AUC score
- False Positive Rate (FPR) and False Negative Rate (FNR)
- 2x2 Confusion Matrix: [[TN, FP], [FN, TP]]
- Low-confidence / uncertainty rate
- Slice-based evaluations by image source and generator family
"""

from __future__ import annotations

from typing import Any, Sequence
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_binary_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_prob_real: Sequence[float] | np.ndarray,
    threshold: float = 0.5,
    uncertain_band: tuple[float, float] = (0.45, 0.55),
) -> dict[str, Any]:
    """Compute comprehensive classification metrics for 0=FAKE, 1=REAL."""
    y_true_arr = np.asarray(y_true, dtype=int)
    y_prob_arr = np.asarray(y_prob_real, dtype=float)

    total = len(y_true_arr)
    if total == 0:
        return {"error": "No samples to evaluate"}

    # Binary predictions at chosen threshold
    y_pred = (y_prob_arr >= threshold).astype(int)

    tp = int(np.sum((y_true_arr == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true_arr == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred == 0)))

    total_real = tp + fn
    total_fake = tn + fp

    accuracy = float((tp + tn) / total) if total > 0 else 0.0

    # Rates
    fpr = float(fp / total_fake) if total_fake > 0 else 0.0
    fnr = float(fn / total_real) if total_real > 0 else 0.0

    # Class REAL metrics
    prec_real = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_real = float(tp / total_real) if total_real > 0 else 0.0
    f1_real = (
        float(2 * prec_real * rec_real / (prec_real + rec_real))
        if (prec_real + rec_real) > 0
        else 0.0
    )

    # Class FAKE metrics
    prec_fake = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_fake = float(tn / total_fake) if total_fake > 0 else 0.0
    f1_fake = (
        float(2 * prec_fake * rec_fake / (prec_fake + rec_fake))
        if (prec_fake + rec_fake) > 0
        else 0.0
    )

    macro_f1 = float((f1_real + f1_fake) / 2.0)

    # ROC-AUC (requires at least one sample of each class)
    if len(np.unique(y_true_arr)) > 1:
        try:
            auc = float(roc_auc_score(y_true_arr, y_prob_arr))
        except Exception:
            auc = 0.5
    else:
        auc = None

    # Uncertainty calculation
    low_b, high_b = uncertain_band
    uncertain_mask = (y_prob_arr >= low_b) & (y_prob_arr <= high_b)
    uncertain_count = int(np.sum(uncertain_mask))
    uncertain_rate = float(uncertain_count / total) if total > 0 else 0.0

    return {
        "total_samples": total,
        "true_real": total_real,
        "true_fake": total_fake,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": [[tn, fp], [fn, tp]],  # [[True FAKE as FAKE, True FAKE as REAL], [True REAL as FAKE, True REAL as REAL]]
        "accuracy": round(accuracy, 4),
        "false_positive_rate": round(fpr, 4),
        "false_negative_rate": round(fnr, 4),
        "precision_real": round(prec_real, 4),
        "recall_real": round(rec_real, 4),
        "f1_real": round(f1_real, 4),
        "precision_fake": round(prec_fake, 4),
        "recall_fake": round(rec_fake, 4),
        "f1_fake": round(f1_fake, 4),
        "macro_f1": round(macro_f1, 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "uncertain_count": uncertain_count,
        "uncertain_rate": round(uncertain_rate, 4),
        "uncertain_band": [low_b, high_b],
    }


def evaluate_slices_by_source_and_generator(
    records: Sequence[dict[str, Any]],
    probabilities: Sequence[float],
    threshold: float = 0.5,
    uncertain_band: tuple[float, float] = (0.45, 0.55),
) -> dict[str, Any]:
    """Slice evaluation across image sources, generator families, and modalities."""
    sources: dict[str, list[int]] = {}
    generators: dict[str, list[int]] = {}
    modalities: dict[str, list[int]] = {}

    for idx, rec in enumerate(records):
        src = rec.get("source", "unknown")
        gen = rec.get("generator", "unknown")
        mod = rec.get("modality", "")
        sources.setdefault(src, []).append(idx)
        generators.setdefault(gen, []).append(idx)
        if mod:
            modalities.setdefault(mod, []).append(idx)

    y_true_all = [r["label"] for r in records]

    # Global metrics
    overall = compute_binary_metrics(y_true_all, probabilities, threshold, uncertain_band)

    # Per-source metrics
    by_source = {}
    for src_name, indices in sources.items():
        y_slice = [y_true_all[i] for i in indices]
        p_slice = [probabilities[i] for i in indices]
        by_source[src_name] = compute_binary_metrics(y_slice, p_slice, threshold, uncertain_band)

    # Per-generator metrics
    by_generator = {}
    for gen_name, indices in generators.items():
        y_slice = [y_true_all[i] for i in indices]
        p_slice = [probabilities[i] for i in indices]
        by_generator[gen_name] = compute_binary_metrics(y_slice, p_slice, threshold, uncertain_band)

    # Per-modality metrics
    by_modality = {}
    for mod_name, indices in modalities.items():
        y_slice = [y_true_all[i] for i in indices]
        p_slice = [probabilities[i] for i in indices]
        by_modality[mod_name] = compute_binary_metrics(y_slice, p_slice, threshold, uncertain_band)

    res = {
        "overall": overall,
        "by_source": by_source,
        "by_generator": by_generator,
    }
    if by_modality:
        res["by_modality"] = by_modality
    return res
