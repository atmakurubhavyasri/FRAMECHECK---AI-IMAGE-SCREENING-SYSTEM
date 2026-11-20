"""Confidence calibration and uncertainty band determination using VALIDATION data only.

Implements:
1. Expected Calibration Error (ECE) and reliability binning.
2. Low-confidence uncertainty margin calibration ([uncertain_low, uncertain_high]).
3. Exporting calibration configuration for inference services and the frontend.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence
import numpy as np


def compute_ece(
    y_true: Sequence[int] | np.ndarray,
    y_prob_real: Sequence[float] | np.ndarray,
    n_bins: int = 10,
) -> tuple[float, list[dict[str, Any]]]:
    """Compute Expected Calibration Error (ECE) and reliability bin stats.

    For binary classification, confidence is max(p, 1-p), and accuracy is whether
    the top-1 prediction matches y_true.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    p_real = np.asarray(y_prob_real, dtype=float)

    preds = (p_real >= 0.5).astype(int)
    confidences = np.maximum(p_real, 1.0 - p_real)
    accuracies = (preds == y_true_arr).astype(float)

    bin_boundaries = np.linspace(0.5, 1.0, n_bins + 1)
    bin_stats = []
    ece = 0.0
    total = len(y_true_arr)

    for i in range(n_bins):
        bin_low = bin_boundaries[i]
        bin_high = bin_boundaries[i + 1]
        mask = (confidences >= bin_low) & (confidences < bin_high if i < n_bins - 1 else confidences <= bin_high)
        count = int(np.sum(mask))

        if count > 0:
            avg_acc = float(np.mean(accuracies[mask]))
            avg_conf = float(np.mean(confidences[mask]))
            ece += (count / total) * abs(avg_acc - avg_conf)
        else:
            avg_acc = 0.0
            avg_conf = float((bin_low + bin_high) / 2.0)

        bin_stats.append({
            "bin_range": [round(bin_low, 3), round(bin_high, 3)],
            "count": count,
            "accuracy": round(avg_acc, 4),
            "confidence": round(avg_conf, 4),
        })

    return round(float(ece), 4), bin_stats


def calibrate_uncertainty_thresholds(
    y_true_val: Sequence[int] | np.ndarray,
    y_prob_val: Sequence[float] | np.ndarray,
    target_certain_accuracy: float = 0.85,
    candidate_margins: Sequence[float] = (0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12),
) -> dict[str, Any]:
    """Calibrate the uncertainty band [0.5 - delta, 0.5 + delta] strictly on validation data.

    Finds the optimal margin delta that separates high-error borderline predictions
    from confident decisions.
    """
    y_true_arr = np.asarray(y_true_val, dtype=int)
    p_real = np.asarray(y_prob_val, dtype=float)

    ece, bin_stats = compute_ece(y_true_arr, p_real)

    best_delta = 0.05
    best_score = -1.0
    margin_evaluations = []

    for delta in candidate_margins:
        low_b = 0.5 - delta
        high_b = 0.5 + delta

        uncertain_mask = (p_real >= low_b) & (p_real <= high_b)
        certain_mask = ~uncertain_mask

        uncertain_count = int(np.sum(uncertain_mask))
        certain_count = int(np.sum(certain_mask))

        if certain_count > 0:
            preds_certain = (p_real[certain_mask] >= 0.5).astype(int)
            certain_acc = float(np.mean(preds_certain == y_true_arr[certain_mask]))
        else:
            certain_acc = 0.0

        if uncertain_count > 0:
            preds_uncertain = (p_real[uncertain_mask] >= 0.5).astype(int)
            uncertain_acc = float(np.mean(preds_uncertain == y_true_arr[uncertain_mask]))
        else:
            uncertain_acc = 0.5

        margin_evaluations.append({
            "delta": delta,
            "band": [round(low_b, 3), round(high_b, 3)],
            "uncertain_count": uncertain_count,
            "uncertain_accuracy": round(uncertain_acc, 4),
            "certain_accuracy": round(certain_acc, 4),
            "uncertain_rate": round(uncertain_count / len(y_true_arr), 4),
        })

        # Score balances improving certain-accuracy while avoiding flagging too many images as uncertain
        # Favor delta where uncertain accuracy is near chance (~50%) and certain accuracy is high
        acc_gain = certain_acc - uncertain_acc
        uncertain_ratio = uncertain_count / len(y_true_arr)
        if uncertain_ratio < 0.25:  # avoid flagging more than 25% of validation as uncertain
            score = acc_gain - (uncertain_ratio * 0.5)
            if score > best_score:
                best_score = score
                best_delta = delta

    calibrated_config = {
        "calibrated_on": "validation_set_only",
        "validation_samples": len(y_true_arr),
        "expected_calibration_error": ece,
        "selected_margin_delta": round(best_delta, 3),
        "uncertain_low": round(0.5 - best_delta, 3),
        "uncertain_high": round(0.5 + best_delta, 3),
        "bin_statistics": bin_stats,
        "margin_search": margin_evaluations,
    }

    return calibrated_config


def save_calibration_config(config: dict[str, Any], filepath: Path | str = "models/calibration_config.json") -> Path:
    out = Path(filepath)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Confidence calibration config saved to: {out}")
    return out
