"""Evaluation CLI for reporting accuracy, macro F1, ROC-AUC, FPR, FNR, confusion matrix,
and slice-based breakdowns by image source and generator family.

Usage:
  python scripts/evaluate_detector.py --model models/framecheck-ai-detector
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from src.evaluation.evaluator import run_full_evaluation


def print_metrics_table(title: str, metrics: dict) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    print(f"  Total Samples Evaluated : {metrics.get('total_samples', 0)}")
    print(f"  Accuracy                : {metrics.get('accuracy', 0.0):.4f}")
    print(f"  Macro F1 Score          : {metrics.get('macro_f1', 0.0):.4f}")
    print(f"  ROC-AUC                 : {metrics.get('roc_auc', 'N/A')}")
    print(f"  Precision (REAL)        : {metrics.get('precision_real', 0.0):.4f}")
    print(f"  Recall (REAL)           : {metrics.get('recall_real', 0.0):.4f}")
    print(f"  Precision (FAKE)        : {metrics.get('precision_fake', 0.0):.4f}")
    print(f"  Recall (FAKE)           : {metrics.get('recall_fake', 0.0):.4f}")
    print(f"  False Positive Rate     : {metrics.get('false_positive_rate', 0.0):.4%}")
    print(f"  False Negative Rate     : {metrics.get('false_negative_rate', 0.0):.4%}")
    print(f"  Uncertain / Borderline  : {metrics.get('uncertain_rate', 0.0):.2%} ({metrics.get('uncertain_count', 0)} samples)")

    cm = metrics.get("confusion_matrix")
    if cm and len(cm) == 2 and len(cm[0]) == 2:
        tn, fp = cm[0]
        fn, tp = cm[1]
        print("\n  Confusion Matrix:")
        print(f"                    Predicted FAKE   Predicted REAL")
        print(f"    Actual FAKE :       {tn:<16} {fp:<16}")
        print(f"    Actual REAL :       {fn:<16} {tp:<16}")
    print(f"{'='*70}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FrameCheck detector against untouched test sets")
    parser.add_argument("--model", type=str, default="models/framecheck-ai-detector")
    parser.add_argument("--manifest", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--camera-manifest", type=str, default="imagesdataset/camera_test/manifest.json")
    parser.add_argument("--output", type=str, default="models/evaluation_results.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Load calibration band if available
    calib_file = PROJECT_ROOT / "models" / "calibration_config.json"
    uncertain_band = (0.45, 0.55)
    if calib_file.is_file():
        try:
            with open(calib_file, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                uncertain_band = (c_data.get("uncertain_low", 0.45), c_data.get("uncertain_high", 0.55))
                print(f"Loaded calibrated uncertainty band: [{uncertain_band[0]:.2f}, {uncertain_band[1]:.2f}]")
        except Exception:
            pass

    results = run_full_evaluation(
        model_path=args.model,
        manifest_path=args.manifest,
        camera_manifest_path=args.camera_manifest,
        output_path=args.output,
        device=args.device,
        uncertain_band=uncertain_band,
    )

    evals = results.get("evaluations", {})
    if "held_out_test_split" in evals:
        test_eval = evals["held_out_test_split"]
        print_metrics_table("OVERALL UNTOUCHED HELD-OUT TEST METRICS", test_eval["overall"])

        print("\nBREAKDOWN BY IMAGE SOURCE:")
        for src_name, s_metrics in test_eval.get("by_source", {}).items():
            print_metrics_table(f"SOURCE SLICE: {src_name}", s_metrics)

        print("\nBREAKDOWN BY GENERATOR:")
        for gen_name, g_metrics in test_eval.get("by_generator", {}).items():
            print_metrics_table(f"GENERATOR SLICE: {gen_name}", g_metrics)

    if "user_camera_test_set" in evals:
        cam_eval = evals["user_camera_test_set"]
        print_metrics_table("USER CAMERA TEST SET (AUTHENTIC UNSEEN PHOTOS)", cam_eval["overall"])
        for src_name, s_metrics in cam_eval.get("by_source", {}).items():
            print_metrics_table(f"CAMERA SOURCE: {src_name}", s_metrics)


if __name__ == "__main__":
    main()
