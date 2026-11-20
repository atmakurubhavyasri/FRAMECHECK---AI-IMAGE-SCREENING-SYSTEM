"""CLI entry point for the complete, reproducible training and evaluation pipeline.

Usage:
  python scripts/train_pipeline.py --epochs 3 --batch-size 16 --seed 42
  python scripts/train_pipeline.py --skip-train   # Calibrate & evaluate existing checkpoint
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

# Force PyTorch
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")

from src.calibration.calibrator import calibrate_uncertainty_thresholds, save_calibration_config
from src.data.camera_split import curate_camera_test_set
from src.data.dataset import build_curated_dataset_manifest
from src.evaluation.evaluator import DetectorEvaluator, run_full_evaluation
from src.training.trainer import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="FrameCheck Complete Real-vs-AI Pipeline")
    parser.add_argument("--base-model", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--output-dir", type=str, default="models/framecheck-ai-detector")
    parser.add_argument("--manifest-path", type=str, default="data/dataset_manifest.json")
    parser.add_argument("--camera-manifest-path", type=str, default="imagesdataset/camera_test/manifest.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-samples-per-class", type=int, default=1000)
    parser.add_argument("--val-samples-per-class", type=int, default=200)
    parser.add_argument("--test-samples-per-class", type=int, default=300)
    parser.add_argument("--skip-train", action="store_true", help="Skip training and run calibration/evaluation on existing checkpoint")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("FrameCheck Real-vs-AI Detector Pipeline")
    print(f"Base model: {args.base_model} | Seed: {args.seed}")
    print("=" * 70)

    # Step 1: Ensure User Camera Test Set is Curated & Deduplicated
    camera_manifest = PROJECT_ROOT / args.camera_manifest_path
    if not camera_manifest.is_file():
        print("\n[Step 1/5] Curating separate user camera test set...")
        curate_camera_test_set(output_dir=camera_manifest.parent)
    else:
        print(f"\n[Step 1/5] User camera test set verified at: {camera_manifest}")

    # Step 2: Build / Verify Multi-Source Dataset Manifest
    manifest_file = PROJECT_ROOT / args.manifest_path
    if not manifest_file.is_file():
        print("\n[Step 2/5] Building balanced multi-source dataset with cross-split deduplication...")
        cifake_train = PROJECT_ROOT / "imagesdataset" / "train"
        cifake_test = PROJECT_ROOT / "imagesdataset" / "test"
        manifest_data = build_curated_dataset_manifest(
            cifake_train_dir=cifake_train,
            cifake_test_dir=cifake_test,
            output_manifest_path=manifest_file,
            train_samples_per_class=args.train_samples_per_class,
            val_samples_per_class=args.val_samples_per_class,
            test_samples_per_class=args.test_samples_per_class,
            seed=args.seed,
        )
    else:
        print(f"\n[Step 2/5] Loading existing dataset manifest: {manifest_file}")
        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

    train_records = manifest_data["splits"]["train"]
    val_records = manifest_data["splits"]["validation"]
    test_records = manifest_data["splits"]["held_out_test"]

    out_model_dir = PROJECT_ROOT / args.output_dir

    # Step 3: Model Training
    if not args.skip_train:
        print(f"\n[Step 3/5] Starting training loop ({args.epochs} epochs)...")
        train_model(
            train_records=train_records,
            val_records=val_records,
            base_model_name=args.base_model,
            output_dir=str(out_model_dir),
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            seed=args.seed,
            device=args.device,
        )
    else:
        print("\n[Step 3/5] Skipping training as requested (--skip-train). Using existing checkpoint.")

    # Step 4: Confidence Calibration (Strictly on Validation Data Only)
    print("\n[Step 4/5] Calibrating confidence thresholds on VALIDATION data...", flush=True)
    evaluator = DetectorEvaluator(str(out_model_dir), device=args.device or "cpu")
    val_y_true = [r["label"] for r in val_records]
    val_probs = evaluator.predict_records_probabilities(val_records, batch_size=32)

    calib_config = calibrate_uncertainty_thresholds(val_y_true, val_probs)
    calib_path = save_calibration_config(calib_config, PROJECT_ROOT / "models" / "calibration_config.json")
    uncertain_band = (calib_config["uncertain_low"], calib_config["uncertain_high"])
    print(f"  Calibrated uncertainty band: [{uncertain_band[0]:.2f}, {uncertain_band[1]:.2f}]", flush=True)

    # Step 5: Untouched Test Split & Camera Test Split Evaluation
    print("\n[Step 5/5] Evaluating on UNTOUCHED held-out test set & camera test set...")
    eval_results = run_full_evaluation(
        model_path=str(out_model_dir),
        manifest_path=str(manifest_file),
        camera_manifest_path=str(camera_manifest),
        output_path=str(PROJECT_ROOT / "models" / "evaluation_results.json"),
        device=args.device or "cpu",
        uncertain_band=uncertain_band,
    )

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION COMPLETE!")
    print(f"Model saved to: {out_model_dir}")
    print(f"Calibration saved to: {calib_path}")
    print(f"Evaluation report saved to: {PROJECT_ROOT / 'models' / 'evaluation_results.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
