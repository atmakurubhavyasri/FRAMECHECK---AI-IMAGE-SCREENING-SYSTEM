"""Evaluate the backend's shared inference path on fresh, locally held-out images.

The user's camera photos stay on this machine and are never sent to a third party.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from backend import app as backend

SAMPLE_PER_CLASS = 71


def score(paths: list[Path], expected: str) -> dict:
    correct = 0
    uncertain = 0
    predicted = Counter()
    probabilities = []
    errors = []
    for path in paths:
        try:
            result = backend.analyze_image(path.read_bytes())
        except Exception as exc:
            errors.append(type(exc).__name__)
            continue
        label = result["label"]
        predicted[label] += 1
        correct += label == expected
        uncertain += label == "UNCERTAIN"
        probabilities.append(float(result["real_probability"]))
    evaluated = sum(predicted.values())
    return {
        "expected_label": expected,
        "samples": len(paths),
        "evaluated": evaluated,
        "correct": int(correct),
        "accuracy": correct / evaluated if evaluated else None,
        "uncertain": uncertain,
        "predicted_counts": dict(predicted),
        "mean_real_probability": sum(probabilities) / len(probabilities) if probabilities else None,
        "errors": dict(Counter(errors)),
    }


def main() -> None:
    fake_dir = ROOT / "imagesdataset" / "test" / "FAKE"
    real_dir = ROOT / "imagesdataset" / "test" / "REAL"
    manifest_path = ROOT / "imagesdataset" / "camera_test" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    camera_paths = sorted(ROOT / item["path"] for item in manifest["records"] if item["label_name"] == "REAL")
    rng = random.Random(42)
    fake_paths = sorted(fake_dir.glob("*.jpg"))
    real_paths = sorted(real_dir.glob("*.jpg"))
    if min(len(fake_paths), len(real_paths), len(camera_paths)) < SAMPLE_PER_CLASS:
        raise RuntimeError("Not enough local images for the planned 71-image groups")
    fake_paths = rng.sample(fake_paths, SAMPLE_PER_CLASS)
    real_paths = rng.sample(real_paths, SAMPLE_PER_CLASS)

    # Use exactly the checkpoint and calibration selected by backend/app.py.
    from transformers import pipeline
    print(f"Loading website checkpoint: {backend.HF_MODEL_ID}", flush=True)
    backend.pipe = pipeline("image-classification", model=backend.HF_MODEL_ID, device=-1, top_k=2)
    calibration = backend.load_calibration_config()
    backend.model_temperature = max(0.05, min(20.0, float(calibration.get("temperature", 1.0))))
    health = backend.health()
    if not health["model_loaded"]:
        raise RuntimeError(f"Model failed to load: {health}")
    print("Model ready; evaluating 71 held-out items per group…", flush=True)

    results = {
        "model_id": health["model_id"],
        "temperature": health["temperature"],
        "calibration": health["calibration"],
        "cifake_test_balanced_71_per_class": {
            "fake": score(fake_paths, "FAKE"),
            "real": score(real_paths, "REAL"),
        },
        "personal_camera_holdout_71_real": score(camera_paths, "REAL"),
        "limitations": [
            "CIFAKE test images are 32x32; camera photographs are full-resolution, so results are separate slices.",
            "The camera slice contains real images only; it measures false-positive behavior, not balanced accuracy.",
            "This is a local exploratory check, not an independent modern-generator benchmark.",
            "The 600-image TIGAS unseen-generator benchmark is reported separately in tigas_multisource_evaluation_large.json.",
        ],
    }
    output = ROOT / "models" / "web_api_fresh_image_evaluation.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))
    print("Saved evaluation to", output)


if __name__ == "__main__":
    main()
