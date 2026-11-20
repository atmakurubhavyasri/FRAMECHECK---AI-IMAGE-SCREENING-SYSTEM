"""Fine-tune FrameCheck on a multi-generator TIGAS sample in Colab.

The dataset is downloaded selectively from Hugging Face. Entire source folders
for Midjourney, DALL-E 2, and Stable Diffusion XL are reserved for OOD testing;
they are never used for training, validation, or early stopping.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import disable_progress_bars


SEED = 42
DATASET_ID = "H1merka/TIGAS_dataset"
BASE_MODEL = "google/vit-base-patch16-224-in21k"
HOLDOUT_SOURCES = {"Midjourney", "DALLE2", "sd_xl"}
SOURCE_FOLDERS = [
    "ADM", "art001", "biggan", "DALLE2", "gaugan", "Glide",
    "Midjourney", "sd14", "sd15_1", "sd15_2", "sd_xl", "stargan", "VQDM",
]
MAX_TRAIN_PER_CLASS_SOURCE = 250
MAX_VALIDATION_PER_CLASS_SOURCE = 50
MAX_TEST_PER_CLASS_SOURCE = 100
EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
OUTPUT_DIR = Path("/content/framecheck-tigas-vit-large")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_from_image_path(value: str) -> str:
    parts = str(value).replace("\\", "/").split("/")
    return parts[1] if len(parts) > 1 else "unknown"


def select_balanced_splits(annotation_path: Path) -> tuple[list[dict], list[dict], list[dict], dict]:
    table = pd.read_csv(annotation_path)
    table["source"] = table["image_path"].map(source_from_image_path)
    table = table[table["source"].isin(SOURCE_FOLDERS)].copy()
    table["repo_path"] = table["image_path"].map(
        lambda value: "train/" + str(value).replace("\\", "/").lstrip("/")
    )
    table["label"] = 1 - table["label"].astype(int)  # canonical labels: 0=FAKE, 1=REAL

    train_rows: list[dict] = []
    validation_rows: list[dict] = []
    test_rows: list[dict] = []
    counts: dict[str, dict[str, int]] = {}

    for source in SOURCE_FOLDERS:
        source_rows = table[table["source"] == source]
        counts[source] = {}
        for label in (0, 1):
            class_rows = source_rows[source_rows["label"] == label].sample(frac=1, random_state=SEED)
            if source in HOLDOUT_SOURCES:
                chosen = class_rows.head(MAX_TEST_PER_CLASS_SOURCE)
                test_rows.extend(chosen.to_dict("records"))
                counts[source]["fake" if label == 0 else "real"] = len(chosen)
            else:
                val_rows = class_rows.head(MAX_VALIDATION_PER_CLASS_SOURCE)
                train_part = class_rows.iloc[len(val_rows):].head(MAX_TRAIN_PER_CLASS_SOURCE)
                validation_rows.extend(val_rows.to_dict("records"))
                train_rows.extend(train_part.to_dict("records"))
                counts[source]["fake" if label == 0 else "real"] = len(train_part)
                counts[source]["validation_fake" if label == 0 else "validation_real"] = len(val_rows)

    random.Random(SEED).shuffle(train_rows)
    random.Random(SEED).shuffle(validation_rows)
    random.Random(SEED).shuffle(test_rows)
    if not train_rows or not validation_rows or not test_rows:
        raise RuntimeError("Dataset selection produced an empty split. Check the TIGAS files and source folders.")
    if len({r["source"] for r in train_rows} & HOLDOUT_SOURCES):
        raise RuntimeError("A held-out generator leaked into training.")
    return train_rows, validation_rows, test_rows, {"source_counts": counts}


def fetch_selected_images(records: list[dict], cache_dir: Path) -> tuple[list[dict], int]:
    """Fetch only the sampled files, with bounded concurrency and retry on HF throttling."""
    unique = {row["repo_path"] for row in records}
    print(f"Fetching {len(unique):,} selected images (not whole source folders)...", flush=True)

    def fetch(repo_path: str) -> tuple[str, str]:
        last_error = None
        for attempt in range(3):
            try:
                local = hf_hub_download(
                    repo_id=DATASET_ID, repo_type="dataset", revision="main",
                    filename=repo_path, cache_dir=str(cache_dir),
                )
                return repo_path, local
            except Exception as error:
                last_error = error
                # Respect rate limits and transient network failures.
                time.sleep((2 ** attempt) + random.random())
        raise RuntimeError(f"Failed to download {repo_path}: {last_error}")

    downloaded: dict[str, str] = {}
    errors = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(fetch, name) for name in unique]
        for number, future in enumerate(as_completed(futures), 1):
            try:
                repo_path, local_path = future.result()
                downloaded[repo_path] = local_path
            except Exception as error:
                errors.append(str(error))
            if number % 100 == 0 or number == len(futures):
                print(f"Downloaded {number:,}/{len(futures):,}; failed={len(errors)}", flush=True)

    available = []
    for row in records:
        local = downloaded.get(row["repo_path"])
        if local:
            row["path"] = local
            available.append(row)
    if errors:
        print("Sample download failures (first 3):", errors[:3], flush=True)
    return available, len(errors)


class ImageRecords(Dataset):
    def __init__(self, records: list[dict], processor):
        self.records = records
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(record["path"]) as raw:
            image = ImageOps.exif_transpose(raw).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixel_values, int(record["label"])


def run_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.inference_mode():
        for pixels, labels in loader:
            outputs = model(pixel_values=pixels.to(device)).logits.float()
            all_logits.append(outputs.cpu())
            all_labels.extend(labels.numpy().tolist())
    return torch.cat(all_logits), np.asarray(all_labels, dtype=int)


def fit_temperature(logits: torch.Tensor, labels: np.ndarray) -> float:
    logits = logits.float()
    targets = torch.tensor(labels, dtype=torch.long)
    log_temperature = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.05, max_iter=100)
    criterion = nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits / log_temperature.exp().clamp(0.05, 20.0), targets)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0).item())


def compute_ece(labels: np.ndarray, probabilities: np.ndarray, bins: int = 15) -> float:
    confidence = np.maximum(probabilities, 1 - probabilities)
    prediction = (probabilities >= 0.5).astype(int)
    correct = prediction == labels
    ece = 0.0
    edges = np.linspace(0.5, 1.0, bins + 1)
    for index in range(bins):
        mask = (confidence >= edges[index]) & (confidence < edges[index + 1] if index < bins - 1 else confidence <= edges[index + 1])
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return ece


def summarize(labels: np.ndarray, p_real: np.ndarray, records: list[dict], temperature: float) -> dict:
    pred = (p_real >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred, labels=[0, 1]).ravel().tolist()
    slices = {}
    for generator in sorted({r["generator"] for r in records}):
        inds = np.asarray([i for i, r in enumerate(records) if r["generator"] == generator])
        if len(inds) == 0:
            continue
        y, p = labels[inds], p_real[inds]
        yp = (p >= 0.5).astype(int)
        stn, sfp, sfn, stp = confusion_matrix(y, yp, labels=[0, 1]).ravel().tolist()
        slices[generator] = {
            "n": int(len(inds)), "accuracy": float(accuracy_score(y, yp)),
            "macro_f1": float(f1_score(y, yp, average="macro", zero_division=0)),
            "false_fake_rate_on_real": float(sfn / (stp + sfn)) if stp + sfn else None,
            "false_real_rate_on_fake": float(sfp / (stn + sfp)) if stn + sfp else None,
        }
    return {
        "samples": int(len(labels)), "temperature": temperature,
        "accuracy": float(accuracy_score(labels, pred)),
        "macro_f1": float(f1_score(labels, pred, average="macro", zero_division=0)),
        "precision_real": float(precision_score(labels, pred, pos_label=1, zero_division=0)),
        "recall_real": float(recall_score(labels, pred, pos_label=1, zero_division=0)),
        "precision_fake": float(precision_score(labels, pred, pos_label=0, zero_division=0)),
        "recall_fake": float(recall_score(labels, pred, pos_label=0, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, p_real)) if len(np.unique(labels)) == 2 else None,
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else None,
        "confusion_matrix_fake_real": [[tn, fp], [fn, tp]],
        "ece": compute_ece(labels, p_real), "by_held_out_generator": slices,
    }


def main():
    disable_progress_bars()
    seed_everything(SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("This notebook needs a Colab GPU runtime. Choose Runtime > Change runtime type > T4 GPU.")
    print("GPU:", torch.cuda.get_device_name(0))

    print("Downloading TIGAS annotations, then selecting a balanced sample...")
    annotation_path = Path(hf_hub_download(
        repo_id=DATASET_ID, repo_type="dataset", revision="main",
        filename="train/annotations01.csv", cache_dir="/content/tigas-cache",
    ))
    train_records, val_records, test_records, split_info = select_balanced_splits(annotation_path)
    all_records = train_records + val_records + test_records
    all_records, failed_downloads = fetch_selected_images(all_records, Path("/content/tigas-cache"))
    # Restore the split boundaries after dropping any unavailable remote files.
    by_path = {row["repo_path"]: row for row in all_records}
    train_records = [by_path[row["repo_path"]] for row in train_records if row["repo_path"] in by_path]
    val_records = [by_path[row["repo_path"]] for row in val_records if row["repo_path"] in by_path]
    test_records = [by_path[row["repo_path"]] for row in test_records if row["repo_path"] in by_path]
    seen: set[str] = set()
    dedup_counts = {}
    for split_name, rows in (("test", test_records), ("validation", val_records), ("train", train_records)):
        kept = []
        for row in rows:
            digest = hashlib.sha256(Path(row["path"]).read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            row["sha256"] = digest
            row["generator"] = row["source"]
            row["label_name"] = "REAL" if row["label"] == 1 else "FAKE"
            row["source"] = "TIGAS"
            kept.append(row)
        dedup_counts[split_name] = len(rows) - len(kept)
        if split_name == "test":
            test_records = kept
        elif split_name == "validation":
            val_records = kept
        else:
            train_records = kept
    split_info["exact_duplicates_removed"] = dedup_counts
    split_info["failed_downloads"] = failed_downloads
    print("Samples:", {"train": len(train_records), "validation": len(val_records), "ood_test": len(test_records)})
    print("Exact duplicates removed:", split_info["exact_duplicates_removed"])
    print("OOD generators held out completely:", sorted(HOLDOUT_SOURCES))

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageClassification.from_pretrained(
        BASE_MODEL, num_labels=2, id2label={0: "FAKE", 1: "REAL"},
        label2id={"FAKE": 0, "REAL": 1}, ignore_mismatched_sizes=True,
    )
    device = torch.device("cuda")
    model.to(device)
    train_loader = DataLoader(ImageRecords(train_records, processor), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ImageRecords(val_records, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(ImageRecords(test_records, processor), batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    best_val_loss = float("inf")
    best_state = None
    history = []
    started = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_loss = total_n = 0
        for pixels, labels in train_loader:
            pixels, labels = pixels.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(pixel_values=pixels).logits
                loss = nn.functional.cross_entropy(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach()) * len(labels)
            total_n += len(labels)
        val_logits, val_labels = run_logits(model, val_loader, device)
        val_loss = float(nn.functional.cross_entropy(val_logits, torch.tensor(val_labels)).item())
        val_prediction = val_logits.argmax(dim=1).numpy()
        stats = {"epoch": epoch + 1, "train_loss": total_loss / max(total_n, 1), "val_loss": val_loss,
                 "val_accuracy": float(accuracy_score(val_labels, val_prediction)),
                 "val_macro_f1": float(f1_score(val_labels, val_prediction, average="macro", zero_division=0))}
        history.append(stats)
        print(stats, flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint.")
    model.load_state_dict(best_state)
    val_logits, val_labels = run_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_labels)
    val_p_real = torch.softmax(val_logits / temperature, dim=1)[:, 1].numpy()
    ece_before = compute_ece(val_labels, torch.softmax(val_logits, dim=1)[:, 1].numpy())
    ece_after = compute_ece(val_labels, val_p_real)

    # Select an uncertainty interval using validation predictions only.
    best_band = [0.45, 0.55]
    for margin in np.arange(0.02, 0.251, 0.01):
        certain = (val_p_real < 0.5 - margin) | (val_p_real > 0.5 + margin)
        if certain.sum() and float(((val_p_real[certain] >= 0.5) == val_labels[certain]).mean()) >= 0.85:
            best_band = [round(0.5 - margin, 3), round(0.5 + margin, 3)]
            break

    test_logits, test_labels = run_logits(model, test_loader, device)
    test_p_real = torch.softmax(test_logits / temperature, dim=1)[:, 1].numpy()
    report = {
        "dataset": DATASET_ID, "dataset_revision": "main (resolved by Hugging Face at run time)",
        "base_model": BASE_MODEL, "seed": SEED, "held_out_generators": sorted(HOLDOUT_SOURCES),
        "split_counts": {"train": len(train_records), "validation": len(val_records), "ood_test": len(test_records)},
        "split_details": split_info, "epochs": history, "training_seconds": round(time.time() - started, 1),
        "validation": {"ece_before_temperature_scaling": ece_before, "ece_after_temperature_scaling": ece_after,
                       "temperature": temperature, "uncertain_band": best_band},
        "ood_test": summarize(test_labels, test_p_real, test_records, temperature),
        "limitations": "TIGAS held-out source evaluation is a research benchmark, not proof of real-world authenticity. Personal camera images were not uploaded or evaluated in this run.",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    processor.save_pretrained(OUTPUT_DIR)
    (OUTPUT_DIR / "calibration_config.json").write_text(json.dumps({
        "method": "validation-only temperature scaling",
        "temperature": temperature,
        "ece_before": ece_before,
        "ece_after": ece_after,
        "uncertain_low": best_band[0],
        "uncertain_high": best_band[1],
        "validation_samples": len(val_records),
    }, indent=2))
    (OUTPUT_DIR / "evaluation_results.json").write_text(json.dumps(report, indent=2))
    (OUTPUT_DIR / "training_metadata.json").write_text(json.dumps({
        "base_model": BASE_MODEL, "dataset": DATASET_ID, "seed": SEED,
        "train_samples": len(train_records), "validation_samples": len(val_records),
        "test_samples": len(test_records), "held_out_generators": sorted(HOLDOUT_SOURCES),
        "epochs": history, "seconds": report["training_seconds"],
    }, indent=2))
    print(json.dumps(report, indent=2))
    print("Saved model and evaluation files to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
