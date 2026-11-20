# FrameCheck — AI Image Screening

An image screening tool that estimates whether an uploaded image may be AI-generated.
The FastAPI backend uses a compact quantized TorchScript FrameCheck ViT for regular
images and a second specialized ViT only for images no larger than 64×64 pixels. The
CIFAKE Keras model is kept as a separate historical baseline artifact.

---

## Model and training status

FrameCheck provides an end-to-end, reproducible training and evaluation pipeline for real-vs-AI image detection built with Hugging Face Vision Transformers (`google/vit-base-patch16-224-in21k`).

The backend currently uses the packaged `models/framecheck-tigas-vit-large/framecheck_model_int8.torchscript.pt` for regular images and routes images up to 64×64 pixels to `models/framecheck-dima-vit/framecheck_model_int8.torchscript.pt`. Both artifacts are quantized TorchScript models, so the API does not download weights on startup. The main checkpoint uses its validation temperature and uncertainty band; the small-image model uses no post-hoc calibration. The original float32 checkpoints remain excluded from Git. The legacy CIFAKE CNN (`models/cifake_detector.keras`) is archived for historical comparison only.

**AIDE research candidate:** its MIT-licensed source is vendored under `third_party/AIDE`; see [`docs/aide-experiment.md`](docs/aide-experiment.md) for the workspace patch, weight/data licensing boundaries, and CPU deployment constraints. It is not the active website model.

**Colab training status:** a new Hugging Face training notebook is prepared at [`notebooks/FrameCheck_RealWorld_Colab_Training.ipynb`](notebooks/FrameCheck_RealWorld_Colab_Training.ipynb), but it has not been run and no new checkpoint has been trained or installed. After a Colab run passes its held-out-generator gate, the safe importer installs it and the backend automatically selects it on restart. Until then, the existing model remains active. The notebook's dataset is research/non-commercial; this candidate is not licensed for commercial deployment.

### Measured benchmark results

The benchmark entries below use [`scripts/evaluate_detector.py`](scripts/evaluate_detector.py) and [`notebooks/evaluate_comparison.py`](notebooks/evaluate_comparison.py) on isolated test sets. The current website routing check is documented separately.

#### Current website routing check (October 26, 2026)

The website's shared backend inference function was run on 300 FAKE and 300 REAL images sampled with seed 42 from `imagesdataset/test`. The deployed low-resolution branch was used for these 32×32 images. UNCERTAIN results count as incorrect.

| Slice | Correct / Evaluated | Accuracy | Notes |
|---|---:|---:|---|
| CIFAKE test sample | 581 / 600 | **96.83%** | 3 UNCERTAIN; 16 incorrect binary predictions |
| Personal camera holdout | 61 / 71 | **85.92%** | 9 labeled FAKE; 1 UNCERTAIN; all images are REAL |

On the same CIFAKE sample, the previous single-model website checkpoint scored 404 / 600 (**67.33%**). The specialized model was kept off full-resolution images: used alone on the 71 camera photos it labeled 63 FAKE and only 2 REAL. Routing by original image dimensions avoids that severe camera false-alarm behavior. These results describe these particular datasets and do not establish accuracy on arbitrary images or current image generators.

#### 0. Expanded TIGAS cross-generator run (new default checkpoint)

This Colab run used 5,000 training images, 1,000 validation images, and 600 images from three generator families withheld from training (200 each from DALLE2, Midjourney, and SDXL). The best checkpoint was selected by validation loss; this is a research benchmark, not a deployment guarantee.

| Metric | Score |
|---|---:|
| **Held-out accuracy** | **73.67%** |
| **Macro F1** | **0.7352** |
| **ROC-AUC** | **0.8019** |
| **AI images misclassified as REAL** | **33.67% (101 / 300)** |
| **REAL images misclassified as FAKE** | **19.00% (57 / 300)** |
| **Held-out expected calibration error** | **0.1456** |

Per-generator accuracy: DALLE2 70.0%, Midjourney 68.5%, and SDXL 82.5% (200 images per family). This checkpoint is not reliable enough for authoritative authenticity decisions.

Full run details: [`models/tigas_multisource_evaluation_large.json`](models/tigas_multisource_evaluation_large.json). The model artifact is stored in Google Drive at `My Drive/FrameCheck/framecheck-tigas-vit-large-20261025` and locally at `models/framecheck-tigas-vit-large/`.

#### 1. Untouched Held-Out Test Set (600 images: 300 Real CIFAR-10, 300 Fake Stable Diffusion v1.4)

| Metric | Score | Detail |
|---|---|---|
| **Accuracy** | **63.00%** | 378 / 600 correct predictions |
| **Macro F1** | **0.6032** | Balanced harmonic mean |
| **ROC-AUC** | **0.7663** | Area under ROC curve |
| **Precision (REAL)** | **77.08%** | Fraction of real predictions that are true photographs |
| **Recall (REAL)** | **37.00%** | Fraction of actual photographs correctly identified |
| **Precision (FAKE)** | **58.55%** | Fraction of AI predictions that are synthetic |
| **Recall (FAKE)** | **89.00%** | Fraction of synthetic images detected |
| **False Positive Rate (FPR)** | **11.00%** | 33 synthetic images falsely passed as Real |
| **False Negative Rate (FNR)** | **63.00%** | **189 real images falsely flagged as AI!** |
| **Calibrated Uncertainty Rate** | **5.17%** | 31 images fall within the calibrated band $[0.46, 0.54]$ |

**Confusion Matrix (Held-Out Test Set):**
```
                    Predicted FAKE    Predicted REAL
Actual FAKE (SD1.4)     267 (TN)          33 (FP)
Actual REAL (CIFAR)     189 (FN)         111 (TP)
```

#### 2. Earlier Single-Checkpoint Website Check (Before Dimension Routing)

The current backend checkpoint and its shared image-inference function were checked on deterministic local samples; no personal images were sent to a third party. The full outputs are saved in [`models/web_api_fresh_image_evaluation.json`](models/web_api_fresh_image_evaluation.json).

| Slice | Correct / Evaluated | Accuracy | Notes |
|---|---:|---:|---|
| CIFAKE FAKE | 48 / 71 | 67.61% | 3 UNCERTAIN; 20 labeled REAL |
| CIFAKE REAL | 41 / 71 | 57.75% | 2 UNCERTAIN; 28 labeled FAKE |
| CIFAKE combined | 89 / 142 | 62.68% | Counts UNCERTAIN as not correct; 137 / 142 received a binary label |
| Personal camera REAL | 57 / 71 | 80.28% | 13 labeled FAKE; 1 UNCERTAIN; all 71 are real, so this measures false alarms only |

CIFAKE images are 32×32 and should not be conflated with full-resolution camera photos. This exploratory local check uses the website's shared inference function but does not replace a deployed HTTP/service test or the 600-image unseen-generator benchmark.

#### 3. Earlier-model User Camera Test Set (71 Authentic Personal Photos across 24 Sessions)

A dedicated test set of 71 personal camera photographs was curated from personal camera rolls across 24 distinct capture sessions. These results apply to the earlier local model, not the expanded TIGAS checkpoint. The images are strictly held out and never used for training or calibration:

| Modality Slice | Total Images | Correct (REAL) | False AI Alarms (FNR) | Recall |
|---|---|---|---|---|
| **Phone Camera** | 47 | 30 | 17 (36.17%) | 63.83% |
| **Webcam** | 10 | 9 | 1 (10.00%) | 90.00% |
| **Screenshots** | 14 | 5 | 9 (64.29%) | 35.71% |
| **Overall Camera Test** | **71** | **44** | **27 (38.03%)** | **61.97%** |

> [!CAUTION]
> **Real-World Reliability Warning:** The model suffers an empirical **38.03% False Fake Alarm rate on real camera photos** (nearly 4 out of every 10 authentic personal photos are falsely flagged as synthetic AI). Do NOT claim this model is reliable for production or forensic verification on real-world photos.

#### 4. Validation-Only Confidence Calibration

- Calibrated strictly on the 400 validation samples (`models/calibration_config.json`).
- **Expected Calibration Error (ECE):** **0.2192**.
- **Calibrated Uncertainty Band:** $[0.46, 0.54]$ (margin $\delta = 0.04$).
- When $0.46 \le P(\text{REAL}) \le 0.54$, the backend returns `"label": "UNCERTAIN"` and the frontend displays an honest **BORDERLINE / INCONCLUSIVE** status.

---

## Dataset composition & leakage prevention

The training pipeline constructs a multi-source balanced dataset with rigorous deduplication and leakage barriers:

1. **Sources & Generators:**
   - **Real Photographs:** Sourced from CIFAR-10 real subsets and user camera rolls with metadata and source labels preserved (`imagesdataset/camera_test/manifest.json`).
   - **Synthetic Images:** Sourced from Stable Diffusion v1.4 text-to-image generations (`imagesdataset/`).
2. **Deduplication:**
   - SHA-256 hash matching removes exact byte duplicates.
   - Difference hashing (`dHash`) with a Hamming distance threshold $\le 4$ removes visually near-duplicate images and burst shots.
   - Cross-split purging guarantees no visual duplicates bridge the train, validation, or test sets.
3. **Session-Grouped Splitting:**
   - User camera photos are grouped into 24 distinct timestamped sessions so burst photos from the same scene never leak across splits.
   - The test splits (both benchmark and personal camera sets) are completely untouched during all training and model selection steps.

---

## Reproducible training & evaluation pipeline

### 1. Run the full training pipeline

To regenerate the balanced dataset manifest, train the Vision Transformer, calibrate uncertainty on validation data, and run evaluations:

```powershell
# Windows PowerShell
.venv\Scripts\python scripts/train_pipeline.py --train --batch-size 16 --epochs 3 --lr 5e-5 --seed 42

# Or run with default settings
.venv\Scripts\python scripts/train_pipeline.py
```

Options:
- `--train`: Execute the full training loop (defaults to evaluating existing checkpoint if omitted).
- `--dataset-size`: Total multi-source dataset size (default: 3000 -> 2000 train, 400 val, 600 test).
- `--epochs`: Number of fine-tuning epochs (default: 3).
- `--batch-size`: Batch size per step (default: 16).
- `--lr`: Learning rate with AdamW and cosine decay (default: 5e-5).
- `--seed`: Deterministic random seed (default: 42).

### 2. Run standalone evaluations

To evaluate any Hugging Face model checkpoint across the untouched held-out test set and the personal camera test set:

```powershell
.venv\Scripts\python scripts/evaluate_detector.py --model models/framecheck-ai-detector --batch-size 32
```

Outputs full sliced metrics by source, generator, and modality, and saves structured JSON to `models/evaluation_results.json`.

### 3. Interactive Jupyter Notebook

For step-by-step interactive exploration, open [`notebooks/train_broad_detector.ipynb`](notebooks/train_broad_detector.ipynb), which contains full dataset exploration, training loops, validation calibration, and sliced test analyses.

---

## Running locally

### Prerequisites

- Python **3.12** (installed at `.venv` in the project root — see below)
- Any static file server for the frontend (see options below)

### 1 — Activate the virtual environment

The `.venv` with Python 3.12 is already created in the project root.

```powershell
# Windows PowerShell — activate
.venv\Scripts\Activate.ps1

# macOS / Linux / WSL
source .venv/bin/activate
```

If the venv doesn't exist yet (fresh clone):

```powershell
# Create it with Python 3.12
py -3.12 -m venv .venv --prompt framecheck
.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

> The backend uses PyTorch/Transformers for ViT inference. See `backend/requirements.txt`
> for the CPU-only PyTorch installation steps.

### 2 — Configure environment variables (optional)

```powershell
copy backend\.env.example backend\.env
# Edit backend\.env if the model lives at a non-default path,
# or if you need to add a different frontend origin to CORS.
```

The backend auto-reads `backend/.env` via `python-dotenv`.

### 3 — Start the backend API

```powershell
# From the project root (not from inside backend/)
.venv\Scripts\python -m uvicorn backend.app:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`.
Check `http://127.0.0.1:8000/api/health` in a browser to confirm it started.

Expected response when the model **is loaded**:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "C:\\...\\models\\framecheck-ai-detector",
  "model_error": ""
}
```

Expected response when the model **is unavailable**:

```json
{
  "status": "model_unavailable",
  "model_loaded": false,
  "model_id": "...",
  "model_error": "Could not load model: ..."
}
```

### 4 — Serve the frontend

The frontend is plain HTML/CSS/JS — no build step needed.

**Option A — VS Code Live Server** (recommended for development)
- Install the "Live Server" extension in VS Code
- Right-click `frontend/index.html` → "Open with Live Server"
- Default URL: `http://127.0.0.1:5500`

**Option B — Python's built-in server**

```powershell
python -m http.server 5500 --directory frontend
```

Then open `http://localhost:5500` in a browser.

**Option C — Any other static server**
- Whatever origin you use, add it to `FRAMECHECK_ALLOWED_ORIGINS` in `backend/.env`
  (comma-separated, no trailing slash), then restart the backend.

### 5 — Verify the connection

- The topbar chip should show a **green dot + "MODEL READY"** if everything is working.
- A **yellow dot + "MODEL MISSING"** means the backend is running but the detector model weights failed to load.
- A **red dot + "BACKEND OFFLINE"** means the API is not reachable (not started, wrong port, or CORS mismatch).

---

## Anonymous inference and deployment

FrameCheck does not require sign-in and does not save uploaded images or results.
The browser sends an image to the inference service for analysis; the service keeps
the image only for the duration of that request. Uploads are limited to 5 MB, and
the free CPU service processes one image at a time to avoid excessive memory use.

[`render.yaml`](render.yaml) defines the Render CPU inference service. The service
serves the frontend and API from one origin and needs no OAuth provider or database.
The Netlify site proxies `/api/*` to the Render service. Keep datasets and original
training weights out of Git; deploy the compact int8 TorchScript model artifact.

Render Free sleeps after inactivity and has limited memory and CPU, so the first
request may take longer while the service wakes and loads the model. It is suitable
for a small demo, not sustained high-volume traffic. [Render free limits](https://render.com/docs/free)
and [Render compute plans](https://render.com/docs/compute-plans).

### Local configuration

No secrets are needed. Start the backend with
`uvicorn backend.app:app --host 127.0.0.1 --port 8000`. For a separate local frontend,
set `FRAMECHECK_API_URL` in `frontend/config.js` to `http://127.0.0.1:8000`.

---

## API reference

### `GET /api/health`

Returns the backend and model status.

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "models/framecheck-tigas-vit-large",
  "model_error": "",
  "authentication_required": false,
  "uploads_saved": false
}
```

### `POST /api/predict`

Accepts an anonymous `multipart/form-data` request with a single field `file` (PNG,
JPG, WEBP, or GIF, max 5 MB). Returns the screening estimate and does not save the
image or result. If another image is already being analyzed, the service returns
HTTP 429 and asks the caller to retry shortly.

**Definitive classification (200):**

```json
{
  "label": "FAKE",
  "real_probability": 0.087432,
  "fake_probability": 0.912568,
  "score": 0.912568
}
```

**Borderline / Inconclusive classification (200):**

When calibrated validation bounds fall within $[0.46, 0.54]$, the backend returns an honest indeterminate label:

```json
{
  "label": "UNCERTAIN",
  "real_probability": 0.512030,
  "fake_probability": 0.487970,
  "score": 0.512030,
  "calibrated_band": [0.46, 0.54]
}
```

`label` is `"REAL"`, `"FAKE"`, or `"UNCERTAIN"`. Probabilities are fractions in `[0, 1]`.

**Model not ready (503):**

```json
{
  "detail": "The detector model is not ready. Check /api/health for details."
}
```

---

## Project structure

```
AI FAKE DETECTION/
├── frontend/
│   ├── index.html              — Modern web UI with upload zone and robo-pet
│   ├── styles.css              — Premium CSS design system (uncertain states, meters)
│   ├── app.js                  — Upload, health check polling, result rendering
│   └── config.js               — API base URL configuration
├── backend/
│   ├── app.py                  — FastAPI inference service with calibrated uncertainty
│   ├── requirements.txt        — Backend dependencies
│   └── .env.example            — Environment variable template
├── src/                        — Modular training and evaluation pipeline
│   ├── calibration/
│   │   └── calibrator.py       — Validation-only ECE & uncertainty band optimizer
│   ├── data/
│   │   ├── camera_split.py     — Curates user camera test set with session grouping
│   │   ├── dataset.py          — Multi-source balanced dataset with leakage barriers
│   │   └── deduplication.py    — SHA-256 and perceptual dHash cross-split deduplication
│   ├── evaluation/
│   │   ├── evaluator.py        — Batched PyTorch inference engine (batch_size=32)
│   │   └── metrics.py          — Acc, Macro F1, ROC-AUC, FPR, FNR, confusion matrix & slices
│   └── training/
│       └── trainer.py          — Reproducible PyTorch ViT fine-tuning loop with AdamW
├── scripts/
│   ├── train_pipeline.py       — CLI driver for manifest creation, training & evaluation
│   └── evaluate_detector.py    — Standalone evaluation CLI with formatted tables & JSON export
├── models/
│   ├── framecheck-ai-detector/ — Fine-tuned ViT model checkpoint (safetensors, config)
│   ├── calibration_config.json — Validation-only calibrated parameters & ECE bins
│   ├── evaluation_results.json — Sliced empirical metrics across benchmarks & camera tests
│   └── cifake_detector.keras   — Archived CIFAKE CNN baseline
├── data/
│   └── dataset_manifest.json   — Leakage-free dataset index (train, val, test)
├── imagesdataset/
│   └── camera_test/            — 71 authentic user camera photos across 24 capture sessions
├── notebooks/
│   ├── evaluate_comparison.py  — Comparative benchmark evaluation script
│   ├── train_broad_detector.ipynb — End-to-end interactive training & calibration notebook
│   └── evaluate_cifake.ipynb   — CIFAKE CNN evaluation and leakage analysis
├── MODEL_CARD.md               — Hugging Face model card with documented empirical limitations
└── README.md                   — Project documentation and execution instructions
```

---

## CORS notes

The backend only accepts requests from origins listed in `FRAMECHECK_ALLOWED_ORIGINS`.
The default allows `http://127.0.0.1:5500` and `http://localhost:5500` (VS Code Live Server).
Add your actual serving origin before deploying.

---

## Limitations, caveats, and non-production notice

> [!WARNING]
> **DO NOT CLAIM THIS MODEL IS RELIABLE ON REAL-WORLD PHOTOGRAPHY.**
> Rigorous empirical evaluation demonstrates significant performance degradation when moving from synthetic training benchmarks to real-world photography.

1. **High False Fake Alarm Rate on Real Photos (38.03% FNR):**
   - On the untouched test set of 71 personal camera photos across 24 distinct sessions, the model misclassified **27 authentic photographs as AI-generated (38.03% False Fake Alarm Rate)**.
   - Sliced by capture modality:
     - Phone camera photos: **36.17%** false fake alarm rate (17 / 47 misclassified).
     - Screenshots: **64.29%** false fake alarm rate (9 / 14 misclassified).
     - Webcam photos: **10.00%** false fake alarm rate (1 / 10 misclassified).
   - Real-world photos exhibit lens flare, ISP tone mapping, sensor noise, compression artifacts, and varied lighting conditions that the model frequently mistakes for synthetic diffusion generation artifacts.

2. **Severe False Negative Rate on Low-Resolution Images (63.00% FNR):**
   - On 300 held-out CIFAR-10 real images, the model misclassified **189 real images as AI-generated (63.00% FNR)**.
   - Upsampling low-resolution images to 224×224 patch grids introduces high-frequency bicubic interpolation artifacts that trigger the detector's synthetic pattern recognition.

3. **Generator Domain Narrowness:**
   - Synthetic training samples were generated using Stable Diffusion v1.4. Detectors trained predominantly on one architecture struggle to generalize to newer generative paradigms such as Flux, Midjourney v6, DALL-E 3, or video diffusion keyframes.

4. **Calibrated Uncertainty is Mandatory, Not Optional:**
   - To mitigate overconfident false accusations, the system uses validation-only confidence calibration (`models/calibration_config.json`). Any image with a predicted probability in $[0.46, 0.54]$ is labeled `"UNCERTAIN"`, and the frontend displays a prominent **"BORDERLINE / INCONCLUSIVE"** warning advising manual human review.

5. **Screening Aid Only — Not Forensic Proof:**
   - This tool is strictly an exploratory screening aid. It must **never** be used as standalone evidence for copyright disputes, academic integrity hearings, legal proceedings, journalism verification, or automated content moderation. Continuous human oversight and multi-modal provenance inspection (C2PA metadata, cryptographic signatures, historical context) remain essential.
