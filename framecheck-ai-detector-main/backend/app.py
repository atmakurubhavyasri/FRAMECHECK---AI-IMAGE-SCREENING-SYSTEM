"""HTTP inference service — FrameCheck AI image detector.

Production inference uses packaged, dynamically quantized TorchScript ViTs so
the service does not need Transformers or float32 checkpoints at runtime. A
specialized model is selected only for images no larger than 64×64. Each model's
label mapping and available calibration are read from local configuration files.
"""

from __future__ import annotations

import os
import json
import threading

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.staticfiles import StaticFiles

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB, to bound public inference requests
MAX_IMAGE_PIXELS = 16_000_000  # keep decoded photos within the free CPU service's memory
ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}
inference_slot = threading.BoundedSemaphore(1)

pipe: Any | None = None
low_resolution_pipe: Any | None = None
torch_runtime: Any | None = None
numpy_runtime: Any | None = None
model_lock = threading.Lock()
low_resolution_load_lock = threading.Lock()
model_load_error: str = ""
model_temperature: float = 1.0
LOW_RESOLUTION_MODEL_DIR = PROJECT_ROOT / "models" / "framecheck-dima-vit"
LOW_RESOLUTION_TORCHSCRIPT_MODEL = LOW_RESOLUTION_MODEL_DIR / "framecheck_model_int8.torchscript.pt"
LOW_RESOLUTION_MODEL_LABELS: dict[str, str] = {"0": "REAL", "1": "FAKE"}
low_resolution_model_temperature: float = 1.0
TRAINED_MODEL_DIR = PROJECT_ROOT / "models" / "framecheck-realworld-vit"
FALLBACK_MODEL_DIR = PROJECT_ROOT / "models" / "framecheck-tigas-vit-large"
MODEL_DIR = (
    TRAINED_MODEL_DIR
    if (TRAINED_MODEL_DIR / "framecheck_model_int8.torchscript.pt").is_file()
    else FALLBACK_MODEL_DIR
)
TORCHSCRIPT_MODEL = MODEL_DIR / "framecheck_model_int8.torchscript.pt"
MODEL_ID = "framecheck-colab-realworld-int8" if MODEL_DIR == TRAINED_MODEL_DIR else "framecheck-tigas-vit-large-int8"
LOW_RESOLUTION_MODEL_ID = "dima806/ai_vs_real_image_detection-int8"
MODEL_LABELS: dict[str, str] = {"0": "FAKE", "1": "REAL"}

# ── Startup / shutdown ────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(_: FastAPI):
    global pipe, low_resolution_pipe, torch_runtime, numpy_runtime, model_load_error, model_temperature
    global MODEL_LABELS
    try:
        import numpy as np
        import torch

        torch.set_num_threads(max(1, int(os.getenv("FRAMECHECK_TORCH_THREADS", "1"))))
        torch.set_num_interop_threads(1)
        if not TORCHSCRIPT_MODEL.is_file():
            raise FileNotFoundError(f"Packaged inference model is missing: {TORCHSCRIPT_MODEL}")
        print(f"Loading packaged quantized model: {TORCHSCRIPT_MODEL}")
        pipe = torch.jit.load(str(TORCHSCRIPT_MODEL), map_location="cpu").eval()
        torch_runtime = torch
        numpy_runtime = np
        try:
            model_config = json.loads((MODEL_DIR / "config.json").read_text(encoding="utf-8"))
            MODEL_LABELS = {str(k): str(v) for k, v in model_config.get("id2label", MODEL_LABELS).items()}
        except (OSError, ValueError, TypeError):
            MODEL_LABELS = {"0": "FAKE", "1": "REAL"}
        calibration = load_calibration_config()
        model_temperature = max(0.05, min(20.0, float(calibration.get("temperature", 1.0))))
        model_load_error = ""
        print(f"Model ready: {MODEL_ID} (temperature={model_temperature:.4f})")
        print(f"Low-resolution model will load on first matching image: {LOW_RESOLUTION_MODEL_ID}")
    except Exception as exc:
        pipe = None
        model_load_error = f"{type(exc).__name__}: {exc}"
        print(f"Model load FAILED: {model_load_error}")
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="FrameCheck prediction API", version="2.0.0", lifespan=lifespan)

render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
default_origins = "http://127.0.0.1:5500,http://localhost:5500,https://framecheck-image-screening.onrender.com"
if render_url:
    default_origins += f",{render_url}"
origins = [
    o.strip()
    for o in os.getenv("FRAMECHECK_ALLOWED_ORIGINS", default_origins).split(",")
    if o.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

DEFAULT_CALIBRATION_FILE = PROJECT_ROOT / "models" / "calibration_config.json"


def calibration_file(model_dir: Path = MODEL_DIR) -> Path:
    """Use calibration colocated with the packaged deployment model."""
    local_calibration = model_dir / "calibration_config.json"
    return local_calibration if local_calibration.is_file() else DEFAULT_CALIBRATION_FILE


def load_calibration_config(model_dir: Path = MODEL_DIR) -> dict[str, Any]:
    try:
        return json.loads(calibration_file(model_dir).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, TypeError):
        return {}


def get_uncertain_bounds(model_dir: Path = MODEL_DIR) -> tuple[float, float]:
    """Retrieve calibrated uncertainty bounds from file or environment variables."""
    low = float(os.getenv("FRAMECHECK_UNCERTAIN_LOW", "0.45"))
    high = float(os.getenv("FRAMECHECK_UNCERTAIN_HIGH", "0.55"))
    c_data = load_calibration_config(model_dir)
    low = float(c_data.get("uncertain_low", low))
    high = float(c_data.get("uncertain_high", high))
    if not (0.0 <= low < high <= 1.0):
        return 0.45, 0.55
    return low, high


def load_low_resolution_model() -> None:
    """Load the small-image checkpoint only when a matching upload arrives."""
    global low_resolution_pipe, LOW_RESOLUTION_MODEL_LABELS, low_resolution_model_temperature
    if low_resolution_pipe is not None or not LOW_RESOLUTION_TORCHSCRIPT_MODEL.is_file():
        return
    with low_resolution_load_lock:
        if low_resolution_pipe is not None:
            return
        try:
            low_resolution_pipe = torch_runtime.jit.load(
                str(LOW_RESOLUTION_TORCHSCRIPT_MODEL), map_location="cpu"
            ).eval()
            low_res_config = json.loads(
                (LOW_RESOLUTION_MODEL_DIR / "config.json").read_text(encoding="utf-8")
            )
            LOW_RESOLUTION_MODEL_LABELS = {
                str(k): str(v)
                for k, v in low_res_config.get("id2label", LOW_RESOLUTION_MODEL_LABELS).items()
            }
            low_res_calibration = load_calibration_config(LOW_RESOLUTION_MODEL_DIR)
            low_resolution_model_temperature = max(
                0.05, min(20.0, float(low_res_calibration.get("temperature", 1.0)))
            )
            print(f"Low-resolution model ready: {LOW_RESOLUTION_MODEL_ID}")
        except Exception as exc:
            low_resolution_pipe = None
            print(f"Low-resolution model unavailable: {type(exc).__name__}: {exc}")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> dict[str, Any]:
    low, high = get_uncertain_bounds()
    calibration = load_calibration_config()
    return {
        "status": "ok" if pipe is not None else "model_unavailable",
        "model_loaded": pipe is not None,
        "model_id": MODEL_ID,
        "low_resolution_model_loaded": low_resolution_pipe is not None,
        "low_resolution_model_available": LOW_RESOLUTION_TORCHSCRIPT_MODEL.is_file(),
        "low_resolution_model_id": LOW_RESOLUTION_MODEL_ID if low_resolution_pipe is not None else None,
        "model_error": model_load_error,
        "temperature": model_temperature,
        "calibration": {
            "uncertain_low": low,
            "uncertain_high": high,
            "has_calibrated_file": bool(
                calibration.get(
                    "calibrated",
                    calibration_file().is_file()
                    and calibration.get("method", "").lower() != "none",
                )
            ),
            "calibration_method": calibration.get("method", "none"),
        },
        "low_resolution_calibration": {
            "temperature": low_resolution_model_temperature,
            "has_calibrated_file": bool(
                load_calibration_config(LOW_RESOLUTION_MODEL_DIR).get("calibrated", False)
            ),
            "calibration_method": load_calibration_config(LOW_RESOLUTION_MODEL_DIR).get(
                "method", "none"
            ),
        },
        "authentication_required": False,
        "uploads_saved": False,
    }


# ── Predict ───────────────────────────────────────────────────────────────────
def open_image(raw: bytes) -> Image.Image:
    """Decode raw bytes → RGB PIL image, honoring EXIF orientation."""
    try:
        with Image.open(BytesIO(raw)) as img:
            if img.width * img.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="This image has too many pixels to analyze. Choose a smaller image.",
                )
            return ImageOps.exif_transpose(img).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(
            status_code=422, detail="The uploaded file is not a readable image."
        ) from exc


@app.post("/api/predict")
async def predict(
    response: Response, file: UploadFile = File(...)
) -> dict[str, Any]:
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=415, detail="Upload a PNG, JPG, WEBP, or GIF image."
        )

    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be 5 MB or smaller.")
    if not raw:
        raise HTTPException(status_code=422, detail="The uploaded image is empty.")
    if pipe is None or torch_runtime is None or numpy_runtime is None:
        raise HTTPException(
            status_code=503,
            detail="The detector model is not ready. Check /api/health for details.",
        )

    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    if not inference_slot.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="The detector is busy with another image. Please try again shortly.",
            headers={"Retry-After": "5"},
        )
    try:
        return await run_in_threadpool(analyze_image, raw)
    finally:
        inference_slot.release()


def analyze_image(raw: bytes) -> dict[str, Any]:
    """Run the shared inference path used by the HTTP route and local evaluation."""
    if pipe is None or torch_runtime is None or numpy_runtime is None:
        raise HTTPException(
            status_code=503,
            detail="The detector model is not ready. Check /api/health for details.",
        )

    image = open_image(raw)
    if max(image.size) <= 64:
        load_low_resolution_model()
    selected_pipe = pipe
    selected_model_labels = MODEL_LABELS
    selected_temperature = model_temperature
    calibration_dir = MODEL_DIR
    if max(image.size) <= 64 and low_resolution_pipe is not None:
        selected_pipe = low_resolution_pipe
        selected_model_labels = LOW_RESOLUTION_MODEL_LABELS
        selected_temperature = low_resolution_model_temperature
        calibration_dir = LOW_RESOLUTION_MODEL_DIR

    try:
        resized = image.resize((224, 224), resample=Image.Resampling.BILINEAR)
        pixels = numpy_runtime.asarray(resized, dtype=numpy_runtime.float32) / 127.5 - 1.0
        tensor = torch_runtime.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).contiguous()
        with model_lock, torch_runtime.inference_mode():
            probabilities = torch_runtime.softmax(selected_pipe(tensor)[0], dim=-1).tolist()
    except Exception as exc:
        raise HTTPException(
            status_code=500, detail="The detector could not analyze this image."
        ) from exc

    # Build a label→score lookup (case-insensitive)
    scores = {
        selected_model_labels.get(str(index), str(index)).upper(): float(probability)
        for index, probability in enumerate(probabilities)
    }
    real_probability = scores.get("REAL", scores.get("HUMAN", 0.0))
    fake_probability = scores.get("FAKE", scores.get("ARTIFICIAL", 0.0))

    # Normalise in case scores don't sum to exactly 1.0
    total = real_probability + fake_probability
    if total > 0:
        real_probability /= total
        fake_probability /= total

    # Pipeline scores are probabilities. Temperature scaling is equivalent to
    # applying softmax(logits / T), using p**(1/T) followed by normalization.
    if selected_temperature != 1.0:
        inv_temperature = 1.0 / selected_temperature
        real_scaled = max(real_probability, 1e-12) ** inv_temperature
        fake_scaled = max(fake_probability, 1e-12) ** inv_temperature
        scaled_total = real_scaled + fake_scaled
        real_probability = real_scaled / scaled_total
        fake_probability = fake_scaled / scaled_total

    # Return UNCERTAIN within this model's configured indeterminate band.
    UNCERTAIN_LOW, UNCERTAIN_HIGH = get_uncertain_bounds(calibration_dir)

    if UNCERTAIN_LOW <= real_probability <= UNCERTAIN_HIGH:
        label = "UNCERTAIN"
        confidence = round(real_probability, 6)
    elif real_probability > UNCERTAIN_HIGH:
        label = "REAL"
        confidence = real_probability
    else:
        label = "FAKE"
        confidence = fake_probability

    return {
        "label": label,
        "real_probability": round(real_probability, 6),
        "fake_probability": round(fake_probability, 6),
        "score": round(confidence, 6),
        "calibrated_band": [UNCERTAIN_LOW, UNCERTAIN_HIGH],
    }


# Serve the frontend and API from the same origin.
app.mount("/", StaticFiles(directory=PROJECT_ROOT / "frontend", html=True), name="frontend")
