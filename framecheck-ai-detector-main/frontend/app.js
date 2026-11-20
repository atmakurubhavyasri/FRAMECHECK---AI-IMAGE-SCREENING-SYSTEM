/**
 * FrameCheck — frontend logic
 *
 * Connects the upload UI to the FastAPI backend at FRAMECHECK_API_URL.
 * The backend serves a ViT image-screening model and reports health
 * dynamically to ensure transparent status for users.
 */

const API_BASE_URL = (window.FRAMECHECK_API_URL || "").replace(/\/$/, "");
const MAX_FILE_SIZE = 5 * 1024 * 1024;
const ALLOWED_TYPES = new Set(["image/png", "image/jpeg", "image/webp", "image/gif"]);

/* ── DOM refs ─────────────────────────────────────────────────────────── */
const input            = document.querySelector("#image-input");
const dropzone         = document.querySelector("#dropzone");
const uploadContent    = document.querySelector("#upload-content");
const previewContent   = document.querySelector("#preview-content");
const previewImage     = document.querySelector("#preview-image");
const previewName      = document.querySelector("#preview-name");
const previewSize      = document.querySelector("#preview-size");
const analyzeButton    = document.querySelector("#analyze-button");
const buttonLabel      = document.querySelector("#button-label");
const chooseImageButton  = document.querySelector("#choose-image");
const changeImageButton  = document.querySelector("#change-image");
const formMessage      = document.querySelector("#form-message");
const resultSection    = document.querySelector("#result-section");
const roboPet          = document.querySelector("#robo-pet");
const petHello         = roboPet.querySelector(".pet-hello");
const backendStatus    = document.querySelector("#backend-status");
const backendStatusLabel = document.querySelector("#backend-status-label");

let selectedFile = null;
let previewUrl   = null;
let healthPollTimer = null;
let activeAnalysisController = null;
let analysisId = 0;

document.querySelector("#year").textContent = new Date().getFullYear();

/* ── Utilities ────────────────────────────────────────────────────────── */
function showMessage(message, isError = false) {
  formMessage.textContent = message;
  formMessage.classList.toggle("error", isError);
}

function formatSize(bytes) {
  return bytes >= 1024 * 1024
    ? `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

function percent(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) return null;
  return Math.round(Math.max(0, Math.min(1, n > 1 ? n / 100 : n)) * 100);
}

/* ── Robo-pet moods ───────────────────────────────────────────────────── */
function setPetMood(mood) {
  roboPet.classList.remove("is-real", "is-fake", "is-scanning", "is-uncertain");
  switch (mood) {
    case "real":
      roboPet.classList.add("is-real");
      petHello.textContent = "Nice one! ✨";
      break;
    case "fake":
      roboPet.classList.add("is-fake");
      petHello.textContent = "Grr… synthetic!";
      break;
    case "uncertain":
      roboPet.classList.add("is-uncertain");
      petHello.textContent = "Hmm… inconclusive!";
      break;
    case "scanning":
      roboPet.classList.add("is-scanning");
      petHello.textContent = "Scanning…";
      break;
    default:
      petHello.textContent = "Ready to scan!";
  }
}

/* ── Backend health check ─────────────────────────────────────────────── */
/**
 * Polls GET /api/health and updates the topbar status chip.
 * States: checking → ok | degraded | error
 *   ok       – backend running, model loaded
 *   degraded – backend running, model not yet installed
 *   error    – backend unreachable or unexpected response
 */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, {
      method: "GET",
      credentials: "omit",
      cache: "no-store",
      signal: AbortSignal.timeout(5000),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    if (data.model_loaded) {
      setBackendStatus("ok", "MODEL READY");
    } else {
      // Backend running but model file missing – inform the user clearly
      setBackendStatus("degraded", "MODEL MISSING");
    }
  } catch {
    setBackendStatus("error", "BACKEND OFFLINE");
  }
}

function setBackendStatus(state, label) {
  if (!backendStatus) return;
  backendStatus.dataset.backendStatus = state;
  if (backendStatusLabel) backendStatusLabel.textContent = label;
}

/** Poll health every 30 s so the chip stays current without constant requests. */
function startHealthPolling() {
  checkHealth();
  clearInterval(healthPollTimer);
  healthPollTimer = setInterval(checkHealth, 30_000);
}

/* ── File selection / preview ─────────────────────────────────────────── */
function setFile(file) {
  if (!file) return;
  if (!ALLOWED_TYPES.has(file.type)) {
    showMessage("Choose a PNG, JPG, WEBP, or GIF image.", true);
    return;
  }
  if (file.size > MAX_FILE_SIZE) {
    showMessage("This image is larger than 5 MB. Choose a smaller file.", true);
    return;
  }
  cancelAnalysis();
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  previewImage.src = previewUrl;
  previewName.textContent = file.name;
  previewSize.textContent = formatSize(file.size);
  uploadContent.hidden = true;
  previewContent.hidden = false;
  analyzeButton.disabled = false;
  showMessage("");
  resetResult();
}

function clearFile() {
  cancelAnalysis();
  selectedFile = null;
  input.value = "";
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = null;
  previewImage.removeAttribute("src");
  previewContent.hidden = true;
  uploadContent.hidden = false;
  analyzeButton.disabled = true;
  showMessage("");
  resetResult();
}

/** Stop any older prediction before showing a different image. */
function cancelAnalysis() {
  analysisId += 1;
  activeAnalysisController?.abort();
  activeAnalysisController = null;
  analyzeButton.classList.remove("is-loading");
  buttonLabel.textContent = "Analyze image";
}

/* ── Result display ───────────────────────────────────────────────────── */
function resetResult() {
  resultSection.hidden = true;
  resultSection.classList.remove("is-fake", "is-uncertain");
  document.querySelector("#result-meter-fill").style.width = "0%";
  setPetMood("idle");
}

function displayResult(data) {
  const label = String(data.label || data.prediction || "").toLowerCase();
  const isUncertain = label === "uncertain" || label === "borderline" || label === "inconclusive";
  const isFake = !isUncertain && (label === "fake" || label === "ai-generated" || label === "synthetic");
  const fakeProbability = percent(data.fake_probability ?? data.fakeProbability);
  const realProbability = percent(data.real_probability ?? data.realProbability);

  let score;
  if (isUncertain) {
    score = realProbability ?? 50;
  } else if (isFake) {
    score = fakeProbability ?? percent(data.score);
  } else {
    score = realProbability ?? percent(data.score);
  }

  if (!label || score === null) {
    throw new Error("The analysis service returned an incomplete result.");
  }

  resultSection.classList.remove("is-fake", "is-uncertain");
  if (isUncertain) {
    resultSection.classList.add("is-uncertain");
    setPetMood("uncertain");
    document.querySelector("#result-status").textContent = "BORDERLINE / INCONCLUSIVE";
    document.querySelector("#result-title").textContent = "Inconclusive result";
    document.querySelector("#result-description").textContent =
      "The visual patterns show mixed or borderline signals between photographic camera noise and synthetic generation. Check provenance and image context.";
    document.querySelector("#result-score-value").textContent = score;
    document.querySelector("#result-score-label").textContent = `${realProbability}% real / ${fakeProbability}% AI`;
  } else if (isFake) {
    resultSection.classList.add("is-fake");
    setPetMood("fake");
    document.querySelector("#result-status").textContent = "SCREENING COMPLETE";
    document.querySelector("#result-title").textContent = "May be AI-generated";
    document.querySelector("#result-description").textContent =
      "The model found patterns commonly associated with synthetic images.";
    document.querySelector("#result-score-value").textContent = score;
    document.querySelector("#result-score-label").textContent = "synthetic estimate";
  } else {
    setPetMood("real");
    document.querySelector("#result-status").textContent = "SCREENING COMPLETE";
    document.querySelector("#result-title").textContent = "Likely real";
    document.querySelector("#result-description").textContent =
      "The model found patterns closer to the authentic photographs in its training data.";
    document.querySelector("#result-score-value").textContent = score;
    document.querySelector("#result-score-label").textContent = "real-image estimate";
  }

  document.querySelector("#result-meter-fill").style.width = `${score}%`;
  resultSection.hidden = false;
  resultSection.scrollIntoView({ behavior: "smooth", block: "center" });
}

/* ── Analysis request ─────────────────────────────────────────────────── */
async function analyze() {
  if (!selectedFile || analyzeButton.disabled) return;

  /* Enter loading state */
  const requestId = ++analysisId;
  const fileBeingAnalyzed = selectedFile;
  const controller = new AbortController();
  activeAnalysisController = controller;
  analyzeButton.disabled = true;
  analyzeButton.classList.add("is-loading");
  buttonLabel.textContent = "Analyzing…";
  resetResult();                 // hides previous result and resets pet to idle first
  showMessage("Sending image to the analysis service…");
  setPetMood("scanning");        // then put pet into scanning state

  const timeoutId  = window.setTimeout(() => controller.abort(), 60_000);
  const body = new FormData();
  body.append("file", fileBeingAnalyzed, fileBeingAnalyzed.name);

  try {
    const response = await fetch(`${API_BASE_URL}/api/predict`, {
      method: "POST",
      credentials: "omit",
      cache: "no-store",
      body,
      signal: controller.signal,
    });

    const payload = await response.json().catch(() => ({}));

    if (requestId !== analysisId || fileBeingAnalyzed !== selectedFile) return;

    if (response.status === 503) {
      // Detector model is unavailable or still loading
      throw new Error(
        "⚠ Detector model unavailable. " +
        "Ensure a configured model checkpoint is installed or the Hugging Face fallback can be reached, then restart the backend.",
      );
    }

    if (!response.ok) {
      throw new Error(payload.detail || payload.error || `Analysis failed (${response.status}).`);
    }

    displayResult(payload);
    showMessage("Analysis complete — see the estimate below.");
    // Refresh health chip after a successful prediction
    checkHealth();
  } catch (error) {
    if (requestId !== analysisId || fileBeingAnalyzed !== selectedFile) return;
    const message =
      error.name === "AbortError"
        ? "The analysis took too long. Please try again."
        : error.message || "Could not reach the analysis service. Please try again later.";
    showMessage(message, true);
    setPetMood("idle");
  } finally {
    window.clearTimeout(timeoutId);
    if (requestId === analysisId) {
      activeAnalysisController = null;
      analyzeButton.disabled = !selectedFile;
      analyzeButton.classList.remove("is-loading");
      buttonLabel.textContent = "Analyze image";
    }
  }
}

/* ── Event listeners ──────────────────────────────────────────────────── */
dropzone.addEventListener("click", (event) => {
  if (event.target.closest("button")) return;
  input.click();
});
chooseImageButton.addEventListener("click", () => input.click());
changeImageButton.addEventListener("click", () => input.click());
input.addEventListener("change", () => setFile(input.files?.[0]));
analyzeButton.addEventListener("click", analyze);

for (const eventName of ["dragenter", "dragover"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  });
}
for (const eventName of ["dragleave", "drop"]) {
  dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  });
}
dropzone.addEventListener("drop", (event) => setFile(event.dataTransfer?.files?.[0]));

const removeButton = document.querySelector("#remove-image");
removeButton.addEventListener("click", (event) => {
  event.stopPropagation();
  clearFile();
});
/* ── Boot ─────────────────────────────────────────────────────────────── */
startHealthPolling();
