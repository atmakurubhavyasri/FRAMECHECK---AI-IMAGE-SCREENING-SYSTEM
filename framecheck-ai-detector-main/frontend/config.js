// Production serves the frontend and API through the same Netlify origin.
// For local Live Server, point at the FastAPI backend on the loopback interface.
if (typeof window.FRAMECHECK_API_URL === "undefined") {
  const localPreview = ["localhost", "127.0.0.1"].includes(window.location.hostname)
    && window.location.port !== "8000";
  const apiHost = window.location.hostname === "localhost" ? "localhost" : "127.0.0.1";
  window.FRAMECHECK_API_URL = localPreview ? `http://${apiHost}:8000` : "";
}
