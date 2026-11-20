# FrameCheck Model Card

**Current local checkpoint:** `models/framecheck-tigas-vit-large/` (ViT-Base patch16-224; 85.8M parameters)
**Task:** Experimental binary image screening (`FAKE` / `REAL`) with a validation-calibrated `UNCERTAIN` band.
**Readiness:** Research/demo system only. It is not suitable as proof of authenticity or for automated high-stakes decisions.

## Model and training

- Base architecture: `google/vit-base-patch16-224-in21k`.
- Dataset: `H1merka/TIGAS_dataset`; the run selected 5,000 training images, 1,000 validation images, and 600 held-out images from generator families excluded from training (200 each for DALLE2, Midjourney, and SDXL).
- The epoch-2 checkpoint was selected by lowest validation loss. The epoch-3 checkpoint had higher validation loss, despite higher training accuracy.
- Validation temperature: `1.081684`; validation ECE after scaling: `0.0364`; `UNCERTAIN` interval: `[0.48, 0.52]`.
- Run details: [`models/tigas_multisource_evaluation_large.json`](models/tigas_multisource_evaluation_large.json).

## Held-out results

| Metric | Result |
|---|---:|
| Accuracy | 73.67% (442 / 600) |
| Macro F1 | 0.7352 |
| ROC-AUC | 0.8019 |
| AI images classified as real | 33.67% (101 / 300) |
| Real images classified as AI | 19.00% (57 / 300) |
| Held-out ECE | 0.1456 |

Per-generator accuracy: DALLE2 70.0%, Midjourney 68.5%, SDXL 82.5% (200 images each). This benchmark does not establish reliability on other generators, edits, social-media recompression, or modern real-world photography.

The latest local website-inference check is saved at [`models/web_api_fresh_image_evaluation.json`](models/web_api_fresh_image_evaluation.json). On its 71-image CIFAKE samples, accuracy was 67.6% on fake thumbnails and 57.7% on real thumbnails. On 71 full-resolution user camera images, 57 were labeled real, 13 fake, and 1 uncertain. These small, separate slices are exploratory and do not establish general real-world performance.

## Deployment behavior

The backend selects this checkpoint when its local weight file exists. The 343 MB weights are excluded from Git. The Render blueprint initially uses the public Hugging Face fallback `dima806/ai_vs_real_image_detection` until the trained checkpoint is published to a model repository and configured through `FRAMECHECK_HF_MODEL`. The fallback is a different model and should not be described using the TIGAS metrics above.

## Limitations and responsible use

- On the held-out generator benchmark, one in three tested AI images was classified as real.
- The checkpoint also falsely flagged 19% of tested real images as AI.
- The 71-photo camera check is too small and includes only real images; its 80.3% real-label rate measures false alarms on this sample only.
- Models learn dataset-specific visual patterns; compression, resizing, edits, camera pipelines, generator changes, and domain shifts can change behavior substantially.
- Treat outputs as model estimates. Check provenance and context independently. Do not use this detector as forensic proof, sole evidence, or an autonomous moderation/disciplinary decision.

Personal camera images and the generated local dataset manifest are excluded from the repository. The training dataset's license and redistribution terms must be checked before publishing model weights.
