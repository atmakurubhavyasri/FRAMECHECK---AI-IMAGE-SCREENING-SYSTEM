# Model weights

The current local Hugging Face ViT fine-tune is in `framecheck-ai-detector/` and is
excluded from Git because its weights are large. The FastAPI backend uses it
automatically when present; set `FRAMECHECK_HF_MODEL` to override it.

`cifake_detector.keras` is a separate, small CIFAKE baseline artifact, also excluded
from Git. It is not loaded by the current ViT backend.

The CIFAKE baseline can be downloaded from its Colab session at `/content/cifake_detector.keras` with:

```python
from google.colab import files
files.download('/content/cifake_detector.keras')
```

Then put the downloaded file here.
