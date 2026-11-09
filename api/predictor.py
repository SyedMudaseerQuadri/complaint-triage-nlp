"""Model loading + prediction. Prefers the fine-tuned DistilBERT, falls back to the TF-IDF baseline."""
import json
import os
from pathlib import Path

import joblib
import numpy as np

from src.preprocess import clean_text

DEFAULT_MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parents[1] / "models"))


class BaselinePredictor:
    def __init__(self, pipeline, labels, model_version):
        self.pipeline, self.labels, self.model_version = pipeline, list(labels), model_version

    def predict_proba(self, text: str) -> np.ndarray:
        return self.pipeline.predict_proba([clean_text(text)])[0]


class TransformerPredictor:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        meta = json.loads((model_dir / "labels.json").read_text())
        self.labels, self.model_version = meta["labels"], meta["model_version"]
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.dev)

    def predict_proba(self, text: str) -> np.ndarray:
        torch = self._torch
        enc = self.tok(clean_text(text), truncation=True, max_length=256, return_tensors="pt").to(self.dev)
        with torch.no_grad():
            logits = self.model(**enc).logits[0].float()
        return torch.softmax(logits, -1).cpu().numpy()


def load_predictor(models_dir: Path | None = None, backend: str | None = None):
    """backend: 'auto' (default) | 'distilbert' | 'baseline' (env MODEL_BACKEND)."""
    models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
    backend = (backend or os.environ.get("MODEL_BACKEND", "auto")).lower()
    if backend in ("auto", "distilbert") and (models_dir / "distilbert" / "labels.json").exists():
        try:
            return TransformerPredictor(models_dir / "distilbert")
        except ImportError:
            if backend == "distilbert":
                raise
    if backend == "distilbert":
        raise FileNotFoundError(f"No fine-tuned model in {models_dir / 'distilbert'}")
    path = models_dir / "baseline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"No model found. Train one first (see README): missing {path}")
    blob = joblib.load(path)
    return BaselinePredictor(blob["pipeline"], blob["labels"], blob["model_version"])
