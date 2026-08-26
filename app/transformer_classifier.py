"""Wraps a fine-tuned DistilBERT topic classifier, loaded from a local
directory -- and reports honestly when it isn't there yet, rather than
crashing the service.

Why "loaded from a local directory" instead of downloading a model: this
project's transformer was fine-tuned in a separate Colab notebook
(notebooks/finetune_distilbert_ag_news.ipynb), not in this codebase's own
runtime -- the sandbox this project was originally built in can't reach
huggingface.co to download pretrained weights (see README). Loading
`from_pretrained()` on a local path never touches the network, so once the
fine-tuned weights are copied into MODEL_DIR, this service runs and serves
predictions fully offline -- it just isn't there on a fresh checkout until
someone runs the notebook and drops the output in.

Same honesty pattern as this portfolio's Emotion Insight Assistant, which
surfaces `weights_loaded: false` rather than silently predicting from
random-initialized weights and pretending the numbers mean something.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "ag_news_distilbert"


@dataclass
class TransformerPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class TransformerClassifier:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        self.loaded = False
        self.load_error: str | None = None
        self._tokenizer = None
        self._model = None
        self._id2label: dict[int, str] = {}
        self._try_load()

    def _try_load(self) -> None:
        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            self.load_error = (
                f"No fine-tuned weights found at {self.model_dir}. "
                "Run notebooks/finetune_distilbert_ag_news.ipynb and copy its output here."
            )
            return
        try:
            import torch  # noqa: F401 -- imported lazily so the whole app doesn't require torch just to boot
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
            self._model.eval()
            self._id2label = {int(k): v for k, v in self._model.config.id2label.items()}
            self.loaded = True
        except Exception as exc:  # noqa: BLE001 -- any load failure should degrade gracefully, not crash the app
            self.load_error = f"Found {self.model_dir} but failed to load it: {exc}"

    def predict(self, text: str) -> TransformerPrediction:
        if not self.loaded:
            raise RuntimeError(self.load_error or "Transformer model not loaded")
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: list[str], batch_size: int = 32) -> list[TransformerPrediction]:
        """Same model, batched -- for bulk scoring (e.g. scripts/evaluate_transformer.py
        running all 7,600 test-set rows) rather than the single-request API path.
        Batching the tokenizer + forward pass, instead of looping `predict()` one row at
        a time, is what keeps a full-test-set evaluation to well under a minute on CPU
        instead of several minutes -- the model and math are identical either way, this
        only changes how many rows go through the network at once.
        """
        if not self.loaded:
            raise RuntimeError(self.load_error or "Transformer model not loaded")

        import torch

        results: list[TransformerPrediction] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            inputs = self._tokenizer(batch, return_tensors="pt", truncation=True, max_length=256, padding=True)
            with torch.no_grad():
                logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)
            for row in probs:
                best_idx = int(torch.argmax(row))
                results.append(
                    TransformerPrediction(
                        label=self._id2label[best_idx],
                        confidence=float(row[best_idx]),
                        probabilities={self._id2label[i]: float(p) for i, p in enumerate(row)},
                    )
                )
        return results
