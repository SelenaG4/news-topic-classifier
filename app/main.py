"""News Topic Classifier -- a real TF-IDF/logistic-regression baseline
(trained and evaluated on real AG News data at startup) shown side by side
with a fine-tuned DistilBERT transformer, once one has been fine-tuned
externally and dropped into models/ (see README -- this sandbox can't reach
Hugging Face Hub to fine-tune one in-session).
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import baseline
from app.transformer_classifier import TransformerClassifier

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="News Topic Classifier",
    description=(
        "A real classical-ML baseline (TF-IDF + logistic regression) compared against a "
        "fine-tuned DistilBERT transformer on AG News topic classification -- measured, "
        "not assumed."
    ),
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_baseline_result: baseline.BaselineResult | None = None
_transformer: TransformerClassifier | None = None


@app.on_event("startup")
def _startup() -> None:
    global _baseline_result, _transformer
    _baseline_result = baseline.train_and_evaluate()
    _transformer = TransformerClassifier()


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "baseline_loaded": _baseline_result is not None,
        "transformer_loaded": bool(_transformer and _transformer.loaded),
        "transformer_status": None if not _transformer else (
            "loaded" if _transformer.loaded else _transformer.load_error
        ),
    }


@app.get("/classes")
def classes() -> dict:
    return {"classes": list(baseline.CLASS_NAMES.values())}


@app.get("/baseline/metrics")
def baseline_metrics() -> dict:
    if _baseline_result is None:
        raise HTTPException(status_code=503, detail="Baseline not trained yet")
    m = _baseline_result.metrics
    return {
        "accuracy": m.accuracy,
        "macro_f1": m.macro_f1,
        "per_class_f1": m.per_class_f1,
        "confusion_matrix": m.confusion_matrix,
        "class_order": list(baseline.CLASS_NAMES.values()),
        "n_train": m.n_train,
        "n_test": m.n_test,
    }


class PredictIn(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)


@app.post("/predict")
def predict(payload: PredictIn) -> dict:
    if _baseline_result is None:
        raise HTTPException(status_code=503, detail="Baseline not trained yet")

    result = {"baseline": _baseline_result.predict(payload.text)}

    if _transformer and _transformer.loaded:
        pred = _transformer.predict(payload.text)
        result["transformer"] = {
            "label": pred.label,
            "confidence": pred.confidence,
            "probabilities": pred.probabilities,
        }
    else:
        result["transformer"] = None
        result["transformer_status"] = (
            _transformer.load_error if _transformer else "Transformer not initialized"
        )

    return result
