"""The classical baseline every fine-tuned transformer has to actually beat
before "we fine-tuned a transformer" earns its complexity: TF-IDF features
plus logistic regression, trained and evaluated on the real AG News data in
this repo, not a toy sample.

This is the "naive/drift baseline" of this project, same role as the naive
forecaster in swiss-economic-timeseries or the greedy heuristic in
bedding-franchise-erp -- fast, unglamorous, and the yardstick the more
sophisticated method has to clear.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

CLASS_NAMES = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}


def _load_csv(path: Path) -> tuple[list[str], list[int]]:
    df = pd.read_csv(path)
    # Title + description together is what a real headline-triage system
    # would actually have available at inference time -- using only one or
    # the other would be an easier, less representative task.
    texts = (df["title"].fillna("") + ". " + df["description"].fillna("")).tolist()
    labels = df["class_id"].astype(int).tolist()
    return texts, labels


@dataclass
class BaselineMetrics:
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    confusion_matrix: list[list[int]]
    n_train: int
    n_test: int


@dataclass
class BaselineResult:
    pipeline: Pipeline = field(repr=False)
    metrics: BaselineMetrics

    def predict(self, text: str) -> dict:
        proba = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        ranked = sorted(zip(classes, proba), key=lambda cp: -cp[1])
        return {
            "label": CLASS_NAMES[int(ranked[0][0])],
            "confidence": float(ranked[0][1]),
            "probabilities": {CLASS_NAMES[int(c)]: float(p) for c, p in zip(classes, proba)},
        }


def train_and_evaluate(
    train_csv: Path = DATA_DIR / "ag_news_train_sample.csv",
    test_csv: Path = DATA_DIR / "ag_news_test.csv",
) -> BaselineResult:
    train_texts, train_labels = _load_csv(train_csv)
    test_texts, test_labels = _load_csv(test_csv)

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=30000,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                    min_df=2,
                ),
            ),
            ("clf", LogisticRegression(max_iter=1000, C=10.0)),
        ]
    )
    pipeline.fit(train_texts, train_labels)

    predictions = pipeline.predict(test_texts)
    class_order = sorted(CLASS_NAMES)
    per_class = f1_score(test_labels, predictions, labels=class_order, average=None)

    metrics = BaselineMetrics(
        accuracy=float(accuracy_score(test_labels, predictions)),
        macro_f1=float(f1_score(test_labels, predictions, average="macro")),
        per_class_f1={CLASS_NAMES[c]: float(f) for c, f in zip(class_order, per_class)},
        confusion_matrix=confusion_matrix(test_labels, predictions, labels=class_order).tolist(),
        n_train=len(train_labels),
        n_test=len(test_labels),
    )
    return BaselineResult(pipeline=pipeline, metrics=metrics)
