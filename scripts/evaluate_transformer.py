"""Evaluates the fine-tuned DistilBERT transformer against the classical
TF-IDF/logistic-regression baseline on the real held-out AG News test set,
and logs both to MLflow -- same "measured, not assumed" principle as the
rest of this portfolio (see swiss-claims-assistant/scripts/train_risk_model.py
and bedding-franchise-erp's reorder-point simulation for the same pattern).

Why this is a separate script from the notebook: the notebook
(notebooks/finetune_distilbert_ag_news.ipynb) is where the *training*
happens, on Colab's GPU -- fine-tuning DistilBERT on a CPU-only sandbox
would take hours. This script only does *inference* (loading the already
fine-tuned weights and running them over the test set), which is fast even
on CPU, and its job is to independently reproduce the comparison the
notebook printed -- not just trust a number copied out of a notebook cell.

Requires the fine-tuned weights to already be sitting in
models/ag_news_distilbert/ (see README/transformer_classifier.py -- they're
deliberately not committed to git, so this only runs where someone has
actually run the notebook and copied its output in). Degrades with a clear
message, not a crash, if they're not there yet -- and starts no MLflow run
in that case, so a fresh checkout without weights doesn't leave behind a
misleading "failed" run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app import baseline  # noqa: E402
from app.transformer_classifier import TransformerClassifier  # noqa: E402

DOCS_DIR = ROOT / "docs"
CLASS_ORDER = [baseline.CLASS_NAMES[i] for i in sorted(baseline.CLASS_NAMES)]  # World, Sports, Business, Sci/Tech


def _evaluate_transformer(clf: TransformerClassifier, texts: list[str], true_labels: list[str]) -> dict:
    predicted = [p.label for p in clf.predict_batch(texts)]
    per_class = f1_score(true_labels, predicted, labels=CLASS_ORDER, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(true_labels, predicted)),
        "macro_f1": float(f1_score(true_labels, predicted, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "per_class_f1": dict(zip(CLASS_ORDER, (float(f) for f in per_class))),
    }


def _plot_comparison(baseline_pcf: dict, transformer_pcf: dict, out_path: Path) -> None:
    x = range(len(CLASS_ORDER))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - width / 2 for i in x], [baseline_pcf[c] for c in CLASS_ORDER], width, label="Baseline (TF-IDF + LogReg)")
    ax.bar([i + width / 2 for i in x], [transformer_pcf[c] for c in CLASS_ORDER], width, label="Fine-tuned DistilBERT")
    ax.set_xticks(list(x))
    ax.set_xticklabels(CLASS_ORDER)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0.8, 1.0)
    ax.set_title("Per-class F1: baseline vs. fine-tuned transformer (AG News test set)")
    ax.legend()
    fig.tight_layout()
    DOCS_DIR.mkdir(exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    transformer = TransformerClassifier()
    if not transformer.loaded:
        print(f"Transformer not loaded -- {transformer.load_error}")
        print("Nothing to evaluate or log; run the notebook and copy its output into "
              "models/ag_news_distilbert/ first. (No MLflow run started.)")
        return

    print("Training + evaluating classical baseline...")
    baseline_result = baseline.train_and_evaluate()
    bm = baseline_result.metrics

    print("Evaluating fine-tuned transformer on the same held-out test set...")
    test_texts, test_label_ids = baseline._load_csv(ROOT / "data" / "ag_news_test.csv")
    test_labels = [baseline.CLASS_NAMES[i] for i in test_label_ids]
    tm = _evaluate_transformer(transformer, test_texts, test_labels)

    delta_acc = tm["accuracy"] - bm.accuracy
    delta_f1 = tm["macro_f1"] - bm.macro_f1

    print("=" * 60)
    print(f"{'Metric':<15}{'Baseline':>12}{'DistilBERT':>15}{'Delta':>12}")
    print("=" * 60)
    print(f"{'Accuracy':<15}{bm.accuracy:>12.4f}{tm['accuracy']:>15.4f}{delta_acc:>+12.4f}")
    print(f"{'Macro F1':<15}{bm.macro_f1:>12.4f}{tm['macro_f1']:>15.4f}{delta_f1:>+12.4f}")
    for c in CLASS_ORDER:
        d = tm["per_class_f1"][c] - bm.per_class_f1[c]
        print(f"  {c:<10} baseline={bm.per_class_f1[c]:.4f}  distilbert={tm['per_class_f1'][c]:.4f}  delta={d:+.4f}")
    print("=" * 60)

    chart_path = DOCS_DIR / "transformer_vs_baseline.png"
    _plot_comparison(bm.per_class_f1, tm["per_class_f1"], chart_path)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("ag_news_topic_classification")
    with mlflow.start_run(run_name="distilbert_vs_baseline"):
        mlflow.log_params(
            {
                "baseline_model": "TF-IDF + LogisticRegression",
                "transformer_model": "distilbert-base-uncased (fine-tuned)",
                "n_train_baseline": bm.n_train,
                "n_test": bm.n_test,
            }
        )
        mlflow.log_metrics(
            {
                "baseline_accuracy": bm.accuracy,
                "baseline_macro_f1": bm.macro_f1,
                "transformer_accuracy": tm["accuracy"],
                "transformer_macro_f1": tm["macro_f1"],
                "delta_accuracy": delta_acc,
                "delta_macro_f1": delta_f1,
                **{f"baseline_f1_{c}": v for c, v in bm.per_class_f1.items()},
                **{f"transformer_f1_{c}": v for c, v in tm["per_class_f1"].items()},
            }
        )
        mlflow.log_artifact(str(chart_path))
        print(f"Logged to MLflow (sqlite:///mlflow.db, experiment 'ag_news_topic_classification').")
        print(f"View it with: mlflow ui --backend-store-uri sqlite:///mlflow.db")

    print(f"Comparison chart saved to {chart_path}")


if __name__ == "__main__":
    main()
