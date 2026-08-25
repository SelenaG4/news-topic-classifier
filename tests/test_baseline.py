"""Tests the classical baseline against the real AG News data already in
data/ -- no mocks, no synthetic labels. Training runs once per test session
(~14s) and every test below asserts against that one real, measured result.
"""
from __future__ import annotations

import pytest

from app import baseline


@pytest.fixture(scope="module")
def result() -> baseline.BaselineResult:
    return baseline.train_and_evaluate()


def test_trains_on_the_full_real_dataset(result: baseline.BaselineResult) -> None:
    # 40,000-row stratified train sample, full 7,600-row official test set --
    # not a toy subset, so this pins us to the real data files staying real.
    assert result.metrics.n_train == 40000
    assert result.metrics.n_test == 7600


def test_beats_random_guessing_by_a_wide_margin(result: baseline.BaselineResult) -> None:
    # Random guessing across 4 balanced classes is ~0.25 accuracy. The
    # measured run of this exact pipeline got ~0.905 -- 0.85 is a
    # comfortably-real threshold, not a rubber-stamp one.
    assert result.metrics.accuracy > 0.85
    assert result.metrics.macro_f1 > 0.85


def test_no_class_is_badly_neglected(result: baseline.BaselineResult) -> None:
    # All four classes should be learnable from this data -- catches a
    # pipeline that's secretly only good at one or two categories.
    for class_name, f1 in result.metrics.per_class_f1.items():
        assert f1 > 0.75, f"{class_name} F1 unexpectedly low: {f1}"


def test_confusion_matrix_shape_matches_four_classes(result: baseline.BaselineResult) -> None:
    cm = result.metrics.confusion_matrix
    assert len(cm) == 4
    assert all(len(row) == 4 for row in cm)
    # Every real test example lands in exactly one predicted cell.
    assert sum(sum(row) for row in cm) == result.metrics.n_test


def test_predict_returns_a_confident_label_for_an_unambiguous_headline(
    result: baseline.BaselineResult,
) -> None:
    pred = result.predict(
        "Real Madrid beats Barcelona 3-1 in the Champions League final "
        "as fans celebrate across the city."
    )
    assert pred["label"] == "Sports"
    assert pred["confidence"] > 0.5
    assert set(pred["probabilities"]) == {"World", "Sports", "Business", "Sci/Tech"}
    assert abs(sum(pred["probabilities"].values()) - 1.0) < 1e-6


def test_predict_probabilities_are_a_valid_distribution(
    result: baseline.BaselineResult,
) -> None:
    pred = result.predict("The central bank raised interest rates by half a point today.")
    probs = pred["probabilities"]
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert abs(sum(probs.values()) - 1.0) < 1e-6
