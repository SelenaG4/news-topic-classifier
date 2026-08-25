"""End-to-end API tests against the real FastAPI app -- startup actually
trains the real baseline on real data (see tests/test_baseline.py for that
pipeline in isolation), so these tests exercise the full request path with
no mocking of the model layer.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_reports_baseline_loaded_and_transformer_status(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["baseline_loaded"] is True
    # No fine-tuned weights are checked into the repo (see README/.gitignore)
    # -- on a fresh checkout this must degrade gracefully, never crash.
    assert body["transformer_loaded"] is False
    assert body["transformer_status"] is not None
    assert "ag_news_distilbert" in body["transformer_status"]


def test_classes_lists_all_four_categories(client: TestClient) -> None:
    resp = client.get("/classes")
    assert resp.status_code == 200
    assert set(resp.json()["classes"]) == {"World", "Sports", "Business", "Sci/Tech"}


def test_baseline_metrics_are_the_real_measured_numbers(client: TestClient) -> None:
    resp = client.get("/baseline/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert body["n_train"] == 40000
    assert body["n_test"] == 7600
    assert body["accuracy"] > 0.85
    assert body["macro_f1"] > 0.85
    assert len(body["confusion_matrix"]) == 4
    assert set(body["class_order"]) == {"World", "Sports", "Business", "Sci/Tech"}


def test_predict_returns_baseline_and_null_transformer_with_status(client: TestClient) -> None:
    resp = client.post(
        "/predict",
        json={"text": "The stock market rallied today after the Fed held rates steady."},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["baseline"]["label"] in {"World", "Sports", "Business", "Sci/Tech"}
    assert 0.0 <= body["baseline"]["confidence"] <= 1.0
    # No transformer weights present -- must be an explicit null with a
    # human-readable reason, not a missing key or a silent 500.
    assert body["transformer"] is None
    assert "transformer_status" in body
    assert "ag_news_distilbert" in body["transformer_status"]


def test_predict_rejects_empty_text(client: TestClient) -> None:
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422


def test_predict_rejects_missing_text_field(client: TestClient) -> None:
    resp = client.post("/predict", json={})
    assert resp.status_code == 422


def test_landing_page_serves_html(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "News Topic Classifier" in resp.text


def test_openapi_schema_is_available(client: TestClient) -> None:
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    assert resp.json()["info"]["version"] == "1.0.0"
