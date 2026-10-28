import io
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from app import create_app


PROJECT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_DIR / "model"


@pytest.fixture()
def client():
    app = create_app(MODEL_DIR)
    app.config.update(TESTING=True)
    return app.test_client()


def _sample_readings():
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    return [float(value) for value in bundle["imputation_medians"].to_numpy()]


def _csv_bytes(rows, columns=None):
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    columns = columns or bundle["daily_reading_columns"]
    return pd.DataFrame(rows, columns=columns).to_csv(index=False).encode("utf-8")


def test_health_and_metrics_report_saved_model(client):
    health = client.get("/api/health").get_json()
    assert health["server_status"] == "ok"
    assert health["model_loaded"] is True
    assert health["metrics_available"] is True

    response = client.get("/api/metrics")
    payload = response.get_json()
    saved = json.loads((MODEL_DIR / "metrics.json").read_text(encoding="utf-8"))
    assert response.status_code == 200
    assert payload["metrics"]["accuracy"] == saved["metrics"]["accuracy"]
    assert payload["metrics"]["theft_f1"] == saved["metrics"]["theft_f1"]
    assert payload["metadata"]["test_records"] == 5173


def test_evaluation_endpoint_returns_real_test_split_results(client):
    payload = client.get("/api/evaluation").get_json()
    assert payload["available"] is True
    assert payload["confusion_matrix"] == [[4475, 274], [354, 70]]
    assert payload["class_counts"] == {"normal": 4749, "theft_labelled": 424, "total": 5173}
    assert len(payload["roc_curve"]["false_positive_rate"]) > 2
    assert len(payload["precision_recall_curve"]["precision"]) > 2


def test_valid_single_prediction_uses_loaded_pipeline(client):
    response = client.post("/api/predict", json={"readings": _sample_readings()})
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["success"] is True
    assert payload["prediction"] in {"Normal", "Theft-labelled"}
    assert 0 <= payload["theft_probability"] <= 1
    assert payload["threshold"] == pytest.approx(0.11846431169375599)


@pytest.mark.parametrize("readings", [[1, 2], [None] * 26, [-1] * 26])
def test_invalid_single_prediction_returns_clear_error(client, readings):
    response = client.post("/api/predict", json={"readings": readings})
    assert response.status_code == 400
    assert response.get_json()["success"] is False
    assert response.get_json()["error"]


def test_valid_csv_upload_returns_batch_predictions(client):
    sample = _sample_readings()
    csv_data = _csv_bytes([sample, [value * 1.15 for value in sample]])
    response = client.post(
        "/api/predict-csv",
        data={"file": (io.BytesIO(csv_data), "customers.csv")},
        content_type="multipart/form-data",
    )
    payload = response.get_json()
    assert response.status_code == 200
    assert payload["processed"] == 2
    assert len(payload["results"]) == 2
    assert payload["normal_count"] + payload["theft_labelled_count"] == 2
    assert all(0 <= row["theft_probability"] <= 1 for row in payload["results"])


def test_csv_with_missing_or_unexpected_columns_is_rejected(client):
    sample = _sample_readings()
    bundle = joblib.load(MODEL_DIR / "model.joblib")
    wrong_columns = list(bundle["daily_reading_columns"][:-1]) + ["FLAG"]
    csv_data = _csv_bytes([sample], wrong_columns)
    response = client.post(
        "/api/predict-csv",
        data={"file": (io.BytesIO(csv_data), "wrong_schema.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "columns" in response.get_json()["error"].lower()


def test_csv_with_missing_values_is_rejected(client):
    sample = _sample_readings()
    sample[3] = float("nan")
    response = client.post(
        "/api/predict-csv",
        data={"file": (io.BytesIO(_csv_bytes([sample])), "missing.csv")},
        content_type="multipart/form-data",
    )
    assert response.status_code == 400
    assert "missing" in response.get_json()["error"].lower()


def test_missing_model_artifacts_are_reported_safely(tmp_path):
    app = create_app(tmp_path / "empty-model-folder")
    app.config.update(TESTING=True)
    client = app.test_client()

    health = client.get("/api/health").get_json()
    assert health["server_status"] == "ok"
    assert health["model_loaded"] is False
    assert client.get("/api/metrics").get_json()["available"] is False
    prediction = client.post("/api/predict", json={"readings": []})
    assert prediction.status_code == 503
    assert prediction.get_json()["success"] is False


def test_dashboard_renders_the_same_metrics_returned_by_api(client):
    metrics = client.get("/api/metrics").get_json()["metrics"]
    page = client.get("/")
    assert page.status_code == 200
    html = page.get_data(as_text=True)
    for key in ("accuracy", "balanced_accuracy", "theft_precision", "theft_recall", "theft_f1", "roc_auc", "average_precision"):
        assert f'id="metric-{key}"' in html
        assert f"{metrics[key] * 100:.2f}%" in html
        assert f"Exact score: {metrics[key]:.4f}" in html

