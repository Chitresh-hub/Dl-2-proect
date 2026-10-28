"""Flask application for the saved Gaussian Naive Bayes theft model."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import RequestEntityTooLarge


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "model"
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def load_artifacts(model_dir: Path) -> dict[str, Any]:
    """Load the fitted model and exported evaluation artifacts without fitting anything."""
    state: dict[str, Any] = {
        "model_loaded": False,
        "metrics_available": False,
        "model": None,
        "scaler": None,
        "medians": None,
        "reading_columns": [],
        "feature_names": [],
        "threshold": None,
        "metrics": None,
        "metadata": _read_json(model_dir / "metadata.json") or {},
        "evaluation": _read_json(model_dir / "evaluation.json"),
        "model_message": "Model artifacts are not available yet. Run the notebook export cell.",
    }
    metrics_file = _read_json(model_dir / "metrics.json")
    if metrics_file and isinstance(metrics_file.get("metrics"), dict):
        state["metrics"] = metrics_file["metrics"]
        state["metrics_available"] = bool(metrics_file.get("available", True))
        state["metrics_message"] = metrics_file.get("message")
    else:
        state["metrics_message"] = "Evaluation metrics were not exported. Run the notebook evaluation cells."

    model_path = model_dir / "model.joblib"
    if not model_path.is_file():
        return state
    try:
        bundle = joblib.load(model_path)
        required = ("model", "scaler", "imputation_medians", "daily_reading_columns")
        if not isinstance(bundle, dict) or any(key not in bundle for key in required):
            state["model_message"] = "The saved model bundle is missing required preprocessing objects."
            return state
        columns = [str(column) for column in bundle["daily_reading_columns"]]
        if not columns:
            state["model_message"] = "The saved model bundle does not define any reading columns."
            return state
        threshold = float(bundle.get("threshold", state["metadata"].get("threshold", 0.5)))
        if not math.isfinite(threshold) or threshold < 0 or threshold > 1:
            state["model_message"] = "The saved decision threshold is invalid."
            return state
        state.update({
            "model_loaded": True,
            "model": bundle["model"],
            "scaler": bundle["scaler"],
            "medians": bundle["imputation_medians"],
            "reading_columns": columns,
            "feature_names": list(bundle.get("feature_names", [])),
            "threshold": threshold,
            "model_message": "Gaussian Naive Bayes and its saved preprocessing are ready.",
        })
        if not state["metadata"]:
            state["metadata"] = {
                "model_name": "Gaussian Naive Bayes",
                "daily_reading_columns": columns,
                "feature_names": state["feature_names"],
                "threshold": threshold,
            }
        else:
            state["metadata"].setdefault("daily_reading_columns", columns)
            state["metadata"].setdefault("feature_names", state["feature_names"])
            state["metadata"].setdefault("threshold", threshold)
        # The notebook bundle remains the source of the threshold used for prediction.
        if state["metrics"] is not None:
            state["metrics"]["threshold"] = threshold
    except Exception:
        logging.exception("Could not load the saved model artifacts")
        state["model_message"] = "The saved model artifacts could not be loaded. Check the bundle and package versions."
    return state


def _make_features(readings: pd.DataFrame, columns: list[str], medians: Any) -> pd.DataFrame:
    """Replicate the notebook's daily-reading feature engineering exactly."""
    frame = readings.loc[:, columns].copy()
    frame = frame.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame.mask(frame < 0)
    observed = frame.notna().to_numpy()
    missing_share = (~observed).mean(axis=1)
    values = frame.fillna(medians).to_numpy(dtype=float)
    values = np.maximum(values, 0.0)
    logged = np.log1p(values)
    mean_log = logged.mean(axis=1)
    std_log = logged.std(axis=1)
    changes = np.diff(logged, axis=1) if logged.shape[1] > 1 else np.zeros((len(logged), 1))
    time_axis = np.arange(logged.shape[1], dtype=float)
    time_axis -= time_axis.mean()
    denominator = float(np.sum(time_axis**2))
    trend = ((logged - mean_log[:, None]) @ time_axis) / denominator if denominator else np.zeros(len(logged))
    observed_count = np.maximum(observed.sum(axis=1), 1)
    observed_zeros = ((values == 0) & observed).sum(axis=1) / observed_count
    features = pd.DataFrame({
        "mean_log_consumption": mean_log,
        "std_log_consumption": std_log,
        "median_log_consumption": np.median(logged, axis=1),
        "p10_log_consumption": np.quantile(logged, 0.10, axis=1),
        "p90_log_consumption": np.quantile(logged, 0.90, axis=1),
        "log_consumption_cv": std_log / (np.abs(mean_log) + 1e-8),
        "observed_zero_day_share": observed_zeros,
        "missing_day_share": missing_share,
        "mean_abs_daily_change_log": np.abs(changes).mean(axis=1),
        "sharp_drop_share": (changes <= -0.5).mean(axis=1),
        "sharp_rise_share": (changes >= 0.5).mean(axis=1),
        "trend_slope_log": trend,
    })
    return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _validated_reading_frame(value: Any, columns: list[str]) -> pd.DataFrame:
    if isinstance(value, dict):
        missing = [column for column in columns if column not in value]
        extra = [column for column in value if column not in columns]
        if missing or extra:
            details = []
            if missing:
                details.append("missing reading columns: " + ", ".join(missing[:5]))
            if extra:
                details.append("unexpected fields: " + ", ".join(str(item) for item in extra[:5]))
            raise ValueError("; ".join(details))
        values = [value[column] for column in columns]
    elif isinstance(value, list):
        if len(value) != len(columns):
            raise ValueError(f"Expected exactly {len(columns)} daily readings, received {len(value)}.")
        values = value
    else:
        raise ValueError("Provide readings as an ordered list or an object keyed by the required CSV columns.")

    parsed: list[float] = []
    for index, item in enumerate(values):
        if isinstance(item, bool) or item is None or (isinstance(item, str) and not item.strip()):
            raise ValueError(f"Reading {columns[index]} is missing or invalid.")
        try:
            number = float(item)
        except (TypeError, ValueError):
            raise ValueError(f"Reading {columns[index]} must be a number.") from None
        if not math.isfinite(number) or number < 0:
            raise ValueError(f"Reading {columns[index]} must be a finite, non-negative number.")
        parsed.append(number)
    return pd.DataFrame([parsed], columns=columns)


def _predict_rows(state: dict[str, Any], frame: pd.DataFrame) -> list[dict[str, Any]]:
    features = _make_features(frame, state["reading_columns"], state["medians"])
    scaled = state["scaler"].transform(features)
    probabilities = np.asarray(state["model"].predict_proba(scaled), dtype=float)
    classes = list(state["model"].classes_)
    try:
        positive_index = classes.index(1)
    except ValueError:
        raise RuntimeError("The saved model does not contain positive class 1.") from None
    scores = probabilities[:, positive_index]
    threshold = float(state["threshold"])
    return [
        {
            "prediction": "Theft-labelled" if score >= threshold else "Normal",
            "predicted_class": 1 if score >= threshold else 0,
            "theft_probability": float(score),
            "threshold": threshold,
        }
        for score in scores
    ]


def create_app(model_dir: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
    resolved_model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
    state = load_artifacts(resolved_model_dir)
    app.extensions["electricity_model_state"] = state

    @app.get("/")
    def index():
        metric_display: dict[str, dict[str, str]] = {}
        for name, value in (state["metrics"] or {}).items():
            try:
                number = float(value)
                if math.isfinite(number):
                    metric_display[name] = {
                        "percent": f"{number * 100:.2f}%",
                        "exact": f"Exact score: {number:.4f}",
                        "bar": f"{max(0.0, min(100.0, number * 100)):.2f}%",
                    }
            except (TypeError, ValueError):
                continue
        return render_template(
            "index.html",
            initial_metric_display=metric_display,
            initial_metrics_available=state["metrics_available"],
        )

    @app.get("/api/health")
    def health():
        return jsonify({
            "success": True,
            "server_status": "ok",
            "model_loaded": state["model_loaded"],
            "metrics_available": state["metrics_available"],
            "message": state["model_message"],
        })

    @app.get("/api/metrics")
    def metrics():
        metadata = dict(state["metadata"])
        if state["medians"] is not None:
            try:
                metadata["sample_readings"] = [float(value) for value in state["medians"].to_numpy()]
            except AttributeError:
                metadata["sample_readings"] = [float(value) for value in state["medians"]]
        return jsonify({
            "success": True,
            "available": state["metrics_available"],
            "message": state.get("metrics_message"),
            "metrics": state["metrics"],
            "metadata": metadata,
        })

    @app.get("/api/evaluation")
    def evaluation():
        if state["evaluation"] is None:
            return jsonify({
                "success": True,
                "available": False,
                "message": "Test-set chart data was not exported. Run the notebook export cell to generate it.",
            })
        result = dict(state["evaluation"])
        result["success"] = True
        result["available"] = bool(result.get("available", True))
        return jsonify(result)

    @app.post("/api/predict")
    def predict():
        if not state["model_loaded"]:
            return jsonify({"success": False, "error": state["model_message"]}), 503
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or "readings" not in payload:
            return jsonify({"success": False, "error": "Request JSON must include a readings field."}), 400
        try:
            frame = _validated_reading_frame(payload["readings"], state["reading_columns"])
            prediction = _predict_rows(state, frame)[0]
            return jsonify({"success": True, **prediction})
        except ValueError as error:
            return jsonify({"success": False, "error": str(error)}), 400
        except Exception:
            logging.exception("Single-customer prediction failed")
            return jsonify({"success": False, "error": "Prediction failed while processing these readings."}), 500

    @app.post("/api/predict-csv")
    def predict_csv():
        if not state["model_loaded"]:
            return jsonify({"success": False, "error": state["model_message"]}), 503
        uploaded = request.files.get("file")
        if uploaded is None or not uploaded.filename:
            return jsonify({"success": False, "error": "Choose a CSV file to upload."}), 400
        if Path(uploaded.filename).suffix.lower() != ".csv":
            return jsonify({"success": False, "error": "Upload a file with the .csv extension."}), 400
        try:
            frame = pd.read_csv(uploaded, low_memory=False)
            if frame.empty:
                return jsonify({"success": False, "error": "The CSV has no customer records."}), 400
            expected = state["reading_columns"]
            missing = [column for column in expected if column not in frame.columns]
            extra = [str(column) for column in frame.columns if column not in expected]
            if missing or extra or len(frame.columns) != len(expected):
                details = []
                if missing:
                    details.append("missing columns: " + ", ".join(missing[:5]))
                if extra:
                    details.append("unexpected columns: " + ", ".join(extra[:5]))
                return jsonify({
                    "success": False,
                    "error": "CSV columns must match the model's daily-reading schema (" + "; ".join(details) + ").",
                }), 400
            frame = frame.loc[:, expected].apply(pd.to_numeric, errors="coerce")
            values = frame.to_numpy(dtype=float)
            if not np.isfinite(values).all():
                return jsonify({"success": False, "error": "The CSV contains missing or non-numeric readings."}), 400
            if (values < 0).any():
                return jsonify({"success": False, "error": "Daily consumption readings must be non-negative."}), 400
            predictions = _predict_rows(state, frame)
            results = [
                {"record": index + 1, **prediction}
                for index, prediction in enumerate(predictions)
            ]
            theft_count = sum(row["predicted_class"] == 1 for row in results)
            return jsonify({
                "success": True,
                "processed": len(results),
                "normal_count": len(results) - theft_count,
                "theft_labelled_count": theft_count,
                "threshold": state["threshold"],
                "results": results,
            })
        except (pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError):
            return jsonify({"success": False, "error": "The uploaded file could not be read as a CSV."}), 400
        except Exception:
            logging.exception("CSV prediction failed")
            return jsonify({"success": False, "error": "Batch prediction failed while processing this file."}), 500

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(_error):
        return jsonify({"success": False, "error": "The CSV is too large. The upload limit is 8 MB."}), 413

    return app


app = create_app()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
