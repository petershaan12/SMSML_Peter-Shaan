"""FastAPI model server with Prometheus instrumentation."""

from __future__ import annotations

import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "model" / "best_model.joblib"))
PREPROCESSOR_PATH = Path(
    os.getenv("PREPROCESSOR_PATH", BASE_DIR / "model" / "preprocessor.joblib")
)
METADATA_PATH = Path(os.getenv("METADATA_PATH", BASE_DIR / "model" / "metadata.json"))

PREDICTIONS = Counter(
    "model_predictions_total", "Total predictions", ["prediction", "diagnosis"]
)
PREDICTION_ERRORS = Counter("model_prediction_errors_total", "Failed predictions")
PREDICTION_LATENCY = Histogram(
    "model_prediction_latency_seconds",
    "Prediction latency in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)
CONFIDENCE = Histogram(
    "model_prediction_confidence",
    "Predicted-class probability",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)
POSITIVE_RATIO = Gauge("model_benign_prediction_ratio", "Running benign prediction ratio")
INPUT_DRIFT = Gauge(
    "model_input_drift_score", "Mean absolute standardized feature value"
)
LAST_PREDICTION = Gauge(
    "model_last_prediction_timestamp_seconds", "Unix timestamp of the last prediction"
)
MODEL_LOADED = Gauge("model_loaded", "Whether model artifacts loaded successfully")
HTTP_REQUESTS = Counter(
    "model_http_requests_total", "HTTP requests", ["method", "endpoint", "status"]
)
HTTP_LATENCY = Histogram(
    "model_http_request_latency_seconds", "HTTP request latency", ["endpoint"]
)
IN_PROGRESS = Gauge("model_inference_in_progress", "Concurrent inference requests")

model = None
preprocessor = None
feature_columns: list[str] = []
prediction_count = 0
benign_count = 0
counter_lock = Lock()


class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ..., description="Raw feature values ordered according to GET /features"
    )


class PredictionResponse(BaseModel):
    prediction: int
    diagnosis: str
    confidence: float
    benign_probability: float


def load_artifacts() -> None:
    global model, preprocessor, feature_columns
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    feature_columns = metadata["feature_columns"]
    MODEL_LOADED.set(1)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        load_artifacts()
    except Exception:
        MODEL_LOADED.set(0)
        raise
    yield


app = FastAPI(
    title="Breast Cancer Classifier",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def observe_http(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        endpoint = request.url.path
        HTTP_REQUESTS.labels(request.method, endpoint, str(status)).inc()
        HTTP_LATENCY.labels(endpoint).observe(time.perf_counter() - started)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": "loaded"}


@app.get("/features")
def features() -> dict[str, object]:
    return {"count": len(feature_columns), "features": feature_columns}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    global prediction_count, benign_count
    if len(payload.features) != len(feature_columns):
        PREDICTION_ERRORS.inc()
        raise HTTPException(
            status_code=422,
            detail=f"Expected {len(feature_columns)} features, received {len(payload.features)}",
        )

    started = time.perf_counter()
    IN_PROGRESS.inc()
    try:
        raw_frame = pd.DataFrame([payload.features], columns=feature_columns)
        processed = pd.DataFrame(
            preprocessor.transform(raw_frame), columns=feature_columns
        )
        prediction = int(model.predict(processed)[0])
        benign_probability = float(model.predict_proba(processed)[0, 1])
        confidence = max(benign_probability, 1.0 - benign_probability)
        diagnosis = "benign" if prediction == 1 else "malignant"

        with counter_lock:
            prediction_count += 1
            benign_count += prediction
            POSITIVE_RATIO.set(benign_count / prediction_count)

        PREDICTIONS.labels(str(prediction), diagnosis).inc()
        CONFIDENCE.observe(confidence)
        INPUT_DRIFT.set(float(np.abs(processed.to_numpy()).mean()))
        LAST_PREDICTION.set_to_current_time()
        return PredictionResponse(
            prediction=prediction,
            diagnosis=diagnosis,
            confidence=confidence,
            benign_probability=benign_probability,
        )
    except HTTPException:
        raise
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    finally:
        IN_PROGRESS.dec()
        PREDICTION_LATENCY.observe(time.perf_counter() - started)
