from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("MODEL_PATH", BASE_DIR / "artifacts" / "model_bundle.joblib"))

FEATURES = [
    "spo2_mean",
    "systolic_bp_max",
    "systolic_bp_mean",
    "rbc_min",
    "anion_gap_max",
    "diastolic_bp_mean",
    "creatinine_min",
]

app = FastAPI(
    title="SICM research prediction API",
    version="1.0.0",
    description=(
        "Academic research API for an echo-based reduced-LVEF proxy among septic ICU stays. "
        "Not a clinical diagnostic service."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

_bundle: dict[str, Any] | None = None
_load_error: str | None = None


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spo2_mean: float = Field(ge=0, le=100)
    systolic_bp_max: float = Field(gt=0, le=300)
    systolic_bp_mean: float = Field(gt=0, le=300)
    rbc_min: float = Field(gt=0, le=10)
    anion_gap_max: float = Field(ge=0, le=60)
    diastolic_bp_mean: float = Field(gt=0, le=200)
    creatinine_min: float = Field(ge=0, le=30)


def load_bundle() -> None:
    global _bundle, _load_error
    _bundle = None
    _load_error = None
    if not MODEL_PATH.exists():
        _load_error = f"Model artifact not found at {MODEL_PATH}"
        return
    try:
        candidate = joblib.load(MODEL_PATH)
        if not isinstance(candidate, dict):
            raise ValueError("The model bundle must be a dictionary.")
        if candidate.get("feature_names") != FEATURES:
            raise ValueError("Feature order in the model bundle does not match the API schema.")
        pipeline = candidate.get("pipeline")
        if pipeline is None or not hasattr(pipeline, "predict_proba"):
            raise ValueError("The model bundle does not contain a predict_proba pipeline.")
        threshold = float(candidate.get("threshold", 0.5))
        if not 0 < threshold < 1:
            raise ValueError("The saved decision threshold must be between 0 and 1.")
        _bundle = candidate
    except Exception as exc:  # Keep the API online and expose readiness separately.
        _load_error = f"Could not load model artifact: {exc}"


@app.on_event("startup")
def startup_event() -> None:
    load_bundle()


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "SICM research prediction API", "health": "/health", "docs": "/docs"}


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": _bundle is not None,
        "model_name": (_bundle or {}).get("model_name"),
        "message": _load_error,
    }


@app.post("/predict")
def predict(data: PatientFeatures) -> dict[str, Any]:
    if _bundle is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The fitted model is not installed. Export model_bundle.joblib from the project Colab notebook "
                "using model_export/export_model_from_colab.py, then place it in backend/artifacts/."
            ),
        )

    if data.systolic_bp_max < data.systolic_bp_mean:
        raise HTTPException(status_code=422, detail="systolic_bp_max cannot be lower than systolic_bp_mean.")

    row = pd.DataFrame([{name: getattr(data, name) for name in FEATURES}], columns=FEATURES)
    try:
        probability = float(_bundle["pipeline"].predict_proba(row)[0, 1])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {exc}") from exc

    threshold = float(_bundle.get("threshold", 0.5))
    prediction = int(probability >= threshold)
    return {
        "prediction": prediction,
        "risk_probability": probability,
        "threshold": threshold,
        "model_name": _bundle.get("model_name", "Seven-feature Logistic Regression"),
        "interpretation": (
            "The score is at or above the validation-selected model threshold for the reduced-LVEF proxy. "
            "It is not a confirmed SICM diagnosis."
            if prediction == 1
            else "The score is below the validation-selected model threshold for the reduced-LVEF proxy. "
            "This does not rule out cardiac dysfunction or SICM."
        ),
        "endpoint": "LVEF < 50% on an eligible subsequent echocardiogram among septic ICU patients",
        "research_only": True,
    }
