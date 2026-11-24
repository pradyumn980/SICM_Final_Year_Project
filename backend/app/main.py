
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "artifacts" / "model_bundle.joblib"

EXPECTED_FEATURES = [
    "spo2_mean",
    "systolic_bp_max",
    "systolic_bp_mean",
    "rbc_min",
    "anion_gap_max",
    "diastolic_bp_mean",
    "creatinine_min",
]


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="SICM Research Prediction API",
    description=(
        "Research API using a fitted seven-feature "
        "Logistic Regression model to estimate the "
        "probability of LVEF below 50% on an eligible "
        "subsequent echocardiogram among septic ICU patients."
    ),
    version="1.0.0",
)


# Allow local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

model_bundle = None
model_load_error = None


def load_model():
    global model_bundle, model_load_error

    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {MODEL_PATH}"
            )

        bundle = joblib.load(MODEL_PATH)

        required_keys = {
            "pipeline",
            "feature_names",
            "threshold",
            "model_name",
            "endpoint",
            "feature_window",
            "research_only",
        }

        missing_keys = required_keys - set(bundle.keys())

        if missing_keys:
            raise ValueError(
                f"Missing model bundle keys: {missing_keys}"
            )

        if bundle["feature_names"] != EXPECTED_FEATURES:
            raise ValueError(
                "Model feature names or order do not match "
                "the expected seven features."
            )

        if not 0 < float(bundle["threshold"]) < 1:
            raise ValueError("Invalid decision threshold.")

        if not hasattr(bundle["pipeline"], "predict_proba"):
            raise ValueError(
                "Loaded pipeline does not support predict_proba."
            )

        classes = list(
            bundle["pipeline"].named_steps["model"].classes_
        )

        if 1 not in classes:
            raise ValueError(
                "Expected binary target with positive class 1."
            )

        model_bundle = bundle
        model_load_error = None

        print("SICM research model loaded successfully!")
        print("Model:", bundle["model_name"])
        print("Threshold:", bundle["threshold"])

    except Exception as error:
        model_bundle = None
        model_load_error = str(error)

        print("MODEL LOADING FAILED:", model_load_error)


@app.on_event("startup")
def startup_event():
    load_model()


# --------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------

class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    spo2_mean: Optional[float]
    systolic_bp_max: Optional[float]
    systolic_bp_mean: Optional[float]
    rbc_min: Optional[float]
    anion_gap_max: Optional[float]
    diastolic_bp_mean: Optional[float]
    creatinine_min: Optional[float]


# --------------------------------------------------
# ROOT ENDPOINT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "SICM Research Prediction API",
        "status": "running",
        "model_ready": model_bundle is not None,
    }


# --------------------------------------------------
# HEALTH ENDPOINT
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy" if model_bundle is not None else "unavailable",
        "model_ready": model_bundle is not None,
        "model_name": (
            model_bundle["model_name"]
            if model_bundle is not None
            else None
        ),
        "error": model_load_error,
    }


# --------------------------------------------------
# PREDICTION ENDPOINT
# --------------------------------------------------

@app.post("/predict")
def predict(patient: PatientFeatures):

    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is not available."
        )

    try:
        # Convert request into a DataFrame.
        # Preserve the exact feature names and order.
        patient_data = patient.model_dump()

        input_df = pd.DataFrame(
            [[patient_data[name] for name in EXPECTED_FEATURES]],
            columns=EXPECTED_FEATURES,
        )

        # Reject infinite values.
        values = input_df.to_numpy(dtype=float)

        if np.isinf(values).any():
            raise HTTPException(
                status_code=422,
                detail="Infinite feature values are not allowed."
            )

        # Generate probability using the fitted pipeline.
        pipeline = model_bundle["pipeline"]

        classes = list(
            pipeline.named_steps["model"].classes_
        )

        positive_class_index = classes.index(1)

        probabilities = pipeline.predict_proba(input_df)

        positive_probability = float(
            probabilities[0][positive_class_index]
        )

        # Use the exact validation-selected threshold.
        threshold = float(model_bundle["threshold"])

        predicted_label = int(
            positive_probability >= threshold
        )

        return {
            "success": True,
            "model_name": model_bundle["model_name"],

            "prediction": {
                "predicted_label": predicted_label,
                "predicted_outcome": (
                    "LVEF below 50%"
                    if predicted_label == 1
                    else "LVEF not below 50%"
                ),
                "probability_lvef_below_50": round(
                    positive_probability, 6
                ),
                "decision_threshold": threshold,
            },

            "model_metadata": {
                "endpoint": model_bundle["endpoint"],
                "feature_window": model_bundle["feature_window"],
                "research_only": model_bundle["research_only"],
            },

            "notice": (
                "Research prediction only. "
                "Not a clinically validated diagnosis."
            ),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
