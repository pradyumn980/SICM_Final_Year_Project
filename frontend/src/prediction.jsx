
import React, { useState } from "react";
import "./Prediction.css";

const API_URL = import.meta.env.VITE_API_URL;

const featureFields = [
  { name: "spo2_mean", label: "Mean SpO₂", unit: "%" },
  { name: "systolic_bp_max", label: "Maximum Systolic BP", unit: "mmHg" },
  { name: "systolic_bp_mean", label: "Mean Systolic BP", unit: "mmHg" },
  { name: "rbc_min", label: "Minimum RBC", unit: "million/µL" },
  { name: "anion_gap_max", label: "Maximum Anion Gap", unit: "mEq/L" },
  { name: "diastolic_bp_mean", label: "Mean Diastolic BP", unit: "mmHg" },
  { name: "creatinine_min", label: "Minimum Creatinine", unit: "mg/dL" },
];

const initialValues = Object.fromEntries(
  featureFields.map((field) => [field.name, ""])
);

export default function Prediction() {
  const [values, setValues] = useState(initialValues);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleChange = (event) => {
    const { name, value } = event.target;

    setValues((previous) => ({
      ...previous,
      [name]: value,
    }));
  };

  const handleSubmit = async (event) => {
    event.preventDefault();

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const payload = {};

      for (const field of featureFields) {
        const value = values[field.name];

        if (value.trim() === "") {
          payload[field.name] = null;
        } else {
          const number = Number(value);

          if (!Number.isFinite(number)) {
            throw new Error(`Invalid value for ${field.label}`);
          }

          payload[field.name] = number;
        }
      }

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction request failed."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message || "Unable to connect to prediction server."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setValues(initialValues);
    setResult(null);
    setError("");
  };

  return (
    <div className="prediction-page">
      <div className="prediction-container">

        <div className="prediction-header">
          <span className="prediction-tag">
            RESEARCH MODEL
          </span>

          <h1>SICM Prediction</h1>

          <p>
            Enter the seven aggregated clinical features
            required by the trained Logistic Regression model.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="prediction-form">

          <div className="feature-grid">
            {featureFields.map((field) => (
              <div className="feature-field" key={field.name}>
                <label htmlFor={field.name}>
                  {field.label}
                  <span> ({field.unit})</span>
                </label>

                <input
                  id={field.name}
                  name={field.name}
                  type="number"
                  step="any"
                  value={values[field.name]}
                  onChange={handleChange}
                  placeholder="Enter value or leave empty"
                />
              </div>
            ))}
          </div>

          <p className="feature-note">
            Values must be aggregated over the model's
            specified feature window:
            [outcome_time - 36 hours, outcome_time - 12 hours).
            Missing values are handled by the model's median imputer.
          </p>

          <div className="prediction-actions">
            <button
              type="button"
              className="reset-button"
              onClick={handleReset}
            >
              Reset
            </button>

            <button
              type="submit"
              className="predict-button"
              disabled={loading}
            >
              {loading ? "Generating prediction..." : "Generate Prediction"}
            </button>
          </div>
        </form>

        {error && (
          <div className="prediction-error">
            {error}
          </div>
        )}

        {result && (
          <div className="prediction-result">

            <h2>Prediction Result</h2>

            <div className="probability-value">
              {(result.prediction.probability_lvef_below_50 * 100)
                .toFixed(4)}%
            </div>

            <p className="probability-label">
              Estimated probability of LVEF below 50%
            </p>

            <div className="result-row">
              <span>Predicted outcome</span>
              <strong>
                {result.prediction.predicted_outcome}
              </strong>
            </div>

            <div className="result-row">
              <span>Predicted label</span>
              <strong>
                {result.prediction.predicted_label}
              </strong>
            </div>

            <div className="result-row">
              <span>Decision threshold</span>
              <strong>
                {result.prediction.decision_threshold.toFixed(6)}
              </strong>
            </div>

            <div className="result-row">
              <span>Model</span>
              <strong>{result.model_name}</strong>
            </div>

            <div className="research-notice">
              {result.notice}
            </div>

          </div>
        )}

      </div>
    </div>
  );
}
