# SICM Research Risk Assessment — React + FastAPI

This project is a frontend and inference-API scaffold for the seven-feature Logistic Regression candidate from the SICM 59-vs-7 nested-CV comparison.

## Important model and endpoint context

The current project predicts an **echo-based reduced-LVEF proxy** among septic ICU patients:

- Positive class: `outcome_label = 1`, LVEF `< 50%` on an eligible subsequent echocardiogram.
- Negative class: `outcome_label = 0`, LVEF `>= 50%`.
- It is **not** a clinically adjudicated diagnosis of new or reversible sepsis-induced cardiomyopathy.
- Features are aggregates from `[outcome_time - 36 hours, outcome_time - 12 hours)`, a 24-hour interval ending 12 hours before the echo outcome. They are not individual live monitor readings and are not necessarily the first 12 hours after sepsis onset.
- The seven selected fields and order are frozen in the frontend and API:
  1. `spo2_mean`
  2. `systolic_bp_max`
  3. `systolic_bp_mean`
  4. `rbc_min`
  5. `anion_gap_max`
  6. `diastolic_bp_mean`
  7. `creatinine_min`

## Why the first UI uses seven inputs

The nested-CV comparison reported these pooled out-of-fold results for the fixed seven-feature Logistic Regression candidate, at threshold 0.50:

- PR-AUC: `0.414105`
- ROC-AUC: `0.674420`
- Balanced accuracy: `0.622570`
- F1: `0.454609`

This is a compact, comparatively interpretable candidate for the first interface. It was **not** the top pooled PR-AUC model: the 59-feature Hist Gradient Boosting candidate had PR-AUC `0.437794`, but sensitivity at 0.50 was only `0.206049`. The seven-feature Logistic Regression is therefore a practical prototype choice, not a declared clinical winner.

## 1. Export a fitted model from Google Colab

The nested-CV notebook produces out-of-fold scores; those scores are not a saved final estimator. You must fit a final pipeline before the frontend can make real predictions.

1. Open `model_export/export_model_from_colab.py` in Google Colab.
2. Uncomment the Colab authentication / dependency lines at the top if needed, and authenticate to the BigQuery project that has access to `sepsis-project-513115.sicm_project.sicm_36_final_expanded_dataset`.
3. Run the script. It queries only `train` and `validation` rows. It fits median imputation + standard scaling + balanced Logistic Regression on the training partition, then chooses the decision threshold by maximum F1 on the validation partition. It does not query the test split.
4. Download `model_bundle.joblib` from Colab (uncomment `files.download(output_path)`).
5. Put the file at `backend/artifacts/model_bundle.joblib`.

The exporter intentionally creates a new full-training fit of the seven-feature candidate. Its nested-CV metrics summarize the model family during cross-validation; they are not a guarantee of the performance of this refitted artifact on future patients. The exporter prints separate validation metrics for the refitted model.

## 2. Run the FastAPI backend

Python 3.11+ recommended.

```bash
cd backend
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open `http://localhost:8000/health` to check readiness and `http://localhost:8000/docs` for the API documentation. Before the artifact is installed, the API intentionally reports `model_ready: false` and refuses prediction requests rather than returning mock predictions.

## 3. Run the React frontend

Open a second terminal:

```bash
cd frontend
npm install
# Copy .env.example to .env if you need a different API URL.
npm run dev
```

Open `http://localhost:5173`.

## 4. Prediction API contract

`POST /predict` accepts JSON with exactly these seven numeric keys:

```json
{
  "spo2_mean": 96.5,
  "systolic_bp_max": 145,
  "systolic_bp_mean": 112,
  "rbc_min": 3.25,
  "anion_gap_max": 19,
  "diastolic_bp_mean": 66,
  "creatinine_min": 0.9
}
```

The values above are only an example of the request format, not a clinical profile or expected result. The API returns the model probability/score, validation-selected threshold, model label and a non-diagnostic interpretation from the actual fitted pipeline. Treat the score as a research model output; it has not been established as a calibrated clinical risk probability.

## 5. Project structure

```text
SICM_Web_App/
├── frontend/                 # React UI (Vite)
│   ├── src/App.jsx
│   ├── src/styles.css
│   └── .env.example
├── backend/                  # FastAPI inference API
│   ├── app/main.py
│   ├── artifacts/             # Place model_bundle.joblib here
│   └── requirements.txt
└── model_export/
    └── export_model_from_colab.py
```

## Research and safety limitations

- Use only for academic demonstration and software integration testing.
- Model output is a statistical estimate for the project's reduced-LVEF proxy and is not a diagnosis, clinical risk score, or treatment recommendation.
- The selected cohort includes patients with eligible echocardiography; selection bias can limit generalization to all septic ICU patients.
- The dataset had upstream training-partition median imputation. This export pipeline applies another train-fitted imputer, but it cannot undo upstream imputation. A stricter future release should rebuild features with original missing values and fit all imputation within validation folds.
- Previous experiments inspected the test partition; do not describe it as a pristine untouched test set.
- External, temporal and prospective validation; calibration; subgroup performance; sensitivity-focused operating-point review; clinical workflow evaluation; and applicable regulatory review would be needed before any clinical deployment.
