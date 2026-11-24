"""Run this complete cell/script in Google Colab after enabling BigQuery access.

It fits the pre-specified seven-feature Logistic Regression candidate on the
original training split, selects a probability threshold on the validation split
(maximum F1), and exports one pipeline bundle for the FastAPI service.
The test split is deliberately not queried or used here.
"""

# In a Colab cell, uncomment the following lines if required:
# !pip -q install google-cloud-bigquery db-dtypes joblib scikit-learn
# from google.colab import auth, files
# auth.authenticate_user()

from google.cloud import bigquery
import joblib
import numpy as np
import pandas as pd
import sklearn
from datetime import datetime, timezone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
    accuracy_score,
    balanced_accuracy_score,
)

PROJECT_ID = "sepsis-project-513115"
TABLE_ID = "sepsis-project-513115.sicm_project.sicm_36_final_expanded_dataset"
FEATURES = [
    "spo2_mean",
    "systolic_bp_max",
    "systolic_bp_mean",
    "rbc_min",
    "anion_gap_max",
    "diastolic_bp_mean",
    "creatinine_min",
]
LABEL = "outcome_label"

client = bigquery.Client(project=PROJECT_ID)
selected_cols = ", ".join(f"`{c}`" for c in FEATURES)
query = f"""
SELECT dataset_split, {LABEL}, {selected_cols}
FROM `{TABLE_ID}`
WHERE dataset_split IN ('train', 'validation')
"""
df = client.query(query).to_dataframe()

train = df.loc[df.dataset_split.eq("train")].copy()
valid = df.loc[df.dataset_split.eq("validation")].copy()
X_train = train[FEATURES].apply(pd.to_numeric, errors="coerce")
y_train = train[LABEL].astype(int)
X_valid = valid[FEATURES].apply(pd.to_numeric, errors="coerce")
y_valid = valid[LABEL].astype(int)

if len(train) == 0 or len(valid) == 0:
    raise ValueError("Train and validation splits must both contain rows.")
if y_train.nunique() != 2 or y_valid.nunique() != 2:
    raise ValueError("Both train and validation labels must contain classes 0 and 1.")

# Match the model family and preprocessing used by the seven-feature nested-CV
# comparison: median imputation, standard scaling, balanced logistic regression.
pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42,
    )),
])
pipeline.fit(X_train[FEATURES], y_train)

valid_probability = pipeline.predict_proba(X_valid[FEATURES])[:, 1]
precision, recall, thresholds = precision_recall_curve(y_valid, valid_probability)
if len(thresholds):
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    threshold = float(thresholds[int(np.nanargmax(f1_values))])
    threshold = float(np.clip(threshold, 1e-6, 1 - 1e-6))
else:
    threshold = 0.5
valid_prediction = (valid_probability >= threshold).astype(int)

validation_metrics = {
    "n_train": int(len(train)),
    "n_validation": int(len(valid)),
    "validation_prevalence": float(y_valid.mean()),
    "validation_pr_auc": float(average_precision_score(y_valid, valid_probability)),
    "validation_roc_auc": float(roc_auc_score(y_valid, valid_probability)),
    "validation_accuracy_at_selected_threshold": float(accuracy_score(y_valid, valid_prediction)),
    "validation_balanced_accuracy_at_selected_threshold": float(balanced_accuracy_score(y_valid, valid_prediction)),
    "validation_f1_at_selected_threshold": float(f1_score(y_valid, valid_prediction, zero_division=0)),
}

bundle = {
    "pipeline": pipeline,
    "feature_names": FEATURES,
    "threshold": threshold,
    "model_name": "Seven-feature Logistic Regression",
    "endpoint": "LVEF < 50% on an eligible subsequent echocardiogram among septic ICU patients",
    "feature_window": "[outcome_time - 36 hours, outcome_time - 12 hours)",
    "training_table": TABLE_ID,
    "training_split": "train",
    "threshold_selection": "maximum F1 on validation split; test split not used",
    "validation_metrics": validation_metrics,
    "sklearn_version": sklearn.__version__,
    "exported_at_utc": datetime.now(timezone.utc).isoformat(),
    "research_only": True,
}

output_path = "model_bundle.joblib"
joblib.dump(bundle, output_path)
print("Model bundle saved:", output_path)
print("Validation-selected threshold:", round(threshold, 4))
print("Validation audit:", validation_metrics)

# In Colab, uncomment to download the exported file to your computer:
# files.download(output_path)
