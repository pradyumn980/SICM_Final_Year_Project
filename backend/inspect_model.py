
import joblib
import sklearn

bundle = joblib.load("artifacts/model_bundle.joblib")

print("SCIKIT-LEARN:", sklearn.__version__)

print("\n========== MODEL FEATURES ==========")
for i, feature in enumerate(bundle["feature_names"], 1):
    print(f"{i}. {feature}")

print("\n========== PIPELINE ==========")
print(bundle["pipeline"])

print("\n========== THRESHOLD ==========")
print(bundle["threshold"])

print("\n========== VALIDATION METRICS ==========")
for key, value in bundle["validation_metrics"].items():
    print(f"{key}: {value}")

print("\n========== MODEL METADATA ==========")
print("Model:", bundle["model_name"])
print("Endpoint:", bundle["endpoint"])
print("Feature window:", bundle["feature_window"])
print("Research only:", bundle["research_only"])

print("\n========== PIPELINE INPUT ==========")
print("Expected features:", bundle["pipeline"].n_features_in_)
