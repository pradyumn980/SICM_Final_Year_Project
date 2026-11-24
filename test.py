import joblib

model = joblib.load("artifacts/model_bundle.joblib")

print("Model loaded successfully!")
print("Bundle type:", type(model))

if isinstance(model, dict):
    print("Bundle keys:", model.keys())