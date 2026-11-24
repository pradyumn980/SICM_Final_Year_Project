
import joblib
import os

model_path = "artifacts/model_bundle.joblib"

# Check whether the file exists
print("File exists:", os.path.exists(model_path))
print("File size:", round(os.path.getsize(model_path) / 1024, 2), "KB")

# Load model bundle
bundle = joblib.load(model_path)

print("\nModel loaded successfully!")
print("Bundle type:", type(bundle))

# Inspect saved components
if isinstance(bundle, dict):
    print("\nBundle keys:")
    for key, value in bundle.items():
        print(f"{key}: {type(value)}")
        if isinstance(value, (int, float, str)):
            print("   Value:", value)
else:
    print("Bundle contents:", bundle)
