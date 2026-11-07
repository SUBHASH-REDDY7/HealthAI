
from flask import Flask, render_template, request
import joblib
import json
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model = joblib.load(
    os.path.join(BASE_DIR, "models", "svm_diabetes_model.pkl")
)

# Load model configuration
with open(os.path.join(BASE_DIR, "models", "model_config.json")) as f:
    config = json.load(f)

THRESHOLD = config["threshold"]
FEATURES = config["features"]

@app.route("/")
def home():
    return render_template("index.html", features=FEATURES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form[feature]) for feature in FEATURES]

        input_data = np.array(values).reshape(1, -1)

        probability = float(model.predict_proba(input_data)[0][1])

        prediction = (
            "Diabetes Risk Detected"
            if probability >= THRESHOLD
            else "Lower Estimated Risk"
        )

        return render_template(
            "index.html",
            features=FEATURES,
            prediction=prediction,
            probability=round(probability * 100, 2)
        )

    except (ValueError, KeyError):
        return render_template(
            "index.html",
            features=FEATURES,
            error="Please enter valid numerical values."
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
