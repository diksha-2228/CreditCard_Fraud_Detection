import io
import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# Configure paths relative to this file
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "saved_models"
LOG_REG_MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics_summary.json"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

app = Flask(__name__, static_folder="../", template_folder="../")
CORS(app)

# Load models and metrics on startup
print("[startup] Loading models...")
try:
    log_reg_model = joblib.load(LOG_REG_MODEL_PATH)
    print(f"Loaded Logistic Regression model from {LOG_REG_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load Logistic Regression model: {e}")
    log_reg_model = None

try:
    rf_model = joblib.load(RF_MODEL_PATH)
    print(f"Loaded Random Forest model from {RF_MODEL_PATH}")
except Exception as e:
    print(f"Failed to load Random Forest model: {e}")
    rf_model = None

try:
    with open(METRICS_PATH, "r") as f:
        metrics_summary = json.load(f)
    print(f"Loaded metrics summary from {METRICS_PATH}")
except Exception as e:
    print(f"Failed to load metrics summary: {e}")
    metrics_summary = []


@app.route("/")
def index():
    """Serve index.html at root."""
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve other static files (css, js, etc.)."""
    return send_from_directory(app.static_folder, path)


@app.route("/api/health", methods=["GET"])
def health():
    """Simple status check."""
    return jsonify({"status": "ok"})


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Return metrics_summary.json data."""
    return jsonify(metrics_summary)


@app.route("/api/predict", methods=["POST"])
def predict_single():
    """Run a single-row prediction based on the selected model and threshold."""
    if not request.is_json:
        return jsonify({"error": "Request body must be JSON"}), 400

    data = request.json
    model_choice = data.get("model", "random_forest")
    threshold = float(data.get("threshold", 0.5))

    # Pick the appropriate model
    if model_choice in ["random_forest", "random_forest_tuned", "Random Forest (Tuned)"]:
        model = rf_model
        model_name = "random_forest"
    elif model_choice in ["logistic_regression", "Logistic Regression"]:
        model = log_reg_model
        model_name = "logistic_regression"
    else:
        return jsonify({"error": f"Invalid model selection: {model_choice}"}), 400

    if model is None:
        return jsonify({"error": f"Selected model ({model_name}) is not loaded on the server"}), 500

    # Retrieve parameters and format features
    try:
        row_dict = {}
        for col in FEATURE_COLUMNS:
            # Default missing values to 0.0 (like PCA columns or time/amount if omitted)
            val = data.get(col, 0.0)
            row_dict[col] = float(val)

        # Construct dataframe with exact column order
        df_input = pd.DataFrame([row_dict])[FEATURE_COLUMNS]

        # Run inference
        proba = float(model.predict_proba(df_input)[0, 1])
        prediction = int(proba >= threshold)

        return jsonify({
            "prediction": prediction,
            "fraud_probability": round(proba, 6),
            "model_used": model_name
        })
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


@app.route("/api/predict-batch", methods=["POST"])
def predict_batch():
    """Accepts a CSV file upload, runs batch inference, and returns JSON array of predictions."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded under key 'file'"}), 400

    model_choice = request.form.get("model", "random_forest")
    threshold = float(request.form.get("threshold", 0.5))

    # Pick the appropriate model
    if model_choice in ["random_forest", "random_forest_tuned", "Random Forest (Tuned)"]:
        model = rf_model
        model_name = "random_forest"
    elif model_choice in ["logistic_regression", "Logistic Regression"]:
        model = log_reg_model
        model_name = "logistic_regression"
    else:
        return jsonify({"error": f"Invalid model selection: {model_choice}"}), 400

    if model is None:
        return jsonify({"error": f"Selected model ({model_name}) is not loaded on the server"}), 500

    try:
        # Load CSV into Pandas
        stream = io.StringIO(file.stream.read().decode("utf-8"), newline=None)
        df_input = pd.read_csv(stream)
    except Exception as e:
        return jsonify({"error": f"Failed to parse CSV file: {str(e)}"}), 400

    # Validate feature columns are present
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df_input.columns]
    if missing_cols:
        return jsonify({"error": f"CSV missing required feature columns: {missing_cols}"}), 400

    try:
        # Keep exact columns for input to the pipeline
        X = df_input[FEATURE_COLUMNS]
        
        # Calculate probabilities and predictions
        probas = model.predict_proba(X)[:, 1]
        preds = (probas >= threshold).astype(int)

        # Merge predictions back into original data
        df_output = df_input.copy()
        df_output["Predicted_Class"] = preds
        df_output["Fraud_Probability"] = probas.round(6)

        # Convert to JSON records list
        records = df_output.to_dict(orient="records")
        return jsonify(records)

    except Exception as e:
        return jsonify({"error": f"Batch prediction failed: {str(e)}"}), 500


if __name__ == "__main__":
    # Standard Flask port is 5000, serve on local addresses
    app.run(host="127.0.0.1", port=5000, debug=True)
