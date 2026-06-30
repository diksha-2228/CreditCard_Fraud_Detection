# Credit Card Fraud Detection Web UI

This project provides a lightweight, responsive static web frontend (plain HTML/CSS/JS) and a Flask API backend for the **Supervised Learning: Credit Card Fraud Detection Pipeline**.

It allows users to:
1. Load pre-trained models on startup (Logistic Regression and tuned Random Forest).
2. Perform single transaction checks with customizable PCA inputs and instant status alerts.
3. Upload CSV batch files, display paginated diagnostics with top PCA drivers, and download full predictions.
4. Visualize test set evaluation metrics (F1-score and ROC-AUC) using Chart.js.
5. Set a global classification threshold dynamically.

---

## Local Setup & Run Instructions

All instructions assume commands are run from the project root on a Windows machine:
`C:\Users\sheetal singh\OneDrive\Desktop\project2`

### 1. Install Dependencies
Ensure you have the required packages installed. Run:
```cmd
pip install -r requirements.txt
```
*(Dependencies added: `Flask` and `flask-cors` in addition to `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, and `joblib`)*

### 2. Start the Flask Backend Server
To spin up the Flask web and API server, execute:
```cmd
python backend\app.py
```
This loads the pre-trained models from the `saved_models\` directory, handles routing, and listens for requests.

### 3. Open the User Interface
Once the server starts and says `Running on http://127.0.0.1:5000`, choose one of these methods:
* **Via Flask Server (Recommended):** Open your web browser and navigate to:
  `http://127.0.0.1:5000`
* **Direct File Open:** Double-click the `index.html` file in the project directory, or open it in a browser as a local file (`file:///C:/Users/sheetal%20singh/OneDrive/Desktop/project2/index.html`). The frontend will communicate with the API on port 5000 via CORS.

---

## Technical Details

* **Feature Ordering**: The backend strictly aligns feature order with training format:
  `["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]`
* **Threshold Adjustment**: Moving the global slider passes the selected value to the backend prediction endpoints, modifying predictions instantly.
* **PCA Driver Highlighting**: The batch prediction preview extracts the top 3 PCA features with the largest absolute deviation from zero for each row, showing key factors contributing to the risk profile.
