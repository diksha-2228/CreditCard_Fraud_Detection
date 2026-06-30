Markdown
# Credit Card Fraud Detection System

An end-to-end Machine Learning pipeline and interactive web dashboard designed to detect fraudulent credit card transactions in real-time. This project handles the classic challenges of highly imbalanced financial datasets using advanced resampling techniques and compares multiple supervised learning models.

## 🚀 Live Dashboard Features
* **Active Pipeline:** Real-time prediction tracking with an operational pipeline.
* **Single Transaction Check:** Interactive inference form where users can input specific transaction details (Time, Amount, and Anonymized PCA features V1-V28) to analyze instantly.
* **Batch Transaction Analysis:** File drag-and-drop zone allowing bulk CSV uploads for rapid large-scale prediction processing.
* **Dynamic Decision Threshold Slider:** Adjust thresholds dynamically to trade off between catching more fraud (Sensitivity) or reducing false alarms (Strictness).

---

## 📊 Model Performance & Insights

The system evaluates and compares two key classifiers optimized using SMOTE for class balancing. 

| Model | F1-Score (Fraud Class) | ROC-AUC |
| :--- | :---: | :---: |
| **Logistic Regression** | 0.1106 | 0.9712 |
| **Random Forest (Tuned)** | **0.7477** | **0.9805** |

### Performance Insights
* **Random Forest (Tuned)** is the designated production champion model. 
* It successfully captures **83 out of 98** actual fraud cases in evaluation.
* It triggers only **41 false alarms**, yielding a highly robust **0.7477 F1-Score** on the heavily imbalanced minority class.

---

## 📁 Project Structure

```text
├── backend/
│   └── app.py                      # Flask/FastAPI backend API for model inference
├── saved_models/
│   ├── logistic_regression_pipeline.joblib  # Serialized Logistic Regression model
│   ├── random_forest_pipeline.joblib       # Serialized champion Random Forest model
│   └── metrics_summary.json         # Evaluated pipeline validation metrics
├── app.js                          # Dashboard frontend logic & API connectors
├── index.html                      # Interactive UI dashboard layout
├── style.css                       # Dashboard styling component
├── fraud_detection_pipeline.py     # Main ML training, preprocessing & optimization pipeline
├── requirements.txt                # System dependencies and Python libraries
└── .gitignore                      # Prevents large dataset files tracking
🛠️ Tech Stack
Frontend: HTML5, CSS3, JavaScript (Vanilla ES6 for interactive charts and metrics processing)

Backend: Python, Flask / FastAPI

Machine Learning: Scikit-Learn, Joblib, Imbalanced-Learn (SMOTE)

🔧 Installation & Local Setup
1. Clone the Repository
Bash
git clone [https://github.com/diksha-2228/CreditCard_Fraud_Detection.git](https://github.com/diksha-2228/CreditCard_Fraud_Detection.git)
cd CreditCard_Fraud_Detection
2. Set Up Python Environment
Ensure you have Python installed, then create a virtual environment and install dependencies:

Bash
python -m venv venv
# Activate on Windows:
venv\Scripts\activate

pip install -r requirements.txt
3. Run the Pipeline
If you want to re-train or inspect the pipeline setup:

Bash
python fraud_detection_pipeline.py
4. Start the Application
Launch the backend server:

Bash
cd backend
python app.py
Open index.html directly in your browser or run it via a local live server extension to view the dashboard interactively.

🛑 Dataset Note
Due to GitHub file size limitations, the primary training dataset (creditcard.csv, 143.84 MB) is excluded from version control. You can download the official anonymized dataset from Kaggle's Credit Card Fraud Detection Dataset and place it directly in the root directory before running the training script.d strictly aligns feature order with training format:
  `["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]`
* **Threshold Adjustment**: Moving the global slider passes the selected value to the backend prediction endpoints, modifying predictions instantly.
* **PCA Driver Highlighting**: The batch prediction preview extracts the top 3 PCA features with the largest absolute deviation from zero for each row, showing key factors contributing to the risk profile.
