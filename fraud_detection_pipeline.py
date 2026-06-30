"""
Project 2: Supervised Learning - Fraud Detection Pipeline
============================================================

Goal:
    Build and tune a classification model to identify fraudulent credit card
    transactions in a highly imbalanced dataset (the standard European
    cardholders dataset: Time, V1-V28 PCA features, Amount, Class).

Pipeline:
    1. Load data, split features/target, stratified 80/20 train-test split.
    2. Build an imbalanced-learn Pipeline:
         Step 1 -> Scale 'Time' and 'Amount' (RobustScaler, robust to outliers
                   which are common in transaction amounts and fraud cases).
         Step 2 -> SMOTE oversampling (fit ONLY on training folds, never on
                   test data, to avoid data leakage).
         Step 3 -> Classifier (Logistic Regression or Random Forest).
    3. Train + compare Logistic Regression vs. Random Forest.
    4. Hyperparameter-tune the Random Forest with RandomizedSearchCV.
    5. Evaluate strictly with Precision/Recall/F1/ROC-AUC/Confusion Matrix.
       Accuracy is intentionally never used as a decision metric.

Author: Senior DS/MLE assistant
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import RobustScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = Path("creditcard.csv")
TARGET_COL = "Class"
SCALE_COLS = ["Time", "Amount"]

# Where trained models get saved/loaded from. The expensive RandomizedSearchCV
# step (the one that took ~1.5-2 hours) only needs to run ONCE; after that,
# main() reuses the saved file instead of retraining from scratch every run.
MODEL_DIR = Path("saved_models")
LOG_REG_MODEL_PATH = MODEL_DIR / "logistic_regression_pipeline.joblib"
RF_MODEL_PATH = MODEL_DIR / "random_forest_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "metrics_summary.json"


# ----------------------------------------------------------------------------
# 1. DATA LOADING & SPLITTING
# ----------------------------------------------------------------------------
def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the credit card fraud dataset from a local CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find '{path}'. Place creditcard.csv in the working directory."
        )
    df = pd.read_csv(path)
    print(f"[load_data] Loaded dataset with shape: {df.shape}")
    return df


def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COL):
    """Separate predictors (X) from the target label (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    fraud_rate = y.mean() * 100
    print(
        f"[split_features_target] Class distribution -> "
        f"Legit: {(y == 0).sum()} | Fraud: {(y == 1).sum()} "
        f"({fraud_rate:.4f}% fraud)"
    )
    return X, y


def stratified_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2):
    """
    Perform an 80/20 stratified train-test split so both sets preserve the
    same (severe) class imbalance ratio as the original data.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(
        f"[stratified_split] Train size: {X_train.shape[0]} "
        f"(fraud rate {y_train.mean() * 100:.4f}%) | "
        f"Test size: {X_test.shape[0]} (fraud rate {y_test.mean() * 100:.4f}%)"
    )
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------------------
# 2. PIPELINE CONSTRUCTION
# ----------------------------------------------------------------------------
def build_preprocessor(all_columns: list, scale_cols: list = SCALE_COLS) -> ColumnTransformer:
    """
    Build a ColumnTransformer that scales only 'Time' and 'Amount'.
    The V1-V28 columns are already PCA-transformed (roughly standardized by
    construction), so we pass them through untouched to avoid distorting
    their structure.
    """
    passthrough_cols = [c for c in all_columns if c not in scale_cols]

    preprocessor = ColumnTransformer(
        transformers=[
            ("scale_time_amount", RobustScaler(), scale_cols),
            ("passthrough_pca_features", "passthrough", passthrough_cols),
        ]
    )
    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, classifier) -> ImbPipeline:
    """
    Construct an imbalanced-learn Pipeline:
        1. preprocessing (scale Time/Amount, passthrough V1-V28)
        2. SMOTE (training folds ONLY -- imblearn pipelines automatically
           skip resampling at .predict()/.transform() time, which is exactly
           what prevents test-set leakage)
        3. classifier placeholder
    """
    pipeline = ImbPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", classifier),
        ]
    )
    return pipeline


# ----------------------------------------------------------------------------
# 3. MODEL TRAINING & HYPERPARAMETER TUNING
# ----------------------------------------------------------------------------
def train_logistic_regression(preprocessor: ColumnTransformer, X_train, y_train) -> ImbPipeline:
    """Train a baseline Logistic Regression model inside the SMOTE pipeline."""
    print("\n[train_logistic_regression] Training Logistic Regression...")
    log_reg = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = build_pipeline(preprocessor, log_reg)

    start = time.time()
    pipeline.fit(X_train, y_train)
    print(f"[train_logistic_regression] Done in {time.time() - start:.1f}s")
    return pipeline


def tune_random_forest(preprocessor: ColumnTransformer, X_train, y_train) -> ImbPipeline:
    """
    Train a Random Forest inside the SMOTE pipeline and tune its
    hyperparameters with RandomizedSearchCV (faster than exhaustive
    GridSearchCV on a dataset this size, while still searching a wide space).

    Scoring is set to 'f1' (NOT accuracy) so the search itself optimizes for
    minority-class performance, consistent with the project's evaluation
    philosophy.
    """
    print("\n[tune_random_forest] Setting up Random Forest + RandomizedSearchCV...")

    # IMPORTANT: n_jobs=-1 here is intentionally NOT set on the RandomForest
    # itself. RandomizedSearchCV already parallelizes across CPU cores (one
    # core per fit). If the inner model ALSO tries to grab all cores, every
    # fit fights every other fit for the same cores, which can make the
    # search dramatically slower than running single-threaded trees with
    # search-level parallelism. So: parallelize at the search level only.
    rf = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)
    pipeline = build_pipeline(preprocessor, rf)

    # Smaller, cheaper search space than before. Dropping unbounded
    # max_depth (None) and large tree counts (200/300) is what actually
    # buys most of the speedup -- those settings let individual trees grow
    # huge on a SMOTE-balanced ~450k-row training set, which is the real
    # cost driver, not the number of candidates.
    param_distributions = {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [8, 12, 16],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=8,
        scoring="f1",  # optimize for F1, never accuracy
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,   # parallelize ACROSS fits, not within a single tree
        verbose=3,   # higher verbosity -> prints as each fit completes
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    print(f"[tune_random_forest] Search complete in {elapsed:.1f}s")
    print(f"[tune_random_forest] Best CV F1-score: {search.best_score_:.4f}")
    print(f"[tune_random_forest] Best params: {search.best_params_}")

    return search.best_estimator_


# ----------------------------------------------------------------------------
# 4. STRICT EVALUATION (NO ACCURACY)
# ----------------------------------------------------------------------------
def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Evaluate a fitted pipeline on the held-out test set using only metrics
    appropriate for severe class imbalance:
        - Precision / Recall / F1 (via classification_report)
        - F1-score (explicit, minority class)
        - ROC-AUC score (using predicted probabilities)
        - Confusion Matrix

    Accuracy is intentionally NEVER computed or reported.
    """
    print(f"\n{'=' * 70}")
    print(f"EVALUATION: {model_name}")
    print(f"{'=' * 70}")

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- Classification Report (Precision / Recall / F1) ---")
    report = classification_report(y_test, y_pred, target_names=["Legit", "Fraud"], digits=4)
    print(report)

    f1_fraud = f1_score(y_test, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"F1-Score (Fraud class): {f1_fraud:.4f}")
    print(f"ROC-AUC Score:          {roc_auc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    print("\n--- Confusion Matrix ---")
    print("                 Predicted Legit   Predicted Fraud")
    print(f"Actual Legit     {cm[0, 0]:<17d} {cm[0, 1]:<17d}")
    print(f"Actual Fraud     {cm[1, 0]:<17d} {cm[1, 1]:<17d}")

    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Positives (fraud caught):     {tp}")
    print(f"False Negatives (fraud missed):    {fn}")
    print(f"False Positives (legit flagged):   {fp}")
    print(f"True Negatives (legit correct):    {tn}")

    return {
        "model_name": model_name,
        "f1_fraud": f1_fraud,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compare_models(results: list):
    """Print a concise side-by-side comparison of model results, and save
    the summary metrics to JSON so other tools (e.g. the Streamlit app)
    can display real numbers instead of placeholders."""
    print(f"\n{'=' * 70}")
    print("MODEL COMPARISON SUMMARY (Accuracy intentionally excluded)")
    print(f"{'=' * 70}")
    print(f"{'Model':<25}{'F1 (Fraud)':<15}{'ROC-AUC':<15}")
    print("-" * 55)
    for r in results:
        print(f"{r['model_name']:<25}{r['f1_fraud']:<15.4f}{r['roc_auc']:<15.4f}")

    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\nBest model by ROC-AUC: {best['model_name']} ({best['roc_auc']:.4f})")

    metrics_summary = [
        {"model_name": r["model_name"], "f1_fraud": r["f1_fraud"], "roc_auc": r["roc_auc"]}
        for r in results
    ]
    MODEL_DIR.mkdir(exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"[compare_models] Saved metrics summary to '{METRICS_PATH}'")


# ----------------------------------------------------------------------------
# MODEL PERSISTENCE (save once, reuse on every future run)
# ----------------------------------------------------------------------------
def save_model(model, path: Path):
    """Save a fitted pipeline to disk so it doesn't need retraining."""
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Saved to '{path}'")


def load_model(path: Path):
    """Load a previously saved fitted pipeline from disk, if it exists."""
    if path.exists():
        print(f"[load_model] Found existing model at '{path}' -- loading instead of retraining.")
        return joblib.load(path)
    return None


# ----------------------------------------------------------------------------
# MAIN ORCHESTRATION
# ----------------------------------------------------------------------------
def main():
    print("Project 2: Supervised Learning - Fraud Detection Pipeline")
    print("=" * 70)

    # 1. Load & split data
    df = load_data()
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = stratified_split(X, y)

    # 2. Build preprocessing step shared by both models
    preprocessor = build_preprocessor(all_columns=list(X.columns))

    # 3a. Logistic Regression -- reuse saved model if one exists, else train + save
    log_reg_pipeline = load_model(LOG_REG_MODEL_PATH)
    if log_reg_pipeline is None:
        log_reg_pipeline = train_logistic_regression(preprocessor, X_train, y_train)
        save_model(log_reg_pipeline, LOG_REG_MODEL_PATH)

    # 3b. Random Forest (tuned) -- reuse saved model if one exists, else
    # run the expensive RandomizedSearchCV once and save the result.
    rf_pipeline = load_model(RF_MODEL_PATH)
    if rf_pipeline is None:
        rf_pipeline = tune_random_forest(preprocessor, X_train, y_train)
        save_model(rf_pipeline, RF_MODEL_PATH)

    # 4. Strict evaluation (no accuracy) on the untouched test set
    results = []
    results.append(evaluate_model(log_reg_pipeline, X_test, y_test, "Logistic Regression"))
    results.append(evaluate_model(rf_pipeline, X_test, y_test, "Random Forest (Tuned)"))

    compare_models(results)

    print(
        f"\n[main] Tip: models are cached in '{MODEL_DIR}/'. Delete that folder "
        "if you ever want to force a full retrain (e.g. after changing the "
        "hyperparameter search space or getting new data)."
    )


if __name__ == "__main__":
    main()
