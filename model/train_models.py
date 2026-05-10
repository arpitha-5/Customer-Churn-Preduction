"""
PHASE 4 & 5: MODEL BUILDING + OPTIMIZATION
Trains 5 ML models, tunes the best one, saves results.
"""
import os, json
import numpy as np, pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed, skipping.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)
MODEL_DIR = os.path.join(PROJECT_DIR, "model", "saved_models")
PROCESSED_DIR = os.path.join(PROJECT_DIR, "data", "processed")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_processed_data():
    path = os.path.join(PROCESSED_DIR, "processed_data.pkl")
    if not os.path.exists(path):
        from data_preprocessing import preprocess_pipeline
        return preprocess_pipeline()
    return joblib.load(path)


def evaluate_model(model, X_test, y_test, name):
    y_pred = model.predict(X_test)
    metrics = {
        "model": name,
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
    print(f"\n{'─'*40}")
    print(f"  {name}")
    print(f"{'─'*40}")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Confusion Matrix:\n  {metrics['confusion_matrix']}")
    return metrics


def train_all_models():
    print("\n" + "="*60 + "\nPHASE 4: MODEL BUILDING\n" + "="*60)
    data = load_processed_data()
    X_train, X_test = data["X_train"], data["X_test"]
    y_train, y_test = data["y_train"], data["y_test"]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100, use_label_encoder=False,
            eval_metric='logloss', random_state=42
        )

    all_results = []
    for name, model in models.items():
        print(f"\n[TRAIN] Training {name}...")
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test, name)
        all_results.append(metrics)
        joblib.dump(model, os.path.join(MODEL_DIR, f"{name.lower().replace(' ', '_')}.pkl"))

    # Save comparison
    results_path = os.path.join(MODEL_DIR, "model_comparison.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60 + "\n  MODEL COMPARISON SUMMARY\n" + "="*60)
    for r in sorted(all_results, key=lambda x: x['f1_score'], reverse=True):
        print(f"  {r['model']:25s} | F1={r['f1_score']:.4f} | Acc={r['accuracy']:.4f}")

    return all_results, models, data


def optimize_best_model(all_results, models, data):
    print("\n" + "="*60 + "\nPHASE 5: MODEL OPTIMIZATION\n" + "="*60)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    best_name = max(all_results, key=lambda x: x['f1_score'])['model']
    print(f"[INFO] Best base model: {best_name}")

    # Hyperparameter grids
    param_grids = {
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [10, 20, None],
            "min_samples_split": [2, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"],
        },
        "Decision Tree": {
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["rbf", "linear"],
        }
    }

    if best_name in param_grids:
        print(f"[TUNE] Running GridSearchCV for {best_name}...")
        grid = GridSearchCV(
            models[best_name], param_grids[best_name],
            cv=5, scoring='f1', n_jobs=-1, verbose=0
        )
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        print(f"[TUNE] Best params: {grid.best_params_}")
        print(f"[TUNE] Best CV F1: {grid.best_score_:.4f}")
    else:
        best_model = models[best_name]

    final_metrics = evaluate_model(best_model, X_test, y_test, f"{best_name} (Tuned)")

    # Save best model
    best_model_path = os.path.join(MODEL_DIR, "best_model.pkl")
    joblib.dump(best_model, best_model_path)
    print(f"\n✅ Best model saved to {best_model_path}")

    # Save metadata
    meta = {"best_model_name": best_name, "metrics": final_metrics}
    with open(os.path.join(MODEL_DIR, "best_model_meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return best_model


def run_training_pipeline():
    all_results, models, data = train_all_models()
    best_model = optimize_best_model(all_results, models, data)
    return best_model


if __name__ == "__main__":
    run_training_pipeline()
