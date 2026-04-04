"""
CLI Training Script for Game Churn Prediction AI

This script loads the cleaned dataset, trains both a Random Forest and
a Logistic Regression classifier, evaluates them with standard metrics
and 5-fold cross-validation, saves the trained models to disk, and
persists evaluation metrics to metrics.json.

Usage:
    python train.py
"""

import json
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def main():
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("Loading dataset...")
    df = pd.read_csv("data/clean_data.csv")
    print(f"  Dataset shape: {df.shape}")

    # Separate features and target
    target_col = "Churn"
    X = df.drop(columns=[target_col])
    y = df[target_col]
    feature_names = list(X.columns)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")

    metrics_all = {}

    # ------------------------------------------------------------------
    # 2. Train Random Forest
    # ------------------------------------------------------------------
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)

    rf_metrics = {
        "accuracy": round(accuracy_score(y_test, rf_preds), 4),
        "precision": round(precision_score(y_test, rf_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, rf_preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, rf_preds, zero_division=0), 4),
    }

    print("  Running 5-fold cross-validation...")
    rf_cv = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    rf_metrics["cv_scores"] = [round(s, 4) for s in rf_cv.tolist()]
    rf_metrics["cv_mean"] = round(rf_cv.mean(), 4)
    print(f"  CV scores: {rf_metrics['cv_scores']}")
    print(f"  CV mean:   {rf_metrics['cv_mean']}")
    print(f"  Accuracy:  {rf_metrics['accuracy']}")
    print(f"  Precision: {rf_metrics['precision']}")
    print(f"  Recall:    {rf_metrics['recall']}")
    print(f"  F1 Score:  {rf_metrics['f1_score']}")

    metrics_all["random_forest"] = rf_metrics

    # ------------------------------------------------------------------
    # 3. Train Logistic Regression
    # ------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)

    lr_metrics = {
        "accuracy": round(accuracy_score(y_test, lr_preds), 4),
        "precision": round(precision_score(y_test, lr_preds, zero_division=0), 4),
        "recall": round(recall_score(y_test, lr_preds, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, lr_preds, zero_division=0), 4),
    }

    print("  Running 5-fold cross-validation...")
    lr_cv = cross_val_score(lr, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    lr_metrics["cv_scores"] = [round(s, 4) for s in lr_cv.tolist()]
    lr_metrics["cv_mean"] = round(lr_cv.mean(), 4)
    print(f"  CV scores: {lr_metrics['cv_scores']}")
    print(f"  CV mean:   {lr_metrics['cv_mean']}")
    print(f"  Accuracy:  {lr_metrics['accuracy']}")
    print(f"  Precision: {lr_metrics['precision']}")
    print(f"  Recall:    {lr_metrics['recall']}")
    print(f"  F1 Score:  {lr_metrics['f1_score']}")

    metrics_all["logistic_regression"] = lr_metrics

    # ------------------------------------------------------------------
    # 4. Save models
    # ------------------------------------------------------------------
    print("\nSaving models...")
    joblib.dump(rf, "models/random_forest_model.pkl")
    joblib.dump(lr, "models/logistic_regression_model.pkl")
    joblib.dump(feature_names, "models/model_features.pkl")

    # Keep backward-compatible alias
    joblib.dump(rf, "models/churn_model.pkl")
    print("  Saved: models/random_forest_model.pkl")
    print("  Saved: models/logistic_regression_model.pkl")
    print("  Saved: models/model_features.pkl")
    print("  Saved: models/churn_model.pkl (backward-compatible alias)")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
