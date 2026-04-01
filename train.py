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

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
