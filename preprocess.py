"""
Preprocessing Module for Game Churn Prediction AI

Provides reusable data preprocessing functions shared across
the training pipeline (train.py) and the Streamlit application (app.py).

Functions:
    preprocess_data: Full pipeline — handle missing values, drop non-predictive
                     columns, one-hot encode, and align features.
    handle_missing_values: Fill numeric NaNs with median, categorical with mode.
    drop_non_predictive_columns: Remove PlayerID, EngagementLevel, Churn columns.
    align_features: Ensure the dataframe matches the model's expected feature set.
"""

# Commit 1: Initial preprocessing setup
# Commit 2: Added data handling functions
# Commit 3: Implemented feature alignment
# Commit 4: Added preprocessing pipeline

import pandas as pd
import numpy as np


def handle_missing_values(df):
    """
    Fill missing values in a dataframe.

    - Numeric columns: filled with column median.
    - Categorical (object) columns: filled with column mode, or 'Unknown'.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with missing values filled.
    """
    df = df.copy()

    df_numeric = df.select_dtypes(include=[np.number])
    if not df_numeric.empty:
        df[df_numeric.columns] = df[df_numeric.columns].fillna(df_numeric.median())

    for col in df.select_dtypes(include=["object"]).columns:
        mode_vals = df[col].mode()
        if not mode_vals.empty:
            df[col] = df[col].fillna(mode_vals[0])
        else:
            df[col] = df[col].fillna("Unknown")

    return df


def drop_non_predictive_columns(df):
    """
    Remove columns that are not useful for prediction.

    Drops: PlayerID, EngagementLevel, Churn (if present).

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    for col in ["PlayerID", "EngagementLevel", "Churn"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return df


def align_features(df, expected_features):
    """
    Align a one-hot-encoded dataframe to match the model's expected feature list.

    Missing features are filled with 0. Extra features are dropped.

    Args:
        df (pd.DataFrame): One-hot-encoded dataframe.
        expected_features (list): List of feature names the model expects.

    Returns:
        pd.DataFrame: Aligned dataframe ready for prediction.
    """
    aligned = pd.DataFrame(columns=expected_features)
    for col in expected_features:
        if col in df.columns:
            aligned[col] = df[col]
        else:
            aligned[col] = 0
    return aligned


def preprocess_data(df, expected_features):
    """
    End-to-end preprocessing pipeline.

    Steps:
        1. Handle missing values
        2. Drop non-predictive columns
        3. One-hot encode categorical features
        4. Align columns to the trained model's feature set

    Args:
        df (pd.DataFrame): Raw input dataframe.
        expected_features (list): Feature names the model was trained on.

    Returns:
        pd.DataFrame: Preprocessed dataframe ready for model inference.
    """
    df = handle_missing_values(df)
    df = drop_non_predictive_columns(df)
    df = pd.get_dummies(df, drop_first=True)
    df = align_features(df, expected_features)
    return df
