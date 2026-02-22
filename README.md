# Game Churn Prediction AI

## Overview
The **Game Churn Prediction AI** is a complete machine learning system built to predict online player churn. It evaluates gameplay engagement to determine the probability of a player abandoning the game. This project focuses on demonstrating end-to-end Machine Learning workflows, from data cleaning and processing, to model training, and delivering a clean user interface using Streamlit.

## Project Structure
```plaintext
game-churn-prediction-ai/
├── data/
│   ├── raw_data.csv       # Raw Kaggle Game Online Behavior dataset
│   └── clean_data.csv     # Preprocessed data ready for ML training
├── notebooks/
│   ├── data_cleaning.ipynb # Jupyter notebook executing robust data processing actions
│   └── model_training.ipynb# Jupyter notebook showcasing ML evaluation procedures and results
├── models/
│   ├── churn_model.pkl     # Optimized Random Forest Classifier
│   └── model_features.pkl  # List of categorical bindings used for Streamlit mappings
├── app.py                  # Front-end system rendering ML execution with UI elements
├── requirements.txt        # Recommended Python modules/libraries constraint list
└── README.md               # User manual
```

## Features
1. **Data Cleaning & Engineering:** Robust data cleaning techniques that address missing values, feature encoding, and redundant data elimination.
2. **Machine Learning:** Incorporates supervised classification architectures using Random Forest and Logistic Regression. Output is quantified using Accuracy, Precision, Recall, and F-1 metrics.
3. **Interactive UI App:** Seamless user experience leveraging `streamlit` to submit bulk game metrics and dynamically predict player churn probabilities and risk levels (Low, Medium, High).
4. **Insights Dashboard:** Highlights "Feature Importance" providing product-ready perspectives indicating which factors most heavily impact a player churning.

## How to Run Locally

### 1. Requirements
Ensure you have Python 3.8+ installed.

### 2. Setup Virtual Environment (Optional but Recommended)
For Mac / Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

For Windows:
```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Build Dataset & Assets (Optional, models are pre-trained)
To manually generate the assets and run model training scripts:
```bash
python3 build_assets.py
```

### 5. Start the Application
Boot up the Streamlit interface using:
```bash
streamlit run app.py
```

## Dataset Reference
Predict Online Gaming Behavior Dataset (from Kaggle):
https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset

## Tech Stack
- **Data Computation:** `pandas`, `numpy`
- **Machine Learning:** `scikit-learn`
- **Data Visualization:** `matplotlib`, `seaborn`
- **Application Server:** `streamlit`

## Notes for Beginners
The project heavily prioritizes human-readable comments. Each component showcases beginner-friendly steps such as defining evaluation metrics, standardizing the schema structures, and rendering user feedback immediately.
