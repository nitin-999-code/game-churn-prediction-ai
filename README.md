# Game Churn Prediction AI

## Overview

The **Game Churn Prediction AI** is a complete machine learning system built to predict online player churn. It evaluates gameplay engagement to determine the probability of a player abandoning the game and provides AI-powered engagement recommendations through an agentic assistant.

This project demonstrates an end-to-end ML pipeline — from data cleaning and model training, through evaluation and comparison, to a professional Streamlit dashboard with an integrated Agentic AI assistant.

---

## Demo

### Dashboard — KPI Metrics & Confusion Matrix
The Dashboard tab shows real-time KPI cards, model evaluation metrics, cross-validation results, and a confusion matrix heatmap.

### Predictions — Risk Classification Table
Upload a CSV and instantly see per-player churn probabilities with color-coded risk levels.

### Insights — Feature Importance
Visual breakdown of which gameplay factors most heavily influence churn.

### AI Assistant — Engagement Optimization
Select any player to generate a structured retention report with actionable recommendations and a downloadable PDF.

---

## Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | CV Mean |
|---------------------|----------|-----------|--------|----------|---------|
| Random Forest       | 94.5%    | 91.5%     | 86.6%  | 89.0%    | 94.4%   |
| Logistic Regression | 87.4%    | 80.4%     | 67.7%  | 73.5%    | 87.7%   |

---

## Project Structure

```plaintext
game-churn-prediction-ai/
├── agents/
│   └── engagement_agent.py      # Agentic AI engagement optimization assistant
├── knowledge_base/
│   └── strategies.json          # RAG-style retention strategy knowledge base
├── utils/
│   └── report_generator.py      # Structured report & PDF export generator
├── data/
│   ├── raw_data.csv             # Raw Kaggle gaming behavior dataset
│   └── clean_data.csv           # Preprocessed data ready for ML training
├── notebooks/
│   ├── data_cleaning.ipynb      # Data processing notebook
│   └── model_training.ipynb     # Model evaluation notebook
├── models/
│   ├── random_forest_model.pkl  # Trained Random Forest classifier
│   ├── logistic_regression_model.pkl  # Trained Logistic Regression classifier
│   ├── churn_model.pkl          # Backward-compatible model alias
│   └── model_features.pkl       # Feature name list for inference alignment
├── app.py                       # Streamlit dashboard application
├── train.py                     # CLI training script
├── preprocess.py                # Shared preprocessing module
├── metrics.json                 # Saved evaluation & cross-validation metrics
├── requirements.txt             # Python dependencies
├── runtime.txt                  # Python runtime specification
├── architecture.md              # System architecture diagram (Mermaid)
├── .gitignore                   # Git ignore rules
└── README.md                    # This file
```

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/username/game-churn-prediction-ai.git
cd game-churn-prediction-ai
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Mac / Linux
# venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training

Train both models and generate `metrics.json`:

```bash
python train.py
```

**Output:**
- `models/random_forest_model.pkl`
- `models/logistic_regression_model.pkl`
- `models/model_features.pkl`
- `metrics.json`

The script logs accuracy, precision, recall, F1 score, and 5-fold cross-validation results for each model.

---

## Usage

### Run the Streamlit App

```bash
streamlit run app.py
```

### Features

1. **Model Selection** — Switch between Random Forest and Logistic Regression from the sidebar.
2. **KPI Dashboard** — View total players, high-risk count, average churn probability, and model accuracy at a glance.
3. **Evaluation Metrics** — Accuracy, Precision, Recall, F1 Score displayed dynamically from `metrics.json`.
4. **Confusion Matrix** — Visual heatmap comparing predictions against ground truth (when Churn column is present).
5. **Cross-Validation** — 5-fold CV scores and mean accuracy displayed per model.
6. **Prediction Table** — Color-coded risk levels for up to 500 players.
7. **Feature Importance** — Top 10 churn drivers visualized with bar charts (Random Forest) or coefficient magnitudes (Logistic Regression).
8. **Model Comparison Table** — Side-by-side performance metrics for both models.
9. **AI Engagement Assistant** — Agentic AI that analyzes player behavior and generates structured retention recommendations.
10. **PDF Export** — Download engagement reports as professionally formatted PDFs.

---

## Architecture

See [architecture.md](architecture.md) for the full system architecture diagram including:

- User Upload flow
- Churn Prediction Model
- Agentic AI Assistant
- Knowledge Base (RAG)
- Report Generator
- Streamlit UI

---

## Dataset Reference

Predict Online Gaming Behavior Dataset (Kaggle):  
https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset

---

## Tech Stack

| Layer              | Technologies                          |
|--------------------|---------------------------------------|
| Data Processing    | `pandas`, `numpy`                     |
| Machine Learning   | `scikit-learn`                        |
| Visualization      | `matplotlib`, `seaborn`               |
| Application Server | `streamlit`                           |
| Report Generation  | `reportlab`                           |
| AI Agent           | Rule-based reasoning + JSON knowledge base |

---

## Deployment

This project is compatible with **Streamlit Community Cloud** for free public hosting.

---

## Notes

- Models are pre-trained and included in the `models/` directory. You can retrain at any time with `python train.py`.
- The project uses only free-tier, open-source tools — no paid APIs required.
- All code includes comprehensive docstrings and inline comments for readability.
