"""
Game Churn Prediction AI Application

This Streamlit application allows users to:

  Upload player behavior data
  Predict churn risk
  View feature importance
  Generate engagement recommendations
  Download reports

Input:
  CSV dataset with player behavior features

Output:
  Churn predictions
  Risk classification
  Engagement recommendations
"""

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from sklearn.metrics import confusion_matrix

from agents.engagement_agent import generate_recommendations
from utils.report_generator import generate_structured_report, generate_pdf_report
from preprocess import preprocess_data

# ------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------
st.set_page_config(
    page_title="Game Churn Prediction AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme & styling
st.markdown("""
<style>
    /* Dark Theme Core */
    .stApp {
        background-color: #0f172a;
        color: #f8fafc;
    }

    /* Typography */
    h1, h2, h3, h4, h5 {
        color: #f8fafc !important;
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #111827;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #38bdf8;
    }

    /* Tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Buttons */
    div.stButton > button, div.stDownloadButton > button {
        background-color: #2563eb !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        border: none !important;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #1d4ed8 !important;
        color: white !important;
    }

    /* Professional Navbar */
    .stTabs [data-baseweb="tab-list"] {
        justify-content: center;
        gap: 40px;
        margin-top: 10px;
        margin-bottom: 20px;
        border-bottom: 1px solid #2b3a55;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 18px;
        font-weight: 600;
        padding: 12px 18px;
    }

    .stTabs [aria-selected="true"] {
        color: #4da6ff !important;
        border-bottom: 3px solid #4da6ff !important;
        outline: none !important;
        box-shadow: none !important;
    }

    .stTabs [data-baseweb="tab-highlight"] {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------
# HEADER
# ------------------------------------------------
st.title("Game Churn Prediction AI Dashboard")
st.markdown("##### AI-driven player retention and engagement optimization system")
st.markdown("<br>", unsafe_allow_html=True)

# ------------------------------------------------
# CACHING — MODELS & METRICS
# ------------------------------------------------
@st.cache_resource
def load_model_by_name(model_name):
    """Load a specific model from disk by its file name."""
    try:
        model = joblib.load(f"models/{model_name}")
        features = joblib.load("models/model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None


@st.cache_data
def load_metrics():
    """Load training metrics from metrics.json."""
    path = "metrics.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None


@st.cache_data
def cached_preprocess(df_raw, expected_features):
    """Cached wrapper around the shared preprocessing pipeline."""
    return preprocess_data(df_raw, expected_features)


@st.cache_data
def make_predictions(_model, X):
    """Run model inference and return churn probabilities."""
    probabilities = _model.predict_proba(X)
    return probabilities[:, 1]


# ------------------------------------------------
# SIDEBAR
# ------------------------------------------------
st.sidebar.markdown("## Game Churn Prediction AI")
st.sidebar.markdown("""
**Description:**

Predict player churn risk and generate engagement strategies using machine learning and agentic AI.
""")
st.sidebar.markdown("---")

# Model selection dropdown
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Select Prediction Model",
    ["Random Forest", "Logistic Regression"]
)

# Map user choice to file names
model_file_map = {
    "Random Forest": "random_forest_model.pkl",
    "Logistic Regression": "logistic_regression_model.pkl",
}
metrics_key_map = {
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
}

model, expected_features = load_model_by_name(model_file_map[model_choice])

# Fallback: try legacy churn_model.pkl for first-time users who haven't run train.py
if model is None:
    model, expected_features = load_model_by_name("churn_model.pkl")

st.sidebar.markdown("---")
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file with player data", type=["csv"])

st.sidebar.markdown("""
**Required Columns (Example):**
- Age, Gender, Location
- GameGenre, PlayTimeHours
- InGamePurchases, GameDifficulty
- SessionsPerWeek, AvgSessionDurationMinutes
- PlayerLevel, AchievementsUnlocked
""")

# ------------------------------------------------
# FILE HANDLING TO SESSION STATE
# ------------------------------------------------
if uploaded_file is not None:
    try:
        @st.cache_data
        def load_csv(file_bytes):
            return pd.read_csv(file_bytes)
            
        df = load_csv(uploaded_file)
        
        if df.empty:
            st.error("Uploaded file is empty.")
            st.stop()
            
        st.session_state["data"] = df
    except Exception as e:
        st.error(f"Error processing the file: {e}")

# ------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------
if model is None:
    st.error("Model not found! Please run `python train.py` first, or ensure model .pkl files exist in the models/ directory.")
    st.stop()

# LOAD DATA FROM SESSION STATE
if "data" in st.session_state:
    df_raw = st.session_state["data"]
else:
    st.warning("Please upload a dataset first.")
    st.stop()

# PREDICTION PIPELINE
try:
    X = cached_preprocess(df_raw, expected_features)
    churn_probs = make_predictions(model, X)
    
    # Store predictions in Session State safely
    y_pred = (churn_probs >= 0.5).astype(int)
    
    if 'Churn' in df_raw.columns:
        st.session_state["y_true"] = df_raw['Churn']
    else:
        st.session_state["y_true"] = None # To prevent key errors
        
    st.session_state["predictions"] = churn_probs
    st.session_state["y_pred"] = y_pred

    # Attach predictions for dataframe rendering
    df_raw['Churn Probability'] = churn_probs

    def get_risk_level(prob):
        if prob < 0.4: return 'Low'
        elif prob < 0.7: return 'Medium'
        else: return 'High'

    df_raw['Risk Level'] = df_raw['Churn Probability'].apply(get_risk_level)

    # Load training metrics
    all_metrics = load_metrics()
    current_metrics = None
    if all_metrics:
        current_metrics = all_metrics.get(metrics_key_map[model_choice])

    # ------------------------------------------------
    # TAB-BASED NAVIGATION
    # ------------------------------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard",
        "Predictions",
        "Insights",
        "AI Assistant"
    ])

    # ------------------------------------------------
    # TAB 1 — DASHBOARD
    # ------------------------------------------------
    with tab1:
        st.markdown(f"### Key Performance Indicators — {model_choice}")

        total_players = len(df_raw)
        high_risk_players = len(df_raw[df_raw['Risk Level'] == 'High'])
        avg_churn_prob = df_raw['Churn Probability'].mean()

        accuracy_display = f"{current_metrics['accuracy']*100:.1f}%" if current_metrics else "N/A"

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Total Players", total_players)
        m_col2.metric("High Risk Players", high_risk_players)
        m_col3.metric("Average Churn Probability", f"{avg_churn_prob*100:.1f}%")
        m_col4.metric("Model Accuracy", accuracy_display)

        # Evaluation Metrics Row
        if current_metrics:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Model Evaluation Metrics")
            e_col1, e_col2, e_col3, e_col4 = st.columns(4)
            e_col1.metric("Accuracy", f"{current_metrics['accuracy']*100:.1f}%")
            e_col2.metric("Precision", f"{current_metrics['precision']*100:.1f}%")
            e_col3.metric("Recall", f"{current_metrics['recall']*100:.1f}%")
            e_col4.metric("F1 Score", f"{current_metrics['f1_score']*100:.1f}%")

            # Cross-Validation Results
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Cross-Validation Results (5-Fold)")
            cv_scores = current_metrics.get("cv_scores", [])
            cv_mean = current_metrics.get("cv_mean", 0)
            if cv_scores:
                cv_col1, cv_col2 = st.columns([3, 1])
                with cv_col1:
                    cv_df = pd.DataFrame({
                        "Fold": [f"Fold {i+1}" for i in range(len(cv_scores))],
                        "Accuracy": [f"{s*100:.2f}%" for s in cv_scores]
                    })
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                with cv_col2:
                    st.metric("Mean CV Accuracy", f"{cv_mean*100:.2f}%")

        # Confusion Matrix using Safe State Fetching
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Confusion Matrix")
        if "predictions" in st.session_state and st.session_state.get("y_true") is not None:
            y_true = st.session_state["y_true"]
            y_pred = st.session_state["y_pred"]
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Incorporate dark layout into user's requested syntax where possible
            fig, ax = plt.subplots(figsize=(6, 4))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="Blues",
                ax=ax,
                xticklabels=["Not Churned", "Churned"],
                yticklabels=["Not Churned", "Churned"]
            )
            ax.set_xlabel("Predicted", color='#f8fafc')
            ax.set_ylabel("Actual", color='#f8fafc')
            ax.tick_params(colors='#f8fafc')
            plt.title("Confusion Matrix", color='#f8fafc')
            
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Run prediction first to view confusion matrix (Requires 'Churn' column).")

    # ------------------------------------------------
    # TAB 2 — PREDICTIONS
    # ------------------------------------------------
    with tab2:
        st.markdown("### Prediction Results")
        st.markdown("Here is the estimated churn probability and risk level for each player.")

        col1, col2, col3 = st.columns(3)
        with col1: st.success("Low Risk  -- churn probability below 40%")
        with col2: st.warning("Medium Risk  -- churn probability between 40% and 70%")
        with col3: st.error("High Risk  -- churn probability above 70%")

        def color_risk(val):
            color = '#28a745' if val == 'Low' else '#ffc107' if val == 'Medium' else '#dc3545'
            return f'color: {color}; font-weight: bold'

        df_to_show = df_raw.head(500)
        try:
            styled_df = df_to_show[['Churn Probability', 'Risk Level'] + list(df_to_show.columns[:-2])].style.map(
                color_risk, subset=['Risk Level']
            ).format({'Churn Probability': '{:.1%}'}).set_properties(**{'text-align': 'center'})
        except AttributeError:
            styled_df = df_to_show[['Churn Probability', 'Risk Level'] + list(df_to_show.columns[:-2])].style.applymap(
                color_risk, subset=['Risk Level']
            ).format({'Churn Probability': '{:.1%}'}).set_properties(**{'text-align': 'center'})

        st.dataframe(styled_df, use_container_width=True)
        st.caption("Showing top 500 records for performance optimization.")

    # ------------------------------------------------
    # TAB 3 — INSIGHTS
    # ------------------------------------------------
    with tab3:
        st.markdown("### Top Drivers of Player Churn")
        st.markdown("This chart shows which gameplay factors most influence churn risk.")

        @st.cache_data
        def get_feature_importance(_model, features):
            importances = _model.feature_importances_
            return pd.DataFrame({
                'Feature': features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)

        # Feature importance is only available for tree-based models
        if hasattr(model, 'feature_importances_'):
            feat_imp_df = get_feature_importance(model, tuple(expected_features))

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            ax.tick_params(colors='#f8fafc')
            ax.yaxis.label.set_color('#f8fafc')
            ax.xaxis.label.set_color('#f8fafc')
            ax.title.set_color('#f8fafc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#334155')

            sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10), ax=ax, palette='viridis')
            ax.set_title('Top 10 Drivers of Churn')
            st.pyplot(fig)
            plt.close(fig)

            top_features = feat_imp_df.head(3)['Feature'].tolist()
            st.info(f"**Insights:** The model indicates churn is primarily driven by: **{', '.join(top_features)}**.")

            demographic_keywords = ['Age', 'Gender', 'Location']
            engagement_is_key = True
            for feat in top_features:
                if any(dem_key in feat for dem_key in demographic_keywords):
                    engagement_is_key = False
                    break
            if engagement_is_key:
                st.caption("Note: Engagement behavior is more influential than demographics (e.g., age, gender, location).")
        else:
            # Logistic Regression uses coefficients instead
            coefs = model.coef_[0]
            coef_df = pd.DataFrame({
                'Feature': expected_features,
                'Coefficient': np.abs(coefs)
            }).sort_values(by='Coefficient', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6))
            fig.patch.set_facecolor('#111827')
            ax.set_facecolor('#111827')
            ax.tick_params(colors='#f8fafc')
            ax.yaxis.label.set_color('#f8fafc')
            ax.xaxis.label.set_color('#f8fafc')
            ax.title.set_color('#f8fafc')
            for spine in ax.spines.values():
                spine.set_edgecolor('#334155')

            sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10), ax=ax, palette='viridis')
            ax.set_title('Top 10 Feature Coefficients (Absolute)')
            st.pyplot(fig)
            plt.close(fig)

            top_features = coef_df.head(3)['Feature'].tolist()
            st.info(f"**Insights:** The model indicates churn is primarily influenced by: **{', '.join(top_features)}**.")

        # Model Comparison Table
        if all_metrics and len(all_metrics) > 1:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### Model Comparison")
            comparison_rows = []
            for mname, mdata in all_metrics.items():
                comparison_rows.append({
                    "Model": mname.replace("_", " ").title(),
                    "Accuracy": f"{mdata['accuracy']*100:.1f}%",
                    "Precision": f"{mdata['precision']*100:.1f}%",
                    "Recall": f"{mdata['recall']*100:.1f}%",
                    "F1 Score": f"{mdata['f1_score']*100:.1f}%",
                    "CV Mean": f"{mdata['cv_mean']*100:.1f}%",
                })
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)

    # ------------------------------------------------
    # TAB 4 — AI ASSISTANT
    # ------------------------------------------------
    with tab4:
        st.markdown("### Agentic AI Game Engagement Optimization Assistant")
        st.markdown("Select a player from the dataset to generate a structured retention report and engagement recommendations.")

        player_index = st.selectbox("Select Player Index:", df_raw.index)

        if st.button("Generate Engagement Report"):
            with st.spinner("Generating engagement report..."):
                player_data = df_raw.iloc[player_index].to_dict()
                risk_prob = player_data['Churn Probability']
                risk_level = player_data['Risk Level']

                recommendations = generate_recommendations(df_raw.iloc[player_index])
                report = generate_structured_report(player_index, player_data, risk_prob, risk_level, recommendations)

                st.markdown("---")

                st.markdown("### Player Behavior Summary")
                st.write(report.get("Player Behavior Summary", "No summary available."))

                st.markdown("### Churn Risk Interpretation")
                st.write(report.get("Churn Risk Interpretation", "No interpretation available."))

                st.markdown("### Recommended Engagement Actions")
                for item in report.get("Engagement Recommendations", []):
                    st.write(f"* {item}")

                st.markdown("### Supporting References")
                st.write(report.get("Supporting References", "No references available."))

                st.markdown("### Ethical Disclaimer")
                st.info(report.get("Ethical Disclaimer", "These are automated suggestions."))

                # Store bytes for download
                st.session_state['pdf_report'] = generate_pdf_report(report)
                st.session_state['report_idx'] = player_index
                st.markdown("---")

        # Download Section
        st.markdown("### Download Reports")
        col_dl1, col_dl2 = st.columns(2)

        with col_dl1:
            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )

        with col_dl2:
            if 'pdf_report' in st.session_state:
                st.download_button(
                    label="Download Engagement Report PDF",
                    data=st.session_state['pdf_report'],
                    file_name=f"engagement_report_player_{st.session_state['report_idx']}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            else:
                st.info("Generate a report above to unlock PDF download.")

except Exception as e:
    st.error(f"Error processing the data: {e}")


# ------------------------------------------------
# FOOTER
# ------------------------------------------------
st.markdown("---")
st.markdown("<div style='text-align: center; color: gray; padding: 10px;'>Built using Streamlit | Random Forest | Logistic Regression | Agentic AI | 2026</div>", unsafe_allow_html=True)
