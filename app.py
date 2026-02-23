import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration for a better layout
st.set_page_config(page_title="Game Churn Prediction AI", layout="wide")

st.title("Game Churn Prediction AI")
st.markdown("""
Welcome to the Game Churn Prediction AI. This tool allows you to upload player gameplay and engagement data 
to predict which players are at risk of leaving (churning).
""")

# Load the trained model and expected feature names
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/churn_model.pkl")
        features = joblib.load("models/model_features.pkl")
        return model, features
    except FileNotFoundError:
        return None, None

model, expected_features = load_model()



if model is None:
    st.error("Model not found! Please ensure 'models/churn_model.pkl' and 'models/model_features.pkl' exist.")
else:
    #upload-chart 
    st.sidebar.header("1. Upload Data")
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file with player data", type=["csv"])
    
    # output-chart
    st.sidebar.markdown("""
    **Required Columns (Example):**
    - Age, Gender, Location
    - GameGenre, PlayTimeHours
    - InGamePurchases, GameDifficulty
    - SessionsPerWeek, AvgSessionDurationMinutes
    - PlayerLevel, AchievementsUnlocked
    """)
    
    if uploaded_file is not None:
        # Load the user uploaded dataset
        st.subheader("Uploaded Data Preview")
        try:
            df_raw = pd.read_csv(uploaded_file)
            st.dataframe(df_raw.head())
            
            # --- Preprocessing the uploaded data ---
            df_processed = df_raw.copy()
            
            # 1. Handle missing values
            df_processed.fillna(df_processed.median(numeric_only=True), inplace=True)
            for col in df_processed.select_dtypes(include=['object']).columns:
                df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])
            
            # 2. Drop PlayerID if exists
            if 'PlayerID' in df_processed.columns:
                df_processed.drop(columns=['PlayerID'], inplace=True)
                
            # If 'EngagementLevel' or 'Churn' exists, we should likely remove it for prediction
            # but usually prediction data wouldn't have the target. Let's drop them to be safe.
            if 'EngagementLevel' in df_processed.columns:
                df_processed.drop(columns=['EngagementLevel'], inplace=True)
            if 'Churn' in df_processed.columns:
                df_processed.drop(columns=['Churn'], inplace=True)
            
            # 3. One-hot encoding
            df_processed = pd.get_dummies(df_processed, drop_first=True)
            
            # 4. Align features with the trained model
            # We create a dataframe with the expected columns, filled with 0s
            X = pd.DataFrame(columns=expected_features)
            
            # Fill in the data from the processed dataframe
            for col in expected_features:
                if col in df_processed.columns:
                    X[col] = df_processed[col]
                else:
                    X[col] = 0 # Feature missing in the uploaded file, fill with 0
                    
            # Make predictions
            probabilities = model.predict_proba(X)
            # Probability of churning is usually the second class (index 1)
            churn_probs = probabilities[:, 1]
            
            # Attach predictions to the original dataframe to display
            df_raw['Churn Probability'] = churn_probs
            
            # Define Risk Level based on probability
            def get_risk_level(prob):
                if prob < 0.4:
                    return 'Low'
                elif prob < 0.7:
                    return 'Medium'
                else:
                    return 'High'
                    
            df_raw['Risk Level'] = df_raw['Churn Probability'].apply(get_risk_level)
            
            st.subheader("Prediction Results")
            st.markdown("Here is the estimated churn probability and risk level for each player.")
            
            # --- Churn Risk Explanation Box ---
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success("Low Risk  → churn probability below 40%")
            with col2:
                st.warning("Medium Risk  → churn probability between 40% and 70%")
            with col3:
                st.error("High Risk  → churn probability above 70%")
            
            # Style the dataframe for better visualization
            pd.set_option("styler.render.max_elements", 4000000)
            
            def color_risk(val):
                color = 'green' if val == 'Low' else 'orange' if val == 'Medium' else 'red'
                return f'color: {color}; font-weight: bold'
            
            try:
                # pandas >= 2.1.0 uses .map
                styled_df = df_raw[['Churn Probability', 'Risk Level'] + list(df_raw.columns[:-2])].style.map(
                    color_risk, subset=['Risk Level']
                )
            except AttributeError:
                # fallback for older versions
                styled_df = df_raw[['Churn Probability', 'Risk Level'] + list(df_raw.columns[:-2])].style.applymap(
                    color_risk, subset=['Risk Level']
                )
                
            st.dataframe(styled_df)
            
            # --- Feature Importance ---
            st.subheader("Model Feature Importance")
            st.markdown("""
            What drives churn? Check out the most important factors considered by our Random Forest model.
            """)
            
            importances = model.feature_importances_
            feat_imp_df = pd.DataFrame({
                'Feature': expected_features,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feat_imp_df.head(10), ax=ax, palette='viridis')
            ax.set_title('Top 10 Drivers of Churn')
            st.pyplot(fig)
            
            # --- Auto-Generated Feature Importance Insights ---
            top_features = feat_imp_df.head(3)['Feature'].tolist()
            st.info(f"**Insights:** The model indicates churn is primarily driven by: **{', '.join(top_features)}**.")
            
            # Check if demographic features are driving churn
            demographic_keywords = ['Age', 'Gender', 'Location']
            engagement_is_key = True
            for feat in top_features:
                if any(dem_key in feat for dem_key in demographic_keywords):
                    engagement_is_key = False
                    break     
            if engagement_is_key:
                st.caption("Note: Engagement behavior is more influential than demographics (e.g., age, gender, location).")
            
            # Download predicted results
            st.subheader("Download Predictions")
            csv = df_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
            )

        except Exception as e:
            st.error(f"Error processing the file: {e}")
    else:
        st.info("Please upload a CSV file from the sidebar to see predictions.")
