"""
Cricket Score Prediction - Interactive Web Application
================================================================================
This module provides an interactive Streamlit web application for
predicting cricket match final scores in real-time based on mid-innings data.

Features:
  - Live score prediction
  - Upcoming overs forecasting
  - Interactive data visualizations
  - Direct integration with ML models

Machine Learning Topics Covered (Inference):
  - Model Loading & Caching
  - Feature Transformation
  - Prediction Pipeline
  - Data Visualization & Analytics
================================================================================
"""

import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# UI/UX STYLING & THEME CONFIGURATION
# ============================================================================

def apply_custom_styling():
    """Apply custom CSS for enhanced UI/UX design."""
    with open('styles.css', 'r') as css_file:
        css_content = css_file.read()
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ============================================================================
# MODEL & DATA LOADING (WITH CACHING)
# ============================================================================

@st.cache_resource
def load_model_and_preprocessor():
    """
    Load pre-trained model and preprocessor from disk.
    
    Caching Strategy: @st.cache_resource
    Resources persist across reruns, reducing I/O operations.
    
    Returns:
        tuple: (trained_model, preprocessor_pipeline)
    """
    model = joblib.load('data/best_model.pkl')
    preprocessor = joblib.load('data/preprocessor.pkl')
    return model, preprocessor


@st.cache_data
def load_data(path):
    """
    Load cricket dataset from CSV file.
    
    Caching Strategy: @st.cache_data
    Data remains consistent across reruns unless file changes.
    
    Args:
        path (str): Path to CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    return pd.read_csv(path)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(df):
    """
    Engineer cricket-specific features for predictions.
    
    Features:
    - current_run_rate: Runs per over
    - remaining_overs: Overs left (0-20)
    - wickets_in_hand: Available wickets
    - is_death_overs: Binary indicator for critical overs (16-20)
    - projected_score: Estimated final score
    - pressure_factor: Combined wickets & run rate metric
    
    Args:
        df (pd.DataFrame): Input with columns ['runs', 'over', 'wickets']
    
    Returns:
        pd.DataFrame: Enhanced with engineered features
    """
    df['current_run_rate'] = df.apply(
        lambda row: row['runs'] / row['over'] if row['over'] > 0 else 0, 
        axis=1
    )
    df['remaining_overs'] = 20 - df['over']
    df['wickets_in_hand'] = 10 - df['wickets']
    df['is_death_overs'] = (df['over'] >= 16).astype(int)
    df['projected_score'] = df['current_run_rate'] * 20
    df['pressure_factor'] = df['wickets_in_hand'] * df['current_run_rate']
    return df


# ============================================================================
# PREDICTION MODULE
# ============================================================================

def predict_final_score(user_input, model, preprocessor):
    """
    Predict final score with confidence intervals.
    
    Prediction Strategy:
    - Uses staged predictions from gradient boosting
    - Computes min, mean, max across boosting stages
    - Provides range-based prediction
    
    Args:
        user_input (pd.DataFrame): User-provided match data
        model: Trained Gradient Boosting model
        preprocessor: Feature transformer
    
    Returns:
        tuple: (min_pred, mean_pred, max_pred)
    
    Raises:
        Exception: If transformation or prediction fails
    """
    try:
        transformed = preprocessor.transform(user_input)
        staged_preds = list(model.staged_predict(transformed))
        staged_preds = np.array([p[0] for p in staged_preds])
        
        min_pred = int(round(np.min(staged_preds)))
        mean_pred = int(round(np.mean(staged_preds)))
        max_pred = int(round(np.max(staged_preds)))
        
        return min_pred, mean_pred, max_pred
    except Exception as e:
        raise Exception(f"Prediction failed: {e}")


def predict_upcoming_overs(user_input, current_over, model, preprocessor, num_overs=5):
    """
    Forecast final score for upcoming overs.
    
    Methodology:
    - Iterate through next N overs
    - Apply feature engineering for each state
    - Aggregate predictions
    
    Args:
        user_input (pd.DataFrame): Current match state
        current_over (int): Current over number
        model: Trained model
        preprocessor: Feature transformer
        num_overs (int): Future overs to predict
    
    Returns:
        list: Predicted scores for upcoming overs
    
    Raises:
        Exception: If prediction fails
    """
    try:
        upcoming = []
        for i in range(1, num_overs + 1):
            next_over = current_over + i
            if next_over > 20:
                break
            
            temp = user_input.copy()
            temp['over'] = next_over
            temp = add_features(temp)
            temp_transformed = preprocessor.transform(temp)
            pred = model.predict(temp_transformed)[0]
            upcoming.append(pred)
        
        return upcoming
    except Exception as e:
        raise Exception(f"Upcoming overs prediction failed: {e}")



# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """
    Execute the interactive prediction application.
    
    Pipeline:
    1. Apply styling
    2. Load resources
    3. Render sidebar controls
    4. Build user input
    5. Display match situation
    6. Process predictions
    7. Show visualizations
    """
    
    # 1. STYLING
    apply_custom_styling()
    st.title("🏏 IPL Final Score Predictor 🚀")
    
    # 2. LOAD RESOURCES
    dataset = load_data('data/ipl.csv')
    model, preprocessor = load_model_and_preprocessor()

    # ========================================================================
    # USER INPUT FORM (SIDEBAR)
    # ========================================================================
    st.sidebar.header('📝 Match Input')
    
    over = st.sidebar.number_input('Overs Played', 1, 19, 15, step=1)
    runs = st.sidebar.number_input('Current Runs', 0, 300, 130)
    wickets = st.sidebar.number_input('Wickets Lost', 0, 10, 3)
    venue = st.sidebar.selectbox('Venue', sorted(dataset['venue'].unique()))
    bat_team = st.sidebar.selectbox('Batting Team', sorted(dataset['bat_team'].unique()))
    bowl_team = st.sidebar.selectbox('Bowling Team', sorted(dataset['bowl_team'].unique()))

    # ========================================================================
    # BUILD INPUT DATAFRAME
    # ========================================================================
    user_input = pd.DataFrame({
        'over': [over],
        'runs': [runs],
        'wickets': [wickets],
        'venue': [venue],
        'bat_team': [bat_team],
        'bowl_team': [bowl_team]
    })
    User_input = add_features(user_input)

    # ========================================================================
    # DISPLAY MATCH SITUATION
    # ========================================================================
    st.write("### 📋 Match Situation")
    st.markdown('<div class="match-table">', unsafe_allow_html=True)
    st.table(user_input[['over', 'runs', 'wickets', 'venue', 'bat_team', 'bowl_team']])
    st.markdown('</div>', unsafe_allow_html=True)

    # ========================================================================
    # SECTION 1: FINAL SCORE PREDICTION
    # ========================================================================
    if st.button('🔮 Predict Final Score (Min / Max / Avg)'):
        try:
            min_pred, mean_pred, max_pred = predict_final_score(
                User_input, model, preprocessor
            )
            
            st.success(f"🏏 **Average Predicted Final Score:** {mean_pred}")
            st.info(f"🔻 Likely Minimum Score: {min_pred}")
            st.info(f"🔺 Likely Maximum Score: {max_pred}")
        except Exception as e:
            st.error(f"Prediction error: {e}")

    # ========================================================================
    # SECTION 2: UPCOMING OVERS PREDICTION
    # ========================================================================
    if st.button('📈 Show Upcoming Overs Prediction'):
        try:
            upcoming = predict_upcoming_overs(
                User_input, over, model, preprocessor, num_overs=5
            )
            
            if upcoming:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.bar(
                    [f"Over {round(over+i, 1)}" for i in range(1, len(upcoming)+1)],
                    upcoming,
                    color='#14299c',
                    edgecolor='#3b5998',
                    linewidth=2
                )
                ax.set_ylabel('Predicted Total Score', fontsize=12, fontweight='bold')
                ax.set_title('Upcoming Overs Score Progression', fontsize=14, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Graph error: {e}")

    # ========================================================================
    # SECTION 3: DATASET EXPLORATION
    # ========================================================================
    if st.button('📊 View Dataset Sample'):
        st.write("**Dataset Preview** (100 random samples)")
        st.dataframe(dataset.sample(100), use_container_width=True)


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
