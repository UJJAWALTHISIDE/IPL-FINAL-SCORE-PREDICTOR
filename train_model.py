"""
Cricket Score Prediction - Model Training Pipeline
================================================================================
This module implements a machine learning pipeline for predicting cricket match
final scores using the IPL dataset. The pipeline includes feature engineering,
data preprocessing, and gradient boosting regression.

Machine Learning Topics Covered:
  - Feature Engineering
  - Data Preprocessing (Scaling & Encoding)
  - Model Training (Gradient Boosting)
  - Model Evaluation & Validation
  - Model Serialization
================================================================================
"""

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score, mean_absolute_error
import joblib


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_features(df):
    """
    Engineer cricket-specific features for predictive modeling.
    
    Features Created:
    - current_run_rate: Runs scored per over
    - remaining_overs: Overs left in the match (20-over format)
    - wickets_in_hand: Available wickets (10 - lost)
    - is_death_overs: Binary flag for critical overs (16-20)
    - projected_score: Estimated final score at current run rate
    - pressure_factor: Combined effect of wickets and run rate
    
    Args:
        df (pd.DataFrame): Input dataframe with ['over', 'runs', 'wickets']
    
    Returns:
        pd.DataFrame: DataFrame with engineered features
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
# DATA PREPROCESSING
# ============================================================================

def build_preprocessor():
    """
    Build a preprocessing pipeline for feature transformation.
    
    Preprocessing Strategy:
    - Numeric features: StandardScaler (zero mean, unit variance)
    - Categorical features: OneHotEncoder (dummy variable encoding)
    
    Returns:
        ColumnTransformer: Fitted preprocessor object
    """
    numeric_features = [
        'over', 'runs', 'wickets', 'current_run_rate', 'remaining_overs',
        'wickets_in_hand', 'is_death_overs', 'projected_score', 'pressure_factor'
    ]
    
    categorical_features = ['venue', 'bat_team', 'bowl_team']
    
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])
    
    return preprocessor, numeric_features, categorical_features


# ============================================================================
# MODEL TRAINING & EVALUATION
# ============================================================================

def train_model(X_train, y_train):
    """
    Train a Gradient Boosting Regression model.
    
    Algorithm: Gradient Boosting
    Hyperparameters:
    - n_estimators: 50 boosting stages
    - max_depth: 3 (shallow trees to prevent overfitting)
    - learning_rate: 0.05 (slow learning for better generalization)
    
    Args:
        X_train: Training features
        y_train: Training target (final score)
    
    Returns:
        GradientBoostingRegressor: Trained model
    """
    model = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.05,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test set.
    
    Metrics:
    - R² Score: Proportion of variance explained (0-1)
    - MAE: Mean Absolute Error in runs
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
    
    Returns:
        tuple: (r2_score, mean_absolute_error)
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Execute the complete ML pipeline:
    1. Data Loading
    2. Feature Engineering
    3. Data Preprocessing
    4. Train-Test Split
    5. Model Training
    6. Model Evaluation
    7. Model Serialization
    """
    
    # 1. DATA LOADING
    print("✅ Loading dataset...")
    df = pd.read_csv('data/ipl.csv')
    
    # 2. FEATURE ENGINEERING
    print("⚙ Adding cricket features...")
    df = add_features(df)
    
    # 3. DATA PREPROCESSING
    print("⚙ Building preprocessor...")
    preprocessor, numeric_features, categorical_features = build_preprocessor()
    
    # Select all features needed for modeling
    all_features = numeric_features + categorical_features
    target = 'total'
    
    X = df[all_features]
    y = df[target]
    
    # 4. TRAIN-TEST SPLIT
    print("📊 Splitting data (90% train, 10% test)...")
    X_transformed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=0.1, random_state=42
    )
    
    # 5. MODEL TRAINING
    print("🔍 Training Gradient Boosting model...")
    model = train_model(X_train, y_train)
    
    # 6. MODEL EVALUATION
    print("📈 Evaluating model performance...")
    r2, mae = evaluate_model(model, X_test, y_test)
    print(f"📊 Model Performance → R²: {r2:.4f} | MAE: {mae:.2f} runs")
    
    # 7. MODEL SERIALIZATION
    print("💾 Saving model artifacts...")
    joblib.dump(model, 'data/best_model.pkl')
    joblib.dump(preprocessor, 'data/preprocessor.pkl')
    print("🎉 Training Complete! Models saved successfully.")


if __name__ == "__main__":
    main()
