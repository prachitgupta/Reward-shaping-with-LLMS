import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def preprocess(data):
    # Split features and target
    X = data.drop(columns=['action'])
    y = data['action']
    # # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

def train_xgb_model(X_train, X_test, y_train, y_test, save_path):
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy * 100:.2f}%")

    joblib.dump(xgb_model, save_path)
    print(f"Model saved to {save_path}")

    return xgb_model

# Dataset generation function
# Random Forest training function
def train_rf_model(X_train, X_test, y_train, y_test,save_path):
    rf = RandomForestClassifier(random_state=42)

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    # Evaluate the model
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

    # Save the model to the specified path
    joblib.dump(rf_model, save_path)
    print(f"Model saved to {save_path}")

def train_lgbm_model(X_train, X_test, y_train, y_test,save_path):
    lgbm = LGBMClassifier(random_state=42)
    # Train the models
    lgbm_model = lgbm.fit(X_train, y_train)
    lgbm_pred = lgbm.predict(X_test)
    # Evaluate the models
    lgbm_acc = accuracy_score(y_test, lgbm_pred)
    print("LightGBM Accuracy:", lgbm_acc)
    print("\nLightGBM Classification Report:\n", classification_report(y_test, lgbm_pred))

    # Save the model to the specified path
    joblib.dump(lgbm_model, save_path)
    print(f"Model saved to {save_path}")

file_path = os.path.join("datasets", 'highway_dataset_claude_safe_efficient.csv')
# Read the dataset
data = pd.read_csv(file_path)

X_train, X_test, y_train, y_test = preprocess(data)

# Train Random Forest Model
save_path = 'models/rf_claude_try.pkl'
train_xgb_model(X_train, X_test, y_train, y_test, save_path)







