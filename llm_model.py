import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import seaborn as sns
import joblib

def preprocess(data):
    # Split features and target
    X = data.drop(columns=['action'])
    y = data['action']
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(sampling_strategy="auto",  
                  k_neighbors=5,  # Adjust the number of nearest neighbors
                  random_state=42)
    
  
    
    # # Standardize the features
    # scaler = StandardScaler()
    # X_res_scaled = scaler.fit_transform(X_res)
    
    # # Apply PCA
    # pca = PCA(n_components=0.95)  # Retain 95% of variance
    # X_res_pca = pca.fit_transform(X_res_scaled)
    
    # Split the resampled data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_train, X_test, y_train, y_test

def train_xgb_model(X_train, X_test, y_train, y_test, save_path):
#     param_grid = {
#     'n_estimators': [100, 500, 1000],  # Number of trees in the ensemble
#     'max_depth': [3, 6, 9, 12],  # Maximum depth of each tree
#     'learning_rate': [0.01, 0.1, 0.2],  # Step size shrinkage
#     'subsample': [0.5, 0.7, 1.0],  # Proportion of samples used for fitting individual trees
#     'colsample_bytree': [0.5, 0.7, 1.0],  # Proportion of features used for each tree
#     'gamma': [0, 0.1, 0.5, 1],  # Minimum loss reduction required to make a further partition
#     'reg_alpha': [0, 0.1, 1],  # L1 regularization term
#     'reg_lambda': [1, 10, 100]  # L2 regularization term
# }

#     # Initialize the XGBoost classifier
#     xgb = XGBClassifier(eval_metric='mlogloss')

#     # Apply GridSearchCV to search for the best hyperparameters
#     grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, 
#                             cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

#     # Fit the grid search
#     grid_search.fit(X_train, y_train)

#     # View the best hyperparameters
#     print("Best Hyperparameters:", grid_search.best_params_)
    
    xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, class_weight = "balanced")
    xgb_model.fit(X_train, y_train)

    y_pred = xgb_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"xgb Model Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_test.unique(), yticklabels=y_test.unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
    joblib.dump(xgb_model, save_path)
    print(f"Model saved to {save_path}")

    return xgb_model

# Dataset generation function
# Random Forest training function
def train_rf_model(X_train, X_test, y_train, y_test,save_path):

    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(
    bootstrap=False, 
    max_depth=30, 
    max_features='sqrt', 
    min_samples_leaf=1, 
    min_samples_split=5, 
    n_estimators=500,
    random_state=42, class_weight = "balanced"
)

    # # Define the hyperparameters and their values to be searched
    # param_grid = {
    #     'n_estimators': [100, 500, 1000],  # Number of trees in the forest
    #     'max_depth': [10, 20, 30, None],  # Maximum depth of each tree
    #     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split a node
    #     'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required in a leaf node
    #     'max_features': ['auto', 'sqrt'],  # Number of features to consider when looking for the best split
    #     'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
    # }

    # # Apply GridSearchCV to search for the best hyperparameters
    # grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
    #                         cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

    # # Fit the grid search
    # grid_search.fit(X_train, y_train)

    # # View the best hyperparameters
    # print("Best Hyperparameters:", grid_search.best_params_)

    #Predict using the best model
   
    rf_model.fit(X_train, y_train)

    # Step 5: Model Evaluation on Test Data
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=y_test.unique(), yticklabels=y_test.unique())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

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

file_path = os.path.join("datasets", 'claude_1k.csv')
# Read the dataset
data = pd.read_csv(file_path)

X_train, X_test, y_train, y_test = preprocess(data)

# Train Random Forest Model
save_path = 'models/rf.pkl'
train_rf_model(X_train, X_test, y_train, y_test, save_path)







