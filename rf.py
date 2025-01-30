import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def train_rf(data, save_path, use_smote=False):
    # Step 1: Split the data into features and target
    X = data.drop(columns=['action'])
    y = data['action']

    # Step 2: Split the dataset into train, test, and eval (60% train, 20% test, 20% eval)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_test, X_eval, y_test, y_eval = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    if use_smote:
        # Step 3: Apply SMOTE if specified
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        X_train_final, y_train_final = X_train_resampled, y_train_resampled
    else:
        # Step 3: Use class weights instead of SMOTE
        class_weights = compute_class_weight(class_weight='balanced', classes=y_train.unique(), y=y_train)
        class_weights_dict = dict(zip(y_train.unique(), class_weights))
        X_train_final, y_train_final = X_train, y_train

    # Step 4: Train Random Forest Classifier
    rf_model = RandomForestClassifier(random_state=42, class_weight=class_weights_dict if not use_smote else None)
    rf_model.fit(X_train_final, y_train_final)

    # Step 5: Model Evaluation on the Test Data
    y_pred_test = rf_model.predict(X_test)
    accuracy_test = accuracy_score(y_test, y_pred_test)
    print(f"Random Forest Model Accuracy on Test Data: {accuracy_test * 100:.2f}%")

    # Classification report on test data
    class_report_test = classification_report(y_test, y_pred_test, output_dict=True)
    print("Classification Report on Test Data:")
    print(classification_report(y_test, y_pred_test))

    # Confusion matrix for test data
    conf_matrix_test = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
    plt.title('Confusion Matrix on Test Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Step 6: Model Evaluation on the Eval Data (Final Evaluation)
    y_pred_eval = rf_model.predict(X_eval)
    accuracy_eval = accuracy_score(y_eval, y_pred_eval)
    print(f"Random Forest Model Accuracy on Evaluation Data: {accuracy_eval * 100:.2f}%")

    # Classification report on eval data
    class_report_eval = classification_report(y_eval, y_pred_eval, output_dict=True)
    print("Classification Report on Evaluation Data:")
    print(classification_report(y_eval, y_pred_eval))

    # Confusion matrix for eval data
    conf_matrix_eval = confusion_matrix(y_eval, y_pred_eval)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix_eval, annot=True, fmt='d', cmap='Blues', xticklabels=y.unique(), yticklabels=y.unique())
    plt.title('Confusion Matrix on Evaluation Data')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Save the model to the specified path
    joblib.dump(rf_model, save_path)
    print(f"Model saved to {save_path}")

    # Output a simple confirmation message
    print(f"Model trained using {'SMOTE' if use_smote else 'Class Weights'} with Random Forest Classifier.")
    
    # Return the classification report for further analysis if needed
    return class_report_test, class_report_eval

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a Random Forest model using SMOTE or Class Weights.")
    parser.add_argument("--use_smote", action="store_true", help="Use SMOTE to balance the dataset (default: use class weights).")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model.")

    args = parser.parse_args()
    
    # Load the dataset
    data = pd.read_csv(args.data_path)

    # Call the training function with the appropriate parameters
    train_rf(data, save_path=args.save_path, use_smote=args.use_smote)
