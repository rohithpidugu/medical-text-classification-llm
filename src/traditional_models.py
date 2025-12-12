# src/traditional_models.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

# --- 1. Model Definition ---

def get_traditional_models():
    """Initializes the three required traditional classification models."""
    # Use class_weight='balanced' to handle the reported class imbalance
    return {
        'Logistic Regression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced', max_iter=1000),
        'SVM': SVC(kernel='linear', random_state=42, class_weight='balanced', probability=True),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    }

# --- 2. Model Evaluation and Reporting ---

def evaluate_model_performance(model, X_test_vec, y_test, model_name):
    """Calculates and prints core metrics and returns results."""
    
    # Predict on the test set
    y_pred = model.predict(X_test_vec)
    
    # Calculate required metrics: Accuracy and Weighted F1-score
    accuracy = accuracy_score(y_test, y_pred)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"\n--- Results for {model_name} ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted F1-score: {f1_weighted:.4f}")
    
    # Full classification report (for detailed visualization/appendix)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Confusion Matrix (essential for the visual component of the report)
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'Model': model_name, 
        'Accuracy': accuracy, 
        'F1': f1_weighted, 
        'Confusion Matrix': cm
    }

# --- 3. Training Function ---

def train_and_evaluate_baselines(X_train_vec, X_test_vec, y_train, y_test):
    """Trains all models and collects results."""
    models = get_traditional_models()
    results = []
    
    print("\n--- Starting Baseline Model Training ---")
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train the model
        model.fit(X_train_vec, y_train)
        
        # Evaluate and store results
        res = evaluate_model_performance(model, X_test_vec, y_test, name)
        results.append(res)
        
    return pd.DataFrame(results)