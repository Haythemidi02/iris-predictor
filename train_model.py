"""
train_model.py - Train and save the Iris classification model

Run this script FIRST to create the model files that the Streamlit app will use.
Command: python train_model.py
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import json
from pathlib import Path

print("=" * 60)
print("TRAINING IRIS CLASSIFICATION MODEL")
print("=" * 60)

# Set random seed for reproducibility
np.random.seed(42)

# Load the Iris dataset
print("\n1. Loading Iris dataset...")
iris = load_iris()

# Prepare features (X) and target (y)
X = iris.data
y = iris.target

print(f"   Dataset shape: {X.shape}")
print(f"   Number of samples: {X.shape[0]}")
print(f"   Number of features: {X.shape[1]}")
print(f"   Classes: {iris.target_names}")

# Feature scaling (improved)
print("\n2. Scaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, 'scaler.pkl')
print("   ✓ Scaler saved for later use")

# Split data with stratification
print("\n3. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Train improved Random Forest with better hyperparameters
print("\n4. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=200,      # Increased from 100
    max_depth=5,           # Increased from 3 for better fit
    min_samples_split=5,   # Prevent overfitting (new)
    min_samples_leaf=2,    # Prevent overfitting (new)
    random_state=42,
    n_jobs=-1              # Use all CPU cores (new)
)

model.fit(X_train, y_train)
print("   ✓ Model training completed!")

# Cross-validation for better evaluation
print("\n5. Performing cross-validation...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"   CV Mean Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

# Evaluate on test set
print("\n6. Evaluating model performance on test set...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"   Test Accuracy: {accuracy * 100:.2f}%")
print("\n   Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\n   Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display feature importance
print("\n7. Feature Importance:")
feature_importance_dict = {}
for feature, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"   {feature:20s}: {importance:.4f}")
    feature_importance_dict[feature] = float(importance)

# Save the trained model and scaler
print("\n8. Saving model files...")
Path('models').mkdir(exist_ok=True)

model_filename = 'models/iris_model.pkl'
joblib.dump(model, model_filename)
print(f"   ✓ Model saved as: {model_filename}")

# Save comprehensive model metadata
model_info = {
    'feature_names': iris.feature_names.tolist(),
    'target_names': iris.target_names.tolist(),
    'accuracy': float(accuracy),
    'cv_mean_accuracy': float(cv_scores.mean()),
    'cv_std_accuracy': float(cv_scores.std()),
    'feature_importance': feature_importance_dict,
    'n_features': X.shape[1],
    'n_samples': X.shape[0],
    'test_samples': X_test.shape[0],
    'train_samples': X_train.shape[0]
}

info_filename = 'models/model_info.json'
with open(info_filename, 'w') as f:
    json.dump(model_info, f, indent=4)
print(f"   ✓ Model info saved as: {info_filename}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print(f"\nModel Performance Summary:")
print(f"  • Test Accuracy: {accuracy * 100:.2f}%")
print(f"  • Cross-Val Mean: {cv_scores.mean() * 100:.2f}%")
print(f"  • Training Samples: {X_train.shape[0]}")
print(f"\nNext step: Run the Streamlit app with:")
print("  streamlit run app.py")
print("=" * 60)
