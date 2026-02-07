"""
train_model.py - Train and save the Iris classification model

Run this script FIRST to create the model files that the Streamlit app will use.
Command: python train_model.py
"""

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

print("=" * 60)
print("TRAINING IRIS CLASSIFICATION MODEL")
print("=" * 60)

# Set random seed for reproducibility
# This ensures we get the same results every time we run the script
np.random.seed(42)

# Load the Iris dataset
# This dataset contains measurements of 150 iris flowers from 3 species
print("\n1. Loading Iris dataset...")
iris = load_iris()

# Prepare features (X) and target (y)
X = iris.data  # Features: sepal length, sepal width, petal length, petal width
y = iris.target  # Target: species (0=setosa, 1=versicolor, 2=virginica)

print(f"   Dataset shape: {X.shape}")
print(f"   Number of samples: {X.shape[0]}")
print(f"   Number of features: {X.shape[1]}")
print(f"   Classes: {iris.target_names}")

# Split data into training (80%) and testing (20%) sets
# stratify=y ensures each split has the same proportion of each class
print("\n2. Splitting data into train/test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% for testing
    random_state=42,    # For reproducibility
    stratify=y          # Maintain class distribution
)

print(f"   Training samples: {X_train.shape[0]}")
print(f"   Testing samples: {X_test.shape[0]}")

# Initialize and train the Random Forest Classifier
# Random Forest builds multiple decision trees and combines their predictions
print("\n3. Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=100,   # Number of trees in the forest
    max_depth=3,        # Maximum depth of each tree
    random_state=42     # For reproducibility
)

# Train the model on the training data
model.fit(X_train, y_train)
print("   ✓ Model training completed!")

# Evaluate the model on test data
print("\n4. Evaluating model performance...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"   Accuracy: {accuracy * 100:.2f}%")
print("\n   Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Display feature importance
print("\n5. Feature Importance:")
for feature, importance in zip(iris.feature_names, model.feature_importances_):
    print(f"   {feature:20s}: {importance:.4f}")

# Save the trained model to disk
# This allows us to load it later in the Streamlit app without retraining
print("\n6. Saving model files...")
model_filename = 'iris_model.pkl'
joblib.dump(model, model_filename)
print(f"   ✓ Model saved as: {model_filename}")

# Save model metadata (feature names and class names)
# The Streamlit app will use this information
model_info = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names.tolist(),
    'accuracy': accuracy
}
info_filename = 'model_info.pkl'
joblib.dump(model_info, info_filename)
print(f"   ✓ Model info saved as: {info_filename}")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext step: Run the Streamlit app with:")
print("  streamlit run app.py")
print("=" * 60)
