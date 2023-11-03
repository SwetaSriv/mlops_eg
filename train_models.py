import os
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

# Load the Digits dataset for demonstration
digits = datasets.load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=42)

# Train a simple Support Vector Machine (SVM) classifier
model = SVC()
model.fit(X_train, y_train)

# Save the trained model to the /app/models directory inside the container
model_path = '/app/models/digits_model.pkl'
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")

# Copy the trained model to the host machine's volume
os.system(f"cp {model_path} /host/models/digits_model.pkl")

# Keep the container running
while True:
    pass
