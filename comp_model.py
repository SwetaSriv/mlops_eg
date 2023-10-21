# Import necessary libraries
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the MNIST dataset
mnist = datasets.load_digits()
X = mnist.data
y = mnist.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning for the SVM model
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
}
svm_grid_search = GridSearchCV(SVC(), svm_params, cv=5)
svm_grid_search.fit(X_train, y_train)
svm_model = svm_grid_search.best_estimator_

# Hyperparameter tuning for the Decision Tree model
dt_params = {
    'max_depth': [None, 10, 20, 30],
}
dt_grid_search = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=5)
dt_grid_search.fit(X_train, y_train)
dt_model = dt_grid_search.best_estimator_

# Make predictions with both models
svm_predictions = svm_model.predict(X_test)
dt_predictions = dt_model.predict(X_test)

# Calculate accuracy for both models
svm_accuracy = accuracy_score(y_test, svm_predictions)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Calculate the confusion matrix (10x10) for both models
svm_confusion_matrix = confusion_matrix(y_test, svm_predictions)
dt_confusion_matrix = confusion_matrix(y_test, dt_predictions)

# Calculate the 2x2 confusion matrix for the specific question
def compare_predictions(y_true, y_production, y_candidate):
    production_correct = np.logical_and(y_true == y_production, y_true != y_candidate)
    candidate_correct = np.logical_and(y_true != y_production, y_true == y_candidate)
    return np.array([
        [np.sum(production_correct), np.sum(candidate_correct)],
        [0, 0]
    ])

compare_matrix = compare_predictions(y_test, svm_predictions, dt_predictions)

# [Bonus] Calculate macro-average F1 scores
svm_f1 = f1_score(y_test, svm_predictions, average='macro')
dt_f1 = f1_score(y_test, dt_predictions, average='macro')

# Display the results
print(f"Production Model (SVM) Accuracy: {svm_accuracy:.4f}")
print(f"Candidate Model (Decision Tree) Accuracy: {dt_accuracy:.4f}")
print("Confusion Matrix for SVM:")
print(svm_confusion_matrix)
print("Confusion Matrix for Decision Tree:")
print(dt_confusion_matrix)
print("2x2 Comparison Matrix:")
print(compare_matrix)
print(f"[Bonus] Macro-Average F1 for SVM: {svm_f1:.4f}")
print(f"[Bonus] Macro-Average F1 for Decision Tree: {dt_f1:.4f}")
