#!/usr/bin/env python3

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Load the dataset
# X contains the features, y contains the labels (0 or 1)
X, y = load_breast_cancer(return_X_y=True)

# 2. Split the data
# We hold back 30% of the data as a "test set" to measure performance.
# This is a simple (but less robust) way to estimate performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# 3. Set parameters and create the model
# We are manually choosing the parameters for this one example.
# k = 5 neighbors
# weights = 'uniform' (every neighbor gets an equal vote)
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform')

# 4. Train the model
# The .fit() method is the "learning" step.
# For KNN, it's just memorizing the training data.
knn.fit(X_train, y_train)

# 5. Measure performance
# Use the trained model to .predict() the labels for the unseen test set.
y_pred = knn.predict(X_test)

# Compare the predictions (y_pred) to the true labels (y_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- Basic Example ---")
print(f"Data shape: {X.shape}")
print(f"Train data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")
print(f"\nModel: KNN with k=5, weights='uniform'")
print(f"Test Set Accuracy: {accuracy * 100:.2f}%")
