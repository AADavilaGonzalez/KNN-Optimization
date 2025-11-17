import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# Note: We are using X and y from Part 1 (the *full* dataset)
# RandomizedSearchCV will handle its own train/test splits via cross-validation.

# 1. Create the base model
# We just create a blank model with no parameters.
knn_base = KNeighborsClassifier()

# 2. Define the "Parameter Distributions" (The Search Space)
# This is the "map" we are giving to Random Search.
# It will pick random values from these distributions.
param_distributions = {
    # Pick a random integer between 1 and 30 (inclusive)
    'n_neighbors': randint(1, 31),

    # Pick randomly from this list
    'weights': ['uniform', 'distance'],

    # Pick randomly from this list (p=1 is Manhattan, p=2 is Euclidean)
    'p': [1, 2]
}

# 1. Load the dataset
# X contains the features, y contains the labels (0 or 1)
X, y = load_breast_cancer(return_X_y=True)

# 2. Split the data
# We hold back 30% of the data as a "test set" to measure performance.
# This is a simple (but less robust) way to estimate performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
# 3. Set up the Random Search
# This object wraps the model, the parameters, and the evaluation all in one.
random_search = RandomizedSearchCV(
    estimator=knn_base,               # The model to tune
    param_distributions=param_distributions, # The "map" to search
    n_iter=50,                        # The "budget": try 50 random combinations
    cv=10,                            # The "evaluator": use 10-fold CV for each combo
    n_jobs=-1,                        # Use all available CPU cores
 #   random_state=42                   # For reproducible results
)

print(f"\n--- Random Search Example ---")
print(f"Starting Random Search with a budget of 50 iterations...")

# 4. Run the search
# This one .fit() call does ALL the work:
# It will try 50 different (k, weights, p) combinations.
# For EACH combo, it will run a 10-fold CV (so 50 * 10 = 500 total models trained)
# It finds the combo with the highest average CV accuracy.
random_search.fit(X, y)

# 5. Show the results
print("\nSearch complete.")
print(f"Best mean cross-validation score: {random_search.best_score_ * 100:.2f}%")
print("Best parameters found:")
print(random_search.best_params_)

# You can now use this "best_estimator_" as your final, tuned model
final_model = random_search.best_estimator_
