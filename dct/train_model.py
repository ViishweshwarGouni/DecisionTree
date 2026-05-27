import pickle

from sklearn.datasets import load_iris

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# =========================
# Load Dataset
# =========================

iris = load_iris()

x = iris.data

y = iris.target

# =========================
# Train Test Split
# =========================

xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# =========================
# Base Model
# =========================

dt = DecisionTreeClassifier(
    random_state=42
)

# =========================
# Hyperparameter Tuning
# =========================

param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# =========================
# Train Model
# =========================

grid.fit(xtr, ytr)

best_model = grid.best_estimator_

print("\nBest Parameters:")
print(grid.best_params_)

# =========================
# Predictions
# =========================

ypred = best_model.predict(xte)

# =========================
# Metrics
# =========================

accuracy = accuracy_score(yte, ypred)

precision = precision_score(
    yte,
    ypred,
    average='weighted'
)

recall = recall_score(
    yte,
    ypred,
    average='weighted'
)

f1 = f1_score(
    yte,
    ypred,
    average='weighted'
)

cm = confusion_matrix(yte, ypred)

print("\n========== METRICS ==========")

print(f"\nAccuracy : {accuracy:.4f}")

print(f"Precision : {precision:.4f}")

print(f"Recall : {recall:.4f}")

print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(yte, ypred))

# =========================
# Save Metrics
# =========================

metrics = {
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1 Score": f1
}

pickle.dump(
    metrics,
    open("models/metrics.pkl", "wb")
)

# =========================
# Save Model
# =========================

pickle.dump(
    best_model,
    open("models/dt_classifier.pkl", "wb")
)

print("\nModel Saved Successfully!")