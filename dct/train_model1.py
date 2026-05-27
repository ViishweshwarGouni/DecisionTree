import pandas as pd
import pickle

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV
)

from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# =========================
# Load Dataset
# =========================

df = pd.read_csv(
    r"C:\Users\HARSHAVARDHAN\OneDrive\Desktop\TekWorks\phase2\25may\slr\Realestate.csv"
)

print(df.head())

print(df.columns)

# =========================
# Features & Target
# =========================

x = df.drop(
    columns=['No', 'Y house price of unit area']
)

y = df['Y house price of unit area']

print("\nFeature Count:")
print(x.shape[1])

# =========================
# Train Test Split
# =========================

xtr, xte, ytr, yte = train_test_split(
    x,
    y,
    test_size=0.3,
    random_state=42
)

# =========================
# Base Model
# =========================

dt = DecisionTreeRegressor(
    random_state=42
)

# =========================
# Hyperparameter Tuning
# =========================

param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='r2',
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

mae = mean_absolute_error(
    yte,
    ypred
)

mse = mean_squared_error(
    yte,
    ypred
)

rmse = mse ** 0.5

r2 = r2_score(
    yte,
    ypred
)

print("\n========== METRICS ==========")

print(f"\nMAE : {mae:.4f}")

print(f"MSE : {mse:.4f}")

print(f"RMSE : {rmse:.4f}")

print(f"R2 Score : {r2:.4f}")

# =========================
# Save Metrics
# =========================

metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2": r2
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
    open("models/dt_regressor.pkl", "wb")
)

print("\nModel Saved Successfully!")