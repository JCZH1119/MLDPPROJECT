import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Load your dataset
df = pd.read_csv("your_clean_data.csv")

# Use only these 10 features — match Streamlit order
feature_cols = [
    "latitude",  # even if dummy
    "longitude",  # even if dummy
    "closest_mrt_dist",
    "cbd_dist",
    "floor_area_sqm",
    "lease_commence_date",
    "year",
    "dummy_7",  # unused or always 0
    "remaining_lease_years",
    "storey_mid"
]

# Ensure dummy columns exist
if "dummy_7" not in df.columns:
    df["dummy_7"] = 0
if "latitude" not in df.columns:
    df["latitude"] = 0
if "longitude" not in df.columns:
    df["longitude"] = 0

X = df[feature_cols]
y = df["resale_price"]  # or your target column

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Model training
model = RandomForestRegressor(random_state=42, n_jobs=-1)

param_grid = {
    "n_estimators": [100],
    "max_depth": [12]
}

grid_search = GridSearchCV(
    model,
    param_grid,
    cv=KFold(n_splits=3, shuffle=True, random_state=42),
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate
y_pred = best_model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
print(f"R2: {r2_score(y_test, y_pred):.2f}")

# Save the model
joblib.dump(best_model, "random_forest_model.pkl", compress=("xz", 9))
print("Model saved as random_forest_model.pkl")
