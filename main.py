#%% 🚀 Imports and Init
import pandas as pd
import numpy as np
import wandb
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from patsy import dmatrix
from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt
import seaborn as sns

#%% ⚙️ Initialize W&B
wandb.init(project="CO2-Emissions-Regression", name="Poly_vs_Spline", job_type="training")

#%% 📥 Load Data
df = pd.read_csv("data/CO2_Emissions_cleaned.csv")

#%% 🎯 Features and Target
X = df[["Engine Size(L)", "Cylinders", "Fuel Consumption Comb (L/100 km)"]]
y = df["CO2 Emissions(g/km)"]

#%% 🧪 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% 📏 Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% 🧠 Polynomial Regression Pipeline
best_poly_score = float('inf')
best_poly_model = None
best_poly_degree = None

for degree in [2, 3, 4]:
    wandb.run.log_code = True
    poly_pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("regressor", LinearRegression())
    ])
    poly_pipeline.fit(X_train_scaled, y_train)
    preds = poly_pipeline.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    wandb.log({
        f"Poly_deg_{degree}/RMSE": rmse,
        f"Poly_deg_{degree}/MAE": mae,
        f"Poly_deg_{degree}/R2": r2
    })

    if rmse < best_poly_score:
        best_poly_score = rmse
        best_poly_model = poly_pipeline
        best_poly_degree = degree

#%% 🧩 Spline Regression using dmatrix
def spline_features(X, df_spline=3):
    spline_matrix = dmatrix(f"bs(x, df={df_spline}, degree=3, include_intercept=False)",
                            {"x": X["Fuel Consumption Comb (L/100 km)"]}, return_type='dataframe')
    return spline_matrix

X_train_spline = spline_features(X_train)
X_test_spline = spline_features(X_test)

# Add remaining features
X_train_spline_full = pd.concat([X_train_spline, X_train.drop(columns="Fuel Consumption Comb (L/100 km)").reset_index(drop=True)], axis=1)
X_test_spline_full = pd.concat([X_test_spline, X_test.drop(columns="Fuel Consumption Comb (L/100 km)").reset_index(drop=True)], axis=1)

# Scale
X_train_spline_scaled = scaler.fit_transform(X_train_spline_full)
X_test_spline_scaled = scaler.transform(X_test_spline_full)

#%% 🧩 Spline Regression using dmatrix (FIXED)
def spline_features(X, df_spline=3):
    spline_matrix = dmatrix(
        f"bs(x, df={df_spline}, degree=3, include_intercept=False)",
        {"x": X["Fuel Consumption Comb (L/100 km)"]},
        return_type='dataframe'
    )
    return spline_matrix

# Create spline matrices
X_train_spline = spline_features(X_train).reset_index(drop=True)
X_test_spline = spline_features(X_test).reset_index(drop=True)

# Remove any NaNs just in case
X_train_spline = X_train_spline.dropna()
X_test_spline = X_test_spline.dropna()
y_train_reset = y_train.reset_index(drop=True).iloc[:len(X_train_spline)]
y_test_reset = y_test.reset_index(drop=True).iloc[:len(X_test_spline)]

# Add remaining features
X_train_spline_full = pd.concat([
    X_train_spline,
    X_train.drop(columns="Fuel Consumption Comb (L/100 km)").reset_index(drop=True).iloc[:len(X_train_spline)]
], axis=1)

X_test_spline_full = pd.concat([
    X_test_spline,
    X_test.drop(columns="Fuel Consumption Comb (L/100 km)").reset_index(drop=True).iloc[:len(X_test_spline)]
], axis=1)

# Scale
X_train_spline_scaled = scaler.fit_transform(X_train_spline_full)
X_test_spline_scaled = scaler.transform(X_test_spline_full)

#%% 🔀 Train Spline Model
spline_model = Ridge(alpha=1.0)
spline_model.fit(X_train_spline_scaled, y_train_reset)
spline_preds = spline_model.predict(X_test_spline_scaled)

spline_rmse = np.sqrt(mean_squared_error(y_test_reset, spline_preds))
spline_mae = mean_absolute_error(y_test_reset, spline_preds)
spline_r2 = r2_score(y_test_reset, spline_preds)

wandb.log({
    "Spline_RMSE": spline_rmse,
    "Spline_MAE": spline_mae,
    "Spline_R2": spline_r2
})

#%% 🏆 Compare and Save Best Model
if spline_rmse < best_poly_score:
    best_model = spline_model
    best_type = "Spline"
    X_sample = X_test_spline_scaled
else:
    best_model = best_poly_model
    best_type = f"Polynomial (degree={best_poly_degree})"
    X_sample = X_test_scaled

wandb.log({"Best_Model": best_type})
#%% 📢 Print Final Metrics
print(f"\n📊 Polynomial (degree={best_poly_degree})")
print(f"RMSE: {best_poly_score:.2f}")
print(f"MAE:  {mae:.2f}")
print(f"R²:   {r2:.4f}")

print("\n📊 Spline Regression")
print(f"RMSE: {spline_rmse:.2f}")
print(f"MAE:  {spline_mae:.2f}")
print(f"R²:   {spline_r2:.4f}")

#%% 📈 Visualization: Model Performance Comparison
labels = ['RMSE', 'MAE', 'R² Score']
poly_metrics = [best_poly_score, mae, r2]
spline_metrics = [spline_rmse, spline_mae, spline_r2]

x = np.arange(len(labels))
width = 0.35

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width/2, poly_metrics, width, label='Polynomial', color='skyblue')
bars2 = plt.bar(x + width/2, spline_metrics, width, label='Spline', color='salmon')

for bar in bars1 + bars2:
    height = bar.get_height()
    plt.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width()/2, height),
                 xytext=(0, 3), textcoords="offset points", ha='center', va='bottom')

plt.ylabel("Scores")
plt.title("📊 Model Performance: Polynomial vs. Spline Regression")
plt.xticks(x, labels)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.savefig("results/model_comparison_chart.png")
plt.show()

# Log chart to W&B
wandb.log({"Model_Comparison_Chart": wandb.Image("results/model_comparison_chart.png")})
#%% 📈 Visualize Polynomial Regression Fit per Degree

plt.figure(figsize=(14, 8))

degrees = [2, 3, 4]
colors = ['red', 'green', 'blue']
feature_idx = 0  # Let's visualize against "Engine Size(L)"

# Sort test data for clean curve plotting
sort_idx = X_test_scaled[:, feature_idx].argsort()
X_plot = X_test_scaled[sort_idx]
y_plot = y_test.values[sort_idx]

for i, degree in enumerate(degrees):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)[sort_idx]

    plt.plot(X_plot[:, feature_idx], preds, label=f'Degree {degree}', color=colors[i], linewidth=2)

# Plot actual test values
plt.scatter(X_plot[:, feature_idx], y_plot, label='Actual', color='black', alpha=0.3)

plt.title("Polynomial Regression Fit on Test Data")
plt.xlabel("Engine Size (scaled)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/polynomial_fit_comparison.png")
plt.show()

# Log to W&B
wandb.log({"Polynomial_Fit_Comparison": wandb.Image("results/polynomial_fit_comparison.png")})

#%% 🧾 Save Final Metrics as JSON

import json

# 🧠 Collect Metrics
metrics = {
    "Best_Model": best_type,
    f"Poly_deg_{best_poly_degree}_RMSE": round(best_poly_score, 4),
    f"Poly_deg_{best_poly_degree}_MAE": round(mae, 4),
    f"Poly_deg_{best_poly_degree}_R2": round(r2, 4),
    "Spline_RMSE": round(spline_rmse, 4),
    "Spline_MAE": round(spline_mae, 4),
    "Spline_R2": round(spline_r2, 4)
}

# 📁 Ensure directory exists
os.makedirs("results", exist_ok=True)

# 💾 Save locally
metrics_path = "results/metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

print("📊 metrics.json saved in /results")

# 🚀 Log to W&B as Artifact (✅ Windows Safe)
artifact = wandb.Artifact("metrics", type="metrics")
artifact.add_file(metrics_path)
wandb.log_artifact(artifact)

print("✅ metrics.json logged safely to W&B")

#%% 📈 Visualize Polynomial Regression Fit per Degree (Zoomed & Unscaled)

import matplotlib.pyplot as plt

# Use unscaled feature for X-axis (Engine Size)
X_feature_raw = X_test["Engine Size(L)"].values
sort_idx = X_feature_raw.argsort()
X_sorted = X_feature_raw[sort_idx]
y_sorted = y_test.values[sort_idx]

plt.figure(figsize=(14, 8))
degrees = [2, 3, 4]
colors = ['red', 'green', 'blue']

for i, degree in enumerate(degrees):
    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree)),
        ("regressor", LinearRegression())
    ])
    model.fit(X_train_scaled, y_train)

    preds = model.predict(X_test_scaled)[sort_idx]
    plt.plot(X_sorted, preds, label=f'Degree {degree}', color=colors[i], linewidth=2)

# Actual data points
plt.scatter(X_sorted, y_sorted, label='Actual Data', color='black', alpha=0.3)

# Zoom into realistic engine sizes
plt.xlim(1.0, 6.0)

plt.title("🔍 Polynomial Regression Fit vs Engine Size (Zoomed)")
plt.xlabel("Engine Size (L)")
plt.ylabel("CO2 Emissions (g/km)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save and log
plot_path = "results/polynomial_fit_comparison_zoomed.png"
plt.savefig(plot_path)
plt.show()

wandb.log({"Poly_Fit_Zoomed": wandb.Image(plot_path)})

#%% 💾 Save Best Model Safely
os.makedirs("models", exist_ok=True)
model_path = "models/best_model.pkl"

with open(model_path, "wb") as f:
    pickle.dump(best_model, f)

# Use wandb.Artifact instead of wandb.save to avoid symlink issue on Windows
artifact = wandb.Artifact("best_model", type="model")
artifact.add_file(model_path)
wandb.log_artifact(artifact)

print(f"✅ Best model saved and logged to W&B: {best_type}")
wandb.finish()

# %%
