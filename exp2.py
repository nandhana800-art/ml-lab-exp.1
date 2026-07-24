# California Housing Dataset
# ============================================

# --------------------------------------------
# Step 1: Import Libraries
# --------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------------------------
# Step 2: Load Dataset
# --------------------------------------------

df = pd.read_csv("housing.csv")

# Display first 5 rows
print("\nFIRST 5 ROWS OF DATASET\n")
print(df.head())

# --------------------------------------------
# Step 3: Select Features
# --------------------------------------------

# Predictor Variable
X = df['total_rooms'].values

# Target Variable
y = df['median_house_value'].values

# --------------------------------------------
# Step 4: Remove Missing Values
# --------------------------------------------

mask = ~np.isnan(X) & ~np.isnan(y)

X = X[mask]
y = y[mask]

# --------------------------------------------
# Step 5: Normalize Data
# --------------------------------------------

X_mean = np.mean(X)
X_std = np.std(X)

X = (X - X_mean) / X_std

# --------------------------------------------
# Step 6: Initialize Parameters
# --------------------------------------------

theta0 = 0
theta1 = 0

learning_rate = 0.01
iterations = 1000

n = len(X)

# Store cost values
cost_history = []

# --------------------------------------------
# Step 7: Gradient Descent
# --------------------------------------------

for i in range(iterations):

# Predicted values
    y_pred = theta0 + theta1 * X

# Error
error = y - y_pred

# Gradients
d_theta0 = (-2/n) * np.sum(error)

d_theta1 = (-2/n) * np.sum(error * X)

# Update parameters
theta0 = theta0 - learning_rate * d_theta0

theta1 = theta1 - learning_rate * d_theta1

# Cost function
cost = (1/n) * np.sum(error**2)

cost_history.append(cost)

# --------------------------------------------
# Step 8: Final Predictions
# --------------------------------------------

y_pred_final = theta0 + theta1 * X

# --------------------------------------------
# Step 9: Model Parameters
# --------------------------------------------

print("\nMODEL PARAMETERS\n")

print("Theta0 (Intercept):", theta0)

print("Theta1 (Slope):", theta1)

# --------------------------------------------
# Step 10: Evaluation Metrics
# --------------------------------------------

# Mean Squared Error
mse = np.mean((y - y_pred_final)**2)

# R² Score
SS_res = np.sum((y - y_pred_final)**2)

SS_tot = np.sum((y - np.mean(y))**2)

r2 = 1 - (SS_res / SS_tot)

print("\nEVALUATION METRICS\n")

print("Mean Squared Error (MSE):", round(mse, 4))

print("R² Score:", round(r2, 4))

# --------------------------------------------
# Step 11: Plot Regression Line
# --------------------------------------------

plt.figure(figsize=(10,6))

# Scatter plot
plt.scatter(X,
y,
color='blue',
alpha=0.5,
label='Actual Data')

# Regression line
plt.plot(X,
y_pred_final,
color='red',
linewidth=2,
label='Regression Line')

plt.title("Simple Linear Regression using Gradient Descent")

plt.xlabel("Normalized Total Rooms")

plt.ylabel("Median House Value")

plt.legend()

plt.grid(True)

plt.show()

# --------------------------------------------
# Step 12: Plot Cost Convergence
# --------------------------------------------
