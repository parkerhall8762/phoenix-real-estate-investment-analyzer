# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("data/phoenix_housing_raw.csv")

# Preview dataset
print(data.head())

# Show dataset info
print(data.info())

# Show summary statistics
print(data.describe())

# Plot distribution of house prices
plt.figure(figsize=(8,5))
plt.hist(data["median_house_value"], bins=50)
plt.xlabel("House Price")
plt.ylabel("Count")
plt.title("Distribution of House Prices")
plt.show()

# Scatter plot income vs house value
plt.figure(figsize=(8,5))
plt.scatter(data["median_income"], data["median_house_value"], alpha=0.2)
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Income vs House Price")
plt.show()
# Machine Learning Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Prepare features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Convert categorical column to numbers
X = pd.get_dummies(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("Model RMSE:", rmse)
# Random Forest Model
from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_predictions = rf_model.predict(X_test)

rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))

print("Random Forest RMSE:", rf_rmse)
import joblib

# Save the trained Random Forest model
joblib.dump(rf_model, "../models/random_forest_model.pkl")

print("Model saved to models/random_forest_model.pkl")
