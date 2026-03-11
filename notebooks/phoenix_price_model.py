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
