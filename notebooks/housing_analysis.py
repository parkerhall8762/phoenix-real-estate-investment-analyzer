from sklearn.datasets import fetch_california_housing
import pandas as pd
import matplotlib.pyplot as plt

# Load housing dataset
housing = fetch_california_housing()

# Convert dataset to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add housing price column
df["price"] = housing.target

# Show first rows
print("\nFirst rows of dataset:")
print(df.head())

# Show dataset statistics
print("\nDataset summary statistics:")
print(df.describe())

# Create a histogram of house prices
plt.hist(df["price"], bins=50)
plt.title("Distribution of Housing Prices")
plt.xlabel("Price")
plt.ylabel("Number of Houses")

plt.show()
