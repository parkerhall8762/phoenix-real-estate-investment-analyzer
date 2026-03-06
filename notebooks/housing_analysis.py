from sklearn.datasets import fetch_california_housing
import pandas as pd

# Load housing dataset
housing = fetch_california_housing()

# Convert dataset to DataFrame
df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add housing price column
df["price"] = housing.target

# Show first rows
print(df.head())
