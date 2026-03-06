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
# Split data into features and target
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = df.drop("price", axis=1)
y = df["price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
mae = mean_absolute_error(y_test, predictions)

print("\nModel Performance")
print("Mean Absolute Error:", mae)
# Feature importance
importance = pd.Series(model.coef_, index=X.columns)

print("\nFeature Importance:")
print(importance.sort_values(ascending=False))
# Example prediction for a new property
example_property = [[8.3, 41, 6.9, 1.1, 322, 2.5, 37.8, -122.2]]

predicted_price = model.predict(example_property)

print("\nPredicted house price for example property:", predicted_price[0])
