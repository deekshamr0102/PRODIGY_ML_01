# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

print("Train dataset shape:", train.shape)
print("Test dataset shape:", test.shape)

# Select useful features
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "GarageCars",
    "OverallQual",
    "YearBuilt"
]

# Training data
X = train[features]
y = train["SalePrice"]

# Test data
X_test = test[features]

# Handle missing values
X = X.fillna(X.mean())
X_test = X_test.fillna(X_test.mean())

# Create machine learning model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X, y)

print("Model trained successfully")

# Predict house prices
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

# Save predictions
submission.to_csv("submission.csv", index=False)

print("submission.csv file created successfully!")

# -------------------------------
# GRAPHS SECTION
# -------------------------------

# 1️⃣ Scatter Plot: Living Area vs Sale Price
plt.figure()
plt.scatter(train["GrLivArea"], train["SalePrice"])
plt.xlabel("Living Area (Square Feet)")
plt.ylabel("Sale Price")
plt.title("House Price vs Living Area")
plt.show()

# 2️⃣ Histogram: Price Distribution
plt.figure()
plt.hist(train["SalePrice"], bins=30)
plt.xlabel("Sale Price")
plt.ylabel("Number of Houses")
plt.title("Distribution of House Prices")
plt.show()

# 3️⃣ Box Plot: Bedrooms vs Price
plt.figure()
train.boxplot(column="SalePrice", by="BedroomAbvGr")
plt.xlabel("Number of Bedrooms")
plt.ylabel("Sale Price")
plt.title("Bedrooms vs House Price")
plt.suptitle("")
plt.show()

# 4️⃣ Heatmap: Feature Correlation
numeric_train = train.select_dtypes(include=['number'])

plt.figure(figsize=(12,8))
sns.heatmap(numeric_train.corr(), cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()
