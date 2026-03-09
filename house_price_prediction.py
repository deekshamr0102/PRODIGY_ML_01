# Import libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load datasets
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Select important features
features = ["GrLivArea", "BedroomAbvGr", "FullBath"]

# Training data
X = train[features]
y = train["SalePrice"]

# Test data
X_test = test[features]

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict house prices
predictions = model.predict(X_test)

# Create submission file
submission = pd.DataFrame({
    "Id": test["Id"],
    "SalePrice": predictions
})

submission.to_csv("submission.csv", index=False)

print("Prediction file created successfully!")

numeric_train = train.select_dtypes(include=['number'])

corr = numeric_train.corr()

plt.figure(figsize=(12,8))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()