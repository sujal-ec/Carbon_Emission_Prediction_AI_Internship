# Carbon Emission Prediction
# Submitted by: Sujal Raina
# Roll No: 231101023
# College: GCET Jammu
# AICTE Internship - June 2025

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("data.csv")  # Ensure the CSV is in the same directory
print(df.head())

# Basic info and null value check
print(df.info())
print(df.isnull().sum())

# Describe the data
print(df.describe())

# Split dataset into features and target
X = df.drop('Carbon Emission', axis=1)
y = df['Carbon Emission']

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

# Visualization of Actual vs Predicted
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Carbon Emission")
plt.ylabel("Predicted Carbon Emission")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
