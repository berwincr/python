#Data loading and preparation

import pandas as pd
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("house_regression_lab_dataset.csv")

print("First Five Rows")
print(dataset.head())

print("\nLast Five Rows")
print(dataset.tail())

print("\nNumber of Rows and Columns")
print(dataset.shape)

print("\nColumn Names")
print(dataset.columns)

print("\nData Types")
print(dataset.dtypes)

print("\nMissing Values")
print(dataset.isnull().sum())

X = dataset[["Area_sqft", "Bedrooms", "Age_years"]]
y = dataset["Price_lakhs"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Set Shape")
print(X_train.shape)
print(y_train.shape)

print("\nTesting Set Shape")
print(X_test.shape)
print(y_test.shape)