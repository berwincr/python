#Ridge regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


dataset = pd.read_csv("house_regression_lab_dataset.csv")
X = dataset[["Area_sqft", "Bedrooms", "Age_years"]]
y = dataset["Price_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)

ridge = Ridge(alpha=1.0)

ridge.fit(X_train, y_train)

print("Ridge Intercept")
print(ridge.intercept_)

print("\nRidge Coefficients")
for feature, coef in zip(X.columns, ridge.coef_):
    print(f"{feature} : {coef}")

y_pred = ridge.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

comparison = pd.DataFrame({
    "Feature": X.columns,
    "Linear Regression": model.coef_,
    "Ridge Regression": ridge.coef_
})

print("\nCoefficient Comparison")
print(comparison)