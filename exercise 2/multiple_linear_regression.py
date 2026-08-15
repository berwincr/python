#Multiple linear regression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

b0 = model.intercept_
b1, b2, b3 = model.coef_

print("Intercept (b0):", b0)
print("Coefficient b1 (Area):", b1)
print("Coefficient b2 (Bedrooms):", b2)
print("Coefficient b3 (Age):", b3)

print("\nRegression Equation")
print(f"Price = {b0:.4f} + ({b1:.4f} * Area_sqft) + ({b2:.4f} * Bedrooms) + ({b3:.4f} * Age_years)")

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nEvaluation Metrics")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred,
    "Error": y_test.values - y_pred
})

print("\nPrediction Results")
print(results)