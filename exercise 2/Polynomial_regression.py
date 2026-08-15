#Polynomial resgression

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
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


poly = PolynomialFeatures(degree=2)

X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

feature_names = poly.get_feature_names_out(X.columns)

print("Polynomial Feature Names")
for feature in feature_names:
    print(feature)

model = LinearRegression()
model.fit(X_train_poly, y_train)

print("\nIntercept")
print(model.intercept_)

print("\nCoefficients")
for feature, coef in zip(feature_names, model.coef_):
    print(f"{feature} : {coef}")

y_pred_poly = model.predict(X_test_poly)

mae = mean_absolute_error(y_test, y_pred_poly)
mse = mean_squared_error(y_test, y_pred_poly)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_poly)

print("\nEvaluation Metrics")
print("MAE :", mae)
print("MSE :", mse)
print("RMSE :", rmse)
print("R2 Score :", r2)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred_poly,
    "Error": y_test.values - y_pred_poly
})

print("\nPrediction Results")
print(results)