#Final model comparison

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
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

results = []

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)

results.append([
    "Linear Regression",
    mean_absolute_error(y_test, y_pred_lr),
    mean_squared_error(y_test, y_pred_lr),
    np.sqrt(mean_squared_error(y_test, y_pred_lr)),
    r2_score(y_test, y_pred_lr)
])

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

pr = LinearRegression()
pr.fit(X_train_poly, y_train)
y_pred_pr = pr.predict(X_test_poly)

results.append([
    "Polynomial Regression",
    mean_absolute_error(y_test, y_pred_pr),
    mean_squared_error(y_test, y_pred_pr),
    np.sqrt(mean_squared_error(y_test, y_pred_pr)),
    r2_score(y_test, y_pred_pr)
])

# Ridge Regression
ridge = Ridge(alpha=1)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)

results.append([
    "Ridge Regression",
    mean_absolute_error(y_test, y_pred_ridge),
    mean_squared_error(y_test, y_pred_ridge),
    np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
    r2_score(y_test, y_pred_ridge)
])

# Lasso Regression
lasso = Lasso(alpha=1, max_iter=10000)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)

results.append([
    "Lasso Regression",
    mean_absolute_error(y_test, y_pred_lasso),
    mean_squared_error(y_test, y_pred_lasso),
    np.sqrt(mean_squared_error(y_test, y_pred_lasso)),
    r2_score(y_test, y_pred_lasso)
])

# Elastic Net
elastic = ElasticNet(alpha=1, l1_ratio=0.5, max_iter=10000)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)

results.append([
    "Elastic Net",
    mean_absolute_error(y_test, y_pred_elastic),
    mean_squared_error(y_test, y_pred_elastic),
    np.sqrt(mean_squared_error(y_test, y_pred_elastic)),
    r2_score(y_test, y_pred_elastic)
])

comparison = pd.DataFrame(results, columns=[
    "Model",
    "MAE",
    "MSE",
    "RMSE",
    "R2"
])

print("Model Comparison")
print(comparison)

coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Linear": lr.coef_,
    "Ridge": ridge.coef_,
    "Lasso": lasso.coef_,
    "Elastic Net": elastic.coef_
})

print("\nCoefficient Comparison")
print(coef_table)