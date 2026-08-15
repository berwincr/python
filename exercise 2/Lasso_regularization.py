#Effect of lasso regularization

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score


dataset = pd.read_csv("house_regression_lab_dataset.csv")
X = dataset[["Area_sqft", "Bedrooms", "Age_years"]]
y = dataset["Price_lakhs"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
alpha_values = [0.01, 0.1, 1, 10, 100]

results = []

for alpha in alpha_values:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)

    y_pred = lasso.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append([
        alpha,
        lasso.coef_[0],
        lasso.coef_[1],
        lasso.coef_[2],
        rmse,
        r2
    ])

results_df = pd.DataFrame(results, columns=[
    "Alpha",
    "Area Coefficient",
    "Bedrooms Coefficient",
    "Age Coefficient",
    "RMSE",
    "R2 Score"
])

print(results_df)