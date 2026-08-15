#Evaluation metrics

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13])

model = LinearRegression()
model.fit(X, y)


y_pred = model.predict(X)


mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("Actual Values     :", y)
print("Predicted Values  :", y_pred)

print("\nEvaluation Metrics")
print("------------------")
print("MSE  :", mse)
print("RMSE :", rmse)
print("MAE  :", mae)
print("R2 Score :", r2)