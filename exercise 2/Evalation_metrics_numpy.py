#Evaluation metrics 


import numpy as np

X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([3, 5, 7, 9, 11, 13]).reshape(-1, 1)


ones = np.ones((X.shape[0], 1))
X_new = np.hstack((ones, X))


beta = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y


y_pred = X_new @ beta

mse = np.mean((y - y_pred) ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(y - y_pred))


ss_total = np.sum((y - np.mean(y)) ** 2)
ss_res = np.sum((y - y_pred) ** 2)
r2 = 1 - (ss_res / ss_total)

print("Regression Coefficients")
print("-----------------------")
print("Intercept :", beta[0, 0])
print("Slope     :", beta[1, 0])

print("\nPredicted Values")
print(y_pred)

print("\nEvaluation Metrics")
print("------------------")
print("MSE      :", mse)
print("RMSE     :", rmse)
print("MAE      :", mae)
print("R2 Score :", r2)
