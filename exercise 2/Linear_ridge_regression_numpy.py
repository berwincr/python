#Calculation of linear and ridge regression using numpy


import numpy as np

X = np.array([[1],
              [2],
              [3],
              [4]])

y = np.array([[1],
              [3],
              [4],
              [8]])

ones = np.ones((X.shape[0], 1))
X_new = np.hstack((ones, X))

beta_linear = np.linalg.inv(X_new.T @ X_new) @ X_new.T @ y

print("Linear Regression Coefficients")
print("------------------------------")
print("Intercept :", beta_linear[0, 0])
print("Slope     :", beta_linear[1, 0])


lambda_val = 1

I = np.eye(X_new.shape[1])
I[0, 0] = 0        

beta_ridge = np.linalg.inv(X_new.T @ X_new + lambda_val * I) @ X_new.T @ y

print("\nRidge Regression Coefficients")
print("-----------------------------")
print("Intercept :", beta_ridge[0, 0])
print("Slope     :", beta_ridge[1, 0])