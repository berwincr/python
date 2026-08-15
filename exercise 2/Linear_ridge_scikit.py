#Calculation of linear and ridge regression using scikit learn

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge

X = np.array([[1],
              [2],
              [3],
              [4]])

y = np.array([[1],
              [3],
              [4],
              [8]])

lr = LinearRegression()
lr.fit(X, y)

print("Linear Regression")
print("Intercept:", lr.intercept_[0])
print("Coefficient:", lr.coef_[0])

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

print("\nRidge Regression")
print("Intercept:", ridge.intercept_[0])
print("Coefficient:", ridge.coef_[0])