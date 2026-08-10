import numpy as np


X = np.array([[1,1, 2],
              [1,2, 1],
              [1,3, 4],
              [1,4, 3],
              [1,5, 5]])


Y = np.array([[5],
              [6],
              [9],
              [10],
              [13]])


a = np.linalg.inv(X.T @ X) @ X.T @ Y

print("Coefficients:")
print("a0 =", a[0][0])
print("a1 =", a[1][0])
print("a2 =", a[2][0])