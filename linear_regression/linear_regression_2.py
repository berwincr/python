import numpy as np

X = np.array([[1,1],
              [1,2],
              [1,3],
              [1,4],
              [1,5]])

Y = np.array([[2],
              [4],
              [5],
              [4],
              [5]])



x_transpose = X.T
x_transpose_x = np.matmul(x_transpose, X)
x_tranpose_x_inverse  = np.linalg.inv(x_transpose_x)
x_transpose_y = np.matmul(x_transpose, Y)
a = np.matmul(x_tranpose_x_inverse, x_transpose_y)

print("Coefficients:")
print(a)