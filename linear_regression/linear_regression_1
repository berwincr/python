import numpy as np


X = np.array([1, 2, 3, 4, 5])
Y = np.array([2, 4, 5, 4, 5])


x_bar = np.mean(X)
y_bar = np.mean(Y)
xy_bar = np.mean(X * Y)
x2_bar = np.mean(X ** 2)


a1 = (xy_bar - x_bar * y_bar) / (x2_bar - x_bar**2)
a0 = y_bar - (a1 * x_bar)

print("a0 =", a0)
print("a1 =", a1)
