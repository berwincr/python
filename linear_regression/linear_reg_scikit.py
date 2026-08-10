from sklearn.linear_model import LinearRegression
import numpy as np

X = np.array([[1, 2],
              [2, 1],
              [3, 4],
              [4, 3],
              [5, 5]])


Y = np.array([5, 6, 9, 10, 13])


model = LinearRegression()
model.fit(X, Y)


print("Intercept (a0):", model.intercept_)
print("a1 and a2:", model.coef_)


pred = model.predict([[6, 7]])
print("Prediction:", pred)