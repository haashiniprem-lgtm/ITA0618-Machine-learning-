
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Sample dataset
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Linear Regression")
print("-----------------")
print("Slope:", model.coef_[0])
print("Intercept:", model.intercept_)

sample = np.array([[6]])
result = model.predict(sample)

print("Input:", 6)
print("Predicted Output:", result[0])
print("Mean Squared Error:", mean_squared_error(y, predictions))
