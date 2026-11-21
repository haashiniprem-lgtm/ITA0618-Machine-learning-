
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)

linear_predictions = linear_model.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

poly_predictions = poly_model.predict(X_poly)

# Compare models
print("Linear vs Polynomial Regression")
print("--------------------------------")

print("Linear Regression MSE:",
      round(mean_squared_error(y, linear_predictions), 2))

print("Polynomial Regression MSE:",
      round(mean_squared_error(y, poly_predictions), 2))

sample = np.array([[7]])

linear_result = linear_model.predict(sample)
poly_result = poly_model.predict(poly.transform(sample))

print("\nInput:", 7)
print("Linear Prediction:", round(linear_result[0], 2))
print("Polynomial Prediction:", round(poly_result[0], 2))
