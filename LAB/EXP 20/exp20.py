
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Monthly sales dataset
months = np.array([
    [1], [2], [3], [4], [5], [6],
    [7], [8], [9], [10], [11], [12]
])

sales = np.array([
    100, 120, 130, 150, 170, 180,
    200, 210, 230, 250, 270, 290
])

# Train Linear Regression model
model = LinearRegression()
model.fit(months, sales)

predictions = model.predict(months)

print("Future Sales Prediction")
print("-----------------------")
print("R2 Score:", round(model.score(months, sales), 3))
print("Mean Squared Error:",
      round(mean_squared_error(sales, predictions), 2))

# Predict next three months
future_months = np.array([[13], [14], [15]])
future_sales = model.predict(future_months)

print("\nFuture Sales Forecast:")

for i in range(len(future_months)):
    print("Month", future_months[i][0],
          "Predicted Sales:", round(future_sales[i], 2))
