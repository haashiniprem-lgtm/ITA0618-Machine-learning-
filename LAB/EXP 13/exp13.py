
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Year': [2015, 2016, 2017, 2018, 2019, 2020,
             2021, 2022, 2023, 2014, 2013, 2012],
    'Kilometers': [80000, 70000, 60000, 55000, 45000, 40000,
                   30000, 20000, 10000, 90000, 100000, 110000],
    'EngineCC': [1200, 1200, 1500, 1500, 1800, 1800,
                 2000, 2000, 2200, 1200, 1000, 1000],
    'Price': [350000, 400000, 500000, 550000, 650000, 700000,
              850000, 950000, 1100000, 300000, 250000, 200000]
}

df = pd.DataFrame(data)

X = df[['Year', 'Kilometers', 'EngineCC']]
y = df['Price']

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)

print("Car Price Prediction")
print("--------------------")
print("Model R2 Score:", round(model.score(X, y), 3))
print("Mean Squared Error:", round(mean_squared_error(y, predictions), 2))

sample = pd.DataFrame(
    [[2021, 25000, 1800]],
    columns=X.columns
)

predicted_price = model.predict(sample)[0]

print("\nCar Details: Year=2021, Kilometers=25000, EngineCC=1800")
print("Predicted Car Price: Rs.", round(predicted_price, 2))
