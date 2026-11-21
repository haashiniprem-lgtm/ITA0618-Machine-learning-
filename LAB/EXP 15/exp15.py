
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

data = {
    'Area': [1000, 1200, 1500, 1800, 2000, 2200,
             2500, 2800, 3000, 1100, 1600, 2400],
    'Bedrooms': [2, 2, 3, 3, 4, 4, 4, 5, 5, 2, 3, 4],
    'Age': [10, 8, 5, 7, 4, 3, 2, 1, 2, 12, 6, 3],
    'Price': [2500000, 2800000, 3500000, 4000000,
              4500000, 4800000, 5500000, 6000000,
              6500000, 2600000, 3700000, 5300000]
}

df = pd.DataFrame(data)

X = df[['Area', 'Bedrooms', 'Age']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("House Price Prediction")
print("----------------------")
print("Mean Absolute Error:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

sample = pd.DataFrame(
    [[1800, 3, 5]],
    columns=X.columns
)

print("Sample Input: Area=1800, Bedrooms=3, Age=5")
print("Predicted House Price: Rs.", round(model.predict(sample)[0], 2))
