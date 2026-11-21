
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = {
    'RAM': [2, 2, 3, 3, 4, 4, 6, 6, 8, 8, 12, 12],
    'Storage': [16, 32, 32, 64, 64, 128, 64, 128, 128, 256, 256, 512],
    'Battery': [2500, 3000, 2800, 3500, 3000, 4000,
                4000, 4500, 4500, 5000, 5000, 5500],
    'Camera': [5, 8, 8, 12, 12, 16, 16, 20, 24, 32, 48, 64],
    'PriceRange': [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
}

df = pd.DataFrame(data)

X = df.drop('PriceRange', axis=1)
y = df['PriceRange']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.34, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Mobile Price Prediction")
print("-----------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

sample = pd.DataFrame(
    [[8, 128, 4500, 48]],
    columns=X.columns
)

result = model.predict(sample)[0]

price_categories = {
    0: "Low Cost",
    1: "Medium Cost",
    2: "High Cost",
    3: "Very High Cost"
}

print("Sample Input: RAM=8GB, Storage=128GB, Battery=4500mAh, Camera=48MP")
print("Predicted Price Category:", price_categories[result])
