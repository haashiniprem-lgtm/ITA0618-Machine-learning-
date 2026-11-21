
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data = {
    'Income': [25000, 50000, 30000, 80000, 45000, 90000,
               20000, 70000, 60000, 35000, 85000, 40000],
    'Age': [22, 35, 25, 45, 30, 50, 21, 42, 38, 28, 48, 32],
    'LoanAmount': [15000, 20000, 10000, 30000, 18000, 25000,
                   12000, 22000, 20000, 14000, 28000, 16000],
    'CreditScore': [550, 750, 600, 800, 700, 820,
                    520, 780, 740, 580, 810, 680],
    'CreditClass': ['Low', 'High', 'Low', 'High', 'Medium', 'High',
                    'Low', 'High', 'High', 'Low', 'High', 'Medium']
}

df = pd.DataFrame(data)

X = df.drop('CreditClass', axis=1)
y = df['CreditClass']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Credit Score Classification")
print("---------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

sample = pd.DataFrame(
    [[45000, 30, 18000, 700]],
    columns=X.columns
)

print("Sample Input:", sample.values.tolist())
print("Predicted Credit Class:", model.predict(sample)[0])
