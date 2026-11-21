
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = load_iris()

# Select two classes for binary classification
X = iris.data[iris.target != 2]
y = iris.target[iris.target != 2]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Perceptron model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

print("Perceptron Based Iris Classification")
print("-------------------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

# Test a new sample
sample = [[5.1, 3.5, 1.4, 0.2]]
sample_scaled = scaler.transform(sample)

result = model.predict(sample_scaled)

print("Sample Input:", sample)
print("Predicted Flower:", iris.target_names[result[0]])
