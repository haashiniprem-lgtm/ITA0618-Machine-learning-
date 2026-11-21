
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()

X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("K-Nearest Neighbours Classification")
print("-----------------------------------")
print("Number of Neighbours:", 3)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")

sample = [[5.1, 3.5, 1.4, 0.2]]
result = model.predict(sample)

print("Sample Input:", sample)
print("Predicted Flower:", iris.target_names[result[0]])
