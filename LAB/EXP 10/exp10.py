
import numpy as np
from sklearn.mixture import GaussianMixture

# Sample dataset
X = np.array([
    [1.0], [1.5], [2.0], [2.5], [3.0],
    [8.0], [8.5], [9.0], [9.5], [10.0]
])

# Gaussian Mixture Model using EM
model = GaussianMixture(
    n_components=2,
    random_state=42,
    n_init=5
)

model.fit(X)

labels = model.predict(X)

print("Expectation Maximization Algorithm")
print("----------------------------------")

print("Number of Clusters:", model.n_components)

print("\nCluster Assignments:")
for i in range(len(X)):
    print("Data:", X[i][0], "Cluster:", labels[i])

print("\nCluster Means:")
print(np.round(model.means_.flatten(), 2))

print("\nModel Converged:", model.converged_)
