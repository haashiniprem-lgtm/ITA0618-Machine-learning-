
import numpy as np

# XOR dataset
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)

# Initialize weights and biases
W1 = np.random.randn(2, 4)
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1)
b2 = np.zeros((1, 1))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

learning_rate = 0.5

# Backpropagation training
for epoch in range(20000):
    hidden = sigmoid(np.dot(X, W1) + b1)
    output = sigmoid(np.dot(hidden, W2) + b2)

    output_error = output - y
    d_output = output_error * output * (1 - output)

    d_hidden = np.dot(d_output, W2.T) * hidden * (1 - hidden)

    W2 -= learning_rate * np.dot(hidden.T, d_output) / len(X)
    b2 -= learning_rate * np.mean(d_output, axis=0, keepdims=True)
    W1 -= learning_rate * np.dot(X.T, d_hidden) / len(X)
    b1 -= learning_rate * np.mean(d_hidden, axis=0, keepdims=True)

print("Artificial Neural Network using Backpropagation")
print("------------------------------------------------")
print("Input\tPredicted Output")

hidden = sigmoid(np.dot(X, W1) + b1)
output = sigmoid(np.dot(hidden, W2) + b2)

for i in range(len(X)):
    print(X[i], "\t", round(output[i][0], 3))

print("\nFinal Classification:")
print((output >= 0.5).astype(int).flatten())
