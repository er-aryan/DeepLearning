# mnist_scratch.py
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Load and preprocess
digits = load_digits()
X = digits.data
y = digits.target.reshape(-1, 1)

X = StandardScaler().fit_transform(X)
y_oh = OneHotEncoder(sparse_output=False).fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_oh, test_size=0.2, random_state=42)

# Activation and derivatives
def relu(x): return np.maximum(0, x)
def relu_deriv(x): return x > 0
def softmax(x): 
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

# Parameters
input_size, hidden_size, output_size = 64, 128, 10
lr = 0.01
epochs = 100

# Weights
W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward pass
    Z1 = X_train @ W1 + b1
    A1 = relu(Z1)
    Z2 = A1 @ W2 + b2
    A2 = softmax(Z2)

    # Backprop
    loss = -np.mean(np.sum(y_train * np.log(A2 + 1e-8), axis=1))
    dZ2 = A2 - y_train
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * relu_deriv(Z1)
    dW1 = X_train.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    # Update
    W1 -= lr * dW1
    b1 -= lr * db1
    W2 -= lr * dW2
    b2 -= lr * db2

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f}")

# Predict and evaluate
Z1_test = X_test @ W1 + b1
A1_test = relu(Z1_test)
Z2_test = A1_test @ W2 + b2
A2_test = softmax(Z2_test)
preds = np.argmax(A2_test, axis=1)
true = np.argmax(y_test, axis=1)
print("✅ Accuracy (Scratch NN):", accuracy_score(true, preds))