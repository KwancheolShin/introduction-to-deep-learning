# Week: Neuron, Activation, and Hidden Layer -- Lab Step 5
# One-hidden-layer network (1 -> m sigmoid -> 1 linear), full-batch gradient descent.
# Run this file from its own folder; figures are saved to ./figures/*.png

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs("figures", exist_ok=True)   # folder for saved figures

x = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6],
             dtype=float).reshape(-1, 1)       # (N, 1)
t = np.array([ 0,  0,  0,  0,  1, 1, 1, 2, 2, 3, 3, 3],
             dtype=float).reshape(-1, 1)       # (N, 1)
N = x.shape[0]

m = 2
W1 = np.random.randn(1, m) * 0.5 + 0.1
b1 = np.random.randn(1, m) * 0.5 + 0.1
W2 = np.random.randn(m, 1) * 0.5
b2 = np.random.randn(1, 1) * 0.5

learning_rate = 0.2
epochs = 20000
loss_history = []

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

for epoch in range(epochs):

    # -----------------------------
    # 1. Forward (all N samples at once)
    # -----------------------------
    a = x @ W1 + b1            # (N, m)
    h = sigmoid(a)             # (N, m)
    y = h @ W2 + b2            # (N, 1)

    error = y - t
    loss = np.mean(error**2)

    # -----------------------------
    # 2. Backward (row i = single-sample backprop of sample i)
    # -----------------------------
    dL_dy = 2.0 * error / N                        # (N, 1)
    dL_dW2 = h.T @ dL_dy                           # (m, 1)  sum over samples
    dL_db2 = np.sum(dL_dy, axis=0, keepdims=True)  # (1, 1)  sum over samples

    dL_dh = dL_dy @ W2.T                           # (N, m)
    dL_da = dL_dh * h * (1.0 - h)                  # (N, m)
    dL_dW1 = x.T @ dL_da                           # (1, m)  sum over samples
    dL_db1 = np.sum(dL_da, axis=0, keepdims=True)  # (1, m)  sum over samples

    # -----------------------------
    # 3. Gradient descent (one update per epoch)
    # -----------------------------
    W1 -= learning_rate * dL_dW1
    b1 -= learning_rate * dL_db1
    W2 -= learning_rate * dL_dW2
    b2 -= learning_rate * dL_db2

    loss_history.append(loss)

    if epoch % 2000 == 0:
        print(f"epoch={epoch:5d}, loss={loss:.6f}")

print("W1 =", W1)
print("b1 =", b1)
print("W2 =", W2.ravel())
print("b2 =", b2)

# prediction curve
x_plot = np.linspace(-7.0, 8.0, 300).reshape(-1, 1)   # (300, 1)
h_plot = sigmoid(x_plot @ W1 + b1)                    # (300, m)
y_plot = h_plot @ W2 + b2                             # (300, 1)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(x, t, "ro", label="data")
plt.plot(x_plot, y_plot, "k", label="network output")
plt.xlabel("x")
plt.ylabel("y")
plt.title("2-Layer Neural Network")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss_history)
plt.xlabel("epoch")
plt.ylabel("MSE loss")
plt.title("Cost History")
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/lab_fit_and_cost.png", dpi=200)
plt.show()

# what does each hidden neuron do?
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
for j in range(m):
    plt.plot(x_plot, h_plot[:, j], "--", label=f"h_{j+1}")
plt.title("hidden layer outputs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for j in range(m):
    plt.plot(x_plot, h_plot[:, j] * W2[j, 0], "--", label=f"h_{j+1} * w2_{j+1}")
plt.plot(x_plot, y_plot, "k", label="y = sum + b2")
plt.title("output layer")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/lab_hidden_units.png", dpi=200)
plt.show()
