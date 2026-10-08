# Week: Multi-class Classification -- Lab (MNIST with NumPy only)
# One hidden layer + softmax output, cross-entropy loss, mini-batch gradient descent.
# Run this file from its own folder; figures are saved to ./figures/*.png

import os
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs("figures", exist_ok=True)

# =====================================================
# Step 1. Load MNIST, flatten, normalize, one-hot encode
# =====================================================
def load_mnist():
    """Return (x_train, t_train), (x_test, t_test): images (N,28,28), labels (N,)."""
    try:
        from tensorflow.keras.datasets import mnist   # if TensorFlow is installed
        return mnist.load_data()
    except ImportError:
        from sklearn.datasets import fetch_openml     # otherwise download from OpenML
        X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
        X = X.reshape(-1, 28, 28).astype(np.uint8)
        y = y.astype(np.int64)
        return (X[:60000], y[:60000]), (X[60000:], y[60000:])

(x_train_o, t_train_o), (x_test_o, t_test_o) = load_mnist()

# each image (28, 28) -> one row vector (1, 784);  pixel values 0..255 -> 0..1
x_train = x_train_o.reshape(x_train_o.shape[0], -1) / 255.0   # (60000, 784)
x_test  = x_test_o.reshape(x_test_o.shape[0], -1) / 255.0     # (10000, 784)

# label k -> one-hot vector (row k of the 10x10 identity matrix)
t_train = np.eye(10)[t_train_o]   # (60000, 10)
t_test  = np.eye(10)[t_test_o]    # (10000, 10)

print("x_train:", x_train.shape, " t_train:", t_train.shape)
print("x_test :", x_test.shape,  " t_test :", t_test.shape)

plt.figure(figsize=(10, 2.4))
for k in range(10):
    plt.subplot(1, 10, k + 1)
    plt.imshow(x_train_o[k], cmap="gray")
    plt.title(f"{t_train_o[k]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("figures/mnist_samples.png", dpi=200, bbox_inches="tight")
plt.show()

# =====================================================
# Step 2. Layers: forward and backward
# =====================================================
class Affine:
    """A = X W + B  for a mini-batch X of shape (h, l)."""
    def __init__(self, W, B):
        self.W = W          # (l, m)
        self.B = B          # (1, m)
        self.X = None
        self.dW = None
        self.dB = None

    def forward(self, X):
        self.X = X
        return X @ self.W + self.B          # (h,l)@(l,m) + (1,m) -> (h, m)

    def backward(self, dA):
        # dA = dL/dA (h, m). It already contains the 1/h of the mini-batch mean,
        # so here we only SUM the contributions of the h samples.
        self.dW = self.X.T @ dA                          # (l,h)@(h,m) -> (l, m)
        self.dB = np.sum(dA, axis=0, keepdims=True)      # sum over rows -> (1, m)
        return dA @ self.W.T                             # dL/dX : (h, l)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, A):
        self.mask = (A > 0)                 # True where the node is active
        return np.where(self.mask, A, 0.0)

    def backward(self, dZ):
        return np.where(self.mask, dZ, 0.0) # pass dZ only where A > 0


class Sigmoid:
    def __init__(self):
        self.Z = None

    def forward(self, A):
        self.Z = 1.0 / (1.0 + np.exp(-A))
        return self.Z

    def backward(self, dZ):
        return dZ * self.Z * (1.0 - self.Z)


def softmax(U):
    U = U - np.max(U, axis=1, keepdims=True)   # same result, avoids overflow of exp
    expU = np.exp(U)
    return expU / np.sum(expU, axis=1, keepdims=True)


class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, U, T):
        self.T = T
        self.Y = softmax(U)                                  # (h, n)
        h = U.shape[0]
        return -np.sum(T * np.log(self.Y + 1e-12)) / h       # mean over the mini-batch

    def backward(self):
        h = self.T.shape[0]
        return (self.Y - self.T) / h        # dL/dU : the ONLY place where we divide by h


class SoftmaxWithSquaredError:
    """Same softmax output, but loss = (1/h) * sum 0.5*(y - t)^2  (for comparison)."""
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, U, T):
        self.T = T
        self.Y = softmax(U)
        h = U.shape[0]
        return 0.5 * np.sum((self.Y - self.T) ** 2) / h

    def backward(self):
        h = self.T.shape[0]
        R = self.Y - self.T                                  # dL/dY (times h)
        # multiply by the softmax Jacobian, row by row, without building it
        return self.Y * (R - np.sum(self.Y * R, axis=1, keepdims=True)) / h

# =====================================================
# Step 3. Two-layer network: X -> Affine -> ReLU -> Affine -> Softmax-CE
# =====================================================
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01,
                 loss="ce"):
        self.params = {
            "W1": weight_init_std * np.random.randn(input_size, hidden_size),
            "B1": np.zeros((1, hidden_size)),
            "W2": weight_init_std * np.random.randn(hidden_size, output_size),
            "B2": np.zeros((1, output_size)),
        }
        self.affine1 = Affine(self.params["W1"], self.params["B1"])
        self.act     = Relu()                    # or Sigmoid()
        self.affine2 = Affine(self.params["W2"], self.params["B2"])
        if loss == "ce":
            self.last = SoftmaxWithCrossEntropy()
        else:                                    # loss == "se"
            self.last = SoftmaxWithSquaredError()

    def predict(self, X):                        # returns the scores U (h, n)
        A = self.affine1.forward(X)
        Z = self.act.forward(A)
        U = self.affine2.forward(Z)
        return U

    def loss(self, X, T):
        U = self.predict(X)
        return self.last.forward(U, T)

    def backward(self):                          # call right after loss()
        dU = self.last.backward()                # (h, n)
        dZ = self.affine2.backward(dU)           # (h, m)
        dA = self.act.backward(dZ)               # (h, m)
        dX = self.affine1.backward(dA)           # (h, l)
        return {"W1": self.affine1.dW, "B1": self.affine1.dB,
                "W2": self.affine2.dW, "B2": self.affine2.dB}

    def accuracy(self, X, T):
        U = self.predict(X)
        return np.mean(np.argmax(U, axis=1) == np.argmax(T, axis=1))

# =====================================================
# Step 4. Gradient check: backprop vs. numerical differentiation
# =====================================================
def numerical_gradient(f, P, eps=1e-5):
    """Central difference of the scalar function f() w.r.t. every entry of array P."""
    grad = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        f_plus = f()
        P[idx] = old - eps
        f_minus = f()
        P[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad

X_small = np.random.randn(6, 4)
T_small = np.eye(3)[np.random.randint(0, 3, size=6)]

for loss_type in ["ce", "se"]:
    small = TwoLayerNet(input_size=4, hidden_size=5, output_size=3,
                        weight_init_std=1.0, loss=loss_type)
    small.loss(X_small, T_small)
    grads = small.backward()
    for key in small.params:
        num = numerical_gradient(lambda: small.loss(X_small, T_small), small.params[key])
        print(f"[{loss_type}] {key}: max |backprop - numerical| = "
              f"{np.max(np.abs(grads[key] - num)):.2e}")

# =====================================================
# Step 5. Mini-batch training
# =====================================================
network = TwoLayerNet(input_size=784, hidden_size=100, output_size=10)

epochs        = 10
batch_size    = 100
learning_rate = 0.1

N = x_train.shape[0]
iter_per_epoch = N // batch_size          # 600 iterations = 1 epoch

loss_history   = []
train_acc_list = []
test_acc_list  = []

for epoch in range(epochs):
    perm = np.random.permutation(N)       # shuffle once per epoch
    for it in range(iter_per_epoch):
        idx = perm[it * batch_size:(it + 1) * batch_size]
        X_batch, T_batch = x_train[idx], t_train[idx]

        loss  = network.loss(X_batch, T_batch)   # forward
        grads = network.backward()               # backward
        for key in network.params:               # gradient descent (in place)
            network.params[key] -= learning_rate * grads[key]

        loss_history.append(loss)

    train_acc = network.accuracy(x_train, t_train)
    test_acc  = network.accuracy(x_test, t_test)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    print(f"epoch {epoch + 1:2d}: loss={loss:.4f}, "
          f"train acc={train_acc:.4f}, test acc={test_acc:.4f}")

plt.figure(figsize=(11, 4))
plt.subplot(1, 2, 1)
plt.plot(loss_history, linewidth=0.6)
plt.xlabel("iteration")
plt.ylabel("cross-entropy loss")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(1, 2, 2)
ep = np.arange(1, epochs + 1)
plt.plot(ep, train_acc_list, "o-", label="train")
plt.plot(ep, test_acc_list, "s--", label="test")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title("Train/Test Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/mnist_loss_acc.png", dpi=200, bbox_inches="tight")
plt.show()

# =====================================================
# Step 6. Look at the predictions
# =====================================================
U_test = network.predict(x_test)
Y_test = softmax(U_test)
pred   = np.argmax(Y_test, axis=1)

plt.figure(figsize=(10, 2.6))
for k in range(10):
    plt.subplot(1, 10, k + 1)
    plt.imshow(x_test_o[k], cmap="gray")
    plt.title(f"pred {pred[k]}\n({Y_test[k, pred[k]]:.2f})", fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig("figures/mnist_predictions.png", dpi=200, bbox_inches="tight")
plt.show()

wrong = np.where(pred != t_test_o)[0]
print("number of wrong predictions:", len(wrong), "/", len(t_test_o))

plt.figure(figsize=(10, 2.6))
for k, i in enumerate(wrong[:10]):
    plt.subplot(1, 10, k + 1)
    plt.imshow(x_test_o[i], cmap="gray")
    plt.title(f"pred {pred[i]}\ntrue {t_test_o[i]}", fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.savefig("figures/mnist_wrong.png", dpi=200, bbox_inches="tight")
plt.show()
