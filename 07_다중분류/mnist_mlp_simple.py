# %% 0. Setup
# Week: Multi-class Classification -- ONE network, trained with model.fit()
# The same layers as mnist_mlp.py, but
#   * MultiLayerNet trains itself: net.fit(...) is a METHOD of the class (like Keras' model.fit)
#   * no experiments: one network [784 -> 100 -> 10], as in mnist_torch.py
# Compare with mnist_torch.py, where the training loop is written OUTSIDE the model.
# Run this file from its own folder; figures are saved to ./figures/*.png
#   Cells: each '# %%' starts a cell. In VS Code, Run Cell (Shift+Enter) sends one cell to the
#   Interactive window (variables stay in the kernel); 'python <file>' still runs the whole file.

import os
import time
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs("figures", exist_ok=True)

# %% 1. Data
# =====================================================
# 1. Data: load, flatten, normalize, one-hot, train/validation split (the same as mnist_mlp.py)
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

(x_all_o, t_all_o), (x_test_o, t_test_o) = load_mnist()

x_all  = x_all_o.reshape(x_all_o.shape[0], -1) / 255.0      # (60000, 784)
x_test = x_test_o.reshape(x_test_o.shape[0], -1) / 255.0    # (10000, 784)
t_all  = np.eye(10)[t_all_o]                                 # (60000, 10)
t_test = np.eye(10)[t_test_o]                                # (10000, 10)

perm = np.random.permutation(x_all.shape[0])
val_idx, train_idx = perm[:5000], perm[5000:]
x_train, t_train = x_all[train_idx], t_all[train_idx]       # (55000, 784)
x_val,   t_val   = x_all[val_idx],   t_all[val_idx]         # (5000, 784)
print("train:", x_train.shape, " validation:", x_val.shape, " test:", x_test.shape)

# %% 2. Layers
# =====================================================
# 2. Layers (the same as mnist_lab.py / mnist_mlp.py)
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
        self.dW = self.X.T @ dA                          # (l,h)@(h,m) -> (l, m)
        self.dB = np.sum(dA, axis=0, keepdims=True)      # sum over rows -> (1, m)
        return dA @ self.W.T                             # dL/dX : (h, l)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, A):
        self.mask = (A > 0)
        return np.where(self.mask, A, 0.0)

    def backward(self, dZ):
        return np.where(self.mask, dZ, 0.0)


def softmax(U):
    U = U - np.max(U, axis=1, keepdims=True)
    expU = np.exp(U)
    return expU / np.sum(expU, axis=1, keepdims=True)


class SoftmaxWithCrossEntropy:
    def __init__(self):
        self.Y = None
        self.T = None

    def forward(self, U, T):
        self.T = T
        self.Y = softmax(U)
        h = U.shape[0]
        return -np.sum(T * np.log(self.Y + 1e-12)) / h

    def backward(self):
        h = self.T.shape[0]
        return (self.Y - self.T) / h        # the ONLY place where we divide by h

# %% 3. Network with a fit() method
# =====================================================
# 3. Network: X -> [Affine -> ReLU] x k -> Affine -> Softmax-loss, and it can TRAIN ITSELF
# =====================================================
class MultiLayerNet:
    def __init__(self, input_size, hidden_sizes, output_size, weight_init_std=0.05):
        sizes = [input_size] + list(hidden_sizes) + [output_size]   # e.g. [784, 100, 10]
        self.params = {}
        self.layers = []
        for i in range(len(sizes) - 1):
            n_in, n_out = sizes[i], sizes[i + 1]
            W = weight_init_std * np.random.randn(n_in, n_out)
            B = np.zeros((1, n_out))
            self.params[f"W{i + 1}"] = W
            self.params[f"B{i + 1}"] = B
            self.layers.append(Affine(W, B))
            if i < len(sizes) - 2:                     # no activation after the last Affine
                self.layers.append(Relu())
        self.last = SoftmaxWithCrossEntropy()

    def predict(self, X):                     # scores U
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def loss(self, X, T):
        return self.last.forward(self.predict(X), T)

    def backward(self):                       # call right after loss()
        d = self.last.backward()
        for layer in reversed(self.layers):
            d = layer.backward(d)
        grads, k = {}, 1
        for layer in self.layers:
            if isinstance(layer, Affine):
                grads[f"W{k}"], grads[f"B{k}"] = layer.dW, layer.dB
                k += 1
        return grads

    def accuracy(self, X, T):
        return np.mean(np.argmax(self.predict(X), axis=1) == np.argmax(T, axis=1))

    def num_params(self):
        return sum(p.size for p in self.params.values())

    def fit(self, x, t, x_val=None, t_val=None, epochs=10, batch_size=100, learning_rate=0.1):
        """Mini-batch SGD on (x, t). Returns the history; calling fit() again CONTINUES training."""
        N = x.shape[0]
        iter_per_epoch = N // batch_size
        hist = {"loss": [], "train_acc": [], "val_acc": []}
        t0 = time.time()
        for epoch in range(epochs):
            perm = np.random.permutation(N)                  # a new order every epoch
            for it in range(iter_per_epoch):
                idx = perm[it * batch_size:(it + 1) * batch_size]
                loss = self.loss(x[idx], t[idx])             # forward
                grads = self.backward()                      # backward
                for key in self.params:                      # update (in place)
                    self.params[key] -= learning_rate * grads[key]
                hist["loss"].append(loss)
            hist["train_acc"].append(self.accuracy(x, t))
            msg = f"epoch {epoch + 1:2d}: loss={loss:.4f}, train acc={hist['train_acc'][-1]:.4f}"
            if x_val is not None:
                hist["val_acc"].append(self.accuracy(x_val, t_val))
                msg += f", val acc={hist['val_acc'][-1]:.4f}"
            print(msg + f" ({time.time() - t0:.0f}s)")
        return hist

# %% 4. Model
np.random.seed(1)
net = MultiLayerNet(784, [100], 10)
print("layers:", " -> ".join(type(layer).__name__ for layer in net.layers))
print("number of parameters:", net.num_params())

# %% 5. Training (takes a while): ONE line, the loop lives inside the class
epochs = 10
hist = net.fit(x_train, t_train, x_val, t_val, epochs=epochs, batch_size=100, learning_rate=0.1)

# %% 6. Test accuracy: ONCE, at the end
test_acc = net.accuracy(x_test, t_test)
print(f"test accuracy: {test_acc:.4f}")

# %% 7. Plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(hist["loss"], linewidth=0.6)
ax[0].set(xlabel="iteration", ylabel="cross-entropy loss", title="Training Loss")
ax[0].grid(True)
ax[1].plot(range(1, epochs + 1), hist["train_acc"], "o--", label="train")
ax[1].plot(range(1, epochs + 1), hist["val_acc"], "o-", label="validation")
ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
ax[1].legend()
ax[1].grid(True)
fig.tight_layout()
fig.savefig("figures/mlp_simple_loss_acc.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 8. Predictions on the first 10 test images
probs = softmax(net.predict(x_test[:10]))
pred = probs.argmax(axis=1)
fig, axes = plt.subplots(1, 10, figsize=(10, 2.6))
for k, a in enumerate(axes):
    a.imshow(x_test_o[k], cmap="gray")
    a.set_title(f"pred {pred[k]}\n({probs[k, pred[k]]:.2f})", fontsize=8)
    a.axis("off")
fig.tight_layout()
fig.savefig("figures/mlp_simple_predictions.png", dpi=200, bbox_inches="tight")
plt.show()
print("predicted:", pred.tolist())
print("true     :", t_test_o[:10].tolist())
