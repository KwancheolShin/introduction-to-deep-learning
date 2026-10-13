# Week: Techniques for Training -- batch normalization
#   (1) gradient check of the BatchNormalization layer
#   (2) with / without batch norm for 16 weight scales (1000 training images, as in the book)
#   (3) with / without batch norm on the full MNIST (the network of the optimizer comparison)
#   * uses mnist_data.py, layers.py and optimizers.py in the same folder
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from layers import MultiLayerNet
from optimizers import SGD

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================
# (1) Gradient check: backprop through BatchNormalization vs. numerical differentiation
# =====================================================
def numerical_gradient(f, P, eps=1e-5):
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

np.random.seed(0)
small = MultiLayerNet(4, [5, 5], 3, weight_init_std=1.0, use_batchnorm=True)
X_small = np.random.randn(6, 4)
T_small = np.eye(3)[np.random.randint(0, 3, size=6)]
small.loss(X_small, T_small)
grads = small.backward()
for key in small.params:
    num = numerical_gradient(lambda: small.loss(X_small, T_small), small.params[key])
    print(f"{key:7s}: max |backprop - numerical| = {np.max(np.abs(grads[key] - num)):.2e}")

# =====================================================
# Data
# =====================================================
x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()

def train(net, X, T, epochs, batch_size, lr, X_eval, T_eval, seed=2):
    """plain SGD; return the mini-batch losses and the accuracy on (X_eval, T_eval) after each epoch"""
    optimizer = SGD(lr=lr)
    np.random.seed(seed)                          # the SAME mini-batches for every network
    N = X.shape[0]
    losses, accs = [], []
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for it in range(N // batch_size):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            losses.append(net.loss(X[idx], T[idx]))
            optimizer.update(net.params, net.backward())
        accs.append(net.accuracy(X_eval, T_eval))
    return losses, accs

# =====================================================
# (2) 16 weight scales, 1000 training images (as in the book)
# =====================================================
x_small, t_small = x_train[:1000], t_train[:1000]
hidden = [100, 100, 100, 100, 100]               # 5 hidden layers, 100 nodes each
weight_scales = np.logspace(0, -4, num=16)       # 1, ..., 0.0001
epochs_small = 20

fig, axes = plt.subplots(4, 4, figsize=(13, 10))
t0 = time.time()
for ax, w in zip(axes.ravel(), weight_scales):
    curves = {}
    for use_bn in (True, False):
        np.random.seed(1)                         # the SAME initial weights
        net = MultiLayerNet(784, hidden, 10, weight_init_std=w, use_batchnorm=use_bn)
        _, accs = train(net, x_small, t_small, epochs_small, 100, 0.01, x_small, t_small)
        curves[use_bn] = accs
    ax.plot(range(1, epochs_small + 1), curves[True], label="Batch Norm", color="tab:red")
    ax.plot(range(1, epochs_small + 1), curves[False], "--", label="without BN", color="tab:blue")
    ax.set_title(f"W std = {w:.4g}", fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
    print(f"std {w:.4g}: train acc after {epochs_small} epochs  BN={curves[True][-1]:.3f}  "
          f"noBN={curves[False][-1]:.3f}")
axes[0, 0].legend(fontsize=9)
for ax in axes[-1]:
    ax.set_xlabel("epoch")
for ax in axes[:, 0]:
    ax.set_ylabel("train accuracy")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bn_weight_scales.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"(2) done in {time.time() - t0:.0f}s")

# =====================================================
# (3) Full MNIST: the network of the optimizer comparison (4 hidden layers, std 0.05, SGD lr 0.01)
# =====================================================
results = {}
for use_bn in (False, True):
    np.random.seed(1)
    net = MultiLayerNet(784, [100, 100, 100, 100], 10, weight_init_std=0.05, use_batchnorm=use_bn)
    results[use_bn] = train(net, x_train, t_train, 5, 128, 0.01, x_val, t_val)
    print(f"full MNIST, batch norm = {use_bn}: val acc per epoch",
          " ".join(f"{a:.4f}" for a in results[use_bn][1]))

def smooth(y, w=50):
    return np.convolve(np.asarray(y), np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for use_bn, style in ((True, "-"), (False, "--")):
    label = "Batch Norm" if use_bn else "without BN"
    ax[0].plot(smooth(results[use_bn][0]), style, label=label)
    ax[1].plot(range(1, 6), results[use_bn][1], "o" + style, label=label)
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("training loss (moving average)")
ax[0].set_yscale("log")
ax[0].set_title("Training Loss (SGD, lr = 0.01)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("validation accuracy")
ax[1].set_title("Validation Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bn_mnist.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
