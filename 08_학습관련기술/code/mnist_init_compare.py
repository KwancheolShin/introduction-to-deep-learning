# Week: Techniques for Training -- compare weight initializations on MNIST
# A 5-hidden-layer ReLU network trained with plain SGD; only the initialization differs.
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

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()

# =====================================================
# Compare the initializations on the same network
# =====================================================
inits = {
    "std = 0.01": 0.01,
    "Xavier":     "xavier",
    "He":         "he",
}
hidden = [100, 100, 100, 100, 100]      # 5 hidden layers, 100 nodes each
epochs = 5
batch_size = 128
learning_rate = 0.01

N = x_train.shape[0]
iter_per_epoch = N // batch_size

loss_hist = {}
val_hist = {}
for name, init in inits.items():
    np.random.seed(1)
    net = MultiLayerNet(784, hidden, 10, activation="relu", loss="ce", weight_init_std=init)
    optimizer = SGD(lr=learning_rate)
    np.random.seed(2)                                   # the SAME mini-batches for every initialization
    loss_hist[name], val_hist[name] = [], []
    t0 = time.time()
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for it in range(iter_per_epoch):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            loss = net.loss(x_train[idx], t_train[idx])
            grads = net.backward()
            optimizer.update(net.params, grads)
            loss_hist[name].append(loss)
        val_hist[name].append(net.accuracy(x_val, t_val))
        print(f"[{name:10s}] epoch {epoch + 1}: loss={loss:.4f}, "
              f"val acc={val_hist[name][-1]:.4f} ({time.time() - t0:.0f}s)")

# =====================================================
# Plots
# =====================================================
def smooth(y, w=50):
    """moving average over w iterations (the mini-batch loss is noisy)"""
    y = np.asarray(y)
    return np.convolve(y, np.ones(w) / w, mode="valid")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for name in inits:
    ax[0].plot(smooth(loss_hist[name]), label=name)
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("training loss (moving average)")
ax[0].set_yscale("log")
ax[0].set_title("Training Loss (SGD, lr = 0.01)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()

for name in inits:
    ax[1].plot(range(1, epochs + 1), val_hist[name], "o-", label=name)
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("validation accuracy")
ax[1].set_title("Validation Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "mnist_init.png"), dpi=200, bbox_inches="tight")
plt.show()

print("\nvalidation accuracy after", epochs, "epochs:")
for name in inits:
    print(f"  {name:10s} {val_hist[name][-1]:.4f}")
