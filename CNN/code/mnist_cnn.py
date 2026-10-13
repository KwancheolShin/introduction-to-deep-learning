# Week: CNN -- train SimpleConvNet on MNIST (numpy only)
#   (1) Conv(30 filters 5x5) -> ReLU -> Pool(2x2) -> Affine(100) -> ReLU -> Affine(10), Adam
#   (2) the same training for an MLP (784 -> 100 -> 10) for comparison
#   (3) the 30 filters of the first layer before and after training
#   (4) what the trained filters respond to: the feature maps of one test image
#   * uses conv_layers.py, layers.py, optimizers.py and mnist_data.py in the same folder
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from layers import MultiLayerNet
from conv_layers import SimpleConvNet
from optimizers import Adam

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()
# a CNN keeps the SHAPE of the image: (N, 784) -> (N, 1, 28, 28) = (images, channels, height, width)
x_train = x_train.reshape(-1, 1, 28, 28)
x_val = x_val.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

epochs = 10
batch_size = 100


def train(net, flatten=False, seed=2):
    """Adam (lr 0.001); return the mini-batch losses and the train / validation accuracy per epoch"""
    shape = (lambda X: X.reshape(X.shape[0], -1)) if flatten else (lambda X: X)   # MLP: (N, 784)
    optimizer = Adam(lr=0.001)
    np.random.seed(seed)                              # the SAME mini-batches for both networks
    N = x_train.shape[0]
    losses, train_acc, val_acc = [], [], []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        perm = np.random.permutation(N)
        for it in range(N // batch_size):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            losses.append(net.loss(shape(x_train[idx]), t_train[idx]))
            optimizer.update(net.params, net.backward())
        train_acc.append(net.accuracy(shape(x_train[:5000]), t_train[:5000]))   # 5000 for speed
        val_acc.append(net.accuracy(shape(x_val), t_val))
        print(f"epoch {epoch:2d} | train acc {train_acc[-1]:.4f} | val acc {val_acc[-1]:.4f} "
              f"| {time.time() - t0:.0f} s", flush=True)
    return losses, train_acc, val_acc


# =====================================================
# (1) SimpleConvNet
# =====================================================
np.random.seed(1)
cnn = SimpleConvNet(input_shape=(1, 28, 28), filter_num=30, filter_size=5,
                    hidden_size=100, output_size=10)
print("SimpleConvNet parameters:", cnn.num_params())
W1_before = cnn.params["W1"].copy()                   # the filters before training
cnn_result = train(cnn)

# =====================================================
# (2) MLP for comparison
# =====================================================
np.random.seed(1)
mlp = MultiLayerNet(784, [100], 10, weight_init_std="he")
print("MLP parameters:", mlp.num_params())
mlp_result = train(mlp, flatten=True)

# test data: once, at the very end
print(f"test acc  CNN {cnn.accuracy(x_test, t_test):.4f}   "
      f"MLP {mlp.accuracy(x_test.reshape(-1, 784), t_test):.4f}")


def smooth(y, w=50):
    return np.convolve(np.asarray(y), np.ones(w) / w, mode="valid")


fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
for (name, result), style in zip((("CNN", cnn_result), ("MLP", mlp_result)), ("-", "--")):
    losses, train_acc, val_acc = result
    ax[0].plot(smooth(losses), style, label=name)
    ax[1].plot(range(1, epochs + 1), val_acc, "o" + style, label=f"{name} validation")
ax[0].set_yscale("log")
ax[0].set(xlabel="iteration", ylabel="training loss (moving average)", title="Training Loss (Adam)")
ax[0].grid(True, which="both", alpha=0.3)
ax[0].legend()
ax[1].set(xlabel="epoch", ylabel="accuracy", title="Validation Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cnn_mnist.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# (3) The 30 filters (5x5) of the first layer: before and after training
# =====================================================
fig, axes = plt.subplots(2, 1, figsize=(12, 3.6))
for ax, W, title in zip(axes, (W1_before, cnn.params["W1"]), ("before training", "after training")):
    ax.axis("off")
    ax.set_title(title)
    inner = ax.inset_axes([0, 0, 1, 0.85])
    inner.axis("off")
    for k in range(30):
        sub = inner.inset_axes([k / 30, 0, 1 / 30 * 0.9, 1])
        sub.imshow(W[k, 0], cmap="gray", interpolation="nearest")   # bright: large weight
        sub.set_xticks([])
        sub.set_yticks([])
fig.savefig(os.path.join(FIG_DIR, "cnn_filters.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# (4) Feature maps: the output of the first Convolution layer for one test image
# =====================================================
image = x_test[:1]                                     # (1, 1, 28, 28)
feature = cnn.layers[0].forward(image)                 # (1, 30, 24, 24)
show = [0, 1, 2, 3, 4, 5, 6, 7]                        # the first 8 filters
fig, axes = plt.subplots(2, len(show) + 1, figsize=(14, 3.6))
axes[0, 0].imshow(image[0, 0], cmap="gray")
axes[0, 0].set_title("input")
axes[1, 0].axis("off")
for col, k in enumerate(show, start=1):
    axes[0, col].imshow(cnn.params["W1"][k, 0], cmap="gray", interpolation="nearest")
    axes[0, col].set_title(f"filter {k}", fontsize=9)
    axes[1, col].imshow(feature[0, k], cmap="gray")
    axes[1, col].set_title("feature map", fontsize=9)
for a in axes.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "cnn_feature_maps.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
