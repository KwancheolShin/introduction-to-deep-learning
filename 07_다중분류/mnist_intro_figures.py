# Week: Multi-class Classification -- figures for the introduction
# (1) a grid of MNIST examples for each class 0..9
# (2) one enlarged image and the same image flattened into a 1 x 784 row vector
# Run this file from its own folder; figures are saved to ./figures/*.png

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

os.makedirs("figures", exist_ok=True)

X, y = fetch_openml("mnist_784", version=1, as_frame=False, return_X_y=True)
X = X.reshape(-1, 28, 28).astype(np.uint8)[:60000]   # training images
y = y.astype(np.int64)[:60000]                       # labels 0..9

# ---------------------------------------------------------------
# (1) grid: 10 classes (rows) x 12 examples (columns)
# ---------------------------------------------------------------
n_cols = 12
fig, axes = plt.subplots(10, n_cols, figsize=(n_cols * 0.62, 10 * 0.62))
for k in range(10):
    idx = np.where(y == k)[0][:n_cols]
    for j in range(n_cols):
        ax = axes[k, j]
        ax.imshow(X[idx[j]], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(f"{k}", rotation=0, fontsize=13, labelpad=12, va="center")
plt.subplots_adjust(wspace=0.06, hspace=0.06)
plt.savefig("figures/mnist_grid.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------
# (2) one image (28 x 28) and its flattened row vector (1 x 784)
# ---------------------------------------------------------------
i = 0                                   # the first training image
img = X[i]
row = img.reshape(1, -1)                # (1, 784)

fig = plt.figure(figsize=(11, 5.2))
gs = fig.add_gridspec(2, 1, height_ratios=[5, 1], hspace=0.35)

ax = fig.add_subplot(gs[0])
ax.imshow(img, cmap="gray", vmin=0, vmax=255)
for r in range(28):
    for c in range(28):
        v = img[r, c]
        if v > 0:
            ax.text(c, r, str(v), ha="center", va="center", fontsize=3.2,
                    color="black" if v > 128 else "white")
ax.set_xticks(np.arange(-0.5, 28, 1), minor=True)
ax.set_yticks(np.arange(-0.5, 28, 1), minor=True)
ax.grid(which="minor", color="gray", linewidth=0.2)
ax.tick_params(which="minor", length=0)
ax.set_xticks([0, 27])
ax.set_xticklabels(["1", "28"])
ax.set_yticks([0, 27])
ax.set_yticklabels(["1", "28"])
ax.set_title(f"one MNIST image: 28 x 28 pixels, label = {y[i]}", fontsize=11)

ax2 = fig.add_subplot(gs[1])
ax2.imshow(row, cmap="gray", aspect="auto", vmin=0, vmax=255)
for r in range(1, 28):
    ax2.axvline(28 * r - 0.5, color="red", linewidth=0.3)
ax2.set_yticks([])
ax2.set_xticks([0, 27, 783])
ax2.set_xticklabels(["$x_1$", "$x_{28}$", "$x_{784}$"], fontsize=10)
ax2.annotate("", xy=(-0.5, 1.15), xytext=(27.5, 1.15), xycoords=("data", "axes fraction"),
             arrowprops=dict(arrowstyle="|-|", color="red", lw=1))
ax2.text(14, 1.3, "image row 1", transform=ax2.get_xaxis_transform(),
         ha="left", va="bottom", fontsize=9, color="red")
ax2.set_title(r"flattened: one row vector $X=(x_1,\ldots,x_{784})$"
              "   (red lines: end of each image row)", fontsize=10, pad=18)

plt.savefig("figures/mnist_flatten.png", dpi=250, bbox_inches="tight")
plt.close()
print("saved figures/mnist_grid.png and figures/mnist_flatten.png; label of image 0 =", y[i])
