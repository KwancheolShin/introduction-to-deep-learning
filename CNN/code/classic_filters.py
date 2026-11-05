# Week: CNN -- hand-designed filters from image processing, applied to a grayscale photo
#   the same multiply-and-add as conv2d_naive in conv_basics.py, with filters chosen by people
#   * needs scikit-image for the test photo (skimage.data.camera, 512x512 grayscale)
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from skimage import data

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)


def conv2d(x, f):
    """like conv2d_naive(x, f, pad=FH//2) without loops, the output keeps the size of x;
    the inside is the same, only the border differs (edge padding here, zero padding there)"""
    p = f.shape[0] // 2
    x = np.pad(x, p, mode="edge")                       # repeat the border pixels (no dark frame)
    windows = sliding_window_view(x, f.shape)           # (H, W, FH, FW): every window at once
    return np.einsum("ijpq,pq->ij", windows, f)         # multiply elementwise, then add


filters = {
    "blur (5x5 average)": np.ones((5, 5)) / 25,
    "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    "Sobel x (vertical edges)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "Sobel y (horizontal edges)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "Laplacian (all edges)": np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]),
    "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
}

x = data.camera().astype(float) / 255                  # (512, 512), values in [0, 1]
out = {name: conv2d(x, f) for name, f in filters.items()}
for name, y in out.items():
    print(f"{name:28s} output {y.shape}, min {y.min():6.2f}, max {y.max():6.2f}")

# edge filters give positive AND negative values: show |y| (bright = strong edge)
show = {
    "original": (x, 0, 1),
    "blur (5x5 average)": (out["blur (5x5 average)"], 0, 1),
    "sharpen": (np.clip(out["sharpen"], 0, 1), 0, 1),
    "emboss": (np.clip(out["emboss"], 0, 1), 0, 1),             # weights add up to 1: flat regions keep their value
    "Sobel x (vertical edges)": (np.abs(out["Sobel x (vertical edges)"]), 0, 1),
    "Sobel y (horizontal edges)": (np.abs(out["Sobel y (horizontal edges)"]), 0, 1),
    "Laplacian (all edges)": (np.abs(out["Laplacian (all edges)"]), 0, 0.5),
    "edge strength": (np.hypot(out["Sobel x (vertical edges)"], out["Sobel y (horizontal edges)"]), 0, 1),
}
fig, axes = plt.subplots(2, 4, figsize=(12, 6.4))
for ax, (name, (img, lo, hi)) in zip(axes.ravel(), show.items()):
    ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(name, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cnn_classic_filters.png"), dpi=130)

# a close-up of the camera and the tripod: the edge filters pick different directions
r, c = slice(60, 260), slice(180, 380)
fig, axes = plt.subplots(1, 3, figsize=(9, 3.3))
for ax, (name, img) in zip(axes, [("original", x),
                                  ("Sobel x: vertical edges", np.abs(out["Sobel x (vertical edges)"])),
                                  ("Sobel y: horizontal edges", np.abs(out["Sobel y (horizontal edges)"]))]):
    ax.imshow(img[r, c], cmap="gray", vmin=0, vmax=1)
    ax.set_title(name, fontsize=11)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "cnn_classic_filters_zoom.png"), dpi=130)
print("saved to", os.path.abspath(FIG_DIR))
