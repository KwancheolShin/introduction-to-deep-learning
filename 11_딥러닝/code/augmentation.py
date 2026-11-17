# augmentation.py -- data augmentation for MNIST (Week: deep learning)
#   augment(xb): a random small rotation and shift of every image in a mini-batch (N, 1, 28, 28)
# Run this file to draw ../figures/augment_examples.png

import numpy as np
import torch
import torch.nn.functional as F


def augment(xb, max_deg=10, max_shift=2):
    """rotate every image by a random angle in [-10, 10] degrees and shift it by up to 2 pixels"""
    n = xb.shape[0]
    ang = (torch.rand(n) * 2 - 1) * max_deg * np.pi / 180
    shift = (torch.rand(n, 2) * 2 - 1) * max_shift * 2 / 28     # in units of the half width
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.stack([torch.stack([cos, -sin, shift[:, 0]], 1),
                         torch.stack([sin, cos, shift[:, 1]], 1)], 1)   # (n, 2, 3)
    grid = F.affine_grid(theta, xb.shape, align_corners=False)
    return F.grid_sample(xb, grid, align_corners=False)          # outside the image: 0


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from mnist_data import load_mnist_split

    FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
    torch.manual_seed(3)
    x_train, t_train, *_ = load_mnist_split()
    i = int(np.argmax(np.argmax(t_train, axis=1) == 2))          # the first "2"
    x = torch.tensor(x_train[i], dtype=torch.float32).reshape(1, 1, 28, 28)
    imgs = torch.cat([x, augment(x.repeat(7, 1, 1, 1))])     # the original and 7 random versions
    fig, axes = plt.subplots(1, 8, figsize=(8, 1.3))
    for k, ax in enumerate(axes):
        ax.imshow(imgs[k, 0].numpy(), cmap="gray_r")
        ax.set_title("original" if k == 0 else f"augmented {k}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(os.path.join(FIG_DIR, "augment_examples.png"), dpi=200, bbox_inches="tight")
