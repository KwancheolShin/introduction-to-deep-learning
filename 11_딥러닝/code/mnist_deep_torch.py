# Week: deep learning -- train DeepConvNet (deep_convnet.py) in PyTorch
#   the same network as the numpy version, but much faster to train
#   python mnist_deep_torch.py                      : all 55000 training images, 20 epochs
#   python mnist_deep_torch.py --augment            : + data augmentation (small rotations and shifts)
#   python mnist_deep_torch.py --n_train 1000 ...   : only the first 1000 training images
#   * uses mnist_data.py (the SAME train / validation / test split as the previous weeks) and augmentation.py
# Results go to ../figures/ and ../results/ (next to main.tex)

import os
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from mnist_data import load_mnist_split
from augmentation import augment

parser = argparse.ArgumentParser()
parser.add_argument("--augment", action="store_true")
parser.add_argument("--n_train", type=int, default=55000)
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
tag = f"n{args.n_train}" + ("_aug" if args.augment else "")

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
RES_DIR = os.path.join(HERE, "..", "results")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(RES_DIR, exist_ok=True)

torch.manual_seed(args.seed)
x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()   # 55000 / 5000 / 10000


def to_tensors(x, t):
    """images (N, 784) -> (N, 1, 28, 28), one-hot labels -> class indices"""
    return (torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28),
            torch.tensor(np.argmax(t, axis=1)))


X_train, Y_train = to_tensors(x_train[:args.n_train], t_train[:args.n_train])
X_val, Y_val = to_tensors(x_val, t_val)
X_test, Y_test = to_tensors(x_test, t_test)


def conv(c_in, c_out, pad=1):
    return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=pad), nn.ReLU()]


model = nn.Sequential(
    *conv(1, 16), *conv(16, 16), nn.MaxPool2d(2),          # -> (N, 16, 14, 14)
    *conv(16, 32), *conv(32, 32, pad=2), nn.MaxPool2d(2),  # -> (N, 32, 8, 8)
    *conv(32, 64), *conv(64, 64), nn.MaxPool2d(2),         # -> (N, 64, 4, 4)
    nn.Flatten(),                                          # -> (N, 1024)
    nn.Linear(64 * 4 * 4, 50), nn.ReLU(), nn.Dropout(0.5),
    nn.Linear(50, 10),
)
for m in model:                                            # He initialization, zero bias (as in numpy)
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
        nn.init.zeros_(m.bias)
n_params = sum(p.numel() for p in model.parameters())
print(f"{tag}: {n_params} parameters, {len(X_train)} training images")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


@torch.no_grad()
def predict(X):
    model.eval()
    return torch.cat([model(X[i:i + 1000]).argmax(dim=1) for i in range(0, len(X), 1000)])


val_acc, epoch_time = [], []
for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    perm = torch.randperm(len(X_train))
    for i in range(0, len(X_train), 100):
        idx = perm[i:i + 100]
        xb, yb = X_train[idx], Y_train[idx]
        if args.augment:
            xb = augment(xb)          # a NEW random version every time
        loss = F.cross_entropy(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    epoch_time.append(time.time() - t0)
    val_acc.append((predict(X_val) == Y_val).float().mean().item())
    print(f"epoch {epoch:2d} | val acc {val_acc[-1]:.4f} | {epoch_time[-1]:.0f} s", flush=True)

pred = predict(X_test)
test_acc = (pred == Y_test).float().mean().item()
print(f"test acc {test_acc:.4f}")

with open(os.path.join(RES_DIR, f"deep_{tag}.json"), "w") as f:
    json.dump({"tag": tag, "n_params": n_params, "val_acc": val_acc, "test_acc": test_acc,
               "epoch_time": epoch_time}, f, indent=1)

if args.n_train == 55000 and not args.augment:
    # (1) the parameters, in the shapes of the numpy DeepConvNet (Linear: (out, in) -> (in, out))
    params, k = {}, 0
    for m in model:
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            k += 1
            W = m.weight.detach().numpy().astype(np.float64)
            B = m.bias.detach().numpy().astype(np.float64)
            params[f"W{k}"] = W if W.ndim == 4 else W.T
            params[f"B{k}"] = B if W.ndim == 4 else B[None, :]
    np.savez(os.path.join(RES_DIR, "deep_convnet_params.npz"), **params)

    # (2) the misclassified test images
    wrong = torch.nonzero(pred != Y_test).ravel().numpy()
    print(f"{len(wrong)} misclassified test images")
    fig, axes = plt.subplots(3, 8, figsize=(8, 3.6))
    for ax, i in zip(axes.flat, wrong):
        ax.imshow(x_test[i].reshape(28, 28), cmap="gray_r")
        ax.set_title(f"{Y_test[i].item()} → {pred[i].item()}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.flat[len(wrong):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "deep_misclassified.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
