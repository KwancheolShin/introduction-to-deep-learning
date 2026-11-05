# Week: CNN -- the same SimpleConvNet in PyTorch
#   Convolution -> nn.Conv2d,  Pooling -> nn.MaxPool2d,  Flatten -> nn.Flatten
#   (the weight of nn.Conv2d has the same shape as ours: (FN, C, FH, FW))
#   * uses mnist_data.py in the same folder: the SAME train / validation / test split

import os
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from mnist_data import load_mnist_split

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()   # 55000 / 5000 / 10000


def to_dataset(x, t):
    """images (N, 784) -> (N, 1, 28, 28) tensors, one-hot labels -> class indices"""
    return TensorDataset(torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28),
                         torch.tensor(np.argmax(t, axis=1)))


train_loader = DataLoader(to_dataset(x_train, t_train), batch_size=100, shuffle=True)
val_loader = DataLoader(to_dataset(x_val, t_val), batch_size=1000)
test_loader = DataLoader(to_dataset(x_test, t_test), batch_size=1000)

model = nn.Sequential(
    nn.Conv2d(1, 30, kernel_size=5),      # (N, 1, 28, 28) -> (N, 30, 24, 24)
    nn.ReLU(),
    nn.MaxPool2d(2),                      # -> (N, 30, 12, 12)
    nn.Flatten(),                         # -> (N, 4320)
    nn.Linear(30 * 12 * 12, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
).to(device)
print(model)
print("number of parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train_one_epoch(model, loader):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def accuracy(model, loader):
    model.eval()
    correct, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return correct / n


epochs = 10
val_acc = []
for epoch in range(1, epochs + 1):
    t0 = time.time()
    train_one_epoch(model, train_loader)
    val_acc.append(accuracy(model, val_loader))
    print(f"epoch {epoch:2d} | val acc {val_acc[-1]:.4f} | {time.time() - t0:.0f} s")
print(f"test acc {accuracy(model, test_loader):.4f}")

# the 30 filters of the first layer: model[0].weight has shape (30, 1, 5, 5)
filters = model[0].weight.detach().cpu().numpy()
fig, axes = plt.subplots(1, 30, figsize=(12, 0.9))
for k, ax in enumerate(axes):
    ax.imshow(filters[k, 0], cmap="gray", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
fig.savefig(os.path.join(FIG_DIR, "cnn_torch_filters.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
