# Week: Techniques for Training -- the same MNIST classifier in PyTorch
#
#   Everything we built by hand has a PyTorch counterpart:
#     Affine                     -> nn.Linear            (its weight is (out, in): the TRANSPOSE of our W)
#     Relu                       -> nn.ReLU
#     BatchNormalization         -> nn.BatchNorm1d       (momentum=0.1 means our momentum 0.9)
#     Dropout                    -> nn.Dropout           (inverted dropout, the same as ours)
#     SoftmaxWithCrossEntropy    -> nn.CrossEntropyLoss  (takes the scores and class INDICES, not one-hot)
#     backward() of every layer  -> loss.backward()      (autograd records the computational graph)
#     optimizers.py              -> torch.optim.SGD, torch.optim.Adam, torch.optim.AdamW, ...
#     train_flg=True / False     -> model.train() / model.eval()
#     weight_init_std="he"       -> nn.init.kaiming_normal_
#
#   * uses mnist_data.py in the same folder: the SAME train / validation / test split as our numpy code
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import copy
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

# =====================================================
# Data: numpy arrays -> tensors -> DataLoader (shuffling and mini-batches for us)
# =====================================================
x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()   # 55000 / 5000 / 10000


def to_dataset(x, t):
    """images (N, 784) and one-hot labels (N, 10) -> a dataset of (image, class index)"""
    return TensorDataset(torch.tensor(x, dtype=torch.float32),
                         torch.tensor(np.argmax(t, axis=1)))


train_loader = DataLoader(to_dataset(x_train, t_train), batch_size=100, shuffle=True)
val_loader = DataLoader(to_dataset(x_val, t_val), batch_size=1000)
test_loader = DataLoader(to_dataset(x_test, t_test), batch_size=1000)

# =====================================================
# Model: [Linear -> BatchNorm -> ReLU -> (Dropout)] x k -> Linear,  He initialization
# =====================================================
def make_mlp(hidden=(100, 100, 100, 100), use_batchnorm=True, dropout_ratio=0.0):
    layers, n_in = [], 784
    for n_out in hidden:
        layers.append(nn.Linear(n_in, n_out))
        if use_batchnorm:
            layers.append(nn.BatchNorm1d(n_out))
        layers.append(nn.ReLU())
        if dropout_ratio > 0:
            layers.append(nn.Dropout(dropout_ratio))
        n_in = n_out
    layers.append(nn.Linear(n_in, 10))            # scores U; softmax is inside the loss
    model = nn.Sequential(*layers)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")   # He: std = sqrt(2 / n_in)
            nn.init.zeros_(m.bias)
    return model


model = make_mlp().to(device)
print(model)
print("number of parameters:", sum(p.numel() for p in model.parameters()))

loss_fn = nn.CrossEntropyLoss()                  # softmax + cross-entropy, averaged over the batch

# AdamW: decay only the weights W (2-D), not B, gamma, beta (1-D) -- the same choice as our code
decay = [p for p in model.parameters() if p.dim() == 2]
no_decay = [p for p in model.parameters() if p.dim() == 1]
optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                               {"params": no_decay, "weight_decay": 0.0}], lr=0.001)

# =====================================================
# Training and evaluation
# =====================================================
def train_one_epoch(model, loader):
    model.train()                                 # BatchNorm: batch statistics, Dropout: on
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = loss_fn(model(xb), yb)             # forward: the graph is recorded
        optimizer.zero_grad()                     # gradients ADD UP in .grad, so clear them first
        loss.backward()                           # backward: dL/dW for every parameter
        optimizer.step()                          # = optimizer.update(params, grads)
        total_loss += loss.item() * len(yb)
        n += len(yb)
    return total_loss / n


@torch.no_grad()                                  # no graph needed: faster, less memory
def accuracy(model, loader):
    model.eval()                                  # BatchNorm: running averages, Dropout: off
    correct, n = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        correct += (model(xb).argmax(dim=1) == yb).sum().item()
        n += len(yb)
    return correct / n


epochs = 15
history = {"loss": [], "train": [], "val": []}
best_val, best_state = 0.0, None
for epoch in range(1, epochs + 1):
    history["loss"].append(train_one_epoch(model, train_loader))
    history["train"].append(accuracy(model, train_loader))
    history["val"].append(accuracy(model, val_loader))
    print(f"epoch {epoch:2d} | loss {history['loss'][-1]:.4f} | "
          f"train acc {history['train'][-1]:.4f} | val acc {history['val'][-1]:.4f}")
    if history["val"][-1] > best_val:             # keep the epoch with the best VALIDATION accuracy
        best_val, best_epoch = history["val"][-1], epoch
        best_state = copy.deepcopy(model.state_dict())

# Only at the very end: the test data, ONCE, with the chosen model
model.load_state_dict(best_state)
print(f"chosen epoch {best_epoch}: val acc {best_val:.4f}, test acc {accuracy(model, test_loader):.4f}")
# torch.save(model.state_dict(), "mnist_torch.pt")   # to keep the trained weights

# =====================================================
# Figure
# =====================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(range(1, epochs + 1), history["loss"], "o-")
ax[0].set_yscale("log")
ax[0].set(xlabel="epoch", ylabel="training loss (epoch mean)", title="Training Loss (AdamW)")
ax[0].grid(True, which="both", alpha=0.3)
ax[1].plot(range(1, epochs + 1), history["train"], "o--", label="train")
ax[1].plot(range(1, epochs + 1), history["val"], "o-", label="validation")
ax[1].axvline(best_epoch, color="gray", linestyle=":", label=f"chosen epoch ({best_epoch})")
ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "torch_mnist.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
