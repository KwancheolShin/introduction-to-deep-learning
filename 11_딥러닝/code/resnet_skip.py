# Week: deep learning -- why skip connections (ResNet) help a very deep network
#   networks on MNIST (images flattened to 784), every block has two layers:
#     plain   : h -> relu(W2 relu(W1 h))
#     residual: h -> relu(h + W2 relu(W1 h))          (a skip connection: + h)
#   the training loss during the first epoch (Adam, the same learning rate) of
#     plain with 3 blocks, plain with 30 blocks, residual with 30 blocks
#   * uses mnist_data.py
# The figure is saved to ../figures/ (next to main.tex)

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from mnist_data import load_mnist_split

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
width = 64


class Net(nn.Module):
    def __init__(self, n_blocks, skip):
        super().__init__()
        self.skip = skip
        self.inp = nn.Linear(784, width)
        self.blocks = nn.ModuleList(
            nn.ModuleList([nn.Linear(width, width), nn.Linear(width, width)]) for _ in range(n_blocks))
        self.out = nn.Linear(width, 10)
        for m in self.modules():                    # He initialization, zero bias
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
        if skip:                                    # each branch starts small: the block starts close to h -> h
            for _, l2 in self.blocks:
                l2.weight.data *= 0.1

    def forward(self, x):
        h = F.relu(self.inp(x))
        for l1, l2 in self.blocks:
            f = l2(F.relu(l1(h)))
            h = F.relu(h + f) if self.skip else F.relu(f)
        return self.out(h)


x_train, t_train, *_ = load_mnist_split()
X = torch.tensor(x_train, dtype=torch.float32)
Y = torch.tensor(np.argmax(t_train, axis=1))

fig, ax = plt.subplots(figsize=(6.5, 3.4))
for label, n_blocks, skip in (("plain, 3 blocks (8 layers)", 3, False),
                              ("plain, 30 blocks (62 layers)", 30, False),
                              ("residual, 30 blocks (62 layers)", 30, True)):
    torch.manual_seed(0)
    net = Net(n_blocks, skip)
    opt = torch.optim.Adam(net.parameters(), lr=0.001)
    perm, losses = torch.randperm(len(X)), []
    for i in range(0, len(X), 100):
        idx = perm[i:i + 100]
        loss = F.cross_entropy(net(X[idx]), Y[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    print(f"{label:32s}: mean loss of the last 50 steps {np.mean(losses[-50:]):.3f}")
    ax.plot(np.convolve(losses, np.ones(20) / 20, mode="valid"), label=label)
ax.set_xlabel("update step (first epoch)")
ax.set_ylabel("training loss (moving average)")
ax.set_ylim(0, 2.5)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "resnet_loss.png"), dpi=200, bbox_inches="tight")
