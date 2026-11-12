# %% 0. Setup
# Week: PINN -- why tanh and not ReLU?
#   A network with random weights, its derivative u'(x) and its second derivative u''(x), all by autograd.
#   tanh: u, u', u'' are all smooth.   ReLU: u is piecewise linear, u' is piecewise constant, u'' = 0.
# Figure: ../figures/tanh_vs_relu.png

import os
# Windows + Anaconda: numpy (MKL) and a pip-installed PyTorch each bring their own OpenMP library,
# and using both in one program stops with "OMP: Error #15".  This line lets it run (a known workaround;
# not needed on Colab or in a clean environment).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib.pyplot as plt
import torch
from torch import nn

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

def derivative(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def make_net(act, seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(1, 16), act(), nn.Linear(16, 16), act(), nn.Linear(16, 1))

# %% 1. Derivatives of two random networks
x = torch.linspace(-2, 2, 801).reshape(-1, 1).requires_grad_(True)
fig, ax = plt.subplots(2, 3, figsize=(12, 5.6), sharex=True)
for row, (name, act) in enumerate([("tanh", nn.Tanh), ("ReLU", nn.ReLU)]):
    net = make_net(act, seed=3)
    u = net(x)
    u_x = derivative(u, x)
    try:
        u_xx = derivative(u_x, x)
    except RuntimeError as e:          # with ReLU, u_x may not depend on x at all
        print(name, "u_xx:", e)
        u_xx = torch.zeros_like(u_x)
    print(f"{name}: max |u_xx| on [-2, 2] = {u_xx.abs().max().item():.3e}")
    for col, (y, lab) in enumerate([(u, "u(x)"), (u_x, "u'(x)"), (u_xx, "u''(x)")]):
        ax[row, col].plot(x.detach().numpy(), y.detach().numpy(), color="C0" if row == 0 else "C3")
        ax[row, col].set_title(f"{name}: {lab}")
        ax[row, col].grid(True, alpha=0.3)
for a in ax[1]:
    a.set_xlabel("x")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/tanh_vs_relu.png", dpi=200, bbox_inches="tight")
plt.show()
