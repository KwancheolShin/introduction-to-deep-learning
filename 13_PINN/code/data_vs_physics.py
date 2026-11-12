# %% 0. Setup
# Week: PINN -- what does the physics add?
#   Target: u*(x) = sin(pi x) on [0, 1], the solution of -u'' = pi^2 sin(pi x), u(0) = u(1) = 0.
#   (A) data only:      8 measurements of u*, all in [0, 0.4]          loss = data
#   (B) physics only:   no measurements, the equation + boundary values  loss = PDE + BC   (the PINN of pinn_poisson_1d.py)
#   (C) data + physics: both                                             loss = data + PDE + BC
#   The same network and the same optimizer for all three.
# Figure: ../figures/data_vs_physics.png

import os
# Windows + Anaconda: numpy (MKL) and a pip-installed PyTorch each bring their own OpenMP library,
# and using both in one program stops with "OMP: Error #15".  This line lets it run (a known workaround;
# not needed on Colab or in a clean environment).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import math
import matplotlib.pyplot as plt
import torch
from torch import nn

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

def f(x):
    return math.pi**2 * torch.sin(math.pi * x)

def u_exact(x):
    return torch.sin(math.pi * x)

def make_net():
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1))

def derivative(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# %% 1. Points
torch.manual_seed(0)
x_d = torch.linspace(0, 0.4, 8).reshape(-1, 1)               # measurement locations
u_d = u_exact(x_d)                                           # measurements (no noise)
x_f = torch.rand(64, 1).requires_grad_(True)                 # collocation points
x_b = torch.tensor([[0.0], [1.0]])
x_test = torch.linspace(0, 1, 1001).reshape(-1, 1)

# %% 2. Train the three versions
def train(use_data, use_physics, n_steps=5000, seed=0):
    torch.manual_seed(seed)
    model = make_net()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(n_steps):
        opt.zero_grad()
        loss = 0.0
        if use_data:
            loss = loss + torch.mean((model(x_d) - u_d)**2)
        if use_physics:
            u = model(x_f)
            u_xx = derivative(derivative(u, x_f), x_f)
            loss = loss + torch.mean((-u_xx - f(x_f))**2) + torch.mean(model(x_b)**2)
        loss.backward()
        opt.step()
    with torch.no_grad():
        u_pred = model(x_test)
        err = (torch.norm(u_pred - u_exact(x_test)) / torch.norm(u_exact(x_test))).item()
    return u_pred.numpy().ravel(), err

results = {}
for name, d, p in [("(A) data only", True, False), ("(B) physics only", False, True), ("(C) data + physics", True, True)]:
    results[name] = train(d, p)
    print(f"{name:20s}: rel. L2 error = {results[name][1]:.3e}")

# %% 3. Plot
fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
xt = x_test.numpy().ravel()
for a, (name, (u_pred, err)) in zip(ax, results.items()):
    a.plot(xt, u_exact(x_test).numpy().ravel(), "k-", lw=4, alpha=0.25, label="exact")
    a.plot(xt, u_pred, "r-", label="network")
    if "data" in name:
        a.plot(x_d.numpy(), u_d.numpy(), "bo", ms=5, label="measurements")
    if "physics" in name:
        a.plot(x_f.detach().numpy(), -0.45 * torch.ones(64).numpy(), "|", color="gray", ms=8,
               label="collocation points")
    a.set_title(f"{name}\nrel. L2 error = {err:.1e}")
    a.set_xlabel("x"); a.grid(True, alpha=0.3)
    a.set_ylim(-0.6, 1.6)
ax[0].set_ylabel("u")
ax[0].legend(loc="lower center", fontsize=8)
ax[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/data_vs_physics.png", dpi=200, bbox_inches="tight")
plt.show()
