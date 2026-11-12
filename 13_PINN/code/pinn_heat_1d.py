# %% 0. Setup
# Week: PINN (2) -- a time-dependent problem: the heat equation
#   u_t = kappa * u_xx      for 0 < x < 1, 0 < t < 1        (kappa = 0.1)
#   u(x, 0) = sin(pi x)                                      (initial condition)
#   u(0, t) = u(1, t) = 0                                    (boundary condition)
#   Exact solution: u*(x, t) = exp(-kappa pi^2 t) sin(pi x)   (used only to measure the error)
#   New compared with pinn_poisson_1d.py: the network has TWO inputs (x, t), and there are THREE loss terms.
# Run this file from its own folder; figures are saved to ../figures/*.png
#   Cells: each '# %%' starts a cell (Run Cell = Shift+Enter in VS Code).

import os
# Windows + Anaconda: numpy (MKL) and a pip-installed PyTorch each bring their own OpenMP library,
# and using both in one program stops with "OMP: Error #15".  This line lets it run (a known workaround;
# not needed on Colab or in a clean environment).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

torch.manual_seed(0)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
# 2000 interior points and two autograd.grad calls per step: a GPU helps here (about 2 minutes on
# an RTX 3050; on a CPU it can take several times longer). On Colab: Runtime > Change runtime type > GPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

kappa = 0.1

def u_exact(x, t):
    return torch.exp(-kappa * math.pi**2 * t) * torch.sin(math.pi * x)

# %% 1. The network u_theta(x, t): R^2 -> R
class MLP(nn.Module):
    def __init__(self, n_in=2, n_out=1, width=32, depth=3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, X):           # X: (N, 2), the columns are x and t
        return self.net(X)

model = MLP().to(device)
print("number of parameters:", sum(p.numel() for p in model.parameters()))

# %% 2. Training points: interior, initial line, boundary lines
N_f, N_0, N_b = 2000, 100, 100

X_f = torch.rand(N_f, 2, device=device).requires_grad_(True)       # (x, t) inside the square

x0  = torch.rand(N_0, 1, device=device)
X_0 = torch.cat([x0, torch.zeros_like(x0)], dim=1) # (x, 0)
u_0 = torch.sin(math.pi * x0)                      # initial values

tb  = torch.rand(N_b, 1, device=device)
xb  = (torch.rand(N_b, 1, device=device) > 0.5).float() # x = 0 or x = 1
X_b = torch.cat([xb, tb], dim=1)                        # (0, t) or (1, t)
u_b = torch.zeros(N_b, 1, device=device)

fig, ax    = plt.subplots(figsize=(4.6, 4.2))
Xf, X0, Xb = X_f.detach().cpu(), X_0.cpu(), X_b.cpu()       # matplotlib needs CPU tensors
ax.plot(Xf[:, 0], Xf[:, 1], ".", ms=2, color="gray", label=f"interior ({N_f})")
ax.plot(X0[:, 0], X0[:, 1], "o", ms=3, color="C0", label=f"initial ({N_0})", clip_on=False)
ax.plot(Xb[:, 0], Xb[:, 1], "s", ms=3, color="C3", label=f"boundary ({N_b})", clip_on=False)
ax.set(xlabel="x", ylabel="t", xlim=(0, 1), ylim=(0, 1), title="Training points")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/heat_points.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 3. Residual: r = u_t - kappa u_xx
def residual(model, X):
    u = model(X)                                     # (N, 1)
    # derivative with respect to both inputs: (N, 2), the columns are [u_x, u_t]
    grads = torch.autograd.grad(u, X, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = grads[:, 0:1], grads[:, 1:2]          # (N, 1) each
    # differentiate u_x again: (N, 2) = [u_xx, u_xt]; keep the first column
    g_x = torch.autograd.grad(u_x, X, torch.ones_like(u_x), create_graph=True)[0]
    u_xx = g_x[:, 0:1]
    return u_t - kappa * u_xx

def loss_terms(model):
    loss_pde = torch.mean(residual(model, X_f)**2)
    loss_ic  = torch.mean((model(X_0) - u_0)**2)
    loss_bc  = torch.mean((model(X_b) - u_b)**2)
    return loss_pde, loss_ic, loss_bc

# evaluation grid
n_grid = 101
xg, tg = torch.meshgrid(torch.linspace(0, 1, n_grid), torch.linspace(0, 1, n_grid), indexing="xy")
X_test = torch.stack([xg.reshape(-1), tg.reshape(-1)], dim=1).to(device)    # (101*101, 2)
U_true = u_exact(X_test[:, 0:1], X_test[:, 1:2])

def rel_l2_error(model):
    with torch.no_grad():
        return (torch.norm(model(X_test) - U_true) / torch.norm(U_true)).item()

# %% 4. Training with Adam
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_steps = 5000
hist = {"step": [], "pde": [], "ic": [], "bc": [], "err": []}
t0 = time.time()
for step in range(n_steps):
    optimizer.zero_grad()
    loss_pde, loss_ic, loss_bc = loss_terms(model)
    loss = loss_pde + loss_ic + loss_bc
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        for k, v in zip(["pde", "ic", "bc"], [loss_pde, loss_ic, loss_bc]):
            hist[k].append(v.item())
        hist["step"].append(step)
        hist["err"].append(rel_l2_error(model))
    if step % 1000 == 0:
        print(f"step {step:5d}: pde {loss_pde.item():.2e}, ic {loss_ic.item():.2e}, bc {loss_bc.item():.2e},"
              f" rel. L2 error {hist['err'][-1]:.2e}  ({time.time() - t0:.0f}s)")
print(f"Adam {n_steps} steps: rel. L2 error = {rel_l2_error(model):.3e}  ({time.time() - t0:.0f}s)")
with torch.no_grad():
    print(f"max |u_theta - u*| on the grid = {(model(X_test) - U_true).abs().max().item():.2e}")


# %% 5. Plots
with torch.no_grad():
    U_pred = model(X_test).reshape(n_grid, n_grid).cpu().numpy()
U_ex = U_true.reshape(n_grid, n_grid).cpu().numpy()
ext  = [0, 1, 0, 1]

fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.8))
for a, Z, title in [(ax[0], U_pred, "PINN  u_theta(x, t)"), (ax[1], U_ex, "exact  u*(x, t)")]:
    im = a.imshow(Z, origin="lower", extent=ext, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    a.set(xlabel="x", ylabel="t", title=title)
    fig.colorbar(im, ax=a)
E = U_pred - U_ex
emax = np.abs(E).max()
im = ax[2].imshow(E, origin="lower", extent=ext, aspect="auto", cmap="RdBu_r", vmin=-emax, vmax=emax)
ax[2].set(xlabel="x", ylabel="t", title="error  u_theta - u*")
fig.colorbar(im, ax=ax[2])
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/heat_field.png", dpi=200, bbox_inches="tight")
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
xs = np.linspace(0, 1, n_grid)
for k, t_val in enumerate([0.0, 0.25, 0.5, 1.0]):
    j = int(round(t_val * (n_grid - 1)))
    ax[0].plot(xs, U_ex[j], "-", lw=4, alpha=0.25, color=f"C{k}")
    ax[0].plot(xs, U_pred[j], "--", color=f"C{k}", label=f"t = {t_val}")
ax[0].set(xlabel="x", ylabel="u", title="Snapshots (thick: exact, dashed: PINN)")
ax[0].legend(); ax[0].grid(True, alpha=0.3)
for k, lab in [("pde", "PDE term"), ("ic", "initial term"), ("bc", "boundary term"), ("err", "rel. L2 error")]:
    ax[1].semilogy(hist["step"], hist[k], "k-" if k == "err" else "-", lw=1 if k == "err" else 1.5, label=lab)
ax[1].set(xlabel="Adam step", title="Loss terms and error")
ax[1].legend(); ax[1].grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/heat_snapshots.png", dpi=200, bbox_inches="tight")
plt.show()
