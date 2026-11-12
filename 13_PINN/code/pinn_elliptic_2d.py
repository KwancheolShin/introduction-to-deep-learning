# %% 0. Setup
# A 2D elliptic equation on the unit disk:
#   -div( sigma(x, y) grad u ) = 0  in  x^2 + y^2 < 1,     u = cos(theta)  on the boundary.
#   sigma is a known conductivity with two smooth bumps (one high, one low).
#   Reference solution: finite elements (fem_disk.py).  Figures: ../figures/elliptic_*.png
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
import fem_disk as fem

torch.manual_seed(0)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# %% 1. The conductivity sigma (the same formula for numpy and torch)
def sigma(x, y, lib=torch):
    return (1 + 1.5 * lib.exp(-((x - 0.35)**2 + (y - 0.2)**2) / (2 * 0.15**2))
              - 0.5 * lib.exp(-((x + 0.35)**2 + (y + 0.25)**2) / (2 * 0.18**2)))

# reference solution by finite elements
p, t, bnd = fem.disk_mesh(h=0.01)
th_b = np.arctan2(p[bnd, 1], p[bnd, 0])
u_fem = fem.solve_dirichlet(p, t, bnd, lambda x, y: sigma(x, y, np), np.cos(th_b))
print("FEM nodes:", len(p))

# %% 2. Network u_theta(x, y): R^2 -> R
model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh(),
                      nn.Linear(64, 64), nn.Tanh(), nn.Linear(64, 1)).to(device)

# %% 3. Points: many interior points (used in minibatches) and boundary points
N_f, batch = 20000, 1000
r  = torch.sqrt(torch.rand(N_f, 1, device=device))              # sqrt: uniform in the disk
th = 2 * math.pi * torch.rand(N_f, 1, device=device)
X_f = torch.cat([r * torch.cos(th), r * torch.sin(th)], dim=1)  # (20000, 2)
N_b = 2000
th_bd = 2 * math.pi * torch.rand(N_b, 1, device=device)
X_b = torch.cat([torch.cos(th_bd), torch.sin(th_bd)], dim=1)    # (2000, 2) on the circle
u_b = torch.cos(th_bd)                                          # Dirichlet data

# %% 4. Residual  div(sigma grad u)
def grad(y, X):
    return torch.autograd.grad(y, X, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def residual(X):
    u = model(X)                                     # (N, 1)
    s = sigma(X[:, 0:1], X[:, 1:2])                  # (N, 1)
    flux = s * grad(u, X)                            # (N, 2) = sigma * (u_x, u_y)
    div = grad(flux[:, 0:1], X)[:, 0:1] + grad(flux[:, 1:2], X)[:, 1:2]   # d/dx (sigma u_x) + d/dy (sigma u_y)
    return div

# %% 5. Training
lam_b   = 10.0
n_steps = 5000
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_steps // 3, gamma=0.3)   # lr x 0.3 every 1/3 of training
P         = torch.tensor(p, dtype=torch.float32, device=device)
U_fem     = torch.tensor(u_fem, dtype=torch.float32, device=device)
hist = {"step": [], "pde": [], "bc": [], "err": []}

t0 = time.time()
for step in range(n_steps + 1):
    X        = X_f[torch.randint(0, N_f, (batch,), device=device)].clone().requires_grad_(True)   # minibatch of interior points
    jb       = torch.randint(0, N_b, (256,), device=device)
    loss_pde = torch.mean(residual(X)**2)
    loss_bc  = torch.mean((model(X_b[jb]) - u_b[jb])**2)
    loss     = loss_pde + lam_b * loss_bc
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 100 == 0:
        with torch.no_grad():
            err = (torch.norm(model(P).ravel() - U_fem) / torch.norm(U_fem)).item()
        hist["step"].append(step); hist["pde"].append(loss_pde.item()); hist["bc"].append(loss_bc.item()); hist["err"].append(err)
    if step % 1000 == 0:
        print(f"step {step:5d}: pde {loss_pde.item():.2e}, bc {loss_bc.item():.2e}, rel. L2 error {err:.2e}  ({time.time() - t0:.0f}s)")

with torch.no_grad():
    u_pinn = model(P).cpu().numpy().ravel()
print(f"rel. L2 error (vs FEM, at {len(p)} nodes) = {np.linalg.norm(u_pinn - u_fem) / np.linalg.norm(u_fem):.2e},"
      f" max abs error = {np.abs(u_pinn - u_fem).max():.2e}")

# %% 6. Plots
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
s_nodes = sigma(p[:, 0], p[:, 1], np)
for a, val, title, cmap in [(ax[0], s_nodes, "conductivity sigma", "viridis"),
                            (ax[1], u_pinn, "PINN u", "RdBu_r"),
                            (ax[2], u_fem, "FEM u (reference)", "RdBu_r")]:
    im = a.tripcolor(p[:, 0], p[:, 1], t, val, shading="gouraud", cmap=cmap)
    fig.colorbar(im, ax=a); a.set_title(title)
emax = np.abs(u_pinn - u_fem).max()
im = ax[3].tripcolor(p[:, 0], p[:, 1], t, u_pinn - u_fem, shading="gouraud", cmap="RdBu_r", vmin=-emax, vmax=emax)
fig.colorbar(im, ax=ax[3]); ax[3].set_title("error  PINN - FEM")
for a in ax:
    a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/elliptic_field.png", dpi=200, bbox_inches="tight")
plt.show()

fig, a = plt.subplots(figsize=(5.5, 3.6))
a.semilogy(hist["step"], hist["pde"], label="PDE term")
a.semilogy(hist["step"], hist["bc"], "--", label="boundary term")
a.semilogy(hist["step"], hist["err"], "k-", lw=1, label="rel. L2 error")
a.set(xlabel="step", title="Training"); a.legend(); a.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/elliptic_loss.png", dpi=200, bbox_inches="tight")
plt.show()
