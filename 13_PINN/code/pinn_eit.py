# %% 0. Setup
# Electrical impedance tomography (EIT) with a PINN, on the unit disk.
#   Unknown conductivity sigma(x, y) > 0.  For j = 1..J the potential u_j satisfies
#       div( sigma grad u_j ) = 0  in the disk,
#   and on the boundary we know the current  g_j = sigma du_j/dn  and the voltage  f_j = u_j.
#   The data are made by finite elements (fem_disk.py) from a "true" sigma that the PINN never sees.
#   Figures: ../figures/eit_sigma.png, eit_loss.png
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows + Anaconda: avoids "OMP: Error #15"
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

# %% 1. Boundary data from the true sigma (finite elements)
def sigma_true(x, y):
    return (1 + 1.5 * np.exp(-((x - 0.35)**2 + (y - 0.2)**2) / (2 * 0.15**2))
              - 0.5 * np.exp(-((x + 0.35)**2 + (y + 0.25)**2) / (2 * 0.18**2)))

gs = [g for n in range(1, 5) for g in (lambda t, n=n: np.cos(n * t), lambda t, n=n: np.sin(n * t))]
J = len(gs)                                                  # 8 current-voltage pairs
p, t, bnd = fem.disk_mesh(h=0.01)
K = fem.assemble(p, t, sigma_true)
th_n = np.mod(np.arctan2(p[bnd, 1], p[bnd, 0]), 2 * np.pi)
order = np.argsort(th_n); th_n = th_n[order]
F_n = np.stack([fem.solve_neumann(p, t, bnd, sigma_true, g, K)[bnd][order] for g in gs], axis=1)

# voltages at N_b equispaced boundary points (trigonometric interpolation of the 628 nodal values)
N_b = 4096
th_b = 2 * np.pi * np.arange(N_b) / N_b
m = len(th_n); kk = np.fft.fftfreq(m, 1 / m)
F = np.real(np.exp(1j * np.outer(th_b - th_n[0], kk)) @ (np.fft.fft(F_n, axis=0) / m))
G = np.stack([g(th_b) for g in gs], axis=1)

X_b = torch.tensor(np.c_[np.cos(th_b), np.sin(th_b)], dtype=torch.float32, device=device)   # boundary points = normals
f_b = torch.tensor(F, dtype=torch.float32, device=device)                                   # (4096, J)
g_b = torch.tensor(G, dtype=torch.float32, device=device)                                   # (4096, J)

# %% 2. Two networks: u_net: (x, y) -> (u_1, ..., u_J)  and  s_net: (x, y) -> sigma
def mlp(n_out, width, depth):
    layers, d = [], 2
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.Tanh()]
        d = width
    layers.append(nn.Linear(d, n_out))
    return nn.Sequential(*layers).to(device)

u_net = mlp(J, 64, 4)
s_net = mlp(1, 64, 3)

def grad(y, X):
    return torch.autograd.grad(y, X, grad_outputs=torch.ones_like(y), create_graph=True)[0]

# %% 3. Interior points
N_f, batch = 20000, 2000
r  = torch.sqrt(torch.rand(N_f, 1))
th = 2 * math.pi * torch.rand(N_f, 1)
X_f = torch.cat([r * torch.cos(th), r * torch.sin(th)], dim=1).to(device)

# %% 4. Loss: PDE residual + voltage data + current data
lam = 100.0
def loss_terms():
    X = X_f[torch.randint(0, N_f, (batch,), device=device)].clone().requires_grad_(True)
    U, S = u_net(X), s_net(X)
    loss_pde = 0.0
    for j in range(J):
        flux = S * grad(U[:, j:j+1], X)
        div = grad(flux[:, 0:1], X)[:, 0:1] + grad(flux[:, 1:2], X)[:, 1:2]
        loss_pde = loss_pde + torch.mean(div**2) / J
    Xb = X_b.clone().requires_grad_(True)
    Ub, Sb = u_net(Xb), s_net(Xb)
    dudn = torch.cat([(grad(Ub[:, j:j+1], Xb) * Xb).sum(dim=1, keepdim=True) for j in range(J)], dim=1)
    loss_D = torch.mean((Ub - f_b)**2)
    loss_N = torch.mean((Sb * dudn - g_b)**2)
    return loss_pde, loss_D, loss_N

# %% 5. Training
gx = np.linspace(-1, 1, 101); GX, GY = np.meshgrid(gx, gx); inside = GX**2 + GY**2 <= 1
X_grid = torch.tensor(np.c_[GX[inside], GY[inside]], dtype=torch.float32, device=device)
s_ref = sigma_true(GX[inside], GY[inside])
def sigma_error():
    with torch.no_grad():
        s = s_net(X_grid).cpu().numpy().ravel()
    return np.linalg.norm(s - s_ref) / np.linalg.norm(s_ref - 1), s

n_steps = 150000
optimizer = torch.optim.Adam(list(u_net.parameters()) + list(s_net.parameters()), lr=1e-3)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=n_steps // 3, gamma=0.3)
hist = {"step": [], "pde": [], "D": [], "N": [], "err": []}
t0 = time.time()
for step in range(n_steps + 1):
    loss_pde, loss_D, loss_N = loss_terms()
    loss = loss_pde + lam * (loss_D + loss_N)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    if step % 1000 == 0:
        err, _ = sigma_error()
        for key, val in [("step", step), ("pde", loss_pde.item()), ("D", loss_D.item()), ("N", loss_N.item()), ("err", err)]:
            hist[key].append(val)
    if step % 5000 == 0:
        print(f"step {step:6d}: pde {loss_pde.item():.2e}, voltage {loss_D.item():.2e}, current {loss_N.item():.2e},"
              f" sigma error {hist['err'][-1]:.3f}  ({time.time() - t0:.0f}s)", flush=True)

err, s_pinn = sigma_error()
print(f"relative error of sigma (w.r.t. |sigma - 1|) = {err:.3f},  range of sigma: [{s_pinn.min():.2f}, {s_pinn.max():.2f}]"
      f"  (true [{s_ref.min():.2f}, {s_ref.max():.2f}]),  time {time.time() - t0:.0f}s")

# %% 6. Plots
fig, ax = plt.subplots(1, 2, figsize=(9.5, 4))
for a, val, title in [(ax[0], s_ref, "true sigma"), (ax[1], s_pinn, "PINN sigma")]:
    Z = np.full(GX.shape, np.nan); Z[inside] = val
    im = a.imshow(Z, origin="lower", extent=[-1, 1, -1, 1], cmap="viridis", vmin=s_ref.min(), vmax=s_ref.max())
    a.set_title(title); a.set_xticks([]); a.set_yticks([])
    fig.colorbar(im, ax=a)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/eit_sigma.png", dpi=200, bbox_inches="tight")
plt.show()

fig, a = plt.subplots(figsize=(5.5, 3.6))
a.semilogy(hist["step"], hist["pde"], label="PDE term")
a.semilogy(hist["step"], hist["D"], "--", label="voltage term")
a.semilogy(hist["step"], hist["N"], ":", label="current term")
a.semilogy(hist["step"], hist["err"], "k-", lw=1, label="sigma error")
a.set(xlabel="step", title="Training"); a.legend(fontsize=8); a.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/eit_loss.png", dpi=200, bbox_inches="tight")
plt.show()
