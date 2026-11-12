# %% 0. Setup
#   Solve  -u''(x) = f(x)  on 0 < x < 1,   u(0) = u(1) = 0,   with f(x) = pi^2 sin(pi x).
#   The exact solution is u*(x) = sin(pi x).  We NEVER show u* to the network: it is used only to measure the error.
# Run this file from its own folder; figures are saved to ../figures/*.png
#   Cells: each '# %%' starts a cell. In VS Code, Run Cell (Shift+Enter) sends one cell to the
#   Interactive window (variables stay in the kernel); 'python <file>' still runs the whole file.

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
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

# %% 1. The problem
def f(x):                                   # right-hand side (source term)
    return math.pi**2 * torch.sin(math.pi * x)

def u_exact(x):                             # only for measuring the error
    return torch.sin(math.pi * x)

# %% 2. The network u_theta(x): R -> R
class MLP(nn.Module):
    """Fully connected network with tanh: smooth, so u'' exists (with ReLU, u'' = 0 almost everywhere)."""
    def __init__(self, n_in=1, n_out=1, width=32, depth=3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

model = MLP().to(device)
print(model)
print("number of parameters:", sum(p.numel() for p in model.parameters()))

# %% 3. Training points: no labels, only locations
N_f = 64
x_f = torch.rand(N_f, 1, device=device)            # collocation points in (0, 1): (64, 1)
x_f.requires_grad_(True)                            # we will differentiate u with respect to x_f
x_b = torch.tensor([[0.0], [1.0]], device=device)   # boundary points: (2, 1)
u_b = torch.zeros(2, 1, device=device)              # boundary values u(0) = u(1) = 0

# %% 4. The PDE residual r(x) = -u''(x) - f(x)
def derivative(y, x):
    """dy/dx for every point. y: (N, 1) computed from x: (N, 1).
    grad_outputs=ones: the i-th output depends only on the i-th input, so the sum trick returns
    one derivative per point. create_graph=True: keep the graph so that we can differentiate again."""
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def residual(model, x):
    u    = model(x)                 # (N, 1)   the original graph
    u_x  = derivative(u, x)         # (N, 1)   a new graph, built on top of the first
    u_xx = derivative(u_x, x)       # (N, 1)   and another one
    return -u_xx - f(x)

# %% 5. The loss: physics + boundary condition
lam_b = 1.0                                              # weight of the boundary term

def loss_terms(model):
    r        = residual(model, x_f)
    loss_pde = torch.mean(r**2)
    loss_bc  = torch.mean((model(x_b) - u_b)**2)
    return loss_pde, loss_bc

x_test = torch.linspace(0, 1, 1001, device=device).reshape(-1, 1)

def rel_l2_error(model):
    with torch.no_grad():
        u_pred = model(x_test)
        u_true = u_exact(x_test)
        return (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()

# %% 6. Training with Adam (re-run cell 2 first to start from scratch)
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)
n_steps    = 5000
hist       = {"step": [], "loss": [], "pde": [], "bc": [], "err": []}
snapshots  = {}    # u_theta(x) at a few steps, for a figure
snap_steps = [0, 25, 50, 100, 200, 1000]

t0 = time.time()
for step in range(n_steps + 1):
    if step in snap_steps:
        with torch.no_grad():
            snapshots[step] = model(x_test).cpu().numpy().ravel()
    if step == n_steps:
        break

    optimizer.zero_grad()
    loss_pde, loss_bc = loss_terms(model)
    loss = loss_pde + lam_b * loss_bc
    loss.backward()               # d(loss)/d(theta), also through the derivative graphs
    optimizer.step()

    if step % 50 == 0:
        hist["step"].append(step)
        hist["loss"].append(loss.item())
        hist["pde"].append(loss_pde.item())
        hist["bc"].append(loss_bc.item())
        hist["err"].append(rel_l2_error(model))
    if step % 1000 == 0:
        print(f"step {step:5d}: loss={loss.item():.3e} (pde {loss_pde.item():.3e}, bc {loss_bc.item():.3e}),"
              f" rel. L2 error={hist['err'][-1]:.3e}  ({time.time() - t0:.1f}s)")

print(f"Adam, {n_steps} steps: rel. L2 error = {rel_l2_error(model):.3e},  time {time.time() - t0:.1f}s")

# %% 7. Plots: loss, solution, error, residual
x_np = x_test.cpu().numpy().ravel()
with torch.no_grad():
    u_pred = model(x_test).cpu().numpy().ravel()
u_true = u_exact(x_test).cpu().numpy().ravel()
x_fine = x_test.clone().requires_grad_(True)
r_fine = residual(model, x_fine).detach().cpu().numpy().ravel()

fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
ax[0, 0].semilogy(hist["step"], hist["loss"], label="total loss")
ax[0, 0].semilogy(hist["step"], hist["pde"], "--", label="PDE term")
ax[0, 0].semilogy(hist["step"], hist["bc"], ":", label="boundary term")
ax[0, 0].semilogy(hist["step"], hist["err"], "k-", lw=1, label="rel. L2 error")
ax[0, 0].set(xlabel="step", title="(a) Loss terms and error")
ax[0, 0].legend(); ax[0, 0].grid(True, which="both", alpha=0.3)

ax[0, 1].plot(x_np, u_true, "k-", lw=3, alpha=0.3, label="exact  sin(pi x)")
ax[0, 1].plot(x_np, u_pred, "r--", label="PINN")
ax[0, 1].plot(x_f.detach().cpu().numpy(), np.zeros(N_f), "|", color="gray", ms=10, label="collocation points")
ax[0, 1].set(xlabel="x", ylabel="u", title="(b) Solution")
ax[0, 1].legend(); ax[0, 1].grid(True, alpha=0.3)

ax[1, 0].plot(x_np, u_pred - u_true)
ax[1, 0].set(xlabel="x", ylabel="u_theta - u*", title="(c) Error")
ax[1, 0].grid(True, alpha=0.3)

ax[1, 1].plot(x_np, r_fine)
ax[1, 1].set(xlabel="x", ylabel="r(x)", title="(d) PDE residual  -u'' - f")
ax[1, 1].grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/poisson_result.png", dpi=200, bbox_inches="tight")
plt.show()

# how the solution develops during training
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(x_np, u_true, "k-", lw=4, alpha=0.25, label="exact")
for k, st in enumerate(snap_steps):
    ax.plot(x_np, snapshots[st], color=plt.cm.viridis(k / (len(snap_steps) - 1)), label=f"step {st}")
ax.set(xlabel="x", ylabel="u_theta(x)", title="PINN during training")
ax.legend(); ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/poisson_snapshots.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 8. (Optional) Polish with L-BFGS, a quasi-Newton method
lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=500, line_search_fn="strong_wolfe")

def closure():                    # L-BFGS evaluates the loss several times per step
    lbfgs.zero_grad()
    loss_pde, loss_bc = loss_terms(model)
    loss = loss_pde + lam_b * loss_bc
    loss.backward()
    return loss

t0 = time.time()
lbfgs.step(closure)
loss_pde, loss_bc = loss_terms(model)
print(f"after L-BFGS: loss={(loss_pde + lam_b * loss_bc).item():.3e},"
      f" rel. L2 error = {rel_l2_error(model):.3e}  ({time.time() - t0:.1f}s)")

# %% 9. For comparison: the classical finite difference method on the same problem
#   -(u[i-1] - 2 u[i] + u[i+1]) / dx^2 = f(x_i),  i = 1..n-1,   u[0] = u[n] = 0   (a tridiagonal linear system)
n = 100
xg = torch.linspace(0, 1, n + 1, dtype=torch.float64)
dx = 1.0 / n
t0 = time.time()
A = (torch.diag(2 * torch.ones(n - 1)) - torch.diag(torch.ones(n - 2), 1) - torch.diag(torch.ones(n - 2), -1)).double() / dx**2
u_fd = torch.zeros(n + 1, dtype=torch.float64)
u_fd[1:-1] = torch.linalg.solve(A, f(xg[1:-1]))
t_fd = time.time() - t0
err_fd = (torch.norm(u_fd - u_exact(xg)) / torch.norm(u_exact(xg))).item()
print(f"finite differences, {n} intervals: rel. L2 error = {err_fd:.3e}  ({t_fd * 1000:.1f} ms)")
