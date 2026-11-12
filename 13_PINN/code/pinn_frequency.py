# %% 0. Setup
# A limitation: high frequencies are hard.
#   -u'' = (k pi)^2 sin(k pi x) on (0, 1),  u(0) = u(1) = 0,   exact solution u* = sin(k pi x).
#   Cell 1: k = 1, 2, 4, 8 with the same network, points (64), optimizer and steps (5000) as pinn_poisson_1d.py.
#   Cell 3: k = 8 again, with longer training and with many collocation points used in minibatches.
#   The residual is divided by (k pi)^2 so that the loss has the same scale for every k.
# Figures: ../figures/frequency.png, frequency_loss.png, frequency_k8.png
import os
# Windows + Anaconda: numpy (MKL) and a pip-installed PyTorch each bring their own OpenMP library,
# and using both in one program stops with "OMP: Error #15".  This line lets it run (a known workaround;
# not needed on Colab or in a clean environment).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import math
import matplotlib.pyplot as plt
import torch
from torch import nn

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using", device)

def make_net():
    return nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 32), nn.Tanh(),
                         nn.Linear(32, 32), nn.Tanh(), nn.Linear(32, 1)).to(device)

def derivative(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def scaled_residual(model, x, k):
    c = (k * math.pi)**2
    u = model(x)
    u_xx = derivative(derivative(u, x), x)
    return (-u_xx - c * torch.sin(k * math.pi * x)) / c

x_test = torch.linspace(0, 1, 1001, device=device).reshape(-1, 1)

def solve(k, n_steps=5000, N_f=64, batch=None, seed=0):
    """N_f collocation points are drawn once.  batch=None: use all of them at every step.
    batch=B: every step uses a random minibatch of B points out of the N_f points."""
    torch.manual_seed(seed)
    model = make_net()
    x_pool = torch.rand(N_f, 1, device=device)
    x_b = torch.tensor([[0.0], [1.0]], device=device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_hist = []
    for step in range(n_steps):
        if batch is None:
            x_f = x_pool
        else:
            x_f = x_pool[torch.randint(0, N_f, (batch,), device=device)]
        x_f = x_f.clone().requires_grad_(True)
        opt.zero_grad()
        r = scaled_residual(model, x_f, k)
        loss = torch.mean(r**2) + torch.mean(model(x_b)**2)
        loss.backward()
        opt.step()
        loss_hist.append(loss.item())

    # residual on all training points (train) and on 1001 points of a fine grid (test)
    x_tr = x_pool.clone().requires_grad_(True)
    x_te = x_test.clone().requires_grad_(True)
    res_train = torch.mean(scaled_residual(model, x_tr, k)**2).item()
    res_test  = torch.mean(scaled_residual(model, x_te, k)**2).item()
    with torch.no_grad():
        u_pred = model(x_test)
        u_true = torch.sin(k * math.pi * x_test)
        err = (torch.norm(u_pred - u_true) / torch.norm(u_true)).item()
    return dict(u_pred=u_pred.cpu().numpy().ravel(), u_true=u_true.cpu().numpy().ravel(), err=err,
                loss_hist=loss_hist, x_f=x_pool.cpu().numpy().ravel(), res_train=res_train, res_test=res_test)

# %% 1. Run k = 1, 2, 4, 8 (64 points, 5000 steps)
ks = [1, 2, 4, 8]
out = {}
for k in ks:
    t0 = time.time()
    out[k] = solve(k)
    print(f"k = {k}: final loss = {out[k]['loss_hist'][-1]:.2e}, rel. L2 error = {out[k]['err']:.2e}  ({time.time() - t0:.0f}s)")

# %% 2. Plot
xt = x_test.cpu().numpy().ravel()
fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
for a, k in zip(ax, ks):
    o = out[k]
    a.plot(xt, o["u_true"], "k-", lw=4, alpha=0.25, label="exact")
    a.plot(xt, o["u_pred"], "r-", lw=1.2, label="PINN")
    a.plot(o["x_f"], -1.3 + 0 * o["x_f"], "|", color="gray", ms=6)
    a.set_title(f"k = {k}:  sin({k} pi x)\nrel. L2 error = {o['err']:.1e}")
    a.set_xlabel("x"); a.grid(True, alpha=0.3)
for a in ax[:3]:
    a.set_ylim(-1.45, 1.6)
ax[3].set_ylim(min(-1.45, out[8]["u_pred"].min() - 0.2), max(1.6, out[8]["u_pred"].max() + 0.2))   # k = 8 may leave the range of the others
ax[0].legend(loc="upper right", fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/frequency.png", dpi=200, bbox_inches="tight")
plt.show()

fig, a = plt.subplots(figsize=(5.5, 3.6))
for k in ks:
    a.semilogy(out[k]["loss_hist"], lw=1, label=f"k = {k}")
a.set(xlabel="Adam step", ylabel="loss (scaled)", title="Training loss for each frequency")
a.legend(); a.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/frequency_loss.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 3. k = 8: more steps, and many collocation points in minibatches (3 seeds each)
settings = [
    ("64 points, 5000 steps",                         dict(n_steps=5000,  N_f=64)),
    ("64 points, 20000 steps",                        dict(n_steps=20000, N_f=64)),
    ("10000 points, batch 256, 20000 steps",          dict(n_steps=20000, N_f=10000, batch=256)),
]
seeds = [0, 1, 2]
res8 = {}
for name, kw in settings:
    for s in seeds:
        t0 = time.time()
        res8[name, s] = o = solve(8, seed=s, **kw)
        print(f"{name:38s} seed {s}: residual^2 train {o['res_train']:.1e}, test {o['res_test']:.1e},"
              f" rel. L2 error {o['err']:.2e}  ({time.time() - t0:.0f}s)")

fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), sharey=True)
for a, (name, kw) in zip(ax, settings):
    a.plot(xt, res8[name, 0]["u_true"], "k-", lw=4, alpha=0.25, label="exact")
    for s, color in zip(seeds, ["tab:red", "tab:blue", "tab:green"]):
        a.plot(xt, res8[name, s]["u_pred"], color=color, lw=1, label=f"seed {s}: error {res8[name, s]['err']:.2g}")
    a.set_title(name); a.set_xlabel("x"); a.grid(True, alpha=0.3)
    a.legend(fontsize=7, loc="upper left")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/frequency_k8.png", dpi=200, bbox_inches="tight")
plt.show()
