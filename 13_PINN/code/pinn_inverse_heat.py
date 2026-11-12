# %% 0. Setup
# Week: PINN (2) -- an INVERSE problem: find the unknown coefficient kappa from measurements
#   We know the form of the law:  u_t = kappa * u_xx   (0 < x < 1, 0 < t < 1),
#   but NOT kappa, and we do not use the initial or boundary conditions.
#   We have N_d noisy measurements  u_obs = u*(x_i, t_i) + noise,  where u* uses the true kappa = 0.1.
#   Unknowns: the network weights theta AND kappa.     loss = data term + PDE term
#   Experiment: the noise level sigma = 0, 0.01, 0.05.
# Run this file from its own folder; figures are saved to ../figures/*.png
#   Cells: each '# %%' starts a cell (Run Cell = Shift+Enter in VS Code).

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
# 2000 interior points and two autograd.grad calls per step: a GPU helps here (about 2 minutes on
# an RTX 3050; on a CPU it can take several times longer). On Colab: Runtime > Change runtime type > GPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")

kappa_true = 0.1

def u_exact(x, t):
    return torch.exp(-kappa_true * math.pi**2 * t) * torch.sin(math.pi * x)

class MLP(nn.Module):
    def __init__(self, n_in=2, n_out=1, width=32, depth=3):
        super().__init__()
        layers, d = [], n_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.Tanh()]
            d = width
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, X):
        return self.net(X)

# %% 1. One run: network + unknown kappa
def run(sigma, N_d=200, N_f=2000, n_steps=5000, kappa_init=0.5, seed=0):
    torch.manual_seed(seed)
    # measurements
    X_d = torch.rand(N_d, 2, device=device)
    u_d = u_exact(X_d[:, 0:1], X_d[:, 1:2]) + sigma * torch.randn(N_d, 1, device=device)
    # collocation points
    X_f = torch.rand(N_f, 2, device=device).requires_grad_(True)

    model = MLP().to(device)
    kappa = nn.Parameter(torch.tensor(kappa_init, device=device))   # trainable, like a weight
    optimizer = torch.optim.Adam(list(model.parameters()) + [kappa], lr=1e-3)

    kappa_hist = []
    for step in range(n_steps):
        optimizer.zero_grad()
        loss_data = torch.mean((model(X_d) - u_d)**2)
        u = model(X_f)
        g = torch.autograd.grad(u, X_f, torch.ones_like(u), create_graph=True)[0]
        u_x, u_t = g[:, 0:1], g[:, 1:2]
        u_xx = torch.autograd.grad(u_x, X_f, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        loss_pde = torch.mean((u_t - kappa * u_xx)**2)       # kappa enters ONLY here
        loss = loss_data + loss_pde
        loss.backward()                                       # fills model weights' .grad AND kappa.grad
        optimizer.step()
        kappa_hist.append(kappa.item())
    return model, kappa_hist, X_d, u_d

# %% 2. Three noise levels
results = {}
for sigma in [0.0, 0.01, 0.05]:
    t0 = time.time()
    model, kh, X_d, u_d = run(sigma)
    results[sigma] = (model, kh, X_d, u_d)
    err = abs(kh[-1] - kappa_true) / kappa_true
    print(f"sigma = {sigma:4.2f}: kappa = {kh[-1]:.5f}  (true {kappa_true}, rel. error {err:.2%})  ({time.time() - t0:.0f}s)")

# %% 3. Plots: kappa during training, and the measurements
fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9))
for k, (sigma, (model, kh, X_d, u_d)) in enumerate(results.items()):
    ax[0].plot(kh, color=f"C{k}", label=f"noise sigma = {sigma}  (final {kh[-1]:.4f})")
ax[0].axhline(kappa_true, color="k", ls="--", lw=1, label="true kappa = 0.1")
ax[0].set(xlabel="Adam step", ylabel="kappa", title="Estimated kappa during training")
ax[0].legend(fontsize=8); ax[0].grid(True, alpha=0.3)

_, _, X_d, u_d = results[0.05]
X_d, u_d = X_d.cpu(), u_d.cpu()
sc = ax[1].scatter(X_d[:, 0], X_d[:, 1], c=u_d.ravel(), s=14, cmap="viridis")
ax[1].set(xlabel="x", ylabel="t", xlim=(0, 1), ylim=(0, 1), title="Measurements (sigma = 0.05)")
fig.colorbar(sc, ax=ax[1], label="u_obs")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/inverse_kappa.png", dpi=200, bbox_inches="tight")
plt.show()
