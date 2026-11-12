# %%
# PINN, the bare minimum:  -u''(x) = pi^2 sin(pi x) on (0, 1),  u(0) = u(1) = 0.  Exact solution: u*(x) = sin(pi x).
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")   # Windows + Anaconda OpenMP clash (see pinn_poisson_1d.py)
import math
import torch
from torch import nn
import matplotlib.pyplot as plt

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# network u_theta: R -> R, smooth activation (tanh) so that u'' exists
model = nn.Sequential(nn.Linear(1, 32), 
                      nn.Tanh(), 
                      nn.Linear(32, 32), 
                      nn.Tanh(), 
                      nn.Linear(32, 32), 
                      nn.Tanh(), 
                      nn.Linear(32, 1)).to(device)

# training points: locations only, no labels
x_f = torch.rand(64, 1, device=device, requires_grad=True)     # collocation points in (0, 1): (64, 1); we differentiate u with respect to them
x_b = torch.tensor([[0.0], [1.0]], device=device)              # boundary points: (2, 1)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(5000):
    u    = model(x_f)                                                                          # (64, 1)  original graph
    u_x  = torch.autograd.grad(u,   x_f, grad_outputs=torch.ones_like(u),   create_graph=True)[0]   # (64, 1)  du/dx
    u_xx = torch.autograd.grad(u_x, x_f, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]   # (64, 1)  d2u/dx2
    r    = -u_xx - math.pi**2 * torch.sin(math.pi * x_f)                                        # residual of -u'' = f

    loss = torch.mean(r**2) + torch.mean(model(x_b)**2)        # PDE term + boundary term (u(0) = u(1) = 0)

    optimizer.zero_grad()
    loss.backward()            # d(loss)/d(theta), through the original and the derivative graphs
    optimizer.step()
    if step % 1000 == 0:
        print(f"step {step:5d}: loss {loss.item():.3e}")

# compare with the exact solution (used only here, never in training)
x = torch.linspace(0, 1, 1001, device=device).reshape(-1, 1)
with torch.no_grad():
    u_pred = model(x)
u_true = torch.sin(math.pi * x)
print("relative L2 error:", (torch.norm(u_pred - u_true) / torch.norm(u_true)).item())

x, u_true, u_pred = x.cpu(), u_true.cpu(), u_pred.cpu()   # matplotlib needs CPU tensors
plt.plot(x, u_true, "k-", lw=3, alpha=0.4, label="exact sin(pi x)")
plt.plot(x, u_pred, "r--", label="PINN")
plt.xlabel("x"); plt.ylabel("u"); plt.legend(); plt.show()
