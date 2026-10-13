# Week: Techniques for Training -- parameter update (optimizers)
# SGD, Momentum, AdaGrad, RMSProp, Adam on  f(x, y) = x^2 / 20 + y^2
# The optimizer classes are in optimizers.py (same folder).
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt

from optimizers import SGD, Momentum, AdaGrad, RMSProp, Adam   # optimizers.py

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# =====================================================
# The test function  f(x, y) = x^2/20 + y^2  and its gradient
# =====================================================
def f(x, y):
    return x ** 2 / 20.0 + y ** 2

def grad_f(x, y):
    return x / 10.0, 2.0 * y

def minimize(optimizer, start=(-7.0, 2.0), steps=30):
    """Run `steps` updates from `start`; return the path as an array of shape (steps+1, 2)."""
    params = {"x": np.array(start[0]), "y": np.array(start[1])}
    path = [(float(params["x"]), float(params["y"]))]
    for _ in range(steps):
        gx, gy = grad_f(params["x"], params["y"])
        grads = {"x": gx, "y": gy}
        optimizer.update(params, grads)
        path.append((float(params["x"]), float(params["y"])))
    return np.array(path)

# grid for contour plots
xs = np.linspace(-10, 10, 300)
ys = np.linspace(-5, 5, 300)
XX, YY = np.meshgrid(xs, ys)
ZZ = f(XX, YY)

def draw_path(ax, path, title):
    ax.contour(XX, YY, ZZ, levels=np.linspace(0.05, 12, 14) ** 1.5 / 4, colors="gray", linewidths=0.5)
    ax.plot(path[:, 0], path[:, 1], "o-", color="red", markersize=3, linewidth=1)
    ax.plot(0, 0, "+", color="black", markersize=12, markeredgewidth=1.5)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)

# =====================================================
# Figure 1: the function
# =====================================================
fig = plt.figure(figsize=(11, 4))
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.plot_surface(XX, YY, ZZ, cmap="viridis", linewidth=0, alpha=0.9)
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_title(r"$f(x,y)=x^2/20+y^2$")
ax2 = fig.add_subplot(1, 2, 2)
ax2.contour(XX, YY, ZZ, levels=np.linspace(0.05, 12, 14) ** 1.5 / 4, colors="gray", linewidths=0.6)
gx, gy = grad_f(XX[::25, ::25], YY[::25, ::25])
ax2.quiver(XX[::25, ::25], YY[::25, ::25], -gx, -gy, color="tab:blue", width=0.003)
ax2.plot(0, 0, "+", color="black", markersize=12, markeredgewidth=1.5)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_title("contours and  -gradient")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "opt_function.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Figure 2: SGD with three learning rates
# =====================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, lr in zip(axes, [0.5, 0.95, 1.02]):
    path = minimize(SGD(lr=lr))
    draw_path(ax, path, f"SGD, lr = {lr}")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "opt_sgd_lr.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Figure 3: four optimizers, 30 steps from (-7, 2)
# =====================================================
optimizers = {
    "SGD (lr=0.95)":                SGD(lr=0.95),
    "Momentum (lr=0.1, beta=0.9)":  Momentum(lr=0.1, beta=0.9),
    "AdaGrad (lr=1.5)":             AdaGrad(lr=1.5),
    "Adam (lr=0.3)":                Adam(lr=0.3),
}
paths = {name: minimize(opt) for name, opt in optimizers.items()}

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, path) in zip(axes.ravel(), paths.items()):
    draw_path(ax, path, name)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "opt_compare.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Figure 4: f(x_k, y_k) versus k
# =====================================================
fig, ax = plt.subplots(figsize=(6.5, 4))
for name, path in paths.items():
    ax.semilogy(f(path[:, 0], path[:, 1]), "o-", markersize=3, label=name.split(" (")[0])
ax.set_xlabel("iteration k")
ax.set_ylabel(r"$f(x_k, y_k)$")
ax.set_title("value of f along the path")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "opt_loss.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Figure 5: AdaGrad vs RMSProp (AdaGrad slows down, RMSProp does not)
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
draw_path(axes[0], minimize(AdaGrad(lr=0.5), steps=60), "AdaGrad (lr=0.5), 60 steps")
draw_path(axes[1], minimize(RMSProp(lr=0.5, rho=0.9), steps=60), "RMSProp (lr=0.5, rho=0.9), 60 steps")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "opt_adagrad_rmsprop.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Print the first steps and the final points
# =====================================================
for name, path in paths.items():
    print(f"{name:30s} step1=({path[1, 0]:.4f}, {path[1, 1]:.4f})  "
          f"step2=({path[2, 0]:.4f}, {path[2, 1]:.4f})  "
          f"final=({path[-1, 0]:.4f}, {path[-1, 1]:.4f})  f={f(*path[-1]):.2e}")
print("saved figures: opt_function, opt_sgd_lr, opt_compare, opt_loss, opt_adagrad_rmsprop")
