# %%
# What are the EIT data?  One current-voltage pair (g_1, f_1) on the boundary of the unit disk.
#   input : current  g_1(theta) = cos(theta) = sigma du/dn on the boundary
#   output: voltage  f_1(theta) = u on the boundary, for the true sigma (same data as pinn_eit.py)
# For sigma = 1 the voltage would be exactly cos(theta) (u = r cos(theta)); the difference carries the information about sigma.
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import fem_disk as fem

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

def sigma_true(x, y):      # same as pinn_eit.py
    return (1 + 1.5 * np.exp(-((x - 0.35)**2 + (y - 0.2)**2) / (2 * 0.15**2))
              - 0.5 * np.exp(-((x + 0.35)**2 + (y + 0.25)**2) / (2 * 0.18**2)))

# %% 1. Voltage for the current g_1 = cos(theta) (finite elements, as in pinn_eit.py)
g1 = lambda th: np.cos(th)
p, t, bnd = fem.disk_mesh(h=0.01)
u = fem.solve_neumann(p, t, bnd, sigma_true, g1)
th = np.mod(np.arctan2(p[bnd, 1], p[bnd, 0]), 2 * np.pi)
order = np.argsort(th); th = th[order]
f1 = u[bnd][order]                     # measured voltage, true sigma
f1_hom = np.cos(th)                    # voltage for sigma = 1
print(f"{len(th)} boundary nodes;  max |f_1| = {np.abs(f1).max():.4f},  max |f_1 - cos| = {np.abs(f1 - f1_hom).max():.4f}")

# %% 2. Profiles against the central angle
deg = np.degrees(th)
fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].plot(deg, g1(th), color="tab:red")
ax[0].set_title(r"input: current $g_1(\theta)=\sigma\,\partial_n u=\cos\theta$")
ax[1].plot(deg, f1, color="tab:blue", label=r"true $\sigma$ (data)")
ax[1].plot(deg, f1_hom, "k--", lw=1, label=r"if $\sigma\equiv1$: $\cos\theta$")
ax[1].set_title(r"output: voltage $f_1(\theta)=u|_{\partial\Omega}$")
ax[1].legend(fontsize=8)
ax[2].plot(deg, f1 - f1_hom, color="tab:blue")
ax[2].set_title(r"$f_1-\cos\theta$ (effect of the inclusions)")
for a in ax:
    a.set_xlabel(r"central angle $\theta$ (degrees)")
    a.set_xticks(range(0, 361, 90)); a.set_xlim(0, 360)
    a.axhline(0, color="gray", lw=0.5); a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/eit_data_theta.png", dpi=200, bbox_inches="tight")
plt.show()

# %% 3. The same pair in 3D, drawn around the phantom
n = 240
r, a = np.meshgrid(np.linspace(0, 1, 60), np.linspace(0, 2 * np.pi, n))
X, Y = r * np.cos(a), r * np.sin(a)
S = sigma_true(X, Y)
norm = colors.Normalize(S.min(), S.max())
z0 = -1.7                                           # height of the phantom (floor)
ring = np.linspace(0, 2 * np.pi, 400)
fig = plt.figure(figsize=(12, 5.2))
for k, (vals, title, col) in enumerate([(g1(th), r"input: current $g_1=\cos\theta$", "tab:red"),
                                        (f1, r"output: measured voltage $f_1$", "tab:blue")]):
    ax3 = fig.add_subplot(1, 2, k + 1, projection="3d")
    ax3.plot_surface(X, Y, np.full_like(X, z0), facecolors=cm.viridis(norm(S)), shade=False,
                     rstride=2, cstride=2, linewidth=0, antialiased=False)
    ax3.plot(np.cos(ring), np.sin(ring), 0 * ring, color="gray", lw=0.8, ls="--")          # zero level
    ax3.plot(np.cos(th), np.sin(th), vals, color=col, lw=2.2)                               # profile
    for i in range(0, len(th), 16):                                                          # stems
        ax3.plot([np.cos(th[i])] * 2, [np.sin(th[i])] * 2, [0, vals[i]], color=col, lw=0.6, alpha=0.6)
    ax3.plot(np.cos(ring), np.sin(ring), np.full_like(ring, z0), color="k", lw=0.8)
    ax3.text(1.25, 0, z0, r"$\theta=0$", fontsize=9)
    ax3.set_title(title)
    ax3.set_zlim(z0, 1.2); ax3.set_xticks([]); ax3.set_yticks([]); ax3.set_zticks([-1, 0, 1])
    ax3.view_init(elev=22, azim=-60)
sm = cm.ScalarMappable(norm=norm, cmap="viridis")
fig.colorbar(sm, ax=fig.axes, shrink=0.55, pad=0.02, label=r"true $\sigma$ (floor)")
fig.savefig(f"{FIG_DIR}/eit_data_3d.png", dpi=200, bbox_inches="tight")
plt.show()
