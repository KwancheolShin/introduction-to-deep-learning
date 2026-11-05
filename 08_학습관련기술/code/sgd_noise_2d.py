# Appendix: convergence of SGD with a NOISY gradient on  f(x, y) = x^2 / 20 + y^2
#   g_k = grad f(theta_k) + noise,  noise ~ N(0, sigma^2 I)   (a stand-in for the mini-batch noise)
#   (1) constant learning rate          alpha_k = alpha0
#   (2) decaying learning rate          alpha_k = alpha0 / (1 + k / k0)   (Robbins-Monro)
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def f(x, y):
    return x ** 2 / 20.0 + y ** 2

def grad_f(x, y):
    return np.array([x / 10.0, 2.0 * y])

sigma = 1.0          # standard deviation of the gradient noise
alpha0 = 0.2
k0 = 100
steps = 5000
runs = 50            # average f over many runs to see the expected value

schedules = {
    "constant  alpha = 0.2":                 lambda k: alpha0,
    "constant  alpha = 0.05":                lambda k: 0.05,
    "decaying  alpha_k = 0.2 / (1 + k/100)": lambda k: alpha0 / (1 + k / k0),
}

curves = {}
for name, lr in schedules.items():
    rng = np.random.default_rng(0)
    F = np.zeros((runs, steps + 1))
    for r in range(runs):
        theta = np.array([-7.0, 2.0])
        F[r, 0] = f(*theta)
        for k in range(steps):
            g = grad_f(*theta) + sigma * rng.standard_normal(2)    # noisy gradient
            theta = theta - lr(k) * g
            F[r, k + 1] = f(*theta)
    curves[name] = F.mean(axis=0)
    print(f"{name:40s}  mean f after {steps} steps = {curves[name][-1]:.2e}  "
          f"(mean over the last 500 steps = {curves[name][-500:].mean():.2e})")

fig, ax = plt.subplots(figsize=(7.5, 4.2))
for name, c in curves.items():
    ax.loglog(np.arange(1, steps + 2), c, label=name)
ax.set_xlabel("iteration k")
ax.set_ylabel(r"average of $f(\theta_k)$ over 50 runs")
ax.set_title("SGD with noisy gradients (sigma = 1)")
ax.grid(True, which="both", alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "sgd_noise.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
