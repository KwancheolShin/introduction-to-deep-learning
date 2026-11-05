# Week: Techniques for Training -- weight initialization
# Histograms of the hidden-layer activations of a 5-layer network (100 nodes per layer)
# for different weight initializations (no training: forward pass only).
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

def sigmoid(a):
    return 1.0 / (1.0 + np.exp(-a))

def relu(a):
    return np.maximum(0.0, a)

def tanh(a):
    return np.tanh(a)

def forward_activations(activation, weight_std, n_layers=5, n_nodes=100, n_data=1000, seed=0):
    """Feed 1000 random inputs through n_layers layers; return the list of activations Z of each layer."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_data, n_nodes))          # input data (1000, 100)
    Z = X
    activations = []
    for i in range(n_layers):
        n_in = Z.shape[1]
        std = weight_std(n_in) if callable(weight_std) else weight_std
        W = std * rng.standard_normal((n_in, n_nodes))  # (input nodes, output nodes)
        A = Z @ W                                       # bias = 0
        Z = activation(A)
        activations.append(Z)
    return activations

inits = {
    "std = 1":                     1.0,
    "std = 0.01":                  0.01,
    r"Xavier: std = $1/\sqrt{n}$":  lambda n: np.sqrt(1.0 / n),
    r"He: std = $\sqrt{2/n}$":      lambda n: np.sqrt(2.0 / n),
}

def draw(activation, act_name, init_names, fname, bins=30, rng_hist=(0, 1), drop_zero=False):
    fig, axes = plt.subplots(len(init_names), 5, figsize=(13, 2.3 * len(init_names)), squeeze=False)
    for r, init_name in enumerate(init_names):
        acts = forward_activations(activation, inits[init_name])
        for i, Z in enumerate(acts):
            ax = axes[r, i]
            values = Z.ravel()
            if drop_zero:                                   # ReLU: about half of Z is exactly 0
                values = values[values > 0]
            ax.hist(values, bins=bins, range=rng_hist, color="tab:blue")
            ax.set_title(f"layer {i + 1}", fontsize=10)
            ax.set_yticks([])
            if i == 0:
                ax.set_ylabel(init_name, fontsize=11)
        print(f"{act_name:8s} {init_name:30s} std of Z per layer:",
              " ".join(f"{Z.std():.4f}" for Z in acts))
    title = f"activation = {act_name}" + ("   (only Z > 0 is shown)" if drop_zero else "")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

# sigmoid: std=1, std=0.01, Xavier
draw(sigmoid, "sigmoid", ["std = 1", "std = 0.01", r"Xavier: std = $1/\sqrt{n}$"],
     os.path.join(FIG_DIR, "init_sigmoid.png"), rng_hist=(0, 1))

# tanh with Xavier (symmetric around 0)
draw(tanh, "tanh", ["std = 1", r"Xavier: std = $1/\sqrt{n}$"],
     os.path.join(FIG_DIR, "init_tanh.png"), rng_hist=(-1, 1))

# ReLU: std=0.01, Xavier, He
draw(relu, "ReLU", ["std = 0.01", r"Xavier: std = $1/\sqrt{n}$", r"He: std = $\sqrt{2/n}$"],
     os.path.join(FIG_DIR, "init_relu.png"), rng_hist=(0, 3), drop_zero=True)
