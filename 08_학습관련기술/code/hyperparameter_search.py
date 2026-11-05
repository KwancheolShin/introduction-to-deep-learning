# Week: Techniques for Training -- finding good hyperparameters
#   random search for the learning rate and the weight decay (log scale),
#   judged by the VALIDATION data; the test data is used only ONCE at the very end
#   * uses mnist_data.py, layers.py and optimizers.py in the same folder
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from layers import MultiLayerNet
from optimizers import SGD

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()   # 55000 / 5000 / 10000
x_train, t_train = x_train[:500], t_train[:500]   # a small training set -> fast trials
np.seterr(over="ignore", invalid="ignore")        # a too-large learning rate may blow up

epochs = 50
batch_size = 100


def train(lr, weight_decay, seed=2):
    """train a 7-layer network with SGD; return the network and the train / validation accuracy per epoch"""
    np.random.seed(1)                             # the SAME initial weights for every trial
    net = MultiLayerNet(784, [100, 100, 100, 100, 100, 100], 10,
                        weight_init_std="he", weight_decay_lambda=weight_decay)
    optimizer = SGD(lr=lr)
    np.random.seed(seed)                          # the SAME mini-batches for every trial
    N = x_train.shape[0]
    train_acc, val_acc = [], []
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for it in range(N // batch_size):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            net.loss(x_train[idx], t_train[idx])
            optimizer.update(net.params, net.backward())
        train_acc.append(net.accuracy(x_train, t_train))
        val_acc.append(net.accuracy(x_val, t_val))   # VALIDATION data, not test data
    return net, train_acc, val_acc


# =====================================================
# Random search: sample the hyperparameters uniformly on a LOG scale
# =====================================================
n_trials = 100
rng = np.random.default_rng(0)                    # separate random numbers for the sampling
results = []
for trial in range(n_trials):
    lr = 10 ** rng.uniform(-5, 0)                 # 1e-5 ~ 1
    weight_decay = 10 ** rng.uniform(-6, -1)      # 1e-6 ~ 0.1
    _, train_acc, val_acc = train(lr, weight_decay)
    results.append({"lr": lr, "wd": weight_decay, "train": train_acc, "val": val_acc})
    print(f"trial {trial + 1:3d}: val acc {val_acc[-1]:.4f}  lr {lr:.2e}  weight decay {weight_decay:.2e}")

results.sort(key=lambda r: r["val"][-1], reverse=True)
print("=========== best 10 ===========")
for rank, r in enumerate(results[:10], 1):
    print(f"{rank:2d}. val acc {r['val'][-1]:.4f}  train acc {r['train'][-1]:.4f}  "
          f"lr {r['lr']:.2e}  weight decay {r['wd']:.2e}")

print("======= by learning rate =======")
edges = [1e-5, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
for lo, hi in zip(edges[:-1], edges[1:]):
    accs = [r["val"][-1] for r in results if lo <= r["lr"] < hi]
    print(f"lr {lo:.0e} ~ {hi:.0e}: {len(accs):3d} trials  best {max(accs):.4f}  mean {np.mean(accs):.4f}")

# Figure 1: learning curves of the best 20 trials (solid: validation, dashed: train)
fig, axes = plt.subplots(4, 5, figsize=(14, 9), sharex=True, sharey=True)
for rank, (ax, r) in enumerate(zip(axes.ravel(), results[:20]), 1):
    ax.plot(range(1, epochs + 1), r["val"], label="validation")
    ax.plot(range(1, epochs + 1), r["train"], "--", label="train")
    ax.set_title(f"#{rank}  lr={r['lr']:.1e}\nwd={r['wd']:.1e}", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3)
axes[0, 0].legend(fontsize=8, loc="lower right")
for ax in axes[-1]:
    ax.set_xlabel("epoch")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hyper_top20.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# Figure 2: every trial on the (lr, weight decay) plane, colored by the validation accuracy
fig, ax = plt.subplots(figsize=(6.5, 5))
sc = ax.scatter([r["lr"] for r in results], [r["wd"] for r in results],
                c=[r["val"][-1] for r in results], cmap="viridis", vmin=0, vmax=1, s=40)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("learning rate")
ax.set_ylabel("weight decay")
ax.set_title(f"Random search ({n_trials} trials)")
fig.colorbar(sc, label="validation accuracy")
ax.grid(True, which="both", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "hyper_scatter.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# Only at the very end: the test data, ONCE, with the chosen hyperparameters
# =====================================================
best = results[0]
net, _, _ = train(best["lr"], best["wd"])
print(f"chosen: lr {best['lr']:.2e}, weight decay {best['wd']:.2e}  ->  "
      f"validation {best['val'][-1]:.4f}, test {net.accuracy(x_test, t_test):.4f}")
