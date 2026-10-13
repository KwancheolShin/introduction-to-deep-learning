# Week: Techniques for Training -- overfitting, weight decay, dropout
#   (1) overfitting: a deep network trained on only 300 images (as in the book)
#   (2) the same network with weight decay
#   (3) the same network with dropout
#   (4) how strong should the weight decay be?  (chosen with the VALIDATION data)
#   (5) Adam, Adam + L2 (weight decay in the loss), AdamW (decoupled weight decay)
#   * uses mnist_data.py, layers.py and optimizers.py in the same folder
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from

import os
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from layers import MultiLayerNet
from optimizers import SGD, Adam, AdamW

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()
x_train, t_train = x_train[:300], t_train[:300]   # only 300 training images -> overfitting

hidden = [100, 100, 100, 100, 100, 100]           # 6 hidden layers (7 Affine layers)
epochs = 300
batch_size = 100


def train(net, optimizer, seed=2):
    """return the train / test accuracy after each epoch"""
    np.random.seed(seed)                          # the SAME mini-batches (and dropout masks)
    N = x_train.shape[0]
    train_acc, test_acc = [], []
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for it in range(N // batch_size):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            net.loss(x_train[idx], t_train[idx])
            optimizer.update(net.params, net.backward())
        train_acc.append(net.accuracy(x_train, t_train))
        test_acc.append(net.accuracy(x_test, t_test))
    return train_acc, test_acc


def make_net(**options):
    np.random.seed(1)                             # the SAME initial weights
    return MultiLayerNet(784, hidden, 10, weight_init_std="he", **options)


def plot(train_acc, test_acc, title, filename):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, epochs + 1), train_acc, label="train")
    ax.plot(range(1, epochs + 1), test_acc, "--", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200, bbox_inches="tight")
    plt.close(fig)


def report(name, train_acc, test_acc):
    print(f"{name:22s} train {train_acc[-1]:.4f}  test {test_acc[-1]:.4f}  "
          f"gap {train_acc[-1] - test_acc[-1]:.4f}")


# (1) overfitting
acc = train(make_net(), SGD(lr=0.01))
plot(*acc, "No regularization", "overfit.png")
report("no regularization", *acc)

# (2) weight decay
acc = train(make_net(weight_decay_lambda=0.1), SGD(lr=0.01))
plot(*acc, "Weight decay (lambda = 0.1)", "overfit_weight_decay.png")
report("weight decay 0.1", *acc)

# (3) dropout
acc = train(make_net(use_dropout=True, dropout_ratio=0.2), SGD(lr=0.01))
plot(*acc, "Dropout (ratio = 0.2)", "overfit_dropout.png")
report("dropout 0.2", *acc)

# (4) weight decay strength: compare on the VALIDATION data (never choose with the test data)
for lam in (0.0, 0.01, 0.03, 0.1, 0.3):
    net = make_net(weight_decay_lambda=lam)
    train_acc, _ = train(net, SGD(lr=0.01))
    print(f"lambda = {lam:<5}: train {train_acc[-1]:.4f}  validation {net.accuracy(x_val, t_val):.4f}")

# (5) Adam / Adam + L2 / AdamW
runs = {
    "Adam":             (make_net(), Adam(lr=0.001)),
    "Adam + L2 (0.1)":  (make_net(weight_decay_lambda=0.1), Adam(lr=0.001)),
    "AdamW (wd = 0.1)": (make_net(), AdamW(lr=0.001, weight_decay=0.1)),
}
for name, (net, optimizer) in runs.items():
    report(name, *train(net, optimizer))
