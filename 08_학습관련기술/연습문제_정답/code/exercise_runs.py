# Runs for the answer key of the exercises at the end of the notes (Week: Techniques for Training)
#   usage: python exercise_runs.py 5 8 11 14 17     (exercise numbers; no argument = all)
#   * uses mnist_data.py, layers.py and optimizers.py in ../../code
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from mnist_data import load_mnist_split
from layers import MultiLayerNet, BatchNormalization
from optimizers import SGD, Momentum, AdaGrad, RMSProp, Adam

np.seterr(over="ignore", invalid="ignore")
x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()


def sgd_loop(net, optimizer, X, T, epochs, batch_size, seed=2, every_epoch=None):
    np.random.seed(seed)
    N = X.shape[0]
    for epoch in range(epochs):
        perm = np.random.permutation(N)
        for it in range(N // batch_size):
            idx = perm[it * batch_size:(it + 1) * batch_size]
            net.loss(X[idx], T[idx])
            optimizer.update(net.params, net.backward())
        if every_epoch:
            every_epoch(epoch)


# ---------------- Exercise 5: SGD with larger learning rates, and RMSProp ----------------
def ex5():
    runs = {
        "SGD 0.01": SGD(lr=0.01), "SGD 0.05": SGD(lr=0.05), "SGD 0.1": SGD(lr=0.1),
        "Momentum": Momentum(lr=0.01, beta=0.9), "AdaGrad": AdaGrad(lr=0.01),
        "RMSProp": RMSProp(lr=0.001), "Adam": Adam(lr=0.001),
    }
    for name, opt in runs.items():
        np.random.seed(1)
        net = MultiLayerNet(784, [100, 100, 100, 100], 10, activation="relu", loss="ce")
        accs = []
        sgd_loop(net, opt, x_train, t_train, 5, 128,
                 every_epoch=lambda e: accs.append(net.accuracy(x_val, t_val)))
        print(f"ex5 {name:9s} val acc per epoch: " + " ".join(f"{a:.4f}" for a in accs))


# ---------------- Exercise 8: 20 layers, ReLU + Xavier vs ReLU + He ----------------
def ex8():
    for init_name, std_fn in (("Xavier", lambda n: np.sqrt(1.0 / n)), ("He", lambda n: np.sqrt(2.0 / n))):
        rng = np.random.default_rng(0)
        Z = rng.standard_normal((1000, 100))
        stds = []
        for i in range(20):
            W = std_fn(Z.shape[1]) * rng.standard_normal((Z.shape[1], 100))
            Z = np.maximum(0.0, Z @ W)
            stds.append(Z.std())
        stds = np.array(stds)
        ratio = np.exp(np.mean(np.log(stds[1:] / stds[:-1])))
        print(f"ex8 ReLU + {init_name:6s} std per layer: " + " ".join(f"{s:.4g}" for s in stds))
        print(f"ex8 ReLU + {init_name:6s} last layer {stds[-1]:.4g}, mean ratio per layer {ratio:.4f}")


# ---------------- Exercise 11: batch statistics with ONE image at a time ----------------
def ex11():
    np.random.seed(1)
    net = MultiLayerNet(784, [100, 100, 100, 100, 100], 10, weight_init_std=0.1, use_batchnorm=True)
    sgd_loop(net, SGD(lr=0.01), x_train[:1000], t_train[:1000], 20, 100)
    acc_running = net.accuracy(x_val, t_val)
    U = net.predict(x_val, train_flg=True)
    acc_batch = np.mean(np.argmax(U, 1) == np.argmax(t_val, 1))
    preds = []
    for i in range(len(x_val)):
        u = net.predict(x_val[i:i + 1], train_flg=True)
        preds.append(np.argmax(u, axis=1)[0])
    preds = np.array(preds)
    acc_one = np.mean(preds == np.argmax(t_val, 1))
    u0 = net.predict(x_val[0:1], train_flg=True)
    u1 = net.predict(x_val[1:2], train_flg=True)
    counts = np.bincount(preds, minlength=10)
    print(f"ex11 running averages: {acc_running:.4f}, batch statistics of 5000: {acc_batch:.4f}, "
          f"one image at a time: {acc_one:.4f}")
    print("ex11 predicted classes (one at a time), counts:", counts.tolist())
    print("ex11 same scores for two different images:", np.allclose(u0, u1))
    print("ex11 share of the most common true class:", np.bincount(np.argmax(t_val, 1)).max() / len(t_val))


# ---------------- Exercise 14: [500, 500] with dropout p = 0, 0.2, 0.5 ----------------
def ex14():
    xs, ts = x_train[:300], t_train[:300]
    for p in (0.0, 0.2, 0.5):
        np.random.seed(1)
        net = MultiLayerNet(784, [500, 500], 10, weight_init_std="he",
                            use_dropout=p > 0, dropout_ratio=p)
        t0 = time.time()
        sgd_loop(net, SGD(lr=0.01), xs, ts, 300, 100)
        print(f"ex14 p = {p}: train {net.accuracy(xs, ts):.4f}  validation {net.accuracy(x_val, t_val):.4f}  "
              f"test {net.accuracy(x_test, t_test):.4f}  ({time.time() - t0:.0f}s)")


# ---------------- Exercise 17: a narrower random search, 30 trials ----------------
def ex17():
    xs, ts = x_train[:500], t_train[:500]

    def train(lr, wd):
        np.random.seed(1)
        net = MultiLayerNet(784, [100] * 6, 10, weight_init_std="he", weight_decay_lambda=wd)
        sgd_loop(net, SGD(lr=lr), xs, ts, 50, 100)
        return net

    rng = np.random.default_rng(0)
    results = []
    for trial in range(30):
        lr = 10 ** rng.uniform(np.log10(0.05), np.log10(0.25))
        wd = 10 ** rng.uniform(-6, -1)
        net = train(lr, wd)
        results.append((net.accuracy(x_val, t_val), net.accuracy(xs, ts), lr, wd))
        print(f"ex17 trial {trial + 1:2d}: val acc {results[-1][0]:.4f}  lr {lr:.2e}  weight decay {wd:.2e}")
    results.sort(key=lambda r: r[0], reverse=True)
    vals = np.array([r[0] for r in results])
    print(f"ex17 30 trials: best {vals.max():.4f}, mean {vals.mean():.4f}, "
          f"below 0.5: {np.sum(vals < 0.5)}")
    for rank, r in enumerate(results[:5], 1):
        print(f"ex17 {rank}. val {r[0]:.4f}  train {r[1]:.4f}  lr {r[2]:.2e}  weight decay {r[3]:.2e}")
    best = results[0]
    print(f"ex17 chosen: lr {best[2]:.2e}, wd {best[3]:.2e} -> validation {best[0]:.4f}, "
          f"test {train(best[2], best[3]).accuracy(x_test, t_test):.4f}")


todo = [int(a) for a in sys.argv[1:]] or [5, 8, 11, 14, 17]
for k in todo:
    t0 = time.time()
    {5: ex5, 8: ex8, 11: ex11, 14: ex14, 17: ex17}[k]()
    print(f"-- exercise {k} done in {time.time() - t0:.0f}s", flush=True)
