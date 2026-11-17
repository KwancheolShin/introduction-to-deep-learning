# Week: deep learning -- speed and numerical precision of DeepConvNet (numpy)
#   (1) which layers take the time? forward + backward of one mini-batch, layer by layer
#   (2) the same mini-batch in numpy and in PyTorch (CPU)
#   (3) the trained network (../results/deep_convnet_params.npz from mnist_deep_torch.py)
#       evaluated with 64-bit, 32-bit and 16-bit floating point numbers
#   * uses deep_convnet.py, conv_layers.py, layers.py, mnist_data.py
# The figure is saved to ../figures/ (next to main.tex)

import os
import time
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from deep_convnet import DeepConvNet
from conv_layers import Convolution, Pooling
from layers import Affine, Relu, Dropout

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
PARAMS = os.path.join(HERE, "..", "results", "deep_convnet_params.npz")

x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)

# =====================================================
# (1) time of every layer, forward + backward, mini-batch of 100
# =====================================================
np.random.seed(0)
net = DeepConvNet()
X, T = x_train[:100], t_train[:100]
kinds = {Convolution: "Conv", Pooling: "Pool", Relu: "ReLU", Affine: "Affine"}
spent = {}
for rep in range(6):                                  # the first round only warms up
    h = X
    t_fwd = []
    for layer in net.layers:
        t0 = time.perf_counter()
        h = layer.forward(h, True) if isinstance(layer, Dropout) else layer.forward(h)
        t_fwd.append(time.perf_counter() - t0)
    net.last.forward(h, T)
    d = net.last.backward()
    t_bwd = []
    for layer in reversed(net.layers):
        t0 = time.perf_counter()
        d = layer.backward(d)
        t_bwd.append(time.perf_counter() - t0)
    if rep == 0:
        continue
    for layer, tf, tb in zip(net.layers, t_fwd, reversed(t_bwd)):
        kind = kinds.get(type(layer), "other")
        spent[kind] = spent.get(kind, 0.0) + (tf + tb) / 5
total = sum(spent.values())
print(f"numpy, one mini-batch of 100 (forward + backward): {total * 1000:.0f} ms")
for kind, s in sorted(spent.items(), key=lambda kv: -kv[1]):
    print(f"  {kind:7s} {s * 1000:7.1f} ms  {100 * s / total:5.1f} %")

fig, ax = plt.subplots(figsize=(6, 2.2))
order = sorted(spent, key=spent.get)
ax.barh(order, [100 * spent[k] / total for k in order], color="#4c72b0")
for i, k in enumerate(order):
    ax.text(100 * spent[k] / total + 1, i, f"{100 * spent[k] / total:.1f} %", va="center", fontsize=9)
ax.set_xlim(0, 105)
ax.set_xlabel("share of the time for one mini-batch (forward + backward) [%]")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "time_by_layer.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

# =====================================================
# (2) the same work in PyTorch (CPU)
# =====================================================
try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    def conv(c_in, c_out, pad=1):
        return [nn.Conv2d(c_in, c_out, kernel_size=3, padding=pad), nn.ReLU()]

    model = nn.Sequential(*conv(1, 16), *conv(16, 16), nn.MaxPool2d(2),
                          *conv(16, 32), *conv(32, 32, pad=2), nn.MaxPool2d(2),
                          *conv(32, 64), *conv(64, 64), nn.MaxPool2d(2),
                          nn.Flatten(), nn.Linear(1024, 50), nn.ReLU(), nn.Dropout(0.5), nn.Linear(50, 10))
    xb = torch.tensor(X, dtype=torch.float32)
    yb = torch.tensor(np.argmax(T, axis=1))
    times = []
    for rep in range(11):
        t0 = time.perf_counter()
        F.cross_entropy(model(xb), yb).backward()
        times.append(time.perf_counter() - t0)
    print(f"PyTorch (CPU, {torch.get_num_threads()} threads), the same mini-batch: "
          f"{np.median(times[1:]) * 1000:.0f} ms")
except ImportError:
    print("PyTorch is not installed")

# =====================================================
# (3) 64-bit, 32-bit and 16-bit floating point numbers
# =====================================================
if os.path.exists(PARAMS):
    n = 2000                                          # float16 is slow in numpy: 2000 test images
    for dtype in (np.float64, np.float32, np.float16):
        net = DeepConvNet()
        net.load_params(PARAMS)
        net.astype(dtype)
        t0 = time.perf_counter()
        acc = net.accuracy(x_test[:n].astype(dtype), t_test[:n])
        print(f"{np.dtype(dtype).name:8s} ({np.dtype(dtype).itemsize * 8:2d} bit): "
              f"test accuracy (first {n}) {acc:.4f}  | parameters {net.num_params() * np.dtype(dtype).itemsize / 1024:.0f} KB"
              f" | {time.perf_counter() - t0:.1f} s")
else:
    print("run mnist_deep_torch.py first: no", PARAMS)
