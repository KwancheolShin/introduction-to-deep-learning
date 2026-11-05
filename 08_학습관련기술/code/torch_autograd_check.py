# Week: Techniques for Training -- PyTorch autograd vs. our hand-written backward()
#   (1) the computational graph that PyTorch records during the forward pass
#   (2) the same network (Affine -> BatchNorm -> ReLU, twice, then Affine) with the SAME parameters:
#       our numpy MultiLayerNet and a PyTorch nn.Sequential give the SAME loss and gradients
#   * uses layers.py in the same folder

import os
# Windows + Anaconda: numpy (MKL) and a pip-installed PyTorch each bring their own OpenMP library,
# and using both in one program stops with "OMP: Error #15".  This line lets it run; it is a known
# workaround, not a real fix (the real fix: install PyTorch with conda, or use a clean environment).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
from torch import nn

from layers import MultiLayerNet

# =====================================================
# (1) The computational graph: every tensor remembers the operation that made it (grad_fn)
# =====================================================
x = torch.randn(3, 4)                              # mini-batch of 3, 4 inputs
W = torch.randn(4, 2, requires_grad=True)          # requires_grad: "I want dL/dW"
B = torch.zeros(1, 2, requires_grad=True)
A = x @ W + B                                      # Affine, written as in our code
L = torch.relu(A).sum()


def show_graph(fn, depth=0):
    """walk the recorded graph from the loss back to the parameters"""
    if fn is None:                                 # x: no gradient needed, not in the graph
        return
    name = type(fn).__name__
    if name == "AccumulateGrad":                   # a leaf: a parameter whose .grad will be filled
        name += "  -> " + ("W" if fn.variable is W else "B")
    print("    " * depth + name)
    for next_fn, _ in fn.next_functions:
        show_graph(next_fn, depth + 1)


print("graph recorded for L = sum(relu(x @ W + B)):")
show_graph(L.grad_fn)
L.backward()
print("dL/dW from autograd:\n", W.grad)
print("dL/dW by hand, x.T @ (A > 0):\n", x.T @ (A > 0).float())

# =====================================================
# (2) Our MultiLayerNet and PyTorch, the same parameters, the same mini-batch
# =====================================================
np.random.seed(0)
ours = MultiLayerNet(4, [5, 5], 3, weight_init_std="he", use_batchnorm=True)
X = np.random.randn(6, 4)
T = np.eye(3)[np.random.randint(0, 3, size=6)]    # one-hot

theirs = nn.Sequential(
    nn.Linear(4, 5), nn.BatchNorm1d(5, eps=1e-7), nn.ReLU(),
    nn.Linear(5, 5), nn.BatchNorm1d(5, eps=1e-7), nn.ReLU(),
    nn.Linear(5, 3),
).double()                                         # float64, like numpy

linears = [m for m in theirs if isinstance(m, nn.Linear)]
batchnorms = [m for m in theirs if isinstance(m, nn.BatchNorm1d)]
with torch.no_grad():                              # copy OUR parameters into the PyTorch model
    for k, linear in enumerate(linears, 1):
        linear.weight.copy_(torch.from_numpy(ours.params[f"W{k}"].T))      # (out, in) = W.T
        linear.bias.copy_(torch.from_numpy(ours.params[f"B{k}"].ravel()))
    for k, bn in enumerate(batchnorms, 1):
        bn.weight.copy_(torch.from_numpy(ours.params[f"gamma{k}"].ravel()))
        bn.bias.copy_(torch.from_numpy(ours.params[f"beta{k}"].ravel()))

# ours: forward, then backward through every layer by hand
loss_ours = ours.loss(X, T)
grads = ours.backward()

# PyTorch: forward, then ONE call
theirs.train()
loss_theirs = nn.functional.cross_entropy(theirs(torch.from_numpy(X)),
                                          torch.from_numpy(T.argmax(axis=1)))
loss_theirs.backward()

print(f"\nloss  ours {loss_ours:.10f}   PyTorch {loss_theirs.item():.10f}")
for k, linear in enumerate(linears, 1):
    print(f"W{k}     max |difference| = {np.max(np.abs(grads[f'W{k}'] - linear.weight.grad.numpy().T)):.1e}")
    print(f"B{k}     max |difference| = {np.max(np.abs(grads[f'B{k}'].ravel() - linear.bias.grad.numpy())):.1e}")
for k, bn in enumerate(batchnorms, 1):
    print(f"gamma{k} max |difference| = {np.max(np.abs(grads[f'gamma{k}'].ravel() - bn.weight.grad.numpy())):.1e}")
    print(f"beta{k}  max |difference| = {np.max(np.abs(grads[f'beta{k}'].ravel() - bn.bias.grad.numpy())):.1e}")
