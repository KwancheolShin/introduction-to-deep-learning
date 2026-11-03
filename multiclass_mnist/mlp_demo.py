# Small examples for the slides on mnist_mlp.py.
# Uses the layer classes and MultiLayerNet of mnist_mlp.py (sections 2 and 3 only, no training).

import numpy as np

# take sections 2-3 (layers + MultiLayerNet) from mnist_mlp.py without running the training
src = open("mnist_mlp.py", encoding="utf-8").read()
start = src.index("# 2. Layers")
end = src.index("# =====================================================\n# 4. Training")
exec(src[start:end])

# ---- one-hot with np.eye ----
labels = np.array([5, 0, 4])
print("one-hot of", labels)
print(np.eye(10)[labels].astype(int))

# ---- the layer list of MultiLayerNet ----
np.random.seed(1)
net = MultiLayerNet(784, [100, 100], 10)
print("\nlayers:", [type(layer).__name__ for layer in net.layers])
print("last  :", type(net.last).__name__)
for key, p in net.params.items():
    print(f"{key}: {p.shape}")
print("number of parameters:", net.num_params())

# ---- params and the Affine layers share the SAME arrays ----
print("\nparams['W1'] is layers[0].W :", net.params["W1"] is net.layers[0].W)
W_before = net.layers[0].W[0, 0]
net.params["W1"] -= 0.1                          # in-place update
print("in-place  -= : layer sees it ->", net.layers[0].W[0, 0] != W_before)
W_before = net.layers[0].W[0, 0]
net.params["W1"] = net.params["W1"] - 0.1        # a NEW array
print("new array  = : layer sees it ->", net.layers[0].W[0, 0] != W_before)

# ---- shapes along the forward pass (mini-batch of 3) ----
X = np.random.rand(3, 784)
print("\ninput", X.shape)
for layer in net.layers:
    X = layer.forward(X)
    print(f"{type(layer).__name__:8s} -> {X.shape}")

# ---- gradient check on a small network ----
np.random.seed(0)
small = MultiLayerNet(5, [4, 3], 3, weight_init_std=0.5)
for key in small.params:          # nonzero biases: keep ReLU inputs away from the kink at 0
    if key.startswith("B"):
        small.params[key] += 0.1 * np.random.randn(*small.params[key].shape)
Xs = np.random.randn(2, 5)
Ts = np.eye(3)[[0, 2]]
small.loss(Xs, Ts)
grads = small.backward()
eps = 1e-5
print("\ngradient check (max |backprop - numerical|):")
for key, P in small.params.items():
    num = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps; lp = small.loss(Xs, Ts)
        P[idx] = old - eps; lm = small.loss(Xs, Ts)
        P[idx] = old
        num[idx] = (lp - lm) / (2 * eps)
    print(f"{key}: {np.max(np.abs(grads[key] - num)):.2e}")

# ---- why softmax subtracts the max ----
U = np.array([[1000.0, 1001.0, 1002.0]])
with np.errstate(over="ignore", invalid="ignore"):
    naive = np.exp(U) / np.sum(np.exp(U), axis=1, keepdims=True)
print("\nnaive softmax:", naive)
print("softmax      :", np.round(softmax(U), 4))
