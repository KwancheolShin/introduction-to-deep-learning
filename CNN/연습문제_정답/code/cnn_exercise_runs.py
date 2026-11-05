# Runs for the answer key of the exercises at the end of the notes (Week: CNN)
#   usage: python cnn_exercise_runs.py 2 4 5 6 7     (exercise numbers; no argument = all)
#   * uses conv_layers.py, mnist_data.py in ../../code
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "code"))
from conv_layers import im2col, Convolution, Pooling, Flatten, SimpleConvNet
from layers import Affine, Relu, SoftmaxWithCrossEntropy


# ---------------- Exercises 2 and 4: a small convolution, max pooling and its backward ----------------
def ex2_4():
    X = np.array([[1, 0, 2, 1], [0, 3, 1, 0], [2, 1, 0, 4], [1, 0, 1, 2]], float).reshape(1, 1, 4, 4)
    F = np.array([[1, 0], [0, -1]], float).reshape(1, 1, 2, 2)
    conv = Convolution(F, np.zeros(1))
    print("ex2 convolution:\n", conv.forward(X)[0, 0])
    pool = Pooling(2, 2)
    print("ex2 max pooling:\n", pool.forward(X)[0, 0])
    dX = pool.backward(np.array([[1, 2], [3, 4]], float).reshape(1, 1, 2, 2))
    print("ex4 gradient to the input:\n", dX[0, 0])


# ---------------- Exercise 5: 1-D im2col with symbols ----------------
def ex5():
    x = np.arange(1, 6, dtype=float).reshape(1, 1, 1, 5)        # x_1 .. x_5 as 1 .. 5
    col = im2col(x, 1, 3, stride=1, pad=0)
    print("ex5 im2col (entry k means x_k):\n", col)
    print("ex5 how many times x_3 appears:", int(np.sum(col == 3)))


# ---------------- Exercise 6: SimpleConvNet with padding 2 ----------------
class SimpleConvNetPad(SimpleConvNet):
    """SimpleConvNet whose convolution uses padding `pad`"""
    def __init__(self, input_shape=(1, 28, 28), filter_num=30, filter_size=5, pad=2,
                 hidden_size=100, output_size=10, weight_init_std="he"):
        C, H, W = input_shape
        conv_out = H + 2 * pad - filter_size + 1          # 28 + 4 - 5 + 1 = 28
        pool_out = conv_out // 2                          # 14
        flat = filter_num * pool_out * pool_out           # 30 * 14 * 14 = 5880
        n_in = C * filter_size * filter_size
        def std(n):
            return np.sqrt(2.0 / n) if weight_init_std == "he" else weight_init_std
        self.params = {
            "W1": std(n_in) * np.random.randn(filter_num, C, filter_size, filter_size),
            "B1": np.zeros(filter_num),
            "W2": std(flat) * np.random.randn(flat, hidden_size),
            "B2": np.zeros((1, hidden_size)),
            "W3": std(hidden_size) * np.random.randn(hidden_size, output_size),
            "B3": np.zeros((1, output_size)),
        }
        self.layers = [
            Convolution(self.params["W1"], self.params["B1"], pad=pad),   # <- pad
            Relu(),
            Pooling(2, 2),
            Flatten(),
            Affine(self.params["W2"], self.params["B2"]),
            Relu(),
            Affine(self.params["W3"], self.params["B3"]),
        ]
        self.last = SoftmaxWithCrossEntropy()
        self.conv_out, self.pool_out, self.flat = conv_out, pool_out, flat


def numerical_gradient(f, P, eps=1e-5):
    grad = np.zeros_like(P)
    for idx in np.ndindex(P.shape):
        old = P[idx]
        P[idx] = old + eps
        f_plus = f()
        P[idx] = old - eps
        f_minus = f()
        P[idx] = old
        grad[idx] = (f_plus - f_minus) / (2 * eps)
    return grad


def ex6():
    net = SimpleConvNetPad()
    print(f"ex6 MNIST: conv_out {net.conv_out}, pool_out {net.pool_out}, flat {net.flat}, "
          f"parameters {sum(p.size for p in net.params.values())}")
    out = net.predict(np.random.randn(2, 1, 28, 28))
    print("ex6 score shape for 2 images:", out.shape)
    # the same small network as conv_basics.py, now with padding 2 (filter 3x3 on 8x8 -> 10x10 -> 5x5)
    np.random.seed(1)
    small = SimpleConvNetPad(input_shape=(2, 8, 8), filter_num=3, filter_size=3, pad=2,
                             hidden_size=5, output_size=4)
    print(f"ex6 small: conv_out {small.conv_out}, pool_out {small.pool_out}, flat {small.flat}")
    X_small = np.random.randn(5, 2, 8, 8)
    T_small = np.eye(4)[np.random.randint(0, 4, size=5)]
    small.loss(X_small, T_small)
    grads = small.backward()
    for key in small.params:
        num = numerical_gradient(lambda: small.loss(X_small, T_small), small.params[key])
        print(f"ex6 {key}: shape {str(small.params[key].shape):15s} "
              f"max |backprop - numerical| = {np.max(np.abs(grads[key] - num)):.2e}")


# ---------------- Exercise 7: a LeNet-like network vs SimpleConvNet (PyTorch) ----------------
def ex7():
    # torch only here: on this PC, torch and numpy's matrix product in the same process clash (OpenMP)
    import torch
    from torch import nn
    from torch.utils.data import TensorDataset, DataLoader
    from mnist_data import load_mnist_split
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()

    def to_dataset(x, t):
        return TensorDataset(torch.tensor(x, dtype=torch.float32).reshape(-1, 1, 28, 28),
                             torch.tensor(np.argmax(t, axis=1)))

    def models():
        simple = nn.Sequential(
            nn.Conv2d(1, 30, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(30 * 12 * 12, 100), nn.ReLU(), nn.Linear(100, 10))
        lenet = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2), nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120), nn.ReLU(), nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10))
        return {"SimpleConvNet": simple, "LeNet-like": lenet}

    torch.manual_seed(0)
    for name, model in models().items():
        h = torch.zeros(1, 1, 28, 28)
        print(f"ex7 {name} shapes:")
        for layer in model:
            h = layer(h)
            print(f"ex7     {type(layer).__name__:9s} {tuple(h.shape[1:])}")
        print(f"ex7 {name} parameters: {sum(p.numel() for p in model.parameters())}")

    for name, model in models().items():
        torch.manual_seed(0)                                 # the same mini-batch order for both
        for m in model.modules():                            # fresh PyTorch default initialization
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()
        model = model.to(device)
        train_loader = DataLoader(to_dataset(x_train, t_train), batch_size=100, shuffle=True)
        val_loader = DataLoader(to_dataset(x_val, t_val), batch_size=1000)
        test_loader = DataLoader(to_dataset(x_test, t_test), batch_size=1000)
        opt = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()

        @torch.no_grad()
        def acc(loader):
            model.eval()
            c = n = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                c += (model(xb).argmax(1) == yb).sum().item()
                n += len(yb)
            return c / n

        t0 = time.time()
        vals = []
        for epoch in range(10):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = loss_fn(model(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            vals.append(acc(val_loader))
        print(f"ex7 {name} ({device}): val acc per epoch " + " ".join(f"{v:.4f}" for v in vals))
        print(f"ex7 {name}: val {vals[-1]:.4f}, test {acc(test_loader):.4f}, {time.time() - t0:.0f} s")


todo = [int(a) for a in sys.argv[1:]] or [2, 5, 6, 7]
for k in todo:
    {2: ex2_4, 4: ex2_4, 5: ex5, 6: ex6, 7: ex7}[k]()
    print(f"-- exercise {k} done", flush=True)
