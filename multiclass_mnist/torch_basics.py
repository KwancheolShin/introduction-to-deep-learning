# torch_basics.py -- the basic ideas of PyTorch, one small example each
#   1. tensor   2. autograd   3. nn   4. optim   5. utils.data
#   6. train / eval mode   7. decorators
# Every block prints its result; compare them with the slides.

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(0)
print("torch version:", torch.__version__)
print("GPU available:", torch.cuda.is_available())

# ----------------------------------------------------
# 1. Tensor: a numpy-like array (+ GPU + autograd)
# ----------------------------------------------------
print("\n=== 1. tensor ===")
a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
print(a)
print("shape:", a.shape, " dtype:", a.dtype, " device:", a.device)

print(torch.zeros(2, 3))
print(torch.arange(6).reshape(2, 3))
print(torch.randn(2, 3))                       # standard normal

# numpy <-> torch
x_np = np.array([1.0, 2.0, 3.0])               # numpy: float64
x_t = torch.tensor(x_np)
print("from numpy:", x_t, x_t.dtype)
print("float32  :", torch.tensor(x_np, dtype=torch.float32))
print("to numpy :", a.numpy())

# operations: the same as numpy, but `dim` instead of `axis`
W = torch.ones(3, 4)
print("a @ W:\n", a @ W)                       # (2,3) @ (3,4) -> (2,4)
print("a + 10*b (broadcast):\n", a + 10 * torch.tensor([1., 2., 3.]))
print("sum(dim=0):", a.sum(dim=0))
print("sum(dim=1):", a.sum(dim=1))
print("argmax(dim=1):", a.argmax(dim=1))
print("a.T:\n", a.T)
print("a.reshape(3, 2):\n", a.reshape(3, 2))
print("a[1, 2] =", a[1, 2], "  .item() ->", a[1, 2].item())

# device
device = "cuda" if torch.cuda.is_available() else "cpu"
print("a.to(device).device:", a.to(device).device)

# ----------------------------------------------------
# 2. Autograd: forward builds a graph, backward() runs the chain rule
# ----------------------------------------------------
print("\n=== 2. autograd ===")
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x                             # y = x^2 + 3x
y.backward()                                   # dy/dx = 2x + 3
print("y =", y.item(), "  dy/dx =", x.grad.item())

# an affine layer + squared loss, compared with the formula of our lecture
X = torch.tensor([[1., 2.], [3., 4.]])         # (h, l) = (2, 2)
Wt = torch.tensor([[0.5], [-1.0]], requires_grad=True)   # (l, m) = (2, 1)
B = torch.tensor([[0.1]], requires_grad=True)            # (1, m)
T = torch.tensor([[1.], [0.]])
A = X @ Wt + B
L = 0.5 * ((A - T) ** 2).sum()
L.backward()
dA = (A - T).detach()                          # dL/dA
print("autograd dW:", Wt.grad.T, "  formula X^T dA:", (X.T @ dA).T)
print("autograd dB:", B.grad, "  formula sum(dA):", dA.sum(dim=0, keepdim=True))

# gradients are ADDED, not replaced -> zero them before every backward
x = torch.tensor(2.0, requires_grad=True)
for k in range(3):
    (x ** 2).backward()
    print(f"after backward #{k + 1}: x.grad = {x.grad.item()}")
x.grad.zero_()
print("after zero_(): x.grad =", x.grad.item())

with torch.no_grad():                          # no graph is recorded
    z = x * 3
print("inside no_grad: z.requires_grad =", z.requires_grad)

# ----------------------------------------------------
# 3. nn: layers, models, loss functions
# ----------------------------------------------------
print("\n=== 3. nn ===")
fc = nn.Linear(784, 100)
print(fc)
print("weight:", fc.weight.shape, " bias:", fc.bias.shape)   # weight is (out, in)
xb = torch.randn(5, 784)                       # a mini-batch of 5
print("output:", fc(xb).shape)
print("same as x @ W.T + b:", torch.allclose(fc(xb), xb @ fc.weight.T + fc.bias))

relu = nn.ReLU()
print("ReLU:", relu(torch.tensor([-2., 0., 3.])))

model = nn.Sequential(nn.Linear(784, 100), nn.ReLU(), nn.Linear(100, 10))
for name, p in model.named_parameters():
    print(f"  {name:10s} {tuple(p.shape)}")
print("number of parameters:", sum(p.numel() for p in model.parameters()))

# CrossEntropyLoss = softmax + cross entropy, takes LOGITS and integer LABELS
loss_fn = nn.CrossEntropyLoss()
U = torch.tensor([[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]])     # scores of 2 data, 3 classes
t = torch.tensor([0, 1])                                 # class numbers, not one-hot
Y = torch.softmax(U, dim=1)
manual = -torch.log(Y[torch.arange(2), t]).mean()
print("softmax:\n", Y)
print("CrossEntropyLoss:", loss_fn(U, t).item(), "  by hand:", manual.item())

# ----------------------------------------------------
# 4. optim: the update rule
# ----------------------------------------------------
print("\n=== 4. optim ===")
w = torch.tensor([1.0, -2.0], requires_grad=True)
opt = torch.optim.SGD([w], lr=0.1)
loss = (w ** 2).sum()                          # grad = 2w = [2, -4]
opt.zero_grad()
loss.backward()
print("grad:", w.grad)
opt.step()                                     # w <- w - lr * grad
print("after step:", w.detach(), "  by hand:", torch.tensor([1.0, -2.0]) - 0.1 * torch.tensor([2.0, -4.0]))

# ----------------------------------------------------
# 5. utils.data: Dataset and DataLoader
# ----------------------------------------------------
print("\n=== 5. utils.data ===")
X = torch.randn(10, 784)
y = torch.arange(10) % 3
ds = TensorDataset(X, y)
print("len(ds):", len(ds))
x0, y0 = ds[0]
print("ds[0]:", x0.shape, y0)
train_ds, val_ds = random_split(ds, [8, 2])
print("split:", len(train_ds), len(val_ds))
loader = DataLoader(train_ds, batch_size=3, shuffle=True)
for xb, yb in loader:
    print("  batch:", tuple(xb.shape), yb)

# ----------------------------------------------------
# 6. model.train() / model.eval(): a mode flag, not training
# ----------------------------------------------------
print("\n=== 6. train / eval mode ===")
torch.manual_seed(0)
net = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Dropout(0.5))
print("right after creation:", net.training, net[2].training)
net.eval()                                     # every submodule: training = False
print("after eval() :", net.training, net[2].training)
net.train()                                    # every submodule: training = True
print("after train():", net.training, net[2].training)

drop = nn.Dropout(0.5)
x = torch.ones(1, 6)
drop.eval()
print("Dropout, eval :", drop(x))
drop.train()
print("Dropout, train:", drop(x))              # random zeros, survivors * 1/(1-0.5)

# eval() does NOT stop gradient tracking; torch.no_grad() does
lin = nn.Linear(2, 1)
lin.eval()
print("eval only     :", lin(torch.ones(1, 2)).requires_grad)
with torch.no_grad():
    print("with no_grad  :", lin(torch.ones(1, 2)).requires_grad)

# ----------------------------------------------------
# 7. Decorators: @something above a def wraps the function
# ----------------------------------------------------
print("\n=== 7. decorators ===")
def shout(func):                               # takes a function, returns a new one
    def wrapper(*args):
        print("  calling", func.__name__)
        return func(*args)
    return wrapper

@shout                                         # the same as: square = shout(square)
def square(a):
    return a * a

print("square(3) =", square(3))

@torch.no_grad()                               # the whole function runs inside no_grad
def predict(model, x):
    return model(x)

print("inside @torch.no_grad():", predict(lin, torch.ones(1, 2)).requires_grad)
