# Week: CNN -- small PyTorch examples for the slides on mnist_cnn_torch.py
#   (1) the (N, C, H, W) shape   (2) nn.Conv2d: output size, weight shape, a hand-checked example
#   (3) nn.MaxPool2d   (4) nn.Flatten   (5) the shape after every layer of the model

import torch
from torch import nn

torch.manual_seed(0)

# (1) images (N, 784) -> (N, C, H, W)
x = torch.arange(2 * 784, dtype=torch.float32).reshape(2, 784)
print("flat:", tuple(x.shape), "->", tuple(x.reshape(-1, 1, 28, 28).shape))

# (2) nn.Conv2d
conv = nn.Conv2d(1, 30, kernel_size=5)
print("\nweight:", tuple(conv.weight.shape), " bias:", tuple(conv.bias.shape),
      " parameters:", sum(p.numel() for p in conv.parameters()))
img = torch.randn(1, 1, 28, 28)
for k, s, p in [(5, 1, 0), (5, 1, 2), (3, 2, 1)]:
    out = nn.Conv2d(1, 30, kernel_size=k, stride=s, padding=p)(img)
    print(f"kernel {k}, stride {s}, padding {p}: (1, 1, 28, 28) -> {tuple(out.shape)}")

# the 4x4 input and 3x3 filter of the lecture: the output must be [[15, 16], [6, 15]]
X = torch.tensor([[1., 2., 3., 0.],
                  [0., 1., 2., 3.],
                  [3., 0., 1., 2.],
                  [2., 3., 0., 1.]]).reshape(1, 1, 4, 4)
F = torch.tensor([[2., 0., 1.],
                  [0., 1., 2.],
                  [1., 0., 2.]]).reshape(1, 1, 3, 3)
small = nn.Conv2d(1, 1, kernel_size=3, bias=False)
with torch.no_grad():
    small.weight.copy_(F)                       # put our filter into the layer
print("\nconv of the lecture example:\n", small(X).detach().reshape(2, 2))

# (3) nn.MaxPool2d: the maximum of every 2x2 window, stride = kernel size
A = torch.tensor([[1., 2., 1., 0.],
                  [0., 1., 2., 3.],
                  [3., 0., 1., 2.],
                  [2., 4., 0., 1.]]).reshape(1, 1, 4, 4)
print("max pooling:\n", nn.MaxPool2d(2)(A).reshape(2, 2))

# (4) nn.Flatten keeps the first (batch) dimension
print("\nflatten:", tuple(torch.zeros(100, 30, 12, 12).shape), "->",
      tuple(nn.Flatten()(torch.zeros(100, 30, 12, 12)).shape))

# (5) the model of mnist_cnn_torch.py, layer by layer
model = nn.Sequential(
    nn.Conv2d(1, 30, kernel_size=5),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(30 * 12 * 12, 100),
    nn.ReLU(),
    nn.Linear(100, 10),
)
h = torch.randn(100, 1, 28, 28)
print(f"\n{'input':10s} {tuple(h.shape)}")
for layer in model:
    h = layer(h)
    print(f"{type(layer).__name__:10s} {tuple(h.shape)}")
for name, p in model.named_parameters():
    print(f"{name:9s} {str(tuple(p.shape)):16s} {p.numel():7d}")
print("total parameters:", sum(p.numel() for p in model.parameters()))
