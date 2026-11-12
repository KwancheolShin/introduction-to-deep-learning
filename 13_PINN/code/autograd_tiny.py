# %% 0. Setup
# Week: PINN -- the smallest possible example of "differentiate the network with respect to its INPUT"
#   u(x) = w2 * tanh(w1 * x + b1) + b2          (one input, one hidden neuron, one output)
# Everything printed here is used in the lecture notes, and every number is checked by hand.
#   (1) forward pass:            a, h, u
#   (2) derivatives in x:        u_x, u_xx  (autograd vs. the hand formulas)
#   (3) a residual and a loss:   r = u_xx + u   (the ODE u'' + u = 0),  L = r^2
#   (4) backward pass:           dL/dw1, dL/db1, dL/dw2, dL/db2  (autograd vs. the hand formulas)
#   (5) what the graph looks like (walk through grad_fn)
#   (6) create_graph=False: what goes wrong
#   (7) a batch of x: why grad_outputs=torch.ones_like(u) gives one derivative per point
#   (8) tensors and nodes: grad_fn, next_functions, saved values, and how large the derivative graphs are
#   Cells: each '# %%' starts a cell (Run Cell = Shift+Enter in VS Code).

import math
import torch

torch.set_printoptions(precision=5)

# %% 1. Forward pass
w1 = torch.tensor(0.8,  requires_grad=True)
b1 = torch.tensor(-0.2, requires_grad=True)
w2 = torch.tensor(1.5,  requires_grad=True)
b2 = torch.tensor(0.3,  requires_grad=True)

x = torch.tensor(1.0, requires_grad=True)   # we want du/dx, so x must be tracked too

a = w1 * x + b1          # pre-activation
h = torch.tanh(a)        # hidden value
u = w2 * h + b2          # network output
print(f"a = {a.item():.5f},  h = tanh(a) = {h.item():.5f},  u = {u.item():.5f}")

# %% 2. Derivatives with respect to the input x
# create_graph=True: the computation of u_x is itself recorded as a graph,
# so that we can differentiate u_x again (in x for u_xx, and later in the weights for the loss).
u_x,  = torch.autograd.grad(u,   x, create_graph=True)
u_xx, = torch.autograd.grad(u_x, x, create_graph=True)

# the same derivatives by hand: tanh' = 1 - tanh^2 = s,  (tanh)'' = -2 h s
s = 1 - h**2
u_x_hand  = w2 * s * w1
u_xx_hand = w2 * (-2 * h * s) * w1**2
print(f"s = 1 - h^2 = {s.item():.5f}")
print(f"u_x  : autograd {u_x.item():.5f}    hand {u_x_hand.item():.5f}")
print(f"u_xx : autograd {u_xx.item():.5f}   hand {u_xx_hand.item():.5f}")

# %% 3. Residual and loss: the ODE u'' + u = 0 at the single point x = 1
r = u_xx + u             # the original graph (u) and the derivative graph (u_xx) meet here
L = r**2
print(f"r = u_xx + u = {r.item():.5f},  L = r^2 = {L.item():.5f}")

# %% 4. Backward pass: the gradient of L with respect to the weights
L.backward()

# by hand (chain rule).  r = w2 w1^2 (-2 h s) + w2 h + b2,  d(h s)/da = s (1 - 3 h^2)
with torch.no_grad():
    q = -2 * s * (1 - 3 * h**2)                  # d(-2 h s)/da
    dr_dw2 = w1**2 * (-2 * h * s) + h
    dr_db2 = torch.tensor(1.0)
    dr_db1 = w2 * w1**2 * q + w2 * s             # through a, in both branches
    dr_dw1 = w2 * (2 * w1 * (-2 * h * s) + w1**2 * q * x) + w2 * s * x
    for name, p, d in [("w1", w1, dr_dw1), ("b1", b1, dr_db1), ("w2", w2, dr_dw2), ("b2", b2, dr_db2)]:
        print(f"dL/d{name}: autograd {p.grad.item():9.5f}   hand {(2 * r * d).item():9.5f}")
    print(f"(x also gets a gradient, x.grad = {x.grad.item():.5f}, but we never use it)")
    # w1 reaches r along three paths; dr/dw1 is the SUM of the three contributions
    p1 = w2 * s * x                       # w1 -> a -> h -> u -> r
    p2 = w2 * w1**2 * q * x               # w1 -> a -> h -> (-2 h s) -> u_xx -> r
    p3 = w2 * 2 * w1 * (-2 * h * s)       # w1 -> u_xx -> r   (the factor w1^2 in u_xx)
    print(f"-2 h s = {(-2 * h * s).item():.5f},  q = d(-2 h s)/da = {q.item():.5f}")
    print(f"dr/dw1 paths: (1) {p1.item():.5f}  (2) {p2.item():.5f}  (3) {p3.item():.5f}"
          f"  sum {(p1 + p2 + p3).item():.5f},  2 r = {(2 * r).item():.5f}")
    print(f"dr/db1 paths: (1) {(w2 * s).item():.5f}  (2) {(w2 * w1**2 * q).item():.5f}")
    print(f"dr/dw2 = {dr_dw2.item():.5f} = h {h.item():.5f} + w1^2 (-2 h s) {(w1**2 * (-2 * h * s)).item():.5f}")

# %% 5. Look at the graph: walk backwards through grad_fn
def show_graph(t, max_depth=12):
    """Print the backward graph behind tensor t (one line per node, indentation = depth)."""
    seen, keep = set(), []
    def walk(fn, depth):
        if fn is None or depth > max_depth:
            return
        label = type(fn).__name__
        if label == "AccumulateGrad":
            label += f"  (leaf: {leaf_name.get(id(fn.variable), '?')})"
        print("  " * depth + label)
        if id(fn) in seen:
            return
        seen.add(id(fn))
        keep.append(fn)
        for nxt, _ in fn.next_functions:
            walk(nxt, depth + 1)
    walk(t.grad_fn, 0)

def count_nodes(t):
    """Number of distinct nodes in the backward graph behind t (leaves included)."""
    keep, seen, stack = [], set(), [t.grad_fn]   # keep: hold the node objects so their ids stay unique
    while stack:
        fn = stack.pop()
        if fn is None or id(fn) in seen:
            continue
        keep.append(fn)
        seen.add(id(fn))
        stack.extend(n for n, _ in fn.next_functions)
    return len(seen)

# rebuild the graph (backward() above freed the buffers of the old one)
for p in (w1, b1, w2, b2, x):
    p.grad = None
leaf_name = {id(w1): "w1", id(b1): "b1", id(w2): "w2", id(b2): "b2", id(x): "x"}
a = w1 * x + b1
h = torch.tanh(a)
u = w2 * h + b2
u_x,  = torch.autograd.grad(u,   x, create_graph=True)
u_xx, = torch.autograd.grad(u_x, x, create_graph=True)
r = u_xx + u
L = r**2

print("graph behind u:")
show_graph(u)
print("graph behind u_x:")
show_graph(u_x)
for name, t in [("u", u), ("u_x", u_x), ("u_xx", u_xx), ("L", L)]:
    print(f"number of nodes behind {name:4s}: {count_nodes(t)}")

# %% 6. What if create_graph=False?
u_x_nograph, = torch.autograd.grad(u, x, retain_graph=True)     # create_graph=False (default)
print("u_x without create_graph:", u_x_nograph, " requires_grad:", u_x_nograph.requires_grad)
try:
    torch.autograd.grad(u_x_nograph, x)
except RuntimeError as e:
    print("RuntimeError:", e)

# %% 7. A batch of points: u has shape (N, 1), x has shape (N, 1)
torch.manual_seed(0)
net = torch.nn.Sequential(torch.nn.Linear(1, 8), torch.nn.Tanh(), torch.nn.Linear(8, 1))
xs = torch.tensor([[0.0], [0.5], [1.0]], requires_grad=True)
us = net(xs)
# autograd.grad needs a scalar, or a vector v to form  v^T (du/dx).  With v = ones:
#   sum_i du_i/dx_j = du_j/dx_j, because u_i depends only on x_i (the Jacobian is diagonal).
dus, = torch.autograd.grad(us, xs, grad_outputs=torch.ones_like(us), create_graph=True)
J = torch.autograd.functional.jacobian(lambda z: net(z), xs).reshape(3, 3)
print("Jacobian du_i/dx_j (3 x 3):\n", J)
print("diagonal of J:", torch.diagonal(J))
print("autograd.grad with ones:", dus.flatten())

# %% 8. Tensors and nodes: how the graph is stored
m = w1 * x               # one operation per line, so that every intermediate tensor has a name
a = m + b1
h = torch.tanh(a)
p = w2 * h
u = p + b2
for name, t in [("x", x), ("m", m), ("a", a), ("h", h), ("p", p), ("u", u)]:
    print(f"{name}: requires_grad={t.requires_grad}, is_leaf={t.is_leaf}, grad_fn={type(t.grad_fn).__name__}")

print("u.grad_fn.next_functions:", [(type(n).__name__, i) for n, i in u.grad_fn.next_functions])
print("(the first entry is p.grad_fn:", u.grad_fn.next_functions[0][0] is p.grad_fn, ")")
print("leaf node of b2: .variable is b2:", u.grad_fn.next_functions[1][0].variable is b2)
print("saved in m.grad_fn:", [n for n in dir(m.grad_fn) if n.startswith("_saved")],
      m.grad_fn._saved_self.item(), m.grad_fn._saved_other.item())
print("saved in h.grad_fn:", [n for n in dir(h.grad_fn) if n.startswith("_saved")], h.grad_fn._saved_result.item())
print("m.grad_fn(1.0) =", m.grad_fn(torch.tensor(1.0)), " (these are x and w1)")

c = 2 * x                # a constant is not a node
print("(2 * x).grad_fn.next_functions:", [type(n).__name__ for n, _ in c.grad_fn.next_functions])
with torch.no_grad():
    print("inside no_grad: (w1 * x).grad_fn =", (w1 * x).grad_fn)
print("m.detach().grad_fn =", m.detach().grad_fn)

for t in (m, a, h, p):   # keep .grad on the intermediate tensors too (by default only leaves get it)
    t.retain_grad()
for q in (w1, b1, w2, b2, x):
    q.grad = None
u.backward()
print(f"backward of u: p.grad {p.grad.item():.5f}, h.grad {h.grad.item():.5f}, a.grad {a.grad.item():.5f},"
      f" m.grad {m.grad.item():.5f}, x.grad {x.grad.item():.5f}")
try:
    u.backward()
except RuntimeError as e:
    print("second backward: RuntimeError:", str(e)[:60])

for depth in (1, 2, 4, 8):    # size of the graphs for a tanh network of width 16
    layers, d = [], 1
    for _ in range(depth):
        layers += [torch.nn.Linear(d, 16), torch.nn.Tanh()]
        d = 16
    net = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))
    xs = torch.linspace(0, 1, 5).reshape(-1, 1).requires_grad_(True)
    us = net(xs)
    d1, = torch.autograd.grad(us, xs, torch.ones_like(us), create_graph=True)
    d2, = torch.autograd.grad(d1, xs, torch.ones_like(d1), create_graph=True)
    d3, = torch.autograd.grad(d2, xs, torch.ones_like(d2), create_graph=True)
    print(f"hidden layers {depth}: nodes behind u {count_nodes(us)}, u_x {count_nodes(d1)}, u_xx {count_nodes(d2)}, u_xxx {count_nodes(d3)}")
