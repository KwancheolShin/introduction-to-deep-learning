# Appendix: L-BFGS on  f(x, y) = x^2 / 20 + y^2  with torch.optim.LBFGS
import torch

w = torch.tensor([-7.0, 2.0], requires_grad=True)       # (x, y), start point

def f(w):
    return w[0] ** 2 / 20.0 + w[1] ** 2

# max_iter=1: one L-BFGS iteration per optimizer.step() (the default is 20)
optimizer = torch.optim.LBFGS([w], lr=1.0, max_iter=1, history_size=10,
                              line_search_fn="strong_wolfe")

n_eval = 0
def closure():                  # L-BFGS may evaluate f several times per step
    global n_eval
    n_eval += 1
    optimizer.zero_grad()
    loss = f(w)
    loss.backward()
    return loss

for k in range(1, 9):
    optimizer.step(closure)
    print(f"iteration {k}: (x, y) = ({w[0].item():9.6f}, {w[1].item():9.6f}),  "
          f"f = {f(w).item():.3e},  evaluations so far = {n_eval}")
