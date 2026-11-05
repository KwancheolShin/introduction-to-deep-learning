# %% 0. Setup
# Week: Techniques for Training -- the final MNIST classifier, built from OUR numpy modules
#   mnist_data.py  : data (55000 train / 5000 validation / 10000 test)
#   layers.py      : MultiLayerNet with He initialization, batch norm, (dropout), ReLU
#   optimizers.py  : AdamW (Adam + decoupled weight decay on W only)
# The same settings as mnist_torch.py, so the two programs can be compared line by line.
#   * keep the epoch with the best VALIDATION accuracy; the test data is used only ONCE at the end
# Figures are saved to ../figures/ (next to main.tex), wherever this file is run from
#   Cells: each '# %%' starts a cell. In VS Code, Run Cell (Shift+Enter) sends one cell to the
#   Interactive window (variables stay in the kernel); 'python <file>' still runs the whole file.

import os
import time
import copy
import numpy as np
import matplotlib.pyplot as plt

from mnist_data import load_mnist_split
from layers import MultiLayerNet, BatchNormalization
from optimizers import AdamW

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# %% 1. Data
# =====================================================
# Data
# =====================================================
x_train, t_train, x_val, t_val, x_test, t_test = load_mnist_split()   # 55000 / 5000 / 10000
print("train:", x_train.shape, " validation:", x_val.shape, " test:", x_test.shape)

# %% 2. Model and optimizer (re-run this cell to start training from scratch)
# =====================================================
# Model: [Affine -> BatchNorm -> ReLU -> (Dropout)] x 4 -> Affine -> Softmax-loss,  He initialization
# =====================================================
np.random.seed(1)
net = MultiLayerNet(784, [100, 100, 100, 100], 10,
                    activation="relu", loss="ce", weight_init_std="he",
                    use_batchnorm=True, use_dropout=False, dropout_ratio=0.2)
print("layers:", " -> ".join(type(layer).__name__ for layer in net.layers))
print("number of parameters:", net.num_params())

optimizer = AdamW(lr=0.001, weight_decay=0.01)   # decays only the weights W1, W2, ...

# %% 3. Save / restore the network state
# =====================================================
# Save / restore everything the network has learned
#   params: W, B, gamma, beta     +     running mean / var of every BatchNorm layer
#
#   세 가지 copy (이름은 같지만 서로 다른 것):
#     copy.copy(d)      copy 모듈의 함수, 얕은 복사: dict는 새로 만들지만 안의 배열은 원래 것을 공유한다
#                       (학습 중 params[key] -= ... 가 그 배열을 그 자리에서 바꾸므로, 복사본도 같이 바뀐다)
#     copy.deepcopy(d)  copy 모듈의 함수, 깊은 복사: dict도, 안의 배열도 모두 새로 만든다 -> params를 안전하게 저장
#     arr.copy()        numpy 배열 자신의 메서드: 새 메모리를 잡아 숫자 값을 전부 복사한다
#                       (숫자 배열에는 참조가 아니라 숫자가 들어 있으므로, 이것만으로 완전한 복사이다)
#   -> 배열이 든 dict는 copy.deepcopy(...),  배열 하나는 arr.copy()  (torch tensor는 t.clone())
# =====================================================
def get_state(net):
    state = {"params": copy.deepcopy(net.params), "bn": []}
    for layer in net.layers:
        if isinstance(layer, BatchNormalization):
            state["bn"].append((layer.running_mean.copy(), layer.running_var.copy()))
    return state


def set_state(net, state):
    for key in net.params:
        net.params[key][...] = state["params"][key]      # IN PLACE: the layers share these arrays
    bn_layers = [layer for layer in net.layers if isinstance(layer, BatchNormalization)]
    for layer, (mean, var) in zip(bn_layers, state["bn"]):
        layer.running_mean, layer.running_var = mean.copy(), var.copy()

# %% 4. Training (takes a while)
# =====================================================
# Training
# =====================================================
epochs = 15
batch_size = 100
N = x_train.shape[0]
iter_per_epoch = N // batch_size

np.random.seed(2)                                 # mini-batch order (and dropout masks)
history = {"loss": [], "train": [], "val": []}
best_val, best_epoch, best_state = 0.0, 0, None
t0 = time.time()
for epoch in range(1, epochs + 1):
    perm = np.random.permutation(N)
    total_loss = 0.0
    for it in range(iter_per_epoch):
        idx = perm[it * batch_size:(it + 1) * batch_size]
        loss = net.loss(x_train[idx], t_train[idx])   # forward (batch statistics, dropout on)
        grads = net.backward()                        # backward through every layer
        optimizer.update(net.params, grads)           # AdamW step
        total_loss += loss
    history["loss"].append(total_loss / iter_per_epoch)
    history["train"].append(net.accuracy(x_train, t_train))   # running statistics, dropout off
    history["val"].append(net.accuracy(x_val, t_val))
    print(f"epoch {epoch:2d} | loss {history['loss'][-1]:.4f} | "
          f"train acc {history['train'][-1]:.4f} | val acc {history['val'][-1]:.4f} "
          f"({time.time() - t0:.0f}s)")
    if history["val"][-1] > best_val:             # keep the epoch with the best VALIDATION accuracy
        best_val, best_epoch = history["val"][-1], epoch
        best_state = get_state(net)

# %% 5. Test accuracy: ONCE, at the end
# =====================================================
# Only at the very end: the test data, ONCE, with the chosen network
# =====================================================
set_state(net, best_state)
print(f"chosen epoch {best_epoch}: val acc {net.accuracy(x_val, t_val):.4f}, "
      f"test acc {net.accuracy(x_test, t_test):.4f}")

# %% 6. Figure
# =====================================================
# Figure
# =====================================================
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(range(1, epochs + 1), history["loss"], "o-")
ax[0].set_yscale("log")
ax[0].set(xlabel="epoch", ylabel="training loss (epoch mean)", title="Training Loss (AdamW, numpy)")
ax[0].grid(True, which="both", alpha=0.3)
ax[1].plot(range(1, epochs + 1), history["train"], "o--", label="train")
ax[1].plot(range(1, epochs + 1), history["val"], "o-", label="validation")
ax[1].axvline(best_epoch, color="gray", linestyle=":", label=f"chosen epoch ({best_epoch})")
ax[1].set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
ax[1].grid(True, alpha=0.3)
ax[1].legend()
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "numpy_final.png"), dpi=200, bbox_inches="tight")
plt.close(fig)
