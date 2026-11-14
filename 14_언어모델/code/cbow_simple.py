# %% 0. Setup
# Week 1 (2): the CBOW model of word2vec on one sentence, with one-hot inputs (as in the figures).
#   context (the word on the left and on the right) -> W_in -> average -> W_out -> softmax over the words
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from text_utils import preprocess

torch.manual_seed(0)
np.set_printoptions(precision=3, suppress=True)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)

# %% 1. Contexts and targets (window 1)
corpus, word_to_id, id_to_word = preprocess("You say goodbye and I say hello.")
V = len(word_to_id)
contexts = np.array([[corpus[i - 1], corpus[i + 1]] for i in range(1, len(corpus) - 1)])  # (6, 2)
target = corpus[1:-1]                                                                     # (6,)
for c, t in zip(contexts, target):
    print(f"context ({id_to_word[c[0]]}, {id_to_word[c[1]]}) -> target {id_to_word[t]}")

# one-hot vectors: contexts (6, 2, 7), as in the figures
X = torch.nn.functional.one_hot(torch.tensor(contexts), V).float()
y = torch.tensor(target)

# %% 2. The model
class SimpleCBOW(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.W_in = nn.Linear(V, H, bias=False)    # shared by both context words
        self.W_out = nn.Linear(H, V, bias=False)

    def forward(self, X):                          # X: (B, 2, V) one-hot
        h = 0.5 * (self.W_in(X[:, 0]) + self.W_in(X[:, 1]))   # (B, H): average of two rows
        return self.W_out(h)                       # (B, V): scores (softmax is in the loss)

H = 3
model = SimpleCBOW(V, H)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# %% 3. Training: 3000 epochs, minibatch of 3
n_epochs = 3000
hist = []                                          # mean loss of each epoch (2 minibatches = all 6 examples)
for epoch in range(n_epochs):
    perm = torch.randperm(len(y))
    total = 0.0
    for k in range(0, len(y), 3):
        idx = perm[k:k + 3]
        loss = loss_fn(model(X[idx]), y[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    hist.append(total / len(y))
# (say, and) -> goodbye and (and, say) -> i have the same average, so each can get probability 1/2 at best
floor = 2 * np.log(2) / 6
print(f"loss: first epoch {hist[0]:.3f} (log V = {np.log(V):.3f}), epoch 1000 {hist[999]:.3f}, last epoch {hist[-1]:.3f};"
      f"  lowest possible = 2 log 2 / 6 = {floor:.3f}")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(range(1, n_epochs + 1), hist, lw=1.2, label="loss (mean of each epoch)")
ax.axhline(floor, color="tab:red", ls="--", lw=1, label=f"lowest possible loss = {floor:.3f}")
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("loss", fontsize=12); ax.set_ylim(0, 2.1)
ax.set_title("CBOW on one sentence", fontsize=12); ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/cbow_simple_loss.png", dpi=200)
plt.show()

# %% 4. Predictions for the six training examples
with torch.no_grad():
    P = torch.softmax(model(X), dim=1).numpy()                 # (6, 7)
print("P(? | you, goodbye):", {id_to_word[i]: round(float(P[0, i]), 3) for i in range(V)})
for n, (c, t) in enumerate(zip(contexts, target)):
    print(f"context ({id_to_word[c[0]]:7s}, {id_to_word[c[1]]:7s}) -> P({id_to_word[t]:7s}) = {P[n, t]:.3f}")

# %% 5. Word vectors: the rows of W_in
W_in = model.W_in.weight.detach().numpy().T       # nn.Linear stores (H, V); rows of W_in = word vectors
for i in range(V):
    print(f"{id_to_word[i]:8s}", W_in[i])
cos = lambda a, b: float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
print(f"cos(goodbye, i) = {cos(W_in[word_to_id['goodbye']], W_in[word_to_id['i']]):.3f},"
      f"  cos(goodbye, say) = {cos(W_in[word_to_id['goodbye']], W_in[word_to_id['say']]):.3f}")

fig, ax = plt.subplots(figsize=(4.2, 4.2))
vmax = np.abs(W_in).max()
im = ax.imshow(W_in, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
ax.set_yticks(range(V)); ax.set_yticklabels([id_to_word[i] for i in range(V)], fontsize=13)
ax.set_xticks(range(H)); ax.set_xticklabels([f"{k + 1}" for k in range(H)], fontsize=12)
ax.set_xlabel(f"component of the word vector (H = {H})", fontsize=12)
for i in range(V):
    for k in range(H):
        ax.text(k, i, f"{W_in[i, k]:.2f}", ha="center", va="center", fontsize=11,
                color="white" if abs(W_in[i, k]) > 0.6 * vmax else "black")
fig.colorbar(im, ax=ax, shrink=0.8)
ax.set_title("rows of W_in = word vectors", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/cbow_simple_W_in.png", dpi=200)
plt.show()
