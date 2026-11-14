# %% 0. Setup
# Plots for the notes that are not training results:
#   1. the idea of dimensionality reduction (points spread along one direction)
#   2. negative sampling: word frequencies in PTB and the same raised to the power 0.75
#   3. gradients through time in an RNN: repeated multiplication by W_h^T explodes or vanishes
#   4. perplexity: two models' predicted probabilities
#   5. attention: scores and softmax weights
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import matplotlib.pyplot as plt
from text_utils import load_ptb

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
plt.rcParams.update({"font.size": 13})

# %% 1. Dimensionality reduction: 2-D points that are spread mostly along one direction
rng = np.random.default_rng(0)
t = rng.normal(size=60)
P = np.c_[t, 0.6 * t] + 0.12 * rng.normal(size=(60, 2))
u = np.linalg.svd(P - P.mean(0))[2][0]                     # direction of the largest spread
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].scatter(P[:, 0], P[:, 1], s=14, color="tab:blue")
ax[0].set_title("2 numbers per point (x, y)")
ax[1].scatter(P[:, 0], P[:, 1], s=14, color="tab:blue", alpha=0.35)
L = np.linspace(-3, 3, 2)
ax[1].plot(L * u[0], L * u[1], color="tab:red", lw=2, label="new axis")
proj = (P @ u)[:, None] * u
ax[1].scatter(proj[:, 0], proj[:, 1], s=14, color="tab:red")
ax[1].set_title("1 number per point (position on the axis)")
ax[1].legend(loc="upper left")
for a in ax:
    a.set_aspect("equal"); a.set_xlim(-3, 3); a.set_ylim(-2.2, 2.2); a.set_xticks([]); a.set_yticks([])
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/svd_idea.png", dpi=200)
plt.show()

# %% 2. Negative sampling distribution in PTB
corpus, word_to_id, id_to_word = load_ptb("train")
counts = np.bincount(corpus)
p = counts / counts.sum()
q = counts**0.75 / (counts**0.75).sum()
top = np.argsort(-counts)[:8]
rare = [word_to_id[w] for w in ["toyota", "queen", "galvanized"] if w in word_to_id]
idx = list(top) + rare
fig, ax = plt.subplots(figsize=(9, 3.4))
x = np.arange(len(idx))
ax.bar(x - 0.2, p[idx], 0.4, label="frequency P(w)")
ax.bar(x + 0.2, q[idx], 0.4, label="P(w)^0.75, normalized")
ax.set_xticks(x); ax.set_xticklabels([id_to_word[i] for i in idx], rotation=30)
ax.set_yscale("log"); ax.set_ylabel("probability"); ax.legend()
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/neg_sampling_dist.png", dpi=200)
plt.show()
for i in idx:
    print(f"{id_to_word[i]:10s} P = {p[i]:.2e}   P^0.75 normalized = {q[i]:.2e}   ratio {q[i] / p[i]:.2f}")

# %% 3. Gradient through time: dh <- dh W_h^T, 20 times, for a large and a small W_h
N, H, T = 2, 3, 20
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
Q0 = np.linalg.qr(np.random.default_rng(3).normal(size=(H, H)))[0]   # a rotation: keeps lengths
for k, (scale, title) in enumerate([(1.3, "W_h = 1.3 x rotation: explodes"), (0.7, "W_h = 0.7 x rotation: vanishes")]):
    Wh = scale * Q0
    dh = np.ones((N, H))
    norms = []
    for _ in range(T):
        dh = dh @ Wh.T
        norms.append(np.sqrt((dh**2).sum()) / N)
    ax[k].plot(range(1, T + 1), norms, "o-", ms=4)
    ax[k].set_xlabel("time steps back"); ax[k].set_ylabel("size of gradient"); ax[k].set_title(title)
    ax[k].set_xticks([1, 5, 10, 15, 20])
    print(title, ": after 1, 10, 20 steps:", [round(norms[i], 4) for i in (0, 9, 19)])
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/rnn_grad_norm.png", dpi=200)
plt.show()

# %% 4. Perplexity: probability given to the correct word "say"
words = ["you", "say", "goodbye", "and", "i", "hello", "."]
m1 = np.array([0.05, 0.8, 0.05, 0.03, 0.03, 0.02, 0.02])
m2 = np.array([0.15, 0.2, 0.15, 0.15, 0.15, 0.1, 0.1])
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
for a, m, name in [(ax[0], m1, "model 1"), (ax[1], m2, "model 2")]:
    a.bar(words, m, color=["tab:red" if w == "say" else "tab:gray" for w in words])
    a.set_title(f"{name}: P(say) = {m[1]}\nperplexity = 1 / {m[1]} = {1 / m[1]:.2f}", fontsize=12)
    a.tick_params(axis="x", rotation=30)
ax[0].set_ylabel("probability of the next word")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/perplexity_example.png", dpi=200)
plt.show()

# %% 5. Attention: scores h . hs_j and weights softmax(s) for the 4-word input "I am a cat"
hs = np.array([[0.5, 0.0, 0.0],      # I      (rows of hs: made-up numbers, H = 3)
               [0.0, 0.5, 0.0],      # am
               [0.2, 0.5, 0.5],      # a
               [1.0, 1.1, 1.2]])     # cat
h = np.array([1.0, 1.0, 1.0])        # decoder state (made up)
s = hs @ h
a = np.exp(s) / np.exp(s).sum()
print("scores s =", s.round(3), "  weights a =", a.round(3), "  c =", (a @ hs).round(3))
labels = ["I", "am", "a", "cat"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.0))
ax[0].bar(labels, s, color="tab:gray"); ax[0].set_title("scores  s_j = h . hs_j"); ax[0].set_ylim(0, 3.8)
ax[1].bar(labels, a, color="tab:orange"); ax[1].set_title("weights  a = softmax(s)"); ax[1].set_ylim(0, 1)
for i in range(4):
    ax[0].text(i, s[i] + 0.1, f"{s[i]:.1f}", ha="center")
    ax[1].text(i, a[i] + 0.03, f"{a[i]:.2f}", ha="center")
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/attention_softmax.png", dpi=200)
plt.show()
