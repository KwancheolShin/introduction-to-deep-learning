# %% 0. Setup
# Section 4: CBOW with negative sampling on the Penn Treebank (10,000 words).
#   nn.Embedding instead of one-hot x matrix; "is this the right word?" (sigmoid) instead of a softmax over 10,000 words.
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from text_utils import load_ptb, most_similar

torch.manual_seed(0)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 1. Data: contexts (5 words on each side) and targets
window, H, n_neg = 5, 100, 5
batch, n_epochs = 100, 10
corpus, word_to_id, id_to_word = load_ptb("train")
V = len(word_to_id)
idx = np.arange(window, len(corpus) - window)
offsets = np.r_[-window:0, 1:window + 1]                      # -5..-1, 1..5
contexts = torch.tensor(corpus[idx[:, None] + offsets], device=device)   # (N, 10)
target = torch.tensor(corpus[idx], device=device)                        # (N,)
print("contexts:", tuple(contexts.shape), " target:", tuple(target.shape))

# negative samples are drawn with probability proportional to (count)^0.75
counts = np.bincount(corpus, minlength=V).astype(np.float64)
p_neg = torch.tensor(counts**0.75 / (counts**0.75).sum(), device=device, dtype=torch.float32)

# %% 2. The model
class CBOW(nn.Module):
    def __init__(self, V, H):
        super().__init__()
        self.W_in = nn.Embedding(V, H)               # row i = vector of word i (input side)
        self.W_out = nn.Embedding(V, H)              # row i = vector of word i (output side)
        nn.init.normal_(self.W_in.weight, std=0.01)
        nn.init.normal_(self.W_out.weight, std=0.01)

    def forward(self, ctx, words):                   # ctx (B, 10), words (B, K) -> scores (B, K)
        h = self.W_in(ctx).mean(dim=1)               # (B, H): average of the context vectors
        return (self.W_out(words) * h[:, None, :]).sum(dim=-1)   # dot products h . w_out

model = CBOW(V, H).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# %% 3. Training: one positive (the true word) and 5 negatives (random words) per context
hist, t0 = [], time.time()
N = len(target)
for epoch in range(n_epochs):
    perm = torch.randperm(N, device=device)
    run = 0.0
    for k in range(0, N, batch):
        b = perm[k:k + batch]
        neg = torch.multinomial(p_neg, len(b) * n_neg, replacement=True).view(len(b), n_neg)
        words = torch.cat([target[b, None], neg], dim=1)                       # (B, 6)
        labels = torch.zeros_like(words, dtype=torch.float32); labels[:, 0] = 1
        scores = model(contexts[b], words)
        loss = F.binary_cross_entropy_with_logits(scores, labels, reduction="sum") / len(b)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        run += loss.item()
        if (k // batch) % 1000 == 999:
            hist.append(run / 1000); run = 0.0
    print(f"epoch {epoch + 1}: loss {hist[-1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)
print(f"training time: {(time.time() - t0) / 60:.1f} min")

fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(np.arange(1, len(hist) + 1) * 1000, hist)
ax.set_xlabel("step", fontsize=12); ax.set_ylabel("loss (mean of 1000 steps)", fontsize=12)
ax.set_title("CBOW with negative sampling, PTB", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/cbow_ptb_loss.png", dpi=200)
plt.show()

# %% 4. Similar words and analogies
W = model.W_in.weight.detach().cpu().numpy()
for q in ["you", "year", "car", "toyota"]:
    print(q, "->", [(w, round(s, 3)) for w, s in most_similar(q, word_to_id, id_to_word, W)])

def analogy(a, b, c, top=5):
    """a : b = c : ?   ->  words closest to  v_b - v_a + v_c"""
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    q = Wn[word_to_id[b]] - Wn[word_to_id[a]] + Wn[word_to_id[c]]
    sims = Wn @ (q / np.linalg.norm(q))
    out = [(id_to_word[i], round(float(sims[i]), 3)) for i in np.argsort(-sims) if id_to_word[i] not in (a, b, c)]
    return out[:top]

for a, b, c in [("man", "king", "woman"), ("take", "took", "go"), ("car", "cars", "child"), ("good", "better", "bad")]:
    print(f"{a} : {b} = {c} : ?  ->", analogy(a, b, c))
print("how often the words appear in the corpus:",
      {w: int(counts[word_to_id[w]]) for w in ["king", "queen", "man", "woman", "took", "went", "children", "worse"]})

np.save("../data/cbow_ptb_W_in.npy", W)
