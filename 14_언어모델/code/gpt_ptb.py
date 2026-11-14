# %% 0. Setup
# Week 3 (2): a small GPT (decoder-only Transformer) as a word-level language model on the Penn Treebank.
#   Same data and same task as rnnlm_ptb.py: predict the next word at every position.
#   word embedding + position embedding -> 2 blocks [masked self-attention, add & norm, feed forward, add & norm]
#   -> Linear(d -> V).  No RNN: all T positions of a chunk are computed at once, the causal mask hides the future.
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import math
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from text_utils import DATA_DIR, ptb_path

torch.manual_seed(0)
FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 1. Data: one long sequence of word IDs per split (as in rnnlm_ptb.py)
def read(split):
    with open(ptb_path(split), encoding="utf-8") as f:        # downloaded to ../data on the first run
        return f.read().replace("\n", "<eos>").strip().split()

words = {s: read(s) for s in ["train", "valid", "test"]}
vocab = sorted(set(words["train"]))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}
V = len(vocab)
train, valid, test = [torch.tensor([word_to_id[w] for w in words[s]], device=device) for s in ["train", "valid", "test"]]

T = 64                              # context length: the model sees at most 64 words
B = 20                              # chunks per minibatch

def get_batch():
    """B random chunks of T+1 words; input = the first T words, target = the last T words"""
    start = torch.randint(0, len(train) - T - 1, (B,), device=device)
    chunk = train[start[:, None] + torch.arange(T + 1, device=device)]      # (B, T+1)
    return chunk[:, :-1], chunk[:, 1:]

# %% 2. The model
class SelfAttention(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.n_head = n_head
        self.W_Q, self.W_K, self.W_V = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.W_O = nn.Linear(d, d)               # applied after the heads are joined
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        self.register_buffer("future", future)   # future[i, j] is True where j > i

    def forward(self, x):                        # x: (B, T, d)
        B, T, d = x.shape
        # split every vector into n_head parts: (B, T, d) -> (B, heads, T, d/heads)
        split = lambda z: z.view(B, T, self.n_head, -1).transpose(1, 2)
        q, k, v = split(self.W_Q(x)), split(self.W_K(x)), split(self.W_V(x))
        S = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))      # scores q_i . k_j
        S = S.masked_fill(self.future[:T, :T], float("-inf"))    # causal mask
        A = torch.softmax(S, dim=-1)             # (B, heads, T, T), each row sums to 1
        O = (A @ v).transpose(1, 2).reshape(B, T, d)             # heads joined again
        return self.W_O(O)

class Block(nn.Module):
    def __init__(self, d, n_head):
        super().__init__()
        self.attn = SelfAttention(d, n_head)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)

    def forward(self, x):
        x = self.norm1(x + self.attn(x))         # masked self-attention, add & norm
        x = self.norm2(x + self.ff(x))           # feed forward, add & norm
        return x

class GPT(nn.Module):
    def __init__(self, V, d=100, n_head=4, n_layer=2):
        super().__init__()
        self.tok = nn.Embedding(V, d)            # word ID  -> vector
        self.pos = nn.Embedding(T, d)            # position -> vector
        self.blocks = nn.Sequential(*[Block(d, n_head) for _ in range(n_layer)])
        self.out = nn.Linear(d, V)

    def forward(self, x):                        # x: (B, T) word IDs
        h = self.tok(x) + self.pos(torch.arange(x.size(1), device=x.device))  # (B, T, d)
        return self.out(self.blocks(h))          # (B, T, V): scores of the next word

model = GPT(V).to(device)
n_param = sum(p.numel() for p in model.parameters())
parts = {"word embedding": model.tok.weight.numel(), "position embedding": model.pos.weight.numel(),
         "one block": sum(p.numel() for p in model.blocks[0].parameters()), "output layer": sum(p.numel() for p in model.out.parameters())}
print(f"{n_param} parameters:", parts)

# %% 3. Perplexity: every word is predicted from at least T/2 earlier words
def perplexity(model, seq):
    model.eval()
    half = T // 2
    windows = seq.unfold(0, T + 1, half)                            # windows of T+1 words, moved by T/2 words
    total, n = 0.0, 0
    with torch.no_grad():
        for k in range(0, windows.size(0), 256):
            w = windows[k:k + 256]
            loss = F.cross_entropy(model(w[:, :-1]).transpose(1, 2), w[:, 1:], reduction="none")   # (b, T)
            keep = torch.ones_like(loss, dtype=torch.bool)
            keep[:, :half] = False                                  # the first half was scored in the previous window,
            if k == 0:
                keep[0, :half] = True                               # except at the very beginning
            total += loss[keep].sum().item()
            n += int(keep.sum())
    model.train()
    return math.exp(total / n)

# %% 4. Training: Adam, 10 epochs, gradient clipping (the same settings as rnnlm_ptb.py)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
steps_per_epoch = len(train) // (B * T)
hist, secs = [], []
for epoch in range(10):
    t0 = time.time()
    for _ in range(steps_per_epoch):
        x, y = get_batch()
        loss = F.cross_entropy(model(x).reshape(-1, V), y.reshape(-1))     # all B*T positions at once
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    secs.append(time.time() - t0)
    hist.append(perplexity(model, valid))
    print(f"gpt: epoch {epoch + 1}, valid perplexity {hist[-1]:.1f}  ({secs[-1]:.0f} s)", flush=True)
test_ppl = perplexity(model, test)
print(f"gpt: test perplexity {test_ppl:.1f},  {np.mean(secs):.1f} s per epoch,  {steps_per_epoch} steps per epoch")

results = {"GPT": {"valid": hist, "test": test_ppl, "sec_per_epoch": float(np.mean(secs)), "params": n_param}}
path = os.path.join(DATA_DIR, "rnnlm_results.json")                 # written by rnnlm_ptb.py
if os.path.exists(path):
    with open(path) as f:
        results = {**json.load(f), **results}
print(f"{'model':6s} {'parameters':>10s} {'best valid':>10s} {'valid':>7s} {'test':>7s} {'s/epoch':>8s}")
for name, r in results.items():
    print(f"{name:6s} {r['params']:10d} {min(r['valid']):10.1f} {r['valid'][-1]:7.1f} {r['test']:7.1f} {r['sec_per_epoch']:8.1f}")

fig, ax = plt.subplots(figsize=(5.4, 3.4))
for (name, r), m in zip(results.items(), ["o-", "s-", "^-"]):
    ax.plot(range(1, len(r["valid"]) + 1), r["valid"], m, label=name)
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("validation perplexity", fontsize=12); ax.legend(fontsize=11)
ax.set_title("word-level language models on PTB", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/gpt_ppl.png", dpi=200)
plt.show()

# %% 5. Generating text: sample the next word, append it, repeat
def generate(model, prompt, n_words=25, tau=1.0, seed=0):
    """start from the prompt; sample the next word from softmax(scores / tau), append it, repeat"""
    torch.manual_seed(seed)
    model.eval()
    ids = [word_to_id[w] for w in prompt.split()]
    with torch.no_grad():
        for _ in range(n_words):
            x = torch.tensor([ids[-T:]], device=device)        # the whole text so far (at most T words)
            scores = model(x)[0, -1]                           # scores of the next word: the last position
            p = torch.softmax(scores / tau, dim=0)
            ids.append(torch.multinomial(p, 1).item())         # sample one word and append it
    model.train()
    return " ".join(id_to_word[i] for i in ids).replace("<eos>", "/")

prompt = "the company said it"
for tau in [0.5, 1.0, 1.5]:
    for seed in [1, 2]:
        print(f"GPT, temperature {tau}, seed {seed}: {generate(model, prompt, tau=tau, seed=seed)}")
