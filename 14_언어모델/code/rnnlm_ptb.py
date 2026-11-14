# %% 0. Setup
# Week 2 (1): word-level language models on the Penn Treebank -- a plain RNN and an LSTM, same size.
#   Embedding -> RNN or LSTM -> Linear(H -> V); loss = cross-entropy of the next word at every position.
#   Training uses truncated backpropagation through time: 35 time steps per minibatch, the hidden state is
#   carried over to the next minibatch but the graph is cut (detach).
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

# %% 1. Data: one long sequence of word IDs per split, the same vocabulary for all three
def read(split):
    with open(ptb_path(split), encoding="utf-8") as f:        # downloaded to ../data on the first run
        return f.read().replace("\n", "<eos>").strip().split()

words = {s: read(s) for s in ["train", "valid", "test"]}
vocab = sorted(set(words["train"]))
word_to_id = {w: i for i, w in enumerate(vocab)}
id_to_word = {i: w for w, i in word_to_id.items()}
V = len(vocab)
data = {s: torch.tensor([word_to_id[w] for w in ws]) for s, ws in words.items()}
print({s: len(d) for s, d in data.items()}, " V =", V)

def batchify(seq, B):
    """cut one long sequence into B parallel rows: (B, L)"""
    L = len(seq) // B
    return seq[:B * L].view(B, L).to(device)

B, T = 20, 35                       # 20 rows read in parallel, 35 time steps per update
train, valid, test = batchify(data["train"], B), batchify(data["valid"], B), batchify(data["test"], B)
print("train:", tuple(train.shape), " steps per epoch:", math.ceil((train.size(1) - 1) / T))

# %% 2. The model
class RNNLM(nn.Module):
    def __init__(self, V, D=100, H=100, cell="lstm"):
        super().__init__()
        layer = nn.LSTM if cell == "lstm" else nn.RNN
        self.embed = nn.Embedding(V, D)                  # word ID -> vector
        self.rnn = layer(D, H, batch_first=True)
        self.out = nn.Linear(H, V)

    def forward(self, x, state=None):                    # x: (B, T) word IDs
        h, state = self.rnn(self.embed(x), state)        # h: (B, T, H), all hidden states
        return self.out(h), state                        # (B, T, V): scores of the next word

def detach(state):
    return tuple(s.detach() for s in state) if isinstance(state, tuple) else state.detach()

def perplexity(model, seq):
    model.eval()
    total, n, state = 0.0, 0, None
    with torch.no_grad():
        for k in range(0, seq.size(1) - 1, T):
            x, y = seq[:, k:k + T], seq[:, k + 1:k + 1 + T]
            x = x[:, :y.size(1)]
            logits, state = model(x, state)
            total += F.cross_entropy(logits.reshape(-1, V), y.reshape(-1), reduction="sum").item()
            n += y.numel()
    model.train()
    return math.exp(total / n)

# %% 3. Training: Adam, 10 epochs, gradient clipping
def train_model(cell, n_epochs=10):
    torch.manual_seed(0)
    model = RNNLM(V, cell=cell).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    hist, secs = [], []
    for epoch in range(n_epochs):
        t0 = time.time()
        state = None                             # no state yet: the layer starts from zeros
        for k in range(0, train.size(1) - 1, T):
            x, y = train[:, k:k + T], train[:, k + 1:k + 1 + T]      # y: x shifted by one word
            x = x[:, :y.size(1)]
            if state is not None:
                state = detach(state)            # truncated BPTT: keep the values, cut the graph
            logits, state = model(x, state)      # starts from the last state of the previous block
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)    # gradient clipping
            opt.step()
        secs.append(time.time() - t0)
        hist.append(perplexity(model, valid))
        print(f"{cell}: epoch {epoch + 1}, valid perplexity {hist[-1]:.1f}  ({secs[-1]:.0f} s)", flush=True)
    test_ppl = perplexity(model, test)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"{cell}: test perplexity {test_ppl:.1f},  {n_param} parameters,  {np.mean(secs):.1f} s per epoch")
    return model, {"valid": hist, "test": test_ppl, "sec_per_epoch": float(np.mean(secs)), "params": n_param}

rnn_model, rnn_res = train_model("rnn")
lstm_model, lstm_res = train_model("lstm")
with open(os.path.join(DATA_DIR, "rnnlm_results.json"), "w") as f:      # reused by gpt_ptb.py
    json.dump({"RNN": rnn_res, "LSTM": lstm_res}, f, indent=1)

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(range(1, len(rnn_res["valid"]) + 1), rnn_res["valid"], "o-", label="RNN")
ax.plot(range(1, len(lstm_res["valid"]) + 1), lstm_res["valid"], "s-", label="LSTM")
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("validation perplexity", fontsize=12); ax.legend(fontsize=11)
ax.set_title("word-level language models on PTB", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/rnnlm_ppl.png", dpi=200)
plt.show()

# %% 4. Generating text with the LSTM language model
def generate(model, prompt, n_words=25, tau=1.0, seed=0):
    """read the prompt; then sample the next word from softmax(scores / tau), feed it back, repeat"""
    torch.manual_seed(seed)
    model.eval()
    ids = [word_to_id[w] for w in prompt.split()]
    with torch.no_grad():
        logits, state = model(torch.tensor([ids], device=device))        # read the prompt
        for _ in range(n_words):
            p = torch.softmax(logits[0, -1] / tau, dim=0)                # distribution of the next word
            nxt = torch.multinomial(p, 1).item()                         # sample one word from it
            ids.append(nxt)
            logits, state = model(torch.tensor([[nxt]], device=device), state)   # feed it back
    model.train()
    return " ".join(id_to_word[i] for i in ids).replace("<eos>", "/")

prompt = "the company said it"
for tau in [0.5, 1.0, 1.5]:
    for seed in [1, 2]:
        print(f"LSTM, temperature {tau}, seed {seed}: {generate(lstm_model, prompt, tau=tau, seed=seed)}")
print(f"RNN,  temperature 1.0, seed 1: {generate(rnn_model, prompt, seed=1)}")
