# %% 0. Setup
# Section 8: seq2seq learns addition from text, one character at a time.  "57+5" -> "62"
#   encoder: Embedding -> LSTM, its last hidden state h summarizes the question
#   decoder: Embedding -> LSTM (starts from h) -> Linear, generates the answer character by character
#   three versions: baseline, reversed input, reversed input + peeky (h is given to the decoder at every step)
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 1. Data: 50,000 different addition problems with numbers of 1 to 3 digits, as fixed-length strings
random.seed(0)
def rand_num():
    return random.randint(0, 10 ** random.randint(1, 3) - 1)

pairs, seen = [], set()
while len(pairs) < 50000:
    a, b = rand_num(), rand_num()
    key = (min(a, b), max(a, b))             # a+b and b+a count as the same problem
    if key not in seen:                      # every problem appears only once,
        seen.add(key)                        # so no test problem is in the training set
        pairs.append((a, b))
random.shuffle(pairs)

def carries(a, b):
    """True if adding a and b by hand needs a carry (some pair of digits adds up to 10 or more)"""
    while a and b:
        if a % 10 + b % 10 >= 10:
            return True
        a, b = a // 10, b // 10
    return False

problems = [(f"{a}+{b}".ljust(7),            # question, padded with spaces to 7 characters
             ("_" + str(a + b)).ljust(5))    # "_" marks the start of the answer; 5 characters
            for a, b in pairs]
print(problems[:3])

chars = sorted(set("".join(q + a for q, a in problems)))
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
Q = torch.tensor([[c2i[c] for c in q] for q, _ in problems])
A = torch.tensor([[c2i[c] for c in a] for _, a in problems])
n_train = 45000
Q_tr, A_tr, Q_te, A_te = Q[:n_train].to(device), A[:n_train].to(device), Q[n_train:].to(device), A[n_train:].to(device)
has_carry = torch.tensor([carries(a, b) for a, b in pairs[n_train:]], device=device)
print("characters:", chars, " train/test:", n_train, len(Q) - n_train)
print("test problems that are also in the training set:",
      len(set(q for q, _ in problems[:n_train]) & set(q for q, _ in problems[n_train:])))
print(f"test problems with a carry: {int(has_carry.sum())} of {len(has_carry)}")

# %% 2. The model
class Seq2seq(nn.Module):
    def __init__(self, V, D=16, H=128, peeky=False):
        super().__init__()
        self.peeky = peeky
        self.enc_embed = nn.Embedding(V, D)
        self.encoder = nn.LSTM(D, H, batch_first=True)
        self.dec_embed = nn.Embedding(V, D)
        self.decoder = nn.LSTM(D + (H if peeky else 0), H, batch_first=True)
        self.out = nn.Linear(H + (H if peeky else 0), V)

    def encode(self, q):
        _, (h, c) = self.encoder(self.enc_embed(q))
        return h                              # (1, B, H): the whole question in one vector

    def decode_step(self, x, state, h_enc):   # x: (B, t) characters given to the decoder
        e = self.dec_embed(x)
        if self.peeky:                        # give h to every time step
            e = torch.cat([e, h_enc[0][:, None, :].expand(-1, e.size(1), -1)], dim=2)
        o, state = self.decoder(e, state)
        if self.peeky:
            o = torch.cat([o, h_enc[0][:, None, :].expand(-1, o.size(1), -1)], dim=2)
        return self.out(o), state

    def forward(self, q, a_in):               # training: the true answer is the input
        h = self.encode(q)
        logits, _ = self.decode_step(a_in, (h, torch.zeros_like(h)), h)
        return logits

    def generate(self, q, n=4):               # testing: feed back its own output
        h = self.encode(q)
        state = (h, torch.zeros_like(h))      # the decoder starts from the encoder's h
        x = torch.full((q.size(0), 1), c2i["_"], device=q.device)     # the start symbol
        out = []
        for _ in range(n):
            logits, state = self.decode_step(x, state, h)
            x = logits.argmax(dim=2)          # next input: the most probable character
            out.append(x)
        return torch.cat(out, dim=1)

# %% 3. Training and accuracy (exact match on the 5,000 test problems)
def run(reverse, peeky, n_epochs=25, batch=128):
    torch.manual_seed(0)
    model = Seq2seq(len(chars), peeky=peeky).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    q_tr = Q_tr.flip(1) if reverse else Q_tr
    q_te = Q_te.flip(1) if reverse else Q_te
    accs = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        for k in range(0, n_train, batch):
            b = perm[k:k + batch]
            logits = model(q_tr[b], A_tr[b, :-1])                 # input  "_62 "
            loss = F.cross_entropy(logits.reshape(-1, len(chars)), A_tr[b, 1:].reshape(-1))   # target "62  "
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model.generate(q_te)
        model.train()
        accs.append((pred == A_te[:, 1:]).all(dim=1).float().mean().item())
    return model, accs

results = {}
for name, reverse, peeky in [("baseline", False, False), ("reverse", True, False), ("reverse + peeky", True, True)]:
    t0 = time.time()
    model, accs = run(reverse, peeky)
    results[name] = (model, reverse, accs)
    print(f"{name:16s}: accuracy after epoch 5, 10, 25 = {accs[4]:.3f}, {accs[9]:.3f}, {accs[-1]:.3f}  ({time.time() - t0:.0f} s)", flush=True)

fig, ax = plt.subplots(figsize=(5.4, 3.4))
for (name, (_, _, accs)), m in zip(results.items(), ["o", "s", "^"]):
    ax.plot(range(1, len(accs) + 1), accs, marker=m, ms=3, label=name)
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("test accuracy (exact match)", fontsize=12)
ax.set_ylim(0, 1); ax.legend(fontsize=11)
ax.set_title("seq2seq learning addition", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/seq2seq_accuracy.png", dpi=200)
plt.show()

# %% 4. Answers of the best model to test problems (none of them was in the training set)
model, reverse, _ = results["reverse + peeky"]
q_te = Q_te.flip(1) if reverse else Q_te
model.eval()
with torch.no_grad():
    pred = model.generate(q_te)
ok = (pred == A_te[:, 1:]).all(dim=1)
print(f"with a carry: {int(ok[has_carry].sum())} of {int(has_carry.sum())} correct;"
      f"  without: {int(ok[~has_carry].sum())} of {int((~has_carry).sum())} correct")
for i in range(10):
    q = "".join(i2c[int(c)] for c in Q_te[i]).strip()
    t = "".join(i2c[int(c)] for c in A_te[i, 1:]).strip()
    p = "".join(i2c[int(c)] for c in pred[i]).strip()
    print(f"{q:8s} true {t:5s} model {p:5s} {'O' if p == t else 'X'}{'  (carry)' if has_carry[i] else ''}")
