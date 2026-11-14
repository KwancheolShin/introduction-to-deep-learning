# %% 0. Setup
# Week 3 (1): seq2seq with attention converts dates written in many formats to "YYYY-MM-DD".
#   "Friday, August 26, 1983" -> "1983-08-26"
#   decoder step t: score s_j = h_t . hs_j for every encoder position j, a = softmax(s), c = sum_j a_j hs_j,
#   output = Linear([c, h_t]).  Compared with the peeky seq2seq of week 2 (no attention).
import os; os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import time
import random
import datetime
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

FIG_DIR = "../figures"
os.makedirs(FIG_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

# %% 1. Data: 50,000 random dates, each written in one of several formats
random.seed(0)
def write_date(d):
    month, mon, day = d.strftime("%B"), d.strftime("%b"), d.strftime("%A")
    forms = [f"{month} {d.day}, {d.year}", f"{mon} {d.day}, {d.year}", f"{day}, {month} {d.day}, {d.year}",
             f"{d.month}/{d.day}/{d.strftime('%y')}", f"{d.month}/{d.day}/{d.year}"]
    s = random.choice(forms)
    return random.choice([s, s.lower(), s.upper()])

start = datetime.date(1970, 1, 1)                # the 20,001 days from 1970-01-01 to 2024-10-04
days = list(range(20001))
random.shuffle(days)
test_days, train_days = days[:2000], days[2000:] # a test date never appears in the training set, in any format

def make(day_pool, n):
    out = []
    for _ in range(n):
        d = start + datetime.timedelta(days=random.choice(day_pool))
        out.append((write_date(d), "_" + d.isoformat()))
    return out

n_train = 45000
pairs = make(train_days, n_train) + make(test_days, 5000)
for q, a in pairs[:5]:
    print(f"{q:32s} -> {a[1:]}")
print("test dates that are also in the training set:",
      len(set(a for _, a in pairs[:n_train]) & set(a for _, a in pairs[n_train:])))

LQ = max(len(q) for q, _ in pairs)
chars = sorted(set("".join(q + a for q, a in pairs)) | {" "})
c2i = {c: i for i, c in enumerate(chars)}
i2c = {i: c for c, i in c2i.items()}
Q = torch.tensor([[c2i[c] for c in q.ljust(LQ)] for q, _ in pairs]).flip(1)   # reversed input (week 2)
A = torch.tensor([[c2i[c] for c in a] for _, a in pairs])
Q_tr, A_tr, Q_te, A_te = Q[:n_train].to(device), A[:n_train].to(device), Q[n_train:].to(device), A[n_train:].to(device)
print(f"{len(chars)} characters, question length {LQ}, answer length {A.size(1) - 1}")

# %% 2. The models
class Seq2seq(nn.Module):
    def __init__(self, V, D=16, H=256, attention=True):
        super().__init__()
        self.attention = attention
        self.enc_embed, self.dec_embed = nn.Embedding(V, D), nn.Embedding(V, D)
        self.encoder = nn.LSTM(D, H, batch_first=True)
        self.decoder = nn.LSTM(D + (0 if attention else H), H, batch_first=True)   # peeky if no attention
        self.out = nn.Linear(2 * H, V)

    def decode(self, x, state, hs):                   # x: (B, t) decoder inputs, hs: (B, T, H) encoder outputs
        e = self.dec_embed(x)
        h_last = hs[:, -1, :]                         # last encoder hidden state
        if not self.attention:
            e = torch.cat([e, h_last[:, None].expand(-1, e.size(1), -1)], dim=2)
        h, state = self.decoder(e, state)             # (B, t, H): decoder hidden states
        if self.attention:
            s = h @ hs.transpose(1, 2)                # (B, t, T): scores h_t . hs_j
            a = torch.softmax(s, dim=2)               # attention weights, each row sums to 1
            c = a @ hs                                # (B, t, H): context vectors
        else:
            a, c = None, h_last[:, None].expand(-1, h.size(1), -1)
        logits = self.out(torch.cat([c, h], dim=2))
        return logits, state, a

    def forward(self, q, a_in):
        hs, (h, cc) = self.encoder(self.enc_embed(q))
        logits, _, _ = self.decode(a_in, (h, cc), hs)
        return logits

    def generate(self, q, n):
        hs, state = self.encoder(self.enc_embed(q))
        x, out, att = torch.full((q.size(0), 1), c2i["_"], device=q.device), [], []
        for _ in range(n):
            logits, state, a = self.decode(x, state, hs)
            x = logits.argmax(dim=2)
            out.append(x)
            if a is not None:
                att.append(a)
        return torch.cat(out, dim=1), (torch.cat(att, dim=1) if att else None)

# %% 3. Training: 10 epochs each, accuracy = whole answer correct on 5,000 test dates
def run(attention, n_epochs=10, batch=128):
    torch.manual_seed(0)
    model = Seq2seq(len(chars), attention=attention).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    accs = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n_train, device=device)
        for k in range(0, n_train, batch):
            b = perm[k:k + batch]
            logits = model(Q_tr[b], A_tr[b, :-1])
            loss = F.cross_entropy(logits.reshape(-1, len(chars)), A_tr[b, 1:].reshape(-1))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            pred, _ = model.generate(Q_te, A_te.size(1) - 1)
        model.train()
        accs.append((pred == A_te[:, 1:]).all(dim=1).float().mean().item())
        print(f"{'attention' if attention else 'peeky'}: epoch {epoch + 1}, accuracy {accs[-1]:.3f}", flush=True)
    return model, accs

t0 = time.time(); peeky_model, peeky_acc = run(False); print(f"peeky: {time.time() - t0:.0f} s")
t0 = time.time(); att_model, att_acc = run(True); print(f"attention: {time.time() - t0:.0f} s")

fig, ax = plt.subplots(figsize=(5.4, 3.4))
ax.plot(range(1, len(peeky_acc) + 1), peeky_acc, "s-", label="peeky seq2seq")
ax.plot(range(1, len(att_acc) + 1), att_acc, "o-", label="seq2seq + attention")
ax.set_xlabel("epoch", fontsize=12); ax.set_ylabel("test accuracy (exact match)", fontsize=12)
ax.set_ylim(0, 1.02); ax.legend(fontsize=11, loc="lower right")
ax.set_title("date format conversion", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/attention_date_accuracy.png", dpi=200)
plt.show()

# %% 4. Where does the decoder look?  Attention weights for one test date
i = next(k for k, (q, _) in enumerate(pairs[n_train:]) if q[0].isalpha() and "," in q and q.count(",") == 2)
att_model.eval()
with torch.no_grad():
    pred, att = att_model.generate(Q_te[i:i + 1], A_te.size(1) - 1)
q_str = pairs[n_train + i][0]
a_str = "".join(i2c[int(c)] for c in pred[0])
print("question:", q_str, " answer:", a_str, " true:", pairs[n_train + i][1][1:])
W = att[0].cpu().numpy()[:, ::-1]                    # undo the reversal: columns in reading order
W = W[:, :len(q_str)]
for t, ch in enumerate(a_str):                       # for each output character: where is the largest weight?
    j = int(W[t].argmax())
    print(f"  output '{ch}' looks most at input position {j:2d} '{q_str[j]}' (weight {W[t, j]:.2f});"
          f"  weight on the padding after the question: {1 - W[t].sum():.2f}")
fig, ax = plt.subplots(figsize=(9, 3.9))
ax.imshow(W, cmap="Blues", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(q_str))); ax.set_xticklabels(list(q_str), fontsize=13)
ax.set_yticks(range(len(a_str))); ax.set_yticklabels(list(a_str), fontsize=13)
ax.set_xticks(np.arange(-0.5, len(q_str)), minor=True); ax.set_yticks(np.arange(-0.5, len(a_str)), minor=True)
ax.grid(which="minor", color="0.85", lw=0.5); ax.tick_params(which="minor", length=0)
ax.set_xlabel("input (question)", fontsize=12); ax.set_ylabel("output (answer)", fontsize=12)
ax.set_title("attention weights (dark = large)", fontsize=12)
fig.tight_layout()
fig.savefig(f"{FIG_DIR}/attention_date_map.png", dpi=200)
plt.show()
