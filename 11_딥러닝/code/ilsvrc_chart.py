# Week: deep learning -- top-5 error of the ILSVRC classification winners (and VGG, 2nd in 2014)
# The figure is saved to ../figures/ (next to main.tex)

import os
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")

#          label,                       top-5 error [%], layers (None: not a deep CNN)
results = [("2010\nNEC-UIUC",           28.2, None),
           ("2011\nXRCE",               25.8, None),
           ("2012\nAlexNet",            16.4, 8),
           ("2013\nZFNet",              11.7, 8),
           ("2014\nVGG (2nd)",           7.3, 19),
           ("2014\nGoogLeNet",           6.7, 22),
           ("2015\nResNet",              3.6, 152)]

fig, ax = plt.subplots(figsize=(8, 3.4))
labels = [r[0] + ("" if r[2] is None else f"\n{r[2]} layers") for r in results]
errors = [r[1] for r in results]
colors = ["#bbbbbb" if r[2] is None else "#4c72b0" for r in results]
ax.bar(labels, errors, color=colors)
for i, e in enumerate(errors):
    ax.text(i, e + 0.6, f"{e}", ha="center", fontsize=9)
ax.axhline(5.1, ls="--", color="#c44e52", lw=1)
ax.text(0, 5.8, "one trained human: about 5.1", ha="center", fontsize=8, color="#c44e52",
        bbox=dict(fc="white", ec="none", pad=1))
ax.set_ylabel("top-5 error [%]")
ax.set_ylim(0, 31)
ax.tick_params(axis="x", labelsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "ilsvrc_top5.png"), dpi=200, bbox_inches="tight")
