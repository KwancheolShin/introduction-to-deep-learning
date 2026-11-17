# Reading: hardware -- charts from published specifications (no GPU needed)
#   (1) bandwidth of every path data travels, from a hard disk to HBM
#   (2) matmul speed of data-center and consumer GPUs in different number formats
# Values are rounded vendor figures (dense, no sparsity); the RTX 3050 values are
# our own measurements from gpu_bench.py (../results/gpu_bench.json).
# The figures are saved to ../figures/ (next to main.tex)

import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
with open(os.path.join(HERE, "..", "results", "gpu_bench.json"), encoding="utf-8") as f:
    bench = json.load(f)

# =====================================================
# (1) bandwidth ladder [GB/s], one direction
# =====================================================
#        label,                                   GB/s,   kind
paths = [("HDD (3.5 inch)",                          0.25, "storage"),
         ("SATA SSD",                                0.55, "storage"),
         ("NVMe SSD, PCIe 4.0 x4",                   7.0,  "storage"),
         ("PCIe 4.0 x4  (our laptop GPU, measured)", bench["pcie_gbs"]["pinned"], "link"),
         ("PCIe 5.0 x8",                             31.5, "link"),
         ("PCIe 5.0 x16",                            63.0, "link"),
         ("DDR5-5600, 2 channels (CPU memory)",      89.6, "memory"),
         ("GDDR6, RTX 3050 laptop (measured)",       bench["copy_gbs"], "memory"),
         ("NVLink 4, H100 (one direction)",          450,  "link"),
         ("GDDR7, RTX 5090",                         1792, "memory"),
         ("HBM3, H100",                              3350, "memory"),
         ("HBM3e, B200",                             8000, "memory")]
colors = {"storage": "#8c8c8c", "link": "#dd8452", "memory": "#4c72b0"}

fig, ax = plt.subplots(figsize=(7.2, 4.4))
for i, (label, v, kind) in enumerate(paths):
    ax.barh(i, v, color=colors[kind])
    ax.text(v * 1.12, i, f"{v:,.0f}" if v >= 10 else f"{v:.2g}", va="center", fontsize=8)
ax.set_yticks(range(len(paths)), [p[0] for p in paths], fontsize=8)
ax.set_xscale("log")
ax.set_xlim(0.1, 30000)
ax.set_xlabel("bandwidth [GB/s], log scale")
for kind, c in colors.items():
    ax.barh(-10, 1, color=c, label={"storage": "storage", "link": "link between chips",
                                     "memory": "memory to its processor"}[kind])
ax.set_ylim(-0.6, len(paths) - 0.4)
ax.legend(fontsize=8, loc="lower right")
ax.grid(alpha=0.3, axis="x", which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bandwidth_ladder.png"), dpi=200, bbox_inches="tight")

# =====================================================
# (2) peak matmul speed [TFLOPS] by number format
# =====================================================
#          GPU:            FP64,  FP32,  TF32,  BF16
gpus = {"RTX 3050 laptop\n(measured)": [max(v for v in bench["matmul_tflops"][k] if v)
                                         for k in ["FP64", "FP32", "TF32", "BF16"]],
        "RTX 5090":        [1.6,   105,   105,   210],
        "A100":            [19.5,  19.5,  156,   312],
        "H100":            [67,    67,    495,   990],
        "B200":            [40,    80,    1100,  2250]}
fmts = ["FP64", "FP32", "TF32", "BF16"]
fcol = ["#8c8c8c", "#4c72b0", "#55a868", "#c44e52"]
fig, ax = plt.subplots(figsize=(7.2, 3.6))
w = 0.2
for j, (fmt, c) in enumerate(zip(fmts, fcol)):
    xs = [i + (j - 1.5) * w for i in range(len(gpus))]
    ax.bar(xs, [v[j] for v in gpus.values()], w, color=c, label=fmt)
ax.set_xticks(range(len(gpus)), list(gpus), fontsize=8)
ax.set_yscale("log")
ax.set_ylabel("peak speed [TFLOPS], log scale")
ax.legend(fontsize=8, ncol=4, loc="upper left")
ax.grid(alpha=0.3, axis="y", which="both")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "gpu_formats.png"), dpi=200, bbox_inches="tight")
# =====================================================
# (3) bandwidth = width [bits] x rate per wire [Gbit/s] / 8
#     (line-code overhead such as 8b/10b is ignored)
# =====================================================
#        label,                         width, Gbit/s per wire, kind,     label offset (points)
links = [("SATA",                          1,    6,   "storage", (6, -10)),
         ("PCIe 5.0, one lane",            1,   32,   "link",    (6, 4)),
         ("PCIe 5.0 x16",                 16,   32,   "link",    (6, 4)),
         ("DDR5-5600, one channel",       64,    5.6, "memory",  (6, -10)),
         ("DDR5-5600, 2 channels",       128,    5.6, "memory",  (6, 4)),
         ("GDDR6, RTX 3050 laptop",      128,   14,   "memory",  (6, 4)),
         ("GDDR7, RTX 5090",             512,   28,   "memory",  (-40, 8)),
         ("HBM3e, one stack",           1024,    8,   "memory",  (6, -10)),
         ("HBM3e, B200 (8 stacks)",     8192,    8,   "memory",  (-70, 8))]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
ws = [2**k for k in range(0, 15)]
for gbs in [1, 10, 100, 1000, 10000]:          # iso-bandwidth lines: rate = 8 * GB/s / width
    ax.plot(ws, [8 * gbs / w for w in ws], ":", color="#999999", lw=0.8)
    w0 = 8 * gbs / 60                          # where the line leaves the top of the plot
    if 1 <= w0 <= 2**14:
        ax.text(w0 * 1.1, 52, f"{gbs:,} GB/s" if gbs < 1000 else f"{gbs // 1000} TB/s",
                fontsize=7, color="#777777", rotation=-38)
for label, w, r, kind, off in links:
    ax.plot(w, r, "o", color=colors[kind], ms=6)
    ax.annotate(label, (w, r), textcoords="offset points", xytext=off, fontsize=7.5)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlim(0.7, 2**14)
ax.set_ylim(3, 60)
ax.set_xticks([1, 4, 16, 64, 256, 1024, 4096, 16384], ["1", "4", "16", "64", "256", "1024", "4096", "16384"])
ax.set_yticks([4, 8, 16, 32], ["4", "8", "16", "32"])
ax.yaxis.set_minor_formatter(plt.NullFormatter())
ax.xaxis.set_minor_formatter(plt.NullFormatter())
ax.set_xlabel("width: wires (bits) sent at the same time")
ax.set_ylabel("rate per wire [Gbit/s]")
ax.text(1.1, 3.5, "dotted lines: same bandwidth", fontsize=7, color="#777777")
ax.grid(alpha=0.25, which="major")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bus_width_rate.png"), dpi=200, bbox_inches="tight")
print("saved bandwidth_ladder.png, gpu_formats.png, bus_width_rate.png")
