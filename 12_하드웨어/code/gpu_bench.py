# Reading: hardware -- measure your own GPU
#   python gpu_bench.py
#   (1) matrix multiplication speed [TFLOPS] in FP64, FP32, TF32, FP16 and BF16
#   (2) accuracy of the same product in each format, compared with an FP64 reference
#   (3) memory bandwidth inside the GPU [GB/s]: a copy and an elementwise a*x+b
#   (4) PCIe bandwidth CPU -> GPU [GB/s], ordinary (pageable) and pinned memory
# The numbers are saved to ../results/gpu_bench.json and the figures to ../figures/
# A laptop GPU changes its clock with temperature and power, so the numbers move by about 10%.

import json
import os
import time

import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "..", "figures")
OUT = os.path.join(HERE, "..", "results", "gpu_bench.json")

assert torch.cuda.is_available(), "this script needs an NVIDIA GPU"
dev = torch.device("cuda")
torch.manual_seed(0)


def gpu_time(fn, repeat):
    """Average time [s] of fn() on the GPU, measured with CUDA events."""
    for _ in range(3):              # warm-up: first calls pay for setup
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeat):
        fn()
    end.record()
    torch.cuda.synchronize()        # the GPU runs asynchronously: wait for it
    return start.elapsed_time(end) / 1000 / repeat


# =====================================================
# (1) matmul speed: C = A @ B costs 2 n^3 floating-point operations
# =====================================================
# name: (dtype, allow TF32 for float32 matmul?)
formats = {"FP64": (torch.float64, False),
           "FP32": (torch.float32, False),
           "TF32": (torch.float32, True),
           "FP16": (torch.float16, False),
           "BF16": (torch.bfloat16, False)}
sizes = [256, 512, 1024, 2048, 4096]

matmul = {}
for name, (dtype, tf32) in formats.items():
    torch.backends.cuda.matmul.allow_tf32 = tf32
    matmul[name] = []
    for n in sizes:
        if dtype == torch.float64 and n > 2048:     # FP64 is slow on GeForce cards
            matmul[name].append(None)
            continue
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        t = gpu_time(lambda: a @ b, repeat=max(2, int(4e10 / n**3)))
        matmul[name].append(2 * n**3 / t / 1e12)
        print(f"matmul {name:5s} n={n:5d}: {matmul[name][-1]:7.2f} TFLOPS")
torch.backends.cuda.matmul.allow_tf32 = False

# =====================================================
# (2) accuracy: relative error of A @ B against an FP64 reference
# =====================================================
n = 1024
a64 = torch.randn(n, n, device=dev, dtype=torch.float64)
b64 = torch.randn(n, n, device=dev, dtype=torch.float64)
ref = a64 @ b64
error = {}
for name, (dtype, tf32) in formats.items():
    torch.backends.cuda.matmul.allow_tf32 = tf32
    c = (a64.to(dtype) @ b64.to(dtype)).to(torch.float64)
    error[name] = (torch.linalg.norm(c - ref) / torch.linalg.norm(ref)).item()
    print(f"relative error {name:5s}: {error[name]:.2e}")
torch.backends.cuda.matmul.allow_tf32 = False

# =====================================================
# (3) memory bandwidth inside the GPU
# =====================================================
N = 64 * 2**20                                   # 64 M numbers = 256 MB in float32
x = torch.randn(N, device=dev)
y = torch.empty_like(x)
t_copy = gpu_time(lambda: y.copy_(x), repeat=50)
copy_gbs = 2 * 4 * N / t_copy / 1e9              # read 4 bytes + write 4 bytes per number
t_axpy = gpu_time(lambda: torch.add(x, 1.0, alpha=2.0, out=y), repeat=50)   # y = 1 + 2 x
axpy_gbs = 2 * 4 * N / t_axpy / 1e9
axpy_gflops = 2 * N / t_axpy / 1e9               # one multiply + one add per number
print(f"copy          : {copy_gbs:7.1f} GB/s")
print(f"y = 2x + 1    : {axpy_gbs:7.1f} GB/s, {axpy_gflops:7.1f} GFLOPS")

# =====================================================
# (4) PCIe: CPU memory -> GPU memory
# =====================================================
host = torch.randn(N)                            # ordinary (pageable) memory
host_pinned = host.pin_memory()                  # page-locked memory: the GPU can read it directly
def h2d(src):
    t0 = time.perf_counter()
    for _ in range(10):
        y.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    return 10 * 4 * N / (time.perf_counter() - t0) / 1e9
h2d(host); h2d(host_pinned)                      # warm-up
pcie = {"pageable": h2d(host), "pinned": h2d(host_pinned)}
print(f"CPU -> GPU    : pageable {pcie['pageable']:.1f} GB/s, pinned {pcie['pinned']:.1f} GB/s")

# =====================================================
# save the numbers
# =====================================================
p = torch.cuda.get_device_properties(0)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"gpu": p.name, "sms": p.multi_processor_count, "torch": torch.__version__,
               "sizes": sizes, "matmul_tflops": matmul, "rel_error_n1024": error,
               "copy_gbs": copy_gbs, "axpy_gbs": axpy_gbs, "axpy_gflops": axpy_gflops,
               "pcie_gbs": pcie}, f, indent=2)

# =====================================================
# figure 1: matmul speed against the matrix size
# =====================================================
colors = {"FP64": "#8c8c8c", "FP32": "#4c72b0", "TF32": "#55a868", "FP16": "#dd8452", "BF16": "#c44e52"}
fig, ax = plt.subplots(figsize=(6.4, 3.6))
for name in formats:
    pts = [(s, v) for s, v in zip(sizes, matmul[name]) if v is not None]
    ax.plot([s for s, _ in pts], [v for _, v in pts], "o-", color=colors[name], label=name)
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xticks(sizes, [str(s) for s in sizes])
ax.set_xlabel("matrix size n  (n x n times n x n)")
ax.set_ylabel("speed [TFLOPS]")
ax.set_ylim(top=max(matmul["BF16"]) * 3)
ax.set_title(p.name, fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8, ncol=5, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "bench_matmul.png"), dpi=200, bbox_inches="tight")

# =====================================================
# figure 2: measured roofline
#   arithmetic intensity I = FLOPs / bytes moved from memory
#   attainable speed = min(peak compute, I * memory bandwidth)
# =====================================================
bw = copy_gbs / 1000                              # TB/s, measured
fig, ax = plt.subplots(figsize=(6.4, 3.8))
I = [2**k for k in range(-4, 12)]
for name, ls in [("FP32", "-"), ("FP16", "--")]:
    peak = max(v for v in matmul[name] if v is not None)
    ax.plot(I, [min(peak, i * bw) for i in I], ls, color=colors[name], label=f"roof, {name} (best measured {peak:.1f} TFLOPS)")
    # n x n matmul: 2 n^3 FLOPs, reads A and B and writes C: 3 n^2 numbers
    nbytes = {"FP32": 4, "FP16": 2}[name]
    for s, v in zip(sizes, matmul[name]):
        ax.plot(2 * s**3 / (3 * s**2 * nbytes), v, "o", color=colors[name])
ax.plot(axpy_gflops / 1000 / bw, axpy_gflops / 1000, "s", color="black", label="y = 2x + 1 (FP32)")
ax.set_xscale("log", base=2)
ax.set_yscale("log")
ax.set_xlabel("arithmetic intensity [FLOP / byte]")
ax.set_ylabel("speed [TFLOPS]")
ax.text(I[1], I[1] * bw * 1.6, "memory-bound", rotation=30, fontsize=8)
ax.text(I[-5], max(matmul["FP16"]) * 1.25, "compute-bound", fontsize=8)
ax.set_title(f"{p.name}: measured bandwidth {copy_gbs:.0f} GB/s", fontsize=10)
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, "roofline_measured.png"), dpi=200, bbox_inches="tight")
print("saved", OUT)
