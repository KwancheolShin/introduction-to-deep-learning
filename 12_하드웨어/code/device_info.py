# Reading: hardware -- what is inside my computer? (CPU, memory, GPU, software versions)
#   python device_info.py
# Prints the facts that the reading talks about: number of SMs, VRAM, compute capability,
# CUDA / cuDNN versions seen by PyTorch, and the TF32 switches.
# The same text is saved to ../results/device_info.txt

import os
import platform
import subprocess

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "results", "device_info.txt")

lines = []
def say(s=""):
    print(s)
    lines.append(s)

say(f"OS              : {platform.system()} {platform.release()}")
say(f"CPU             : {platform.processor()}")
say(f"logical CPUs    : {os.cpu_count()}")
say(f"Python          : {platform.python_version()}")
say(f"PyTorch         : {torch.__version__}")
say(f"CUDA (PyTorch)  : {torch.version.cuda}")       # the CUDA runtime bundled with PyTorch
say(f"cuDNN           : {torch.backends.cudnn.version()}")
say()

if not torch.cuda.is_available():
    say("no CUDA GPU found")
else:
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        say(f"GPU {i}           : {p.name}")
        say(f"  VRAM          : {p.total_memory / 2**30:.1f} GiB")
        say(f"  SMs           : {p.multi_processor_count}")
        say(f"  compute cap.  : {p.major}.{p.minor}   (sm_{p.major}{p.minor})")
    say()
    # matmul in float32 uses TF32 tensor cores only if this is True (default False)
    say(f"matmul allow_tf32 : {torch.backends.cuda.matmul.allow_tf32}")
    # convolutions in float32 use TF32 if this is True (default True!)
    say(f"cudnn  allow_tf32 : {torch.backends.cudnn.allow_tf32}")
    say(f"bf16 supported    : {torch.cuda.is_bf16_supported()}")
    say()
    # the driver: nvidia-smi shows the newest CUDA version the driver can run
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,clocks.max.memory,"
                            "pcie.link.gen.max,pcie.link.width.max", "--format=csv"],
                           capture_output=True, text=True, check=True)
        say("nvidia-smi:")
        for row in q.stdout.strip().splitlines():
            say("  " + row)
    except (OSError, subprocess.CalledProcessError):
        say("nvidia-smi not found")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
