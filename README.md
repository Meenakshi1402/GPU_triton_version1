%%writefile README.md
# GPU Blackjack Triton 

**GPU-accelerated Blackjack simulator** using Triton + PyTorch.  
- Clean timing: kernel vs wall time + throughput (games/sec)  
- Tiled runs for hundreds of millions of games on Colab’s T4  
- Simple CLI and Python API  

---

## 🚀 Run on Google Colab (recommended)

1) **Enable GPU**  
   Runtime → Change runtime type → **GPU** → Save

2) **Clone & install**

```bash
!git clone https://github.com/Meenakshi1402/GPU_triton.git
%cd GPU_triton
!pip install -e .
```
3) **Sanity check GPU + versions**
```
import torch, triton
print("Torch:", torch.__version__, "| Triton:", triton.__version__)
assert torch.cuda.is_available(), "No CUDA GPU detected."
```
4) **If you see ModuleNotFoundError for gpubj**
```
import sys
sys.path.append("/content/GPU_triton/src")
```

5) **Quick run**
```
from gpubj.runner import run_once
out = run_once(workers=2048, G=1000, R=28, seed=7)
for k, v in out.items(): print(f"{k:22}: {v}")
```

6) **Bigger benchmark (timed, ~67M games on T4)**
```
from gpubj.runner import run_once
import time

t0 = time.perf_counter()
out = run_once(workers=32768, G=2048, R=28, seed=12345)
elapsed = time.perf_counter() - t0

print("="*60)
print("GPU Blackjack Simulation Results")
print("="*60)
for k, v in out.items():
    print(f"{k:22}: {v}")
print("-"*60)
print(f"Elapsed wall-clock time: {elapsed:.3f} s")
print("="*60)
```
Tip: total games = workers × G.
Workers = Threads

7) **For very large totals (e.g. 100M+), prefer the tiled API:**
```
from gpubj.runner import run_tiled

# Run ~100 million games safely on Colab’s GPU
out = run_tiled(total_games_target=100_000_000,
                workers=32768,
                G=1024,
                R=28,
                seed=2025)

for k, v in out.items():
    print(f"{k:22}: {v}")
```
