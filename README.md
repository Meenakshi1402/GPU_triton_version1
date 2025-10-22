# GPU Blackjack Triton (BCG-style, no GA)

Professional GPU-accelerated Blackjack simulator using Triton.
- True rules: hard/soft, dealer S17, **double**, **true splits** up to 4 hands, split-aces stop, BJ 3:2 (single hand).
- Clean timing: CUDA event (kernel ms) + wall clock (s) + throughput.
- Tiled runs to scale to 1e8–1e9 games.
- Tiny tests.

## Quickstart
```bash
pip install -e .
gbj-bench --mode once  --workers 32768 --G 2048 --R 28 --seed 12345
gbj-bench --mode tiled --workers 65536 --G 1024 --R 28 --target 500000000 --seed 2025
