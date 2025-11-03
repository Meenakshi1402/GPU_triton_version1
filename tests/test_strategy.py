"""
File: test_small_gpu.py
Description:
  Unit test for verifying a small GPU run of the Blackjack Triton kernel.
  Ensures that CUDA is available, the kernel runs without error, and
  produces valid timing and game statistics.

"""

import torch
from gpubj.runner import run_once


def test_small_gpu_run():
    """
    Run a small GPU simulation batch to verify kernel correctness and timing.

    Assertions
    ----------
    - CUDA device must be available.
    - The number of simulated hands equals the total number of games.
    - Both kernel and wall-clock times must be greater than zero.

    Notes
    -----
    This test uses a small configuration (workers=1024, G=256)
    for fast validation on CI or local GPU environments.
    """
    assert torch.cuda.is_available(), "No CUDA GPU detected."
    out = run_once(workers=1024, G=256, R=28, seed=7)

    assert out["hands"] == out["total_games"], "Mismatch between hands and games count."
    assert out["kernel_ms"] > 0.0, "Kernel time not recorded properly."
    assert out["wall_s"] > 0.0, "Wall-clock time not recorded properly."
