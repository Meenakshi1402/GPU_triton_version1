import torch
from gpubj.runner import run_once
def test_small_gpu_run():
    assert torch.cuda.is_available()
    out = run_once(workers=1024, G=256, R=28, seed=7)
    assert out["hands"] == out["total_games"]
    assert out["kernel_ms"] > 0.0
    assert out["wall_s"] > 0.0
