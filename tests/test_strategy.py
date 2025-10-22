from gpubj.strategy import build_basic_strategy_table
import torch
def test_strategy_shape():
    t = build_basic_strategy_table(device="cpu")
    assert t.shape == (43,10)
    assert t.dtype == torch.uint8
