"""
File: test_strategy.py
Description:
  Unit test for verifying correctness and consistency of the
  Blackjack Basic Strategy lookup table used by the Triton kernel.

"""

import torch
from gpubj.strategy import build_basic_strategy_table


def test_strategy_shape():
    """
    Validate the shape and data type of the basic strategy table.

    Assertions
    ----------
    - The table must have shape (43, 10).
    - The dtype must be uint8 for GPU compatibility.

    Notes
    -----
    The strategy table is used by the GPU kernel to make decisions.
    Ensuring correct shape and dtype prevents indexing or memory errors.
    """
    t = build_basic_strategy_table(device="cpu")

    assert t.shape == (43, 10), f"Expected shape (43,10), got {tuple(t.shape)}"
    assert t.dtype == torch.uint8, f"Expected dtype uint8, got {t.dtype}"
