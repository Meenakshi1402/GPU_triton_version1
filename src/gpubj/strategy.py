"""
File: strategy.py
Description:
  Builds the Basic Strategy lookup table (LUT) used by the Triton GPU kernel.
  The table encodes the optimal Blackjack decision ('S','H','D','P')
  for each player total/hand type versus dealer upcard (2..A).

  Layout
  ------
  Shape: (43, 10)
    Rows 0..9   : hard totals 21..12
    Rows 10..16 : hard totals 11..5
    Row  17     : pad row (kept for alignment)
    Rows 18..29 : soft hands A,2..A,K
    Rows 30..42 : pairs 2/2..A/A

  Each cell stores an ASCII byte:
    83 = 'S' (Stand)
    72 = 'H' (Hit)
    68 = 'D' (Double)
    80 = 'P' (Split)

  Columns 0..9 correspond to dealer upcards:
    2..9 -> 0..7
    10/J/Q/K -> 8
    Ace -> 9

"""

import torch


# ASCII encodings of strategy actions
S, H, D, P = ord('S'), ord('H'), ord('D'), ord('P')


# Build the (43x10) strategy table
def build_basic_strategy_table(device="cuda"):
    """
    Construct the (43 x 10) Basic Strategy table as a uint8 tensor.

    Parameters
    ----------
    device : str
        Target device for the tensor ('cuda' or 'cpu').

    Returns
    -------
    torch.Tensor
        Tensor of shape (43, 10), dtype uint8, with entries in
        {83 ('S'), 72 ('H'), 68 ('D'), 80 ('P')} representing
        Stand, Hit, Double, and Split respectively.

    Notes
    -----
    - The GPU kernel flattens this table and indexes it with
      (row * 10 + col) where `col` is determined by the dealer upcard.
    - This table defines four sections:
        I.   Hard totals 21..12
        II.  Hard totals 11..5 (+ one padding row)
        III. Soft hands A,2..A,K
        IV.  Pairs 2/2..A/A
    - Keep dtype as uint8 for efficient GPU memory access.
    - Raises a RuntimeError if the shape is not (43,10).
    """

    # Section I: hard totals 21..12
    sec1 = [
        [S,S,S,S,S,S,S,S,S,S],  # 21
        [S,S,S,S,S,S,S,S,S,S],  # 20
        [S,S,S,S,S,S,S,S,S,S],  # 19
        [S,S,S,S,S,S,S,S,S,S],  # 18
        [S,S,S,S,S,S,S,S,S,S],  # 17
        [S,S,S,S,S,H,H,H,H,H],  # 16
        [S,S,S,S,S,H,H,H,H,H],  # 15
        [S,S,S,S,S,H,H,H,H,H],  # 14
        [S,S,S,S,S,H,H,H,H,H],  # 13
        [H,H,S,S,S,H,H,H,H,H],  # 12
    ]

    # Section II: hard totals 11..5 (+ pad row)
    sec2 = [
        [D,D,D,D,D,D,D,D,D,H],  # 11
        [D,D,D,D,D,D,D,D,H,H],  # 10
        [H,D,D,D,D,H,H,H,H,H],  # 9
        [H,H,H,H,H,H,H,H,H,H],  # 8
        [H,H,H,H,H,H,H,H,H,H],  # 7
        [H,H,H,H,H,H,H,H,H,H],  # 6
        [H,H,H,H,H,H,H,H,H,H],  # 5
    ]
    sec2_extra = [[H,H,H,H,H,H,H,H,H,H]]  # pad row 17

    # Section III: soft A,x
    sec3 = [
        [S,S,S,S,S,S,S,S,S,S],  # A,K
        [S,S,S,S,S,S,S,S,S,S],  # A,Q
        [S,S,S,S,S,S,S,S,S,S],  # A,J
        [S,S,S,S,S,S,S,S,S,S],  # A,T
        [S,S,S,S,S,S,S,S,S,S],  # A,9
        [S,S,S,S,S,S,S,S,S,S],  # A,8
        [S,D,D,D,D,S,S,H,H,H],  # A,7
        [H,D,D,D,D,H,H,H,H,H],  # A,6
        [H,H,D,D,D,H,H,H,H,H],  # A,5
        [H,H,D,D,D,H,H,H,H,H],  # A,4
        [H,H,H,D,D,H,H,H,H,H],  # A,3
        [H,H,H,D,D,H,H,H,H,H],  # A,2
    ]

    # Section IV: pairs
    sec4 = [
        [P,P,P,P,P,P,P,P,P,P],  # A,A
        [S,S,S,S,S,S,S,S,S,S],  # T,K
        [S,S,S,S,S,S,S,S,S,S],  # Q,Q
        [S,S,S,S,S,S,S,S,S,S],  # J,J
        [S,S,S,S,S,S,S,S,S,S],  # T,T
        [P,P,P,P,P,S,P,P,S,S],  # 9,9
        [P,P,P,P,P,P,P,P,P,P],  # 8,8
        [P,P,P,P,P,P,H,H,H,H],  # 7,7
        [P,P,P,P,P,H,H,H,H,H],  # 6,6
        [D,D,D,D,D,D,D,D,H,H],  # 5,5
        [H,H,H,P,P,H,H,H,H,H],  # 4,4
        [P,P,P,P,P,P,H,H,H,H],  # 3,3
        [P,P,P,P,P,P,H,H,H,H],  # 2,2
    ]

    # Combine all sections
    table = sec1 + sec2 + sec2_extra + sec3 + sec4
    t = torch.tensor(table, dtype=torch.uint8, device=device)

    # Sanity check for shape
    if t.shape != (43, 10):
        raise RuntimeError(f"Strategy table shape mismatch: {t.shape}")

    return t


__all__ = ["build_basic_strategy_table", "S", "H", "D", "P"]

