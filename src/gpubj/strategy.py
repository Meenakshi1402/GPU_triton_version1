import torch
S, H, D, P = ord('S'), ord('H'), ord('D'), ord('P')

def build_basic_strategy_table(device="cuda"):
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
    # Section II: hard totals 11..5 (+ pad row 17)
    sec2 = [
        [D,D,D,D,D,D,D,D,D,H],  # 11
        [D,D,D,D,D,D,D,D,H,H],  # 10
        [H,D,D,D,D,H,H,H,H,H],  # 9
        [H,H,H,H,H,H,H,H,H,H],  # 8
        [H,H,H,H,H,H,H,H,H,H],  # 7
        [H,H,H,H,H,H,H,H,H,H],  # 6
        [H,H,H,H,H,H,H,H,H,H],  # 5
    ]
    sec2_extra = [[H,H,H,H,H,H,H,H,H,H]]
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
    table = sec1 + sec2 + sec2_extra + sec3 + sec4
    t = torch.tensor(table, dtype=torch.uint8, device=device)
    if t.shape != (43, 10):
        raise RuntimeError("Strategy table shape mismatch")
    return t

__all__ = ["build_basic_strategy_table", "S", "H", "D", "P"]
