"""
File: runner.py
Description:
  High-level orchestration and benchmarking harness for the GPU-accelerated
  Blackjack Triton kernel. Handles RNG setup, kernel launches, timing, and
  aggregation of performance and statistical results.

"""

import time, math, json, argparse, torch
from .strategy import build_basic_strategy_table
from .triton_kernel import blackjack_kernel

# --------------------------------------------------------------------------
# Global constants for hand limits
# --------------------------------------------------------------------------
MAX_HANDS_CONST = 4          # placeholder for split support (unused in kernel)
MAX_HAND_CARDS_CONST = 10    # max cards per hand, loop bound for Triton


# --------------------------------------------------------------------------
# Run a single batch of games
# --------------------------------------------------------------------------
def run_once(workers=32768, G=2048, R=28, seed=12345, device="cuda"):
    """
    Execute one complete batch of Blackjack simulations on GPU.

    Parameters
    ----------
    workers : int
        Number of GPU threads (each thread simulates G games sequentially).
    G : int
        Number of games per worker.
    R : int
        Rank (card) budget per game.
    seed : int
        Random seed for reproducibility.
    device : str
        Compute device ('cuda' or 'cpu').

    Returns
    -------
    dict
        Dictionary containing aggregated statistics:
          - workers, G, R, seed, total_games
          - kernel_ms: pure GPU time (ms)
          - wall_s: total wall-clock time (s)
          - kernel_games_per_s / wall_games_per_s
          - wins, losses, pushes, player/dealer blackjacks, busts, hands
          - total P&L (pl)
    """
    strategy = build_basic_strategy_table(device=device).flatten()
    gen = torch.Generator(device=device).manual_seed(seed)
    ranks = torch.randint(1, 14, (workers*G*R,), device=device, dtype=torch.uint8, generator=gen)

    stats_i = torch.empty((workers, 7), device=device, dtype=torch.int32)
    stats_f = torch.empty((workers,),    device=device, dtype=torch.float32)

    # ---- warm-up / JIT compile ----
    blackjack_kernel[(workers,)](
        strategy, ranks, stats_i, stats_f,
        G=G, R=R, MAX_HANDS=MAX_HANDS_CONST, MAX_HAND_CARDS=MAX_HAND_CARDS_CONST
    )
    torch.cuda.synchronize()

    # ---- timed launch ----
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    t0 = time.perf_counter()
    start_evt.record()

    blackjack_kernel[(workers,)](
        strategy, ranks, stats_i, stats_f,
        G=G, R=R, MAX_HANDS=MAX_HANDS_CONST, MAX_HAND_CARDS=MAX_HAND_CARDS_CONST
    )

    end_evt.record()
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    kernel_ms = start_evt.elapsed_time(end_evt)

    # ---- aggregate statistics ----
    ti = stats_i.sum(dim=0).to(torch.int64).tolist()
    total_pl = stats_f.sum().item()
    total_games = workers * G

    return {
        "workers": workers, "G": G, "R": R, "seed": seed,
        "total_games": int(total_games),
        "kernel_ms": float(kernel_ms),
        "wall_s": float(wall_s),
        "kernel_games_per_s": float(total_games / (kernel_ms / 1e3)),
        "wall_games_per_s": float(total_games / wall_s),
        "wins": int(ti[0]), "losses": int(ti[1]), "pushes": int(ti[2]),
        "player_bj": int(ti[3]), "dealer_bj": int(ti[4]),
        "busts": int(ti[5]), "hands": int(ti[6]),
        "pl": float(total_pl),
    }



# Run tiled batches for very large totals (hundreds of millions of games)
def run_tiled(total_games_target=500_000_000, workers=65536, G=1024, R=28, seed=2025, device="cuda"):
    """
    Execute multiple tiled GPU runs to reach a large total number of games.

    Each tile runs a batch of `workers * G` games; results are accumulated
    until the target count is reached.

    Parameters
    ----------
    total_games_target : int
        Desired total number of simulated games.
    workers : int
        GPU threads per tile.
    G : int
        Games per worker per tile.
    R : int
        Rank budget per game.
    seed : int
        RNG seed.
    device : str
        Compute device ('cuda' or 'cpu').

    Returns
    -------
    dict
        Aggregated statistics and performance metrics identical to `run_once`.
    """
    strategy = build_basic_strategy_table(device=device).flatten()
    totals_i = torch.zeros(7, dtype=torch.int64, device=device)
    total_pl = torch.zeros(1, dtype=torch.float64, device=device)
    gen = torch.Generator(device=device).manual_seed(seed)

    per = workers * G
    reps = math.ceil(total_games_target / per)
    k_ms_total = 0.0
    t0 = time.perf_counter()

    for _ in range(reps):
        ranks = torch.randint(1, 14, (workers*G*R,), device=device, dtype=torch.uint8, generator=gen)
        stats_i = torch.empty((workers, 7), device=device, dtype=torch.int32)
        stats_f = torch.empty((workers,),    device=device, dtype=torch.float32)

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()

        blackjack_kernel[(workers,)](
            strategy, ranks, stats_i, stats_f,
            G=G, R=R, MAX_HANDS=MAX_HANDS_CONST, MAX_HAND_CARDS=MAX_HAND_CARDS_CONST
        )

        e.record()
        torch.cuda.synchronize()
        k_ms_total += s.elapsed_time(e)

        totals_i += stats_i.sum(dim=0).to(torch.int64)
        total_pl += stats_f.sum().to(torch.float64)

    wall_s = time.perf_counter() - t0
    total_games = reps * per
    ti = totals_i.tolist()

    return {
        "workers": workers, "G": G, "R": R, "seed": seed,
        "reps": reps, "total_games": int(total_games),
        "kernel_ms": float(k_ms_total),
        "wall_s": float(wall_s),
        "kernel_games_per_s": float(total_games / (k_ms_total / 1e3)),
        "wall_games_per_s": float(total_games / wall_s),
        "wins": int(ti[0]), "losses": int(ti[1]), "pushes": int(ti[2]),
        "player_bj": int(ti[3]), "dealer_bj": int(ti[4]),
        "busts": int(ti[5]), "hands": int(ti[6]),
        "pl": float(total_pl.item()),
    }



# Command-line interface
def cli():
    """
    Command-line entry point for the GPU Blackjack benchmark.

    Examples
    --------
    Run a single batch:
        gbj-bench --mode once --workers 32768 --G 2048 --R 28

    Run a tiled simulation:
        gbj-bench --mode tiled --target 500000000
    """
    p = argparse.ArgumentParser(prog="gbj-bench", description="BCG-style GPU Blackjack (no GA)")
    p.add_argument("--mode", choices=["once","tiled"], default="once")
    p.add_argument("--workers", type=int, default=32768)
    p.add_argument("--G", type=int, default=2048)
    p.add_argument("--R", type=int, default=28)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--target", type=int, default=500_000_000)
    args = p.parse_args()

    if args.mode == "once":
        out = run_once(workers=args.workers, G=args.G, R=args.R, seed=args.seed)
    else:
        out = run_tiled(total_games_target=args.target, workers=args.workers, G=args.G, R=args.R, seed=args.seed)

    print(json.dumps(out, indent=2))
