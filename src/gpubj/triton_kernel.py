"""
File: triton_kernel.py
Description:
  Triton GPU kernel for massively parallel Blackjack simulation.
  - One program (= worker) plays G games sequentially.
  - Cards are read from a pre-generated ranks buffer (uint8, values 1..13).
  - Player decisions come from a 43x10 Basic Strategy LUT (flattened to 430B).
  - Masked loops emulate break/continue to satisfy Triton constraints.

Notes:
  * This kernel models a single-player, single-hand flow (no splits yet),
    with Hit / Stand / Double and natural blackjack handling (3:2 payout).
  * Splits can be added later via unrolled, masked per-hand state (MAX_HANDS).

"""

import triton
import triton.language as tl

# --------------------------------------------------------------------------
# Decision codes (constexpr so the kernel can read them as compile-time constants)
# --------------------------------------------------------------------------
S = tl.constexpr(83)  # ord('S') - Stand
H = tl.constexpr(72)  # ord('H') - Hit
D = tl.constexpr(68)  # ord('D') - Double
P = tl.constexpr(80)  # ord('P') - Split (unused in single-hand kernel; kept for parity)


# --------------------------------------------------------------------------
# Helper device functions (must be top-level for Triton)
# --------------------------------------------------------------------------
@triton.jit
def add_card(total_min, n_aces, r):
    """
    Add a card rank to a running hand total in "minimal" form.

    Parameters
    ----------
    total_min : tl.int32 (scalar)
        Current minimal total with all aces counted as 1.
    n_aces : tl.int32 (scalar)
        Current count of aces in the hand.
    r : tl.int32/tl.uint8 (scalar)
        Card rank in [1..13], where:
          1  -> Ace
          2..10 -> face value
          11..13 -> J/Q/K (treated as 10)

    Returns
    -------
    (tl.int32, tl.int32)
        Updated (total_min, n_aces).
    """
    is_face = r >= 11
    v = tl.where(is_face, 10, r)
    total_min = total_min + tl.where(r == 1, 1, v)
    n_aces = n_aces + (r == 1)
    return total_min, n_aces


@triton.jit
def value(total_min, n_aces):
    """
    Convert (total_min, n_aces) into the best blackjack value <= 21.

    Notes
    -----
    Promotes up to `use` aces from 1 to 11 (+10 each) without busting.
    If total_min already exceeds 21, we clamp promotion to zero.
    """
    room = 21 - total_min
    room = tl.where(room >= 0, room, 0)
    bonus = room // 10
    use = tl.minimum(n_aces, bonus)
    return total_min + use * 10


@triton.jit
def upcard_col(up):
    """
    Map dealer upcard rank (1..13) to strategy LUT column index (0..9).

    Mapping
    -------
    2..9         -> 0..7
    10/J/Q/K(>=10)-> 8
    Ace(==1)     -> 9
    """
    col = tl.full((), 8, tl.int32)         # default for 10/J/Q/K
    col = tl.where(up == 1, 9, col)        # Ace
    col = tl.where((up >= 2) & (up <= 9), up.to(tl.int32) - 2, col)
    return col


@triton.jit
def sec1_idx(total, col):
    """
    Flattened LUT index for hard totals 21..12 (inclusive).

    Parameters
    ----------
    total : tl.int32
        Current hard total in [12..21].
    col : tl.int32
        Dealer upcard column (0..9).

    Returns
    -------
    tl.int32
        1-D offset into the flattened (43*10=430) table.
    """
    return (21 - total) * 10 + col


@triton.jit
def sec2_idx(total, col):
    """
    Flattened LUT index for hard totals 11..5 (inclusive).

    Parameters
    ----------
    total : tl.int32
        Current hard total in [5..11].
    col : tl.int32
        Dealer upcard column (0..9).
    """
    return (10 + (11 - total)) * 10 + col


@triton.jit
def sec3_idx(other, col):
    """
    Flattened LUT index for two-card soft hands A,other (other in 2..9).

    Parameters
    ----------
    other : tl.int32
        The non-ace rank of a two-card soft hand (2..9).
    col : tl.int32
        Dealer upcard column (0..9).
    """
    return (18 + (13 - other)) * 10 + col


# --------------------------------------------------------------------------
# Main kernel (single-hand per game; doubles allowed; no splits yet)
# --------------------------------------------------------------------------
@triton.jit
def blackjack_kernel(
    strategy_flat,                 # uint8[430] flattened strategy LUT
    ranks_all,                     # uint8[W * (G * R)] pre-generated ranks
    stats_i,                       # int32[W, 7] -> [wins, losses, pushes, pBJ, dBJ, busts, hands]
    stats_f,                       # float32[W]  -> per-worker P&L
    G: tl.constexpr,               # games per worker (constexpr for loop unrolling)
    R: tl.constexpr,               # per-game rank budget (constexpr)
    MAX_HANDS: tl.constexpr,       # kept for API parity (unused here)
    MAX_HAND_CARDS: tl.constexpr,  # per-hand draw upper bound (constexpr)
):
    """
    Simulate `G` blackjack games for this program (worker) and write per-worker stats.

    Parameters
    ----------
    strategy_flat : *device* pointer (uint8[430])
        Flattened (43x10) basic-strategy table with ASCII codes:
          'S' (83), 'H' (72), 'D' (68), 'P' (80).
    ranks_all : *device* pointer (uint8[W * (G * R)])
        Pre-generated card ranks (1..13) for all workers and games.
        This worker reads from a private slice starting at:
          base = pid * (G * R)
    stats_i : *device* pointer (int32[W, 7])
        Per-worker integer counters written in the order:
          [wins, losses, pushes, player_bj, dealer_bj, busts, hands].
    stats_f : *device* pointer (float32[W])
        Per-worker profit/loss accumulator.
    G : tl.constexpr
        Number of games to play sequentially per worker.
    R : tl.constexpr
        Maximum number of ranks available per game (safety bound).
    MAX_HANDS : tl.constexpr
        API placeholder for future split support (unused in this kernel).
    MAX_HAND_CARDS : tl.constexpr
        Upper bound on per-hand draw loop (needed for Triton compilation).

    Notes
    -----
    - Uses masks instead of Python break/continue.
    - Dealer hits until total >= 17.
    - Natural blackjack pays 3:2 (+1.5 * bet).
    - Doubles allowed; no splits implemented in this kernel.
    """
    pid = tl.program_id(0)

    # Per-worker accumulators
    wins   = tl.zeros((), tl.int32)
    losses = tl.zeros((), tl.int32)
    pushes = tl.zeros((), tl.int32)
    pbj    = tl.zeros((), tl.int32)
    dbj    = tl.zeros((), tl.int32)
    busts  = tl.zeros((), tl.int32)
    hands  = tl.zeros((), tl.int32)
    pl     = tl.zeros((), tl.float32)

    # Compute this worker's base offset into ranks_all and initialize draw index
    base = pid * (G * R)
    idx  = tl.zeros((), tl.int32)

    # Each worker plays G games sequentially
    for _g in range(0, G):
        # ------- initial deal (masked loads for safety) -------
        m = idx < (G * R); p1 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); d1 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); p2 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); d2 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)

        upc = upcard_col(d1)

        # ---- dealer state ----
        d_min  = tl.zeros((), tl.int32)
        d_aces = tl.zeros((), tl.int32)
        d_min, d_aces = add_card(d_min, d_aces, d1)
        d_min, d_aces = add_card(d_min, d_aces, d2)
        d_cards = tl.full((), 2, tl.int32)

        # ---- player single hand ----
        p_min  = tl.zeros((), tl.int32)
        p_ace  = tl.zeros((), tl.int32)
        p_nc   = tl.full((), 2, tl.int32)       # number of player cards
        p_bet  = tl.full((), 1.0, tl.float32)   # initial bet = 1 unit

        p_min, p_ace = add_card(p_min, p_ace, p1)
        p_min, p_ace = add_card(p_min, p_ace, p2)
        p_val = value(p_min, p_ace)

        # Natural blackjack (only applies with exactly 2 cards)
        p_natbj = (p_nc == 2) & (p_val == 21)

        # ---- player play with masks (no break) ----
        # We keep a "playing" mask to emulate break/continue behavior
        playing = tl.full((), 1, tl.int1)
        for _hit in range(0, MAX_HAND_CARDS - 2):
            # Stop if busted or if we already stood/doubled
            still = playing & (p_val <= 21)

            # Distinguish two-card soft from others
            is_soft2 = still & (p_nc == 2) & (((p1 == 1) ^ (p2 == 1)))

            # Default decision = Stand
            decision = tl.full((), S, tl.uint8)

            # Soft two-card: use soft section (sec3); otherwise hard sections (sec1/sec2)
            if is_soft2:
                other = tl.where(p1 == 1, p2, p1)
                ix = sec3_idx(other, upc)
                decision = tl.load(strategy_flat + ix)
            else:
                cur = p_val
                ix = tl.where((cur >= 12) & (cur <= 21), sec1_idx(cur, upc), sec2_idx(cur, upc))
                decision = tl.load(strategy_flat + ix)

            # Forbid double after > 2 cards (downgrade to Hit)
            decision = tl.where((decision == D) & (p_nc > 2), tl.full((), H, tl.uint8), decision)

            # Apply under masks
            # STAND: end playing
            playing = tl.where(still & (decision == S), tl.full((), 0, tl.int1), playing)

            # HIT
            do_hit = still & (decision == H)
            if do_hit:
                m = idx < (G * R); c = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
                p_min, p_ace = add_card(p_min, p_ace, c)
                p_nc = p_nc + 1
                p_val = value(p_min, p_ace)

            # DOUBLE: draw once, double bet, then stop
            do_dbl = still & (decision == D)
            if do_dbl:
                m = idx < (G * R); c = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
                p_min, p_ace = add_card(p_min, p_ace, c)
                p_nc = p_nc + 1
                p_val = value(p_min, p_ace)
                p_bet = p_bet * 2.0
                playing = tl.full((), 0, tl.int1)

            # If nobody is playing, the loop continues but does nothing
        # end player loop

        # ---- dealer plays to >= 17 (masking, no break) ----
        dealer_playing = tl.full((), 1, tl.int1)
        for _d in range(0, MAX_HAND_CARDS - 2):
            dv = value(d_min, d_aces)
            dealer_playing = dealer_playing & (dv < 17)
            if dealer_playing:
                m = idx < (G * R); c = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
                d_min, d_aces = add_card(d_min, d_aces, c)
                d_cards = d_cards + 1
        d_val = value(d_min, d_aces)
        d_natbj = (d_cards == 2) & (d_val == 21)

        # ---- settle (single hand) ----
        hands = hands + 1
        busted = p_val > 21

        if busted:
            pl = pl - p_bet; losses = losses + 1; busts = busts + 1
        elif p_natbj & ~d_natbj:
            pl = pl + p_bet * 1.5; pbj = pbj + 1; wins = wins + 1
        elif d_natbj & ~p_natbj:
            pl = pl - p_bet; dbj = dbj + 1; losses = losses + 1
        else:
            if (d_val > 21) | (p_val > d_val):
                pl = pl + p_bet; wins = wins + 1
            elif p_val < d_val:
                pl = pl - p_bet; losses = losses + 1
            else:
                pushes = pushes + 1

    # ---- write per-worker outputs ----
    base_i = pid * 7
    tl.store(stats_i + base_i + 0, wins)
    tl.store(stats_i + base_i + 1, losses)
    tl.store(stats_i + base_i + 2, pushes)
    tl.store(stats_i + base_i + 3, pbj)
    tl.store(stats_i + base_i + 4, dbj)
    tl.store(stats_i + base_i + 5, busts)
    tl.store(stats_i + base_i + 6, hands)
    tl.store(stats_f + pid, pl)

