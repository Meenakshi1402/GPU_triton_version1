import triton
import triton.language as tl

# Before:
# S, H, D, P = ord('S'), ord('H'), ord('D'), ord('P')

# After (constexpr so the kernel can read them):
S = tl.constexpr(83)  # ord('S')
H = tl.constexpr(72)  # ord('H')
D = tl.constexpr(68)  # ord('D')
P = tl.constexpr(80)  # ord('P')


# ---- helper device functions (must be top-level for Triton) ----
@triton.jit
def add_card(total_min, n_aces, r):
    is_face = r >= 11
    v = tl.where(is_face, 10, r)
    total_min = total_min + tl.where(r == 1, 1, v)
    n_aces = n_aces + (r == 1)
    return total_min, n_aces

@triton.jit
def value(total_min, n_aces):
    room = 21 - total_min
    room = tl.where(room >= 0, room, 0)
    bonus = room // 10
    use = tl.minimum(n_aces, bonus)
    return total_min + use * 10

@triton.jit
def upcard_col(up):
    col = tl.full((), 8, tl.int32)  # 10/J/Q/K
    col = tl.where(up == 1, 9, col)  # Ace
    col = tl.where((up >= 2) & (up <= 9), up.to(tl.int32) - 2, col)
    return col

@triton.jit
def sec1_idx(total, col):  # hard totals 21..12
    return (21 - total) * 10 + col

@triton.jit
def sec2_idx(total, col):  # hard totals 11..5
    return (10 + (11 - total)) * 10 + col

@triton.jit
def sec3_idx(other, col):  # soft A,other (two-card soft hands)
    return (18 + (13 - other)) * 10 + col


# ---- main kernel (single-hand per game; doubles allowed; no splits yet) ----
@triton.jit
def blackjack_kernel(
    strategy_flat,   # uint8[430]
    ranks_all,       # uint8[W * (G * R)]
    stats_i,         # int32[W, 7]  -> wins, losses, pushes, pBJ, dBJ, busts, hands
    stats_f,         # float32[W]   -> P&L
    G: tl.constexpr,               # games per worker
    R: tl.constexpr,               # ranks budget per game per worker
    MAX_HANDS: tl.constexpr,       # kept for API parity (unused here)
    MAX_HAND_CARDS: tl.constexpr,  # per-hand upper bound
):
    pid = tl.program_id(0)

    wins   = tl.zeros((), tl.int32)
    losses = tl.zeros((), tl.int32)
    pushes = tl.zeros((), tl.int32)
    pbj    = tl.zeros((), tl.int32)
    dbj    = tl.zeros((), tl.int32)
    busts  = tl.zeros((), tl.int32)
    hands  = tl.zeros((), tl.int32)
    pl     = tl.zeros((), tl.float32)

    base = pid * (G * R)
    idx  = tl.zeros((), tl.int32)

    for _g in range(0, G):
        # ------- initial deal -------
        m = idx < (G * R); p1 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); d1 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); p2 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
        m = idx < (G * R); d2 = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)

        upc = upcard_col(d1)

        # dealer state
        d_min  = tl.zeros((), tl.int32)
        d_aces = tl.zeros((), tl.int32)
        d_min, d_aces = add_card(d_min, d_aces, d1)
        d_min, d_aces = add_card(d_min, d_aces, d2)
        d_cards = tl.full((), 2, tl.int32)

        # player single hand
        p_min  = tl.zeros((), tl.int32)
        p_ace  = tl.zeros((), tl.int32)
        p_nc   = tl.full((), 2, tl.int32)
        p_bet  = tl.full((), 1.0, tl.float32)

        p_min, p_ace = add_card(p_min, p_ace, p1)
        p_min, p_ace = add_card(p_min, p_ace, p2)
        p_val = value(p_min, p_ace)

        # natural detection (single-hand only)
        p_natbj = (p_nc == 2) & (p_val == 21)

        # ---- player play with masks (no break) ----
        # We keep a "playing" mask to emulate break/continue
        playing = tl.full((), 1, tl.int1)
        for _hit in range(0, MAX_HAND_CARDS - 2):
            # stop if busted or stood/doubled previously
            still = playing & (p_val <= 21)
            # compute decision on still-active paths
            # distinguish two-card soft from others
            is_soft2 = still & (p_nc == 2) & (((p1 == 1) ^ (p2 == 1)))

            # default decision = S
            decision = tl.full((), S, tl.uint8)

            # soft two-card: use sec3
            if is_soft2:
                other = tl.where(p1 == 1, p2, p1)
                ix = sec3_idx(other, upc)
                decision = tl.load(strategy_flat + ix)
            else:
                cur = p_val
                ix = tl.where((cur >= 12) & (cur <= 21), sec1_idx(cur, upc), sec2_idx(cur, upc))
                decision = tl.load(strategy_flat + ix)

            # forbid double after >2 cards
            decision = tl.where((decision == D) & (p_nc > 2), tl.full((), H, tl.uint8), decision)

            # apply under mask
            # STAND: end playing
            playing = tl.where(still & (decision == S), tl.full((), 0, tl.int1), playing)

            # HIT
            do_hit = still & (decision == H)
            if do_hit:
                m = idx < (G * R); c = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
                p_min, p_ace = add_card(p_min, p_ace, c)
                p_nc = p_nc + 1
                p_val = value(p_min, p_ace)

            # DOUBLE (only once; then stop)
            do_dbl = still & (decision == D)
            if do_dbl:
                m = idx < (G * R); c = tl.load(ranks_all + base + idx, mask=m, other=1); idx = tl.where(m, idx + 1, idx)
                p_min, p_ace = add_card(p_min, p_ace, c)
                p_nc = p_nc + 1
                p_val = value(p_min, p_ace)
                p_bet = p_bet * 2.0
                playing = tl.full((), 0, tl.int1)

            # if nobody is playing, the loop will continue but do nothing
        # end player loop

        # Dealer plays to >=17 (masking, no break)
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

        # settle (single hand)
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

    # Write out
    base_i = pid * 7
    tl.store(stats_i + base_i + 0, wins)
    tl.store(stats_i + base_i + 1, losses)
    tl.store(stats_i + base_i + 2, pushes)
    tl.store(stats_i + base_i + 3, pbj)
    tl.store(stats_i + base_i + 4, dbj)
    tl.store(stats_i + base_i + 5, busts)
    tl.store(stats_i + base_i + 6, hands)
    tl.store(stats_f + pid, pl)
