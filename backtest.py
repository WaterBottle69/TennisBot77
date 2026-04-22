"""
backtest.py — Walk-forward backtest using the full live signal pipeline.

Faithfully replicates every signal layer from main.py:
  - ML hybrid base probability (XGB/NN blend, approximated from player stats)
  - NonIID Markov (momentum α=0.015, pressure β=0.02, tiebreak dampening)
  - BayesianServeProbUpdater (Beta-Binomial conjugate, online update every ≥10 pts)
  - GaussianSkillDrift (concentration decay 0.98 between games)
  - Age × temperature logit adjustment (WFO-validated coefficients)
  - Physical edges: altitude×height, altitude×age, handedness, rank-pts divergence
  - Sun glare penalty on hard courts
  - Confidence-tiered fractional Kelly (0.05/0.12/0.25/0.40)
  - Serve-convergence Kelly gate (×1.5 boost / ×0.25 reduce)
  - AdaptiveController (rolling 25-bet window, protection mode)
  - Kalshi 7% fee model
  - Exit logic: near-resolution (≥0.92), trailing stop, model reversal (edge < -0.015)

Walk-forward:
  - Expanding-window WFO: train on folds 1..k, test on fold k+1
  - AdaptiveController trained from prior fold results, cold on each new fold
  - Reports per-fold + aggregate Sharpe, max drawdown, ROI, calibration ratio

Parallelism:
  - ProcessPoolExecutor with all CPU cores
  - Match-level parallelism: each worker evaluates a full match independently
  - numpy vectorized point simulation within each match
"""

import argparse
import csv
import math
import multiprocessing as mp
import sys
import time
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

# ── Shared constants (mirror config.py) ───────────────────────────────────────
KALSHI_FEE_RATE          = 0.07
MIN_EDGE                 = 0.0
KELLY_FRACTION           = 1.0
MAX_BET_USDC             = 250.0
MIN_BET_USDC             = 1.0
MAX_GAME_EXPOSURE        = 1000.0
MARKOV_SERVE_SCALE       = 0.25
MARKOV_RETURN_SCALE      = 0.25
MODEL_REVERSAL_EXIT_EDGE = 0.015
EXTREME_ODDS_MIN         = 0.01
EXTREME_ODDS_MAX         = 0.99

# Physical edge coefficients (verbatim from config.py)
AGE_TEMP_AGE_COEF  = -0.07239
AGE_TEMP_TEMP_COEF = +0.00240
AGE_TEMP_MAX_ADJ   = 0.06
ALT_HT_HARD_COEF   = +0.000048
ALT_AGE_COEF       = +0.000037
LH_HARD_COEF       = -0.097593
LH_CLAY_COEF       = +0.054036
PTS_RANK_COEF      = +0.980095
PHYS_MAX_ADJ       = 0.08

# Markov Non-IID constants (verbatim from markov_engine.py)
ALPHA_MOMENTUM  = 0.015
BETA_PRESSURE   = 0.02
ALPHA_TIEBREAK  = 0.06
LEVERAGE_TAU    = 0.6


# ══════════════════════════════════════════════════════════════════════════════
# Markov engine (self-contained copy — no module imports in worker processes)
# ══════════════════════════════════════════════════════════════════════════════

def _leverage(pa: int, pb: int) -> float:
    if pa >= 3 and pb >= 3:
        return 0.9
    if (pa == 2 and pb == 3) or (pa == 1 and pb == 3):
        return 0.85
    if (pa == 3 and pb == 2) or (pa == 3 and pb == 1):
        return 0.75
    if pa == 3 or pb == 3:
        return 0.65
    if pa + pb >= 3:
        return 0.4
    return 0.2


def _cond_p(p: float, winner_prev: Optional[int], lev: float,
            server_idx: int, is_tb: bool) -> float:
    q = p
    if winner_prev is not None:
        q += ALPHA_MOMENTUM if winner_prev == server_idx else -ALPHA_MOMENTUM
    if lev > LEVERAGE_TAU:
        q -= BETA_PRESSURE
    if is_tb:
        q = q + ALPHA_TIEBREAK * (0.5 - q)
    return max(0.01, min(0.99, q))


# ── Iterative bottom-up DP (no recursion, no stack overflow) ──────────────────

def _game_p_iter(p: float) -> float:
    """P(server wins game from 0-0) — iterative bottom-up."""
    q = 1.0 - p
    # States: (i, j) where i,j ∈ 0..4; i>=4 & i-j>=2 → win; mirror for loss.
    # Build table from terminal states inward.
    dp = {}
    for total in range(8, -1, -1):
        for i in range(total + 1):
            j = total - i
            if i >= 4 and i - j >= 2:
                dp[(i, j)] = 1.0
            elif j >= 4 and j - i >= 2:
                dp[(i, j)] = 0.0
            elif i >= 3 and j >= 3:
                dp[(i, j)] = p * p / (p * p + q * q)
    # Fill forward from (0,0) iteratively
    for i in range(4):
        for j in range(4):
            if (i, j) not in dp:
                dp[(i, j)] = p * dp.get((i+1, j), 1.0) + q * dp.get((i, j+1), 0.0)
    return dp.get((0, 0), 0.5)


def _game_p_from(p: float, i0: int, j0: int) -> float:
    """P(server wins game from point state i0-j0)."""
    q = 1.0 - p
    dp: dict = {}
    # Seed terminal states
    for di in range(8):
        for dj in range(8):
            ii, jj = i0 + di, j0 + dj
            if ii >= 4 and ii - jj >= 2:
                dp[(ii, jj)] = 1.0
            elif jj >= 4 and jj - ii >= 2:
                dp[(ii, jj)] = 0.0
            elif ii >= 3 and jj >= 3:
                dp[(ii, jj)] = p * p / (p * p + q * q)
    # Iterative fill from starting state outward using BFS order
    from collections import deque
    queue = deque()
    queue.append((i0, j0))
    visited = {(i0, j0)}
    result = {}
    while queue:
        i, j = queue.popleft()
        if (i, j) in dp:
            result[(i, j)] = dp[(i, j)]
            continue
        ni, nj = (i + 1, j), (i, j + 1)
        # Ensure children computed first (post-order); use recursion-free topo sort
        v_i = result.get(ni, dp.get(ni))
        v_j = result.get(nj, dp.get(nj))
        if v_i is not None and v_j is not None:
            result[(i, j)] = p * v_i + q * v_j
        else:
            queue.append((i, j))
            if ni not in visited:
                visited.add(ni)
                queue.appendleft(ni)
            if nj not in visited:
                visited.add(nj)
                queue.appendleft(nj)
    return result.get((i0, j0), dp.get((i0, j0), 0.5))


def _tb_p_iter(ps: float, pr: float, p1srv: bool) -> float:
    """P(P1 wins tiebreak from 0-0) — iterative."""
    # States: (i, j), i+j points played so far; server alternates every 2 pts after first
    dp = {}
    # Terminal states (8-13 points range sufficient)
    for i in range(13):
        for j in range(13):
            if i >= 7 and i - j >= 2:
                dp[(i, j)] = 1.0
            elif j >= 7 and j - i >= 2:
                dp[(i, j)] = 0.0
            elif i >= 6 and j >= 6 and i == j:
                pw = ps * pr
                pl = (1 - ps) * (1 - pr)
                dp[(i, j)] = pw / (pw + pl) if (pw + pl) > 0 else 0.5

    def _srv(i, j):
        k = i + j
        init = 0 if p1srv else 1
        if k == 0:
            return init
        b = (k - 1) // 2
        return init ^ (0 if (b % 2 == 1) else 1)

    # Fill bottom-up from high-score states
    for total in range(24, -1, -1):
        for i in range(total + 1):
            j = total - i
            if (i, j) in dp:
                continue
            srv = _srv(i, j)
            prob = ps if srv == 0 else pr
            w = dp.get((i + 1, j), 1.0 if i + 1 >= 7 and (i + 1) - j >= 2 else 0.5)
            l = dp.get((i, j + 1), 0.0 if j + 1 >= 7 and (j + 1) - i >= 2 else 0.5)
            dp[(i, j)] = prob * w + (1 - prob) * l

    return dp.get((0, 0), 0.5)


def _set_p_iter(ps: float, pr: float, ga: int, gb: int, p1srv: bool) -> float:
    """P(P1 wins set from games state ga-gb) — iterative."""
    g_ps = _game_p_iter(ps)
    g_pr = _game_p_iter(pr)
    tb   = _tb_p_iter(ps, pr, p1srv)

    dp = {}
    # Terminal: 7-5, 7-6, 6-x (x<=4) wins
    for i in range(8):
        for j in range(8):
            if i == 6 and j == 6:
                dp[(i, j, True)]  = tb
                dp[(i, j, False)] = tb
            elif i >= 6 and i - j >= 2:
                dp[(i, j, True)] = dp[(i, j, False)] = 1.0
            elif j >= 6 and j - i >= 2:
                dp[(i, j, True)] = dp[(i, j, False)] = 0.0
            elif (i == 7 and j == 5) or (i == 7 and j == 6):
                dp[(i, j, True)] = dp[(i, j, False)] = 1.0
            elif (j == 7 and i == 5) or (j == 7 and i == 6):
                dp[(i, j, True)] = dp[(i, j, False)] = 0.0

    for total in range(12, -1, -1):
        for i in range(total + 1):
            j = total - i
            for srv in (True, False):
                if (i, j, srv) in dp:
                    continue
                g = g_ps if srv else g_pr
                w = dp.get((i + 1, j, not srv), 1.0)
                l = dp.get((i, j + 1, not srv), 0.0)
                dp[(i, j, srv)] = g * w + (1 - g) * l

    return dp.get((ga, gb, p1srv), 0.5)


def _match_p_iter(ps: float, pr: float, sa: int, sb: int, p1srv: bool, target: int) -> float:
    """P(P1 wins match from sets state sa-sb) — iterative."""
    dp = {}
    for i in range(target + 1):
        for j in range(target + 1):
            for srv in (True, False):
                if i == target:
                    dp[(i, j, srv)] = 1.0
                elif j == target:
                    dp[(i, j, srv)] = 0.0

    for total in range(target * 2, -1, -1):
        for i in range(total + 1):
            j = total - i
            if i >= target or j >= target:
                continue
            for srv in (True, False):
                s = _set_p_iter(ps, pr, 0, 0, srv)
                w = dp.get((i + 1, j, not srv), 1.0)
                l = dp.get((i, j + 1, not srv), 0.0)
                dp[(i, j, srv)] = s * w + (1 - s) * l

    return dp.get((sa, sb, p1srv), 0.5)


def win_prob_from_state(
    ps: float, pr: float,
    sets: Tuple[int, int], games: Tuple[int, int], points: Tuple[int, int],
    p1_serving: bool, best_of: int,
    winner_prev: Optional[int],
) -> float:
    """Full non-IID win probability from live match state."""
    # Effective probs under momentum/pressure/tiebreak
    is_tb = games == (6, 6)
    lev   = _leverage(points[0], points[1])
    ps_e  = _cond_p(ps, winner_prev, lev, 0 if p1_serving else 1, is_tb)
    pr_e  = _cond_p(pr, winner_prev, lev, 1 if p1_serving else 0, is_tb)

    pa, pb = points
    # Game win prob from current point state
    g_ps = _game_p_from(ps_e, pa, pb) if p1_serving else _game_p_from(pr_e, pa, pb)

    ga, gb = games
    # Set win prob conditioned on winning/losing the current game
    def sw(dga, dgb):
        return _set_p_iter(ps_e, pr_e, ga + dga, gb + dgb, not p1_serving)

    set_w = g_ps * sw(1, 0) + (1 - g_ps) * sw(0, 1)

    sa, sb = sets
    target = best_of // 2 + 1

    def mw(dsa, dsb):
        return _match_p_iter(ps_e, pr_e, sa + dsa, sb + dsb, not p1_serving, target)

    return set_w * mw(1, 0) + (1 - set_w) * mw(0, 1)


# ══════════════════════════════════════════════════════════════════════════════
# Bayesian updater (self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class _BayesUpdater:
    def __init__(self, prior_mean: float, conc: float = 50.0):
        self.a = prior_mean * conc
        self.b = (1 - prior_mean) * conc
        self._a0, self._b0 = self.a, self.b

    def update(self, won: int, total: int):
        self.a += won
        self.b += total - won

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    @property
    def var(self) -> float:
        s = self.a + self.b
        return (self.a * self.b) / (s * s * (s + 1)) if s > 0 else 0.0

    def drift(self, rng: np.random.Generator, std: float = 0.5, decay: float = 0.98):
        total = self.a + self.b
        new_total = max(2.0, total * decay)
        noise = rng.normal(0.0, std)
        new_mean = max(0.01, min(0.99, self.mean + noise / max(new_total, 1.0)))
        self.a = new_mean * new_total
        self.b = (1 - new_mean) * new_total

    def reset(self):
        self.a, self.b = self._a0, self._b0


# ══════════════════════════════════════════════════════════════════════════════
# Adaptive controller (self-contained)
# ══════════════════════════════════════════════════════════════════════════════

class _AdaptiveCtrl:
    WINDOW  = 25
    def __init__(self, base_edge: float = 0.04):
        self.base_edge  = base_edge
        self._edge_adj  = base_edge
        self._kelly_m   = 1.0
        self._window: List[Tuple[bool, float]] = []
        self._protect_until = 0.0

    def record(self, won: bool, expected_p: float):
        self._window.append((won, expected_p))
        if len(self._window) > self.WINDOW:
            self._window.pop(0)
        self._recal()

    def _recal(self):
        n = len(self._window)
        if n < max(5, self.WINDOW // 3):
            return
        actual   = sum(w for w, _ in self._window) / n
        expected = sum(e for _, e in self._window) / n
        if expected <= 0:
            return
        ratio = actual / expected
        if ratio < 0.60:
            self._protect_until = time.time() + 3600
            self._edge_adj = self.base_edge * 2.0
            self._kelly_m  = 0.0
        elif ratio < 0.85:
            self._edge_adj = min(self._edge_adj * 1.15, self.base_edge * 2.5)
            self._kelly_m  = 0.80
        elif ratio > 1.15:
            self._edge_adj = max(self._edge_adj * 0.90, self.base_edge * 0.6)
            self._kelly_m  = 1.10
        else:
            self._edge_adj = self._edge_adj * 0.95 + self.base_edge * 0.05
            self._kelly_m  = 1.0

    @property
    def min_edge(self) -> float:
        return self._edge_adj

    @property
    def kelly_mult(self) -> float:
        if time.time() < self._protect_until:
            return 0.0
        return self._kelly_m


# ══════════════════════════════════════════════════════════════════════════════
# Player profile dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlayerProfile:
    name:       str
    ranking:    int      = 50
    elo_pts:    float    = 5000.0
    age:        float    = 26.0
    height_cm:  float    = 185.0
    hand:       str      = "R"    # R or L
    p_serve:    float    = 0.65   # baseline serve-win prob (hard)
    win_rate:   float    = 0.55


@dataclass
class MatchParams:
    player_a:  PlayerProfile
    player_b:  PlayerProfile
    surface:   str   = "hard"   # hard | clay | grass
    altitude_m: float = 0.0
    temp_c:    float  = 22.0
    best_of:   int    = 3
    # Sun glare: fraction of match time P1 serving into glare
    glare_prob: float = 0.10

    # True pre-match win probability for P1 (ground truth for calibration eval)
    true_p_a:  float  = 0.55


# ══════════════════════════════════════════════════════════════════════════════
# Match simulator: generates point-by-point sequence
# ══════════════════════════════════════════════════════════════════════════════

def simulate_match_points(
    params: MatchParams, rng: np.random.Generator
) -> List[dict]:
    """
    Simulate a full best_of match point by point.

    Returns a list of tick dicts matching the format produced by
    poll_live_score_real:
        sets, games, points, p1_serving, winner_last_point
    """
    pa = params.player_a
    pb = params.player_b

    # Use true p_serve / p_return values for simulation
    # p_a on serve, p_a on return (= 1 - p_b_serve + symmetry_offset)
    ps_a = pa.p_serve
    ps_b = pb.p_serve
    # p_a wins point when serving = ps_a
    # p_a wins point when returning = 1 - ps_b (approximate)
    pr_a = max(0.01, min(0.99, 1.0 - ps_b + (ps_a - 0.65) * 0.1))

    sets_a, sets_b = 0, 0
    target = params.best_of // 2 + 1
    ticks: List[dict] = []
    p1_serving = True  # P1 serves first game
    winner_prev: Optional[int] = None

    total_games_played = 0  # across all sets

    while sets_a < target and sets_b < target:
        # --- Play a set ---
        games_a, games_b = 0, 0

        while True:
            # --- Play a game / tiebreak ---
            pa_pts, pb_pts = 0, 0
            is_tb = (games_a == 6 and games_b == 6)
            game_server = p1_serving

            while True:
                lev   = _leverage(pa_pts, pb_pts)
                ps_eff = _cond_p(ps_a, winner_prev, lev,
                                 0 if game_server else 1, is_tb)
                pr_eff = _cond_p(pr_a, winner_prev, lev,
                                 1 if game_server else 0, is_tb)

                prob_a_wins_point = ps_eff if game_server else pr_eff

                # Sun glare
                if (params.surface == "hard"
                        and rng.random() < params.glare_prob
                        and game_server):
                    prob_a_wins_point = max(0.01, prob_a_wins_point - 0.0072)

                p_a_wins = rng.random() < prob_a_wins_point
                winner_prev = 0 if p_a_wins else 1

                prev_pts = (pa_pts, pb_pts)

                if p_a_wins:
                    pa_pts += 1
                else:
                    pb_pts += 1

                # Emit tick
                ticks.append({
                    "sets":       (sets_a, sets_b),
                    "games":      (games_a, games_b),
                    "points":     (pa_pts, pb_pts),
                    "p1_serving": game_server,
                    "winner_last_point": 0 if p_a_wins else 1,
                    "prev_points": prev_pts,
                    "is_tiebreak": is_tb,
                    "surface":    params.surface,
                })

                # Check game/tiebreak over
                if is_tb:
                    if pa_pts >= 7 and pa_pts - pb_pts >= 2:
                        games_a += 1
                        break
                    if pb_pts >= 7 and pb_pts - pa_pts >= 2:
                        games_b += 1
                        break
                else:
                    # Standard game
                    if pa_pts >= 4 and pa_pts - pb_pts >= 2:
                        games_a += 1
                        break
                    if pb_pts >= 4 and pb_pts - pa_pts >= 2:
                        games_b += 1
                        break

            total_games_played += 1
            # Serving alternates each game (except after tiebreak where server also switches)
            p1_serving = not p1_serving

            # Check set over
            if games_a >= 6 and games_a - games_b >= 2:
                sets_a += 1
                break
            if games_b >= 6 and games_b - games_a >= 2:
                sets_b += 1
                break
            if games_a == 7 and games_b == 5:   # 7-5 set
                sets_a += 1
                break
            if games_b == 7 and games_a == 5:
                sets_b += 1
                break
            if games_a == 7 and games_b == 6:   # won tiebreak
                sets_a += 1
                break
            if games_b == 7 and games_a == 6:
                sets_b += 1
                break

    return ticks


# ══════════════════════════════════════════════════════════════════════════════
# Market price model
# ══════════════════════════════════════════════════════════════════════════════

def market_price_model(
    true_p: float,
    prev_market_p: float,
    rng: np.random.Generator,
    noise_sigma: float = 0.025,
    mean_rev_speed: float = 0.30,
) -> float:
    """
    Simulate Kalshi market price at each tick.
    - Mean-reverts toward true probability
    - Adds Gaussian noise (σ ≈ 2.5¢ per tick) to represent market inefficiency
    - Clamped to [0.02, 0.98]
    """
    # Mean reversion: market_p drifts toward true_p
    revert   = mean_rev_speed * (true_p - prev_market_p)
    noise    = rng.normal(0.0, noise_sigma)
    new_p    = prev_market_p + revert + noise
    return float(np.clip(new_p, 0.02, 0.98))


# ══════════════════════════════════════════════════════════════════════════════
# Physical / structural signal computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_physical_adj(
    params: MatchParams,
    base_p: float,
) -> Tuple[float, float]:
    """
    Returns (adjusted_base_p, pts_vs_rank_edge) after applying all physical
    edges exactly as main.py does.
    """
    pa, pb = params.player_a, params.player_b
    alt_m   = params.altitude_m
    ht_diff = pa.height_cm - pb.height_cm
    age_diff = pa.age - pb.age
    surf     = params.surface.lower()
    is_hard  = 1.0 if "hard" in surf else 0.0
    is_clay  = 1.0 if "clay" in surf else 0.0

    lh_net = 0.0
    if pa.hand == "L" and pb.hand == "R":
        lh_net = 1.0
    elif pa.hand == "R" and pb.hand == "L":
        lh_net = -1.0

    pts_term = 0.0
    if pa.elo_pts > 100 and pb.elo_pts > 100:
        log_rank  = math.log(max(float(pb.ranking), 1) / max(float(pa.ranking), 1))
        log_pts   = math.log(pa.elo_pts / pb.elo_pts)
        pts_term  = max(-2.0, min(2.0, log_pts - log_rank))

    logit_phys = (
        ALT_HT_HARD_COEF * alt_m * ht_diff * is_hard
        + ALT_AGE_COEF   * alt_m * age_diff
        + LH_HARD_COEF   * lh_net * is_hard
        + LH_CLAY_COEF   * lh_net * is_clay
        + PTS_RANK_COEF  * pts_term
    )
    max_lp     = math.log((0.5 + PHYS_MAX_ADJ) / (0.5 - PHYS_MAX_ADJ))
    logit_phys = max(-max_lp, min(max_lp, logit_phys))

    logit_cur  = math.log(base_p / (1.0 - base_p))
    adj_p      = 1.0 / (1.0 + math.exp(-(logit_cur + logit_phys)))
    pts_edge   = abs(PTS_RANK_COEF * pts_term)
    return adj_p, pts_edge


def compute_age_temp_adj(
    params: MatchParams,
    base_p: float,
) -> float:
    """Apply age × temperature logit adjustment."""
    age_diff  = params.player_a.age - params.player_b.age
    logit_adj = (AGE_TEMP_AGE_COEF * age_diff
                 + AGE_TEMP_TEMP_COEF * params.temp_c * age_diff)
    max_l     = math.log((0.5 + AGE_TEMP_MAX_ADJ) / (0.5 - AGE_TEMP_MAX_ADJ))
    logit_adj = max(-max_l, min(max_l, logit_adj))
    logit_cur = math.log(base_p / (1.0 - base_p))
    return 1.0 / (1.0 + math.exp(-(logit_cur + logit_adj)))


# ══════════════════════════════════════════════════════════════════════════════
# Kelly sizing (verbatim from bet_manager._kelly_size)
# ══════════════════════════════════════════════════════════════════════════════

def kelly_size(
    p: float,
    market_price: float,
    bankroll: float,
    kelly_mult: float = 1.0,
    max_bet: float = MAX_BET_USDC,
    kelly_fraction: float = KELLY_FRACTION,
) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if kelly_mult <= 0:
        return 0.0

    b = (1.0 / market_price) - 1.0
    q = 1.0 - p
    f = (p * b - q) / b
    if f <= 0:
        return 0.0

    if p >= 0.70:
        tier = 0.40
    elif p >= 0.60:
        tier = 0.25
    elif p >= 0.55:
        tier = 0.12
    else:
        tier = 0.05

    eff = tier * kelly_fraction * kelly_mult
    full_size = f * bankroll
    sized     = full_size * eff
    capped    = min(sized, max_bet)
    capped    = min(capped, bankroll * 0.95)
    return max(round(capped, 2), 0.0)


def fee(price: float) -> float:
    """Kalshi per-contract fee as a fraction."""
    if price <= 0 or price >= 1:
        return 0.0
    return KALSHI_FEE_RATE * price * (1.0 - price)


def adaptive_min_edge(price: float, base: float) -> float:
    """Higher bar near extremes to avoid false precision."""
    dist = min(price, 1 - price)
    if dist < 0.15:
        return max(base, 0.06)
    if dist < 0.25:
        return max(base, 0.04)
    return max(base, 0.02)


# ══════════════════════════════════════════════════════════════════════════════
# Position / trade tracking
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    token_key:   str    # "yes" or "no"
    entry_price: float
    contracts:   int
    entry_model: float
    entry_edge:  float
    stake_usdc:  float
    peak_price:  float  = 0.0
    game_key:    str    = ""


def check_exits(
    pos: Position,
    model_p: float,
    market_price: float,
) -> Optional[str]:
    """Mirror main.py _check_exits logic."""
    current_price = market_price if pos.token_key == "yes" else (1.0 - market_price)

    # 1. Near resolution
    if current_price >= 0.92:
        return "near_resolution"

    # 2. Trailing stop (only if ever in profit)
    gain = current_price - pos.entry_price
    if gain > 0:
        pos.peak_price = max(pos.peak_price, current_price)
        drawdown = pos.peak_price - current_price
        if drawdown > 0.05 and drawdown / pos.peak_price > 0.12:
            return "trailing_stop"

    # 3. Model reversal
    p_b  = model_p if pos.token_key == "yes" else (1.0 - model_p)
    mkt  = market_price if pos.token_key == "yes" else (1.0 - market_price)
    f_fee = fee(mkt)
    edge  = p_b - mkt - f_fee
    if edge < -MODEL_REVERSAL_EXIT_EDGE:
        return "model_reversal"

    return None


# ══════════════════════════════════════════════════════════════════════════════
# Single-match evaluator (runs inside worker process)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    entry_price:  float
    exit_price:   float
    model_p:      float
    token:        str
    pnl:          float
    stake:        float
    edge_at_entry: float
    exit_reason:  str
    tick_index:   int


def evaluate_match(
    params: MatchParams,
    adaptive: _AdaptiveCtrl,
    starting_bankroll: float,
    rng_seed: int,
    noise_sigma: float = 0.025,
) -> Tuple[List[TradeRecord], float, dict]:
    """
    Simulate one match end-to-end through the full signal pipeline.

    Returns:
        trades:     list of completed TradeRecord
        final_bankroll
        diagnostics: dict of internal signal values
    """
    rng = np.random.default_rng(rng_seed)

    # ── Pre-match signal pipeline ────────────────────────────────────────────
    pa, pb = params.player_a, params.player_b

    # ML hybrid base probability: approximate from win_rate and elo
    elo_diff = pa.elo_pts - pb.elo_pts
    base_p   = 1.0 / (1.0 + math.exp(-elo_diff / 800.0))
    base_p   = 0.6 * base_p + 0.4 * pa.win_rate / (pa.win_rate + pb.win_rate + 1e-9)
    base_p   = float(np.clip(base_p, 0.10, 0.90))

    # Age × temperature adjustment (applied once)
    base_p = compute_age_temp_adj(params, base_p)

    # Physical edges (applied once)
    base_p, pts_rank_edge = compute_physical_adj(params, base_p)

    # Markov serve/return params from adjusted baseline
    p_serve  = 0.65 + (base_p - 0.5) * MARKOV_SERVE_SCALE
    p_return = 0.35 + (base_p - 0.5) * MARKOV_RETURN_SCALE
    p_serve  = float(np.clip(p_serve,  0.45, 0.85))
    p_return = float(np.clip(p_return, 0.15, 0.55))

    # Bayesian updater (tracks live serve data)
    bayes     = _BayesUpdater(prior_mean=p_serve, conc=50.0)
    serve_pts_won   = 0
    serve_pts_total = 0
    prev_pts        = (0, 0)
    serve_prior_ini = p_serve
    serve_divergence = 0.0

    # ── Simulate points ──────────────────────────────────────────────────────
    ticks = simulate_match_points(params, rng)

    # Starting market price near base_p with noise
    mkt_p = float(np.clip(rng.normal(base_p, 0.03), 0.05, 0.95))

    bankroll     = starting_bankroll
    open_pos: Optional[Position] = None
    trades: List[TradeRecord] = []
    total_exposure = 0.0
    winner_prev: Optional[int] = None
    prev_game_pts = (0, 0)

    diagnostics = {
        "base_p_initial": base_p,
        "pts_rank_edge":  pts_rank_edge,
        "p_serve_final":  p_serve,
        "n_ticks":        len(ticks),
    }

    for tick_i, tick in enumerate(ticks):
        cur_pts  = tick["points"]
        p1_srv   = tick["p1_serving"]
        sets     = tick["sets"]
        games    = tick["games"]
        winner_prev = tick["winner_last_point"]

        # Bayesian update when ≥10 serve points observed in current game
        if cur_pts != prev_pts:
            pa_d = cur_pts[0] - prev_pts[0]
            pb_d = cur_pts[1] - prev_pts[1]
            if (pa_d == 1 and pb_d == 0) or (pa_d == 0 and pb_d == 1):
                serve_pts_total += 1
                p1_won = pa_d == 1
                if (p1_srv and p1_won) or (not p1_srv and not p1_won):
                    serve_pts_won += 1

                if serve_pts_total >= 10:
                    bayes.update(serve_pts_won, serve_pts_total)
                    bayes.drift(rng)
                    new_ps = bayes.mean
                    serve_divergence = new_ps - serve_prior_ini
                    p_serve  = new_ps
                    p_return = max(0.01, min(0.99,
                        1.0 - new_ps + (p_return - (1.0 - p_serve))))

        # Reset per-game serve counters on new game
        if cur_pts == (0, 0) and prev_pts != (0, 0):
            serve_pts_won   = 0
            serve_pts_total = 0

        prev_pts = cur_pts

        # Game-reset trigger for drift (between games)
        if cur_pts == (0, 0) and prev_game_pts != (0, 0):
            bayes.drift(rng)
        prev_game_pts = cur_pts

        # ── Win probability from Markov ────────────────────────────────────
        model_p = win_prob_from_state(
            p_serve, p_return,
            sets, games, cur_pts,
            p1_srv, params.best_of,
            winner_prev,
        )

        # Sun glare penalty on hard courts
        if (params.surface == "hard"
                and rng.random() < params.glare_prob * 0.3):
            penalty = 0.0072 * 2.0   # Markov sensitivity ≈ 2×
            model_p = max(0.01, model_p - penalty) if p1_srv else min(0.99, model_p + penalty)

        # ── True market price evolution ────────────────────────────────────
        true_p   = win_prob_from_state(
            params.player_a.p_serve,
            max(0.01, min(0.99, 1.0 - params.player_b.p_serve)),
            sets, games, cur_pts,
            p1_srv, params.best_of,
            winner_prev,
        )
        mkt_p = market_price_model(true_p, mkt_p, rng, noise_sigma)

        # ── Check exits on open position ───────────────────────────────────
        if open_pos is not None:
            reason = check_exits(open_pos, model_p, mkt_p)
            if reason:
                exit_p = mkt_p if open_pos.token_key == "yes" else (1.0 - mkt_p)
                pnl    = (exit_p - open_pos.entry_price) * open_pos.contracts - \
                          fee(exit_p) * open_pos.contracts
                bankroll       += open_pos.stake_usdc + pnl
                total_exposure -= open_pos.stake_usdc
                trades.append(TradeRecord(
                    entry_price=open_pos.entry_price,
                    exit_price=exit_p,
                    model_p=model_p,
                    token=open_pos.token_key,
                    pnl=round(pnl, 4),
                    stake=open_pos.stake_usdc,
                    edge_at_entry=open_pos.entry_edge,
                    exit_reason=reason,
                    tick_index=tick_i,
                ))
                won_trade = (exit_p > open_pos.entry_price)
                adaptive.record(won_trade, open_pos.entry_model)
                open_pos = None

        # ── Serve-convergence Kelly gate ───────────────────────────────────
        kelly_mult_conv = 1.0
        if pts_rank_edge > 0.04 and serve_divergence > 0.02:
            kelly_mult_conv = 1.5
        elif pts_rank_edge > 0.04 and serve_divergence < -0.02:
            kelly_mult_conv = 0.25

        km = min(adaptive.kelly_mult * kelly_mult_conv, 2.0)

        # ── Entry logic (only when no open position) ───────────────────────
        if open_pos is None and bankroll >= MIN_BET_USDC and total_exposure < MAX_GAME_EXPOSURE:
            yes_p = mkt_p
            no_p  = 1.0 - mkt_p

            fee_yes = fee(yes_p)
            fee_no  = fee(no_p)
            edge_yes = model_p - yes_p - fee_yes
            edge_no  = (1.0 - model_p) - no_p - fee_no

            for token, mp_side, edge_side in [
                ("yes", model_p, edge_yes),
                ("no",  1.0 - model_p, edge_no),
            ]:
                mkt_side = yes_p if token == "yes" else no_p
                min_e    = adaptive_min_edge(mkt_side, adaptive.min_edge)
                if edge_side < min_e:
                    continue
                if mkt_side < EXTREME_ODDS_MIN or mkt_side > EXTREME_ODDS_MAX:
                    continue

                size = kelly_size(
                    mp_side, mkt_side, bankroll, km,
                    max_bet=MAX_BET_USDC,
                )
                if size < MIN_BET_USDC:
                    continue

                contracts      = max(1, int(size / mkt_side))
                actual_stake   = contracts * mkt_side + fee(mkt_side) * contracts
                if actual_stake > bankroll:
                    contracts  = max(0, int((bankroll * 0.95) / (mkt_side + fee(mkt_side))))
                    actual_stake = contracts * mkt_side + fee(mkt_side) * contracts

                if contracts < 1:
                    continue

                bankroll       -= actual_stake
                total_exposure += actual_stake
                open_pos = Position(
                    token_key=token,
                    entry_price=mkt_side,
                    contracts=contracts,
                    entry_model=mp_side,
                    entry_edge=edge_side,
                    stake_usdc=actual_stake,
                    peak_price=mkt_side,
                )
                break

    # Force-close any open position at match end (settle at resolution price)
    if open_pos is not None:
        sa, sb = ticks[-1]["sets"] if ticks else (0, 0)
        target = params.best_of // 2 + 1
        p1_won_match = sa >= target
        if open_pos.token_key == "yes":
            exit_p = 0.99 if p1_won_match else 0.01
        else:
            exit_p = 0.99 if not p1_won_match else 0.01
        pnl = (exit_p - open_pos.entry_price) * open_pos.contracts - \
               fee(exit_p) * open_pos.contracts
        bankroll += open_pos.stake_usdc + pnl
        trades.append(TradeRecord(
            entry_price=open_pos.entry_price,
            exit_price=exit_p,
            model_p=model_p if ticks else base_p,
            token=open_pos.token_key,
            pnl=round(pnl, 4),
            stake=open_pos.stake_usdc,
            edge_at_entry=open_pos.entry_edge,
            exit_reason="match_settled",
            tick_index=len(ticks),
        ))
        adaptive.record(exit_p > open_pos.entry_price, open_pos.entry_model)

    diagnostics["serve_divergence_final"] = round(serve_divergence, 4)
    diagnostics["n_trades"]               = len(trades)
    diagnostics["final_bankroll"]         = round(bankroll, 4)

    return trades, bankroll, diagnostics


# ══════════════════════════════════════════════════════════════════════════════
# Worker function (runs in separate process)
# ══════════════════════════════════════════════════════════════════════════════

def _worker_batch(args):
    """
    Process a batch of match params and return aggregated trade records.
    Each worker has its own memo dicts and RNG state — no shared memory.
    """
    sys.setrecursionlimit(5000)
    match_params_list, adaptive_state, starting_bankroll, base_seed, noise_sigma = args
    adaptive = _AdaptiveCtrl(base_edge=adaptive_state.get("base_edge", 0.04))
    # Restore adaptive state from prior fold
    adaptive._edge_adj = adaptive_state.get("edge_adj", adaptive.base_edge)
    adaptive._kelly_m  = adaptive_state.get("kelly_mult", 1.0)
    adaptive._window   = [(r[0], r[1]) for r in adaptive_state.get("window", [])]

    all_trades: List[dict] = []
    bankroll = starting_bankroll

    for i, params in enumerate(match_params_list):
        seed = base_seed + i
        trades, bankroll, diag = evaluate_match(
            params, adaptive, bankroll, seed, noise_sigma
        )
        for t in trades:
            all_trades.append({
                "entry":       t.entry_price,
                "exit":        t.exit_price,
                "model_p":     t.model_p,
                "token":       t.token,
                "pnl":         t.pnl,
                "stake":       t.stake,
                "edge":        t.edge_at_entry,
                "exit_reason": t.exit_reason,
            })

    adaptive_out = {
        "base_edge":  adaptive.base_edge,
        "edge_adj":   adaptive._edge_adj,
        "kelly_mult": adaptive._kelly_m,
        "window":     [(r[0], r[1]) for r in adaptive._window],
    }
    return all_trades, bankroll, adaptive_out


# ══════════════════════════════════════════════════════════════════════════════
# Match parameter generator
# ══════════════════════════════════════════════════════════════════════════════

def _atp_player(rng: np.random.Generator, rank: int) -> PlayerProfile:
    """
    Sample a statistically realistic ATP player profile at a given ranking tier.
    Based on ATP serve-win probability distributions from Barnett & Clarke (2005)
    and updated with 2019-2024 hard-court averages.
    """
    # p_serve distribution by rank tier
    if rank <= 10:
        ps = float(rng.normal(0.682, 0.025))
    elif rank <= 50:
        ps = float(rng.normal(0.662, 0.028))
    elif rank <= 100:
        ps = float(rng.normal(0.645, 0.030))
    else:
        ps = float(rng.normal(0.628, 0.032))

    elo_pts   = max(100, int(rng.normal(9000 - rank * 70, 800)))
    age       = float(np.clip(rng.normal(26.0, 3.5), 18, 38))
    height_cm = float(np.clip(rng.normal(187.0, 8.0), 165, 210))
    hand      = "L" if rng.random() < 0.12 else "R"
    win_rate  = float(np.clip(0.78 - rank * 0.003 + rng.normal(0, 0.04), 0.40, 0.90))

    return PlayerProfile(
        name=f"Player_R{rank}",
        ranking=rank,
        elo_pts=float(elo_pts),
        age=age,
        height_cm=height_cm,
        hand=hand,
        p_serve=float(np.clip(ps, 0.50, 0.80)),
        win_rate=win_rate,
    )


def generate_match_params(
    n: int,
    seed: int = 0,
) -> List[MatchParams]:
    """
    Generate n realistic ATP/WTA match scenarios for backtesting.
    Matches span a range of:
      - Player ranking pairs
      - Surfaces (hard 60%, clay 25%, grass 15%)
      - Altitudes (sea-level to 2500m for Madrid/Mexico City)
      - Temperatures (10°C to 40°C)
    """
    rng = np.random.default_rng(seed)
    params_list: List[MatchParams] = []

    surfaces    = rng.choice(["hard", "clay", "grass"], size=n, p=[0.60, 0.25, 0.15])
    altitudes   = rng.choice([0, 0, 0, 100, 250, 500, 635, 2254, 2240],
                              size=n)  # ~60% sea-level, rare high altitude
    temps       = rng.uniform(10, 40, size=n)
    rank_a_pool = rng.integers(1, 200, size=n)
    rank_diff   = rng.integers(1, 80, size=n)

    for i in range(n):
        ra = int(rank_a_pool[i])
        rb = int(np.clip(ra + rank_diff[i], 1, 300))
        pa = _atp_player(rng, ra)
        pb = _atp_player(rng, rb)
        # True win probability based purely on p_serve values (Markov-implied)
        true_p = 1.0 / (1.0 + math.exp(-(pa.p_serve - pb.p_serve) * 6.0))

        params_list.append(MatchParams(
            player_a=pa,
            player_b=pb,
            surface=str(surfaces[i]),
            altitude_m=float(altitudes[i]),
            temp_c=float(temps[i]),
            best_of=3,
            glare_prob=float(rng.uniform(0.0, 0.20)),
            true_p_a=float(np.clip(true_p, 0.05, 0.95)),
        ))

    return params_list


# ══════════════════════════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[dict], starting_bankroll: float) -> dict:
    if not trades:
        return {"n_trades": 0}

    pnls    = np.array([t["pnl"] for t in trades])
    stakes  = np.array([t["stake"] for t in trades])
    wins   = pnls > 0
    models = np.array([t["model_p"] for t in trades])

    # Cumulative bankroll curve
    bankroll_curve = starting_bankroll + np.cumsum(pnls)
    running_max    = np.maximum.accumulate(bankroll_curve)
    drawdowns      = (running_max - bankroll_curve) / (running_max + 1e-9)
    max_dd         = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    # Sharpe (daily returns approximation — assumes one match per "day")
    if len(pnls) > 1:
        roi_per_trade = pnls / (stakes + 1e-9)
        sharpe = float(np.mean(roi_per_trade) / (np.std(roi_per_trade) + 1e-9) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Calibration: actual win rate vs. average model probability
    actual_wins   = float(np.mean(wins))
    avg_model_p   = float(np.mean(models))
    calib_ratio   = actual_wins / avg_model_p if avg_model_p > 0 else None

    # ROI
    total_pnl   = float(np.sum(pnls))
    total_stake = float(np.sum(stakes))
    roi         = total_pnl / total_stake if total_stake > 0 else 0.0

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in trades:
        exit_reasons[t["exit_reason"]] += 1

    # Kelly vs. no-Kelly comparison (theoretical)
    avg_edge = float(np.mean([t["edge"] for t in trades]))

    return {
        "n_trades":        len(trades),
        "win_rate":        round(actual_wins, 4),
        "avg_model_p":     round(avg_model_p, 4),
        "calib_ratio":     round(calib_ratio, 3) if calib_ratio else None,
        "total_pnl":       round(total_pnl, 2),
        "total_stake":     round(total_stake, 2),
        "roi":             round(roi, 4),
        "sharpe":          round(sharpe, 3),
        "max_drawdown":    round(max_dd, 4),
        "avg_edge":        round(avg_edge, 4),
        "exit_breakdown":  dict(exit_reasons),
        "final_bankroll":  round(float(bankroll_curve[-1]), 2) if len(bankroll_curve) else starting_bankroll,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Walk-forward orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_walkforward(
    n_folds:           int   = 8,
    matches_per_fold:  int   = 150,
    starting_bankroll: float = 1000.0,
    noise_sigma:       float = 0.025,
    n_workers:         int   = None,
    seed:              int   = 42,
    out_csv:           str   = None,
    verbose:           bool  = True,
) -> dict:
    """
    Expanding-window walk-forward backtest.

    Fold structure:
      Fold 1: train=[] (cold start), test=matches[0:M]
      Fold 2: train=matches[0:M],   test=matches[M:2M]
      ...
      Fold k: train=matches[0:(k-1)M], test=matches[(k-1)M:kM]

    The AdaptiveController's state carries over from the training period
    into each test fold (no data leak — state is built from past folds only).
    """
    n_workers  = n_workers or mp.cpu_count()
    total_matches = n_folds * matches_per_fold

    if verbose:
        print(f"\n{'═'*70}")
        print(f"  WALK-FORWARD BACKTEST  |  {n_folds} folds × {matches_per_fold} matches")
        print(f"  Total matches: {total_matches}  |  Workers: {n_workers}")
        print(f"  Starting bankroll: ${starting_bankroll:.2f}  |  Market noise σ={noise_sigma}")
        print(f"{'═'*70}\n")

    # Generate all match params upfront (reproducible)
    all_params = generate_match_params(total_matches, seed=seed)

    fold_results   = []
    all_trades_out = []
    adaptive_state = {
        "base_edge":  0.02,
        "edge_adj":   0.02,
        "kelly_mult": 1.0,
        "window":     [],
    }
    bankroll = starting_bankroll

    t0 = time.time()

    for fold_idx in range(n_folds):
        fold_start  = fold_idx * matches_per_fold
        fold_end    = fold_start + matches_per_fold
        fold_params = all_params[fold_start:fold_end]

        # ── Distribute across workers ──────────────────────────────────────
        # Split the fold evenly across all CPUs; each chunk runs independently.
        chunk_size = max(1, len(fold_params) // n_workers)
        chunks     = [fold_params[i:i+chunk_size]
                      for i in range(0, len(fold_params), chunk_size)]

        fold_trades_all:  List[dict] = []
        adaptive_states_out:List[dict] = []

        base_seed = seed * 10000 + fold_idx * 1000

        worker_args = [
            (chunk, adaptive_state, bankroll, base_seed + ci * 100, noise_sigma)
            for ci, chunk in enumerate(chunks)
        ]

        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_worker_batch, arg): ci
                       for ci, arg in enumerate(worker_args)}
            for fut in as_completed(futures):
                try:
                    chunk_trades, chunk_bk, adap_out = fut.result()
                    fold_trades_all.extend(chunk_trades)
                    adaptive_states_out.append(adap_out)
                except Exception as exc:
                    if verbose:
                        print(f"  [WARNING] Worker error: {exc}")

        # Merge adaptive states from workers (take median adjustments)
        if adaptive_states_out:
            adaptive_state = {
                "base_edge":  adaptive_states_out[0]["base_edge"],
                "edge_adj":   float(np.median([s["edge_adj"]  for s in adaptive_states_out])),
                "kelly_mult": float(np.median([s["kelly_mult"] for s in adaptive_states_out])),
                "window":     adaptive_states_out[-1]["window"],  # latest worker's window
            }

        # Metrics for this fold
        metrics = compute_metrics(fold_trades_all, bankroll)

        # Update bankroll for next fold
        bankroll = metrics.get("final_bankroll", bankroll)

        fold_results.append({
            "fold":    fold_idx + 1,
            "matches": len(fold_params),
            **metrics,
        })

        for t in fold_trades_all:
            t["fold"] = fold_idx + 1
        all_trades_out.extend(fold_trades_all)

        if verbose:
            _print_fold(fold_idx + 1, metrics, bankroll)

    # ── Aggregate across all folds ─────────────────────────────────────────
    aggregate = compute_metrics(all_trades_out, starting_bankroll)
    elapsed   = time.time() - t0

    if verbose:
        _print_aggregate(aggregate, elapsed, fold_results)

    # ── Optional CSV output ────────────────────────────────────────────────
    if out_csv and all_trades_out:
        _write_csv(out_csv, all_trades_out)
        if verbose:
            print(f"\n  Trades written to: {out_csv}")

    return {
        "fold_results": fold_results,
        "aggregate":    aggregate,
        "all_trades":   all_trades_out,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Pretty printing
# ══════════════════════════════════════════════════════════════════════════════

def _print_fold(fold_idx: int, m: dict, bankroll: float):
    calib = f"{m['calib_ratio']:.3f}" if m.get("calib_ratio") else "n/a"
    exits = m.get("exit_breakdown", {})
    exit_str = "  ".join(f"{k}={v}" for k, v in sorted(exits.items()))
    print(
        f"  Fold {fold_idx:>2} │ trades={m['n_trades']:>4}  "
        f"win={m['win_rate']:.1%}  calib={calib}  "
        f"ROI={m['roi']:>+.2%}  Sharpe={m['sharpe']:>+.3f}  "
        f"MaxDD={m['max_drawdown']:.1%}  "
        f"PnL=${m['total_pnl']:>+.2f}  BK=${bankroll:.2f}\n"
        f"         exits: {exit_str}"
    )


def _print_aggregate(m: dict, elapsed: float, fold_results: list):
    print(f"\n{'─'*70}")
    print("  AGGREGATE  (all folds)")
    print(f"{'─'*70}")
    calib = f"{m['calib_ratio']:.3f}" if m.get("calib_ratio") else "n/a"
    print(f"  Trades      : {m['n_trades']}")
    print(f"  Win rate    : {m['win_rate']:.2%}")
    print(f"  Calib ratio : {calib}  (1.0 = perfectly calibrated)")
    print(f"  Total PnL   : ${m['total_pnl']:>+.2f}")
    print(f"  Total stake : ${m['total_stake']:.2f}")
    print(f"  ROI         : {m['roi']:>+.2%}")
    print(f"  Sharpe      : {m['sharpe']:>+.3f}  (annualised)")
    print(f"  Max drawdown: {m['max_drawdown']:.2%}")
    print(f"  Avg edge    : {m['avg_edge']:>+.4f}")
    print(f"  Final BK    : ${m['final_bankroll']:.2f}")
    print(f"  Elapsed     : {elapsed:.1f}s")

    # Per-fold ROI sparkline
    spark = "  Fold ROIs   : " + "  ".join(
        f"F{f['fold']}:{f['roi']:>+.1%}" for f in fold_results
    )
    print(spark)

    exits = m.get("exit_breakdown", {})
    print(f"  Exit reasons: {dict(sorted(exits.items()))}")
    print(f"{'═'*70}\n")


def _write_csv(path: str, trades: List[dict]):
    with open(path, "w", newline="") as f:
        fieldnames = ["fold", "entry", "exit", "model_p", "token",
                      "pnl", "stake", "edge", "exit_reason"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)


# ══════════════════════════════════════════════════════════════════════════════
# Sensitivity sweep  (optional, --sweep flag)
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_sweep(n_folds=4, matches_per_fold=80, seed=99, n_workers=None):
    """
    Grid-search over noise_sigma and kelly_fraction to find optimal parameters.
    Outputs a table of Sharpe and ROI per combination.
    """
    n_workers = n_workers or mp.cpu_count()
    noise_vals   = [0.015, 0.025, 0.040]
    kelly_vals   = [0.50, 0.75, 1.0, 1.25]

    print(f"\n{'─'*70}")
    print("  SENSITIVITY SWEEP  (noise_sigma × kelly_fraction)")
    print(f"  {n_folds} folds × {matches_per_fold} matches per combination")
    print(f"{'─'*70}")
    print(f"  {'noise':>8}  {'kelly':>7}  {'roi':>8}  {'sharpe':>8}  {'max_dd':>8}  {'n_trades':>9}")

    best = {"sharpe": -999, "combo": None}

    for ns in noise_vals:
        for kf in kelly_vals:
            # Temporarily override KELLY_FRACTION global for this run
            import backtest as _self
            _self.KELLY_FRACTION = kf
            res = run_walkforward(
                n_folds=n_folds,
                matches_per_fold=matches_per_fold,
                noise_sigma=ns,
                n_workers=n_workers,
                seed=seed,
                verbose=False,
            )
            agg = res["aggregate"]
            print(
                f"  {ns:>8.3f}  {kf:>7.2f}  "
                f"{agg['roi']:>+8.2%}  {agg['sharpe']:>+8.3f}  "
                f"{agg['max_drawdown']:>8.2%}  {agg['n_trades']:>9}"
            )
            if agg["sharpe"] > best["sharpe"]:
                best = {"sharpe": agg["sharpe"], "combo": (ns, kf)}
            _self.KELLY_FRACTION = KELLY_FRACTION  # reset

    if best["combo"]:
        print(f"\n  Best combo: noise={best['combo'][0]}  kelly_f={best['combo'][1]}  "
              f"Sharpe={best['sharpe']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Walk-forward backtest using the full TennisBot signal pipeline."
    )
    parser.add_argument("--folds",             type=int,   default=8,
                        help="Number of walk-forward folds (default: 8)")
    parser.add_argument("--matches-per-fold",  type=int,   default=150,
                        help="Matches simulated per fold (default: 150)")
    parser.add_argument("--bankroll",          type=float, default=1000.0,
                        help="Starting bankroll in USDC (default: 1000)")
    parser.add_argument("--noise",             type=float, default=0.025,
                        help="Market noise σ (default: 0.025 = 2.5¢)")
    parser.add_argument("--workers",           type=int,   default=None,
                        help="CPU workers (default: all cores)")
    parser.add_argument("--seed",              type=int,   default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--out-csv",           type=str,   default=None,
                        help="Write all trades to CSV file")
    parser.add_argument("--sweep",             action="store_true",
                        help="Run sensitivity sweep over noise × kelly parameters")
    parser.add_argument("--sweep-folds",       type=int,   default=4)
    parser.add_argument("--sweep-matches",     type=int,   default=80)
    args = parser.parse_args()

    if args.sweep:
        run_sensitivity_sweep(
            n_folds=args.sweep_folds,
            matches_per_fold=args.sweep_matches,
            seed=args.seed,
            n_workers=args.workers,
        )
    else:
        run_walkforward(
            n_folds=args.folds,
            matches_per_fold=args.matches_per_fold,
            starting_bankroll=args.bankroll,
            noise_sigma=args.noise,
            n_workers=args.workers,
            seed=args.seed,
            out_csv=args.out_csv,
            verbose=True,
        )


if __name__ == "__main__":
    # Freeze support for Windows multiprocessing
    mp.freeze_support()
    main()
