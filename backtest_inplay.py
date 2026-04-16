"""
In-Play Markov Backtest
========================
Simulates what the live bot would have done during real matches by
replaying synthetic score progression through the Markov engine.

How it works:
  1. Load ATP match CSVs (2022-2024)
  2. For each match, derive serve/return probabilities from real stats
     (w_svpt, w_1stIn, w_1stWon, w_2ndWon, l_svpt etc.)
  3. Simulate a realistic game-by-game score progression using those
     probabilities (same as the live bot does with LiveScore updates)
  4. At each game checkpoint, compute Markov win probability and
     compare to ELO-implied market price
  5. Apply tiered Kelly sizing and settle bets at match end
  6. Report: PnL, ROI, win rate, avg latency-of-bet (how early/late)
"""

import os
import glob
import json
import logging
import warnings
import sys
import numpy as np
import pandas as pd

sys.setrecursionlimit(5000)  # Markov DP needs deeper recursion for match states
sys.path.insert(0, os.path.dirname(__file__))
from markov_engine import LiveMatchState, match_win_prob

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────
ATP_DIR        = os.path.expanduser("~/Downloads/tennis_atp-master")
TEST_YEARS     = range(2022, 2027)   # out-of-sample only
BANKROLL_INIT  = 500.0
VIG            = 0.08                # 8% exchange rake
MIN_EDGE       = 0.04                # 4pp min edge (same as live bot)
ELO_K          = 32
ELO_INIT       = 1500.0

# Confidence tiers — exactly mirrors bet_manager._kelly_size
KELLY_TIERS = [
    (0.70, 0.40),
    (0.60, 0.25),
    (0.55, 0.12),
    (0.00, 0.05),
]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data loading
# ─────────────────────────────────────────────────────────────────────────────
def load_matches() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ATP_DIR, "atp_matches_[12]*.csv")))
    frames = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = year
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    df = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(df):,} matches ({files[0][-8:-4]}–{files[-1][-8:-4]})")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Serve/Return probability extraction from real match stats
# ─────────────────────────────────────────────────────────────────────────────
def extract_serve_probs(row) -> tuple[float, float] | None:
    """
    Derive P(winner wins point on own serve) and P(winner wins point on return)
    from the real in-match serve stats.

    w_1stWon / w_1stIn  → 1st serve win %
    w_2ndWon / (w_svpt - w_1stIn) → 2nd serve win %
    Combined: p_serve = weighted 1st+2nd serve point win prob
    p_return = 1 - l_1stWon/l_1stIn weighted similarly
    """
    try:
        # Winner (P1)
        w_svpt  = float(row["w_svpt"])
        w_1stIn = float(row["w_1stIn"])
        w_1stWon= float(row["w_1stWon"])
        w_2ndWon= float(row["w_2ndWon"])

        # Loser (P2)
        l_svpt  = float(row["l_svpt"])
        l_1stIn = float(row["l_1stIn"])
        l_1stWon= float(row["l_1stWon"])
        l_2ndWon= float(row["l_2ndWon"])

        if w_svpt <= 0 or l_svpt <= 0:
            return None

        w_2ndIn = w_svpt - w_1stIn
        l_2ndIn = l_svpt - l_1stIn

        # P(winner wins point on own serve)
        p_w_serve = (w_1stWon + w_2ndWon) / w_svpt

        # P(winner wins point on loser's serve)
        # = 1 - P(loser wins point on own serve)
        p_l_serve = (l_1stWon + l_2ndWon) / l_svpt
        p_w_return = 1.0 - p_l_serve

        # Clamp to valid range
        p_w_serve  = np.clip(p_w_serve,  0.40, 0.90)
        p_w_return = np.clip(p_w_return, 0.10, 0.60)

        return p_w_serve, p_w_return
    except (ValueError, TypeError, ZeroDivisionError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parse score string into set-by-set game scores
# ─────────────────────────────────────────────────────────────────────────────
def parse_score(score_str: str) -> list[tuple[int, int]] | None:
    """
    Parse '6-3 7-5' or '6-3 3-6 7-6(4)' into [(6,3),(7,5)] etc.
    Returns list of (winner_games, loser_games) per set.
    Skips retirements/walkovers.
    """
    if not isinstance(score_str, str):
        return None
    if any(x in score_str.upper() for x in ["RET", "W/O", "DEF", "UNF"]):
        return None

    sets = []
    for s in score_str.strip().split():
        # Remove tiebreak score e.g. 7-6(4) -> 7-6
        s_clean = s.split("(")[0]
        parts = s_clean.split("-")
        if len(parts) != 2:
            return None
        try:
            w, l = int(parts[0]), int(parts[1])
            if w < 0 or l < 0 or w > 7 or l > 7:
                return None
            sets.append((w, l))
        except ValueError:
            return None
    return sets if len(sets) >= 2 else None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Kelly sizing — mirrors bet_manager exactly
# ─────────────────────────────────────────────────────────────────────────────
def kelly_size(p: float, market_price: float, bankroll: float) -> float:
    if market_price <= 0 or market_price >= 1:
        return 0.0
    b = (1.0 / market_price) - 1.0
    q = 1.0 - p
    f = (p * b - q) / b
    if f <= 0:
        return 0.0

    tier_fraction = 0.05
    for threshold, frac in KELLY_TIERS:
        if p >= threshold:
            tier_fraction = frac
            break

    capped = min(f * bankroll * tier_fraction, bankroll * 0.20)
    return max(round(capped, 2), 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main in-play simulation loop
# ─────────────────────────────────────────────────────────────────────────────
def simulate_inplay(df: pd.DataFrame, elo_ratings: dict) -> dict:
    """
    For each match:
       - Replay game-by-game score through LiveMatchState
       - At the START of each set (clean entry point):
           • Get current Markov win prob
           • Compare to ELO market price
           • Place bet if edge > MIN_EDGE
       - Settle at match end
    """
    bankroll   = BANKROLL_INIT
    history    = [bankroll]
    bets       = []
    bet_log    = []
    cum_pnl    = 0.0

    for idx, row in df.iterrows():
        sets = parse_score(row.get("score", ""))
        if sets is None:
            continue

        probs = extract_serve_probs(row)
        if probs is None:
            continue

        p_serve, p_return = probs
        best_of = int(row.get("best_of", 3))

        # ELO-implied market price (pre-match)
        wid = str(row.get("winner_id", ""))
        lid = str(row.get("loser_id", ""))
        elo_w = elo_ratings.get(wid, ELO_INIT)
        elo_l = elo_ratings.get(lid, ELO_INIT)
        market_prob = 1.0 / (1.0 + 10 ** ((elo_l - elo_w) / 400))

        # Update ELO post-match
        exp_w = 1.0 / (1.0 + 10 ** ((elo_l - elo_w) / 400))
        elo_ratings[wid] = elo_w + ELO_K * (1 - exp_w)
        elo_ratings[lid] = elo_l + ELO_K * (0 - (1 - exp_w))

        # ── In-play replay ────────────────────────────────────────────────
        lms = LiveMatchState(p_serve, p_return)
        placed = False       # only one bet per match (first clear edge)
        bet_at_set = None

        for set_idx, (wg, lg) in enumerate(sets):
            # Snapshot Markov probability BEFORE this set is fed in
            markov_p = lms.win_probability()

            # ELO market adjusted for vig
            mkt_adj = min(market_prob + VIG / 2, 0.95)
            edge    = markov_p - mkt_adj

            # Bet on first set with sufficient edge (mirrors how live bot works)
            if not placed and abs(edge) >= MIN_EDGE:
                # Bet on winner (p=markov_p) or loser (p=1-markov_p)?
                if edge > 0:
                    bet_p, bet_price, bet_wins_if = markov_p, mkt_adj, True
                else:
                    bet_p, bet_price = 1.0 - markov_p, 1.0 - mkt_adj
                    edge = -edge
                    bet_wins_if = False

                stake = kelly_size(bet_p, bet_price, bankroll)
                if stake >= 1.0:
                    placed    = True
                    bet_at_set = set_idx

                    # "Winner wins" = True. bet_wins_if tracks which side we bet.
                    match_won_by_winner = True  # winner always wins in this dataset
                    won = (match_won_by_winner == bet_wins_if)

                    pnl = stake * (1.0 - bet_price) / bet_price if won else -stake
                    bankroll    += pnl
                    cum_pnl     += pnl
                    bankroll     = max(bankroll, 0.01)
                    bets.append(stake)
                    bet_log.append({
                        "edge":       round(edge, 4),
                        "markov_p":   round(markov_p, 4),
                        "market_p":   round(bet_price, 4),
                        "stake":      round(stake, 2),
                        "pnl":        round(pnl, 2),
                        "won":        won,
                        "set_idx":    set_idx,            # 0 = pre-match, 1 = after S1, etc.
                        "surface":    row.get("surface", ""),
                    })

            # Feed this set result into LiveMatchState
            if edge > 0:  # We bet on winner side
                lms.update({
                    "sets":      (lms.match_sets[0] + wg // max(wg, 1),
                                  lms.match_sets[1] + lg // max(wg, 1)) if wg > lg
                                 else (lms.match_sets[0], lms.match_sets[1] + 1),
                    "games":    (0, 0),
                    "points":   (0, 0),
                })
            # Simpler: just count sets won/lost
            winner_sets = sum(1 for (w, l) in sets[:set_idx+1] if w > l)
            loser_sets  = sum(1 for (w, l) in sets[:set_idx+1] if l > w)
            lms.match_sets = (winner_sets, loser_sets)
            lms.current_set_games = (0, 0)
            lms.current_game_points = (0, 0)

        history.append(bankroll)

    n_bets    = len(bets)
    wins      = sum(1 for b in bet_log if b["won"])
    staked    = sum(bets)
    roi       = cum_pnl / staked * 100 if staked > 0 else 0.0

    # By-set timing breakdown
    timing = {}
    for entry in bet_log:
        s = entry["set_idx"]
        label = "Pre-match" if s == 0 else f"After Set {s}"
        timing.setdefault(label, {"bets": 0, "pnl": 0.0})
        timing[label]["bets"] += 1
        timing[label]["pnl"]  += entry["pnl"]

    # By-surface breakdown
    surface_stats = {}
    for entry in bet_log:
        surf = entry.get("surface", "Unknown") or "Unknown"
        surface_stats.setdefault(surf, {"bets": 0, "pnl": 0.0, "wins": 0})
        surface_stats[surf]["bets"] += 1
        surface_stats[surf]["pnl"]  += entry["pnl"]
        if entry["won"]:
            surface_stats[surf]["wins"] += 1

    return {
        "history":        history,
        "bankroll_final": bankroll,
        "n_bets":         n_bets,
        "wins":           wins,
        "win_rate":       wins / n_bets * 100 if n_bets > 0 else 0.0,
        "roi":            roi,
        "cum_pnl":        cum_pnl,
        "staked":         staked,
        "timing":         timing,
        "surface":        surface_stats,
        "avg_edge":       float(np.mean([b["edge"] for b in bet_log])) if bet_log else 0.0,
        "avg_stake":      float(np.mean(bets)) if bets else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Main
# ─────────────────────────────────────────────────────────────────────────────
def build_elo(df: pd.DataFrame) -> dict:
    elo = {}
    warmup = df[df["year"] < 2022]
    for _, row in warmup.iterrows():
        wid, lid = str(row.get("winner_id","")), str(row.get("loser_id",""))
        ew = elo.get(wid, ELO_INIT)
        el = elo.get(lid, ELO_INIT)
        exp_w = 1.0 / (1.0 + 10 ** ((el - ew) / 400))
        elo[wid] = ew + ELO_K * (1 - exp_w)
        elo[lid] = el + ELO_K * (0 - (1 - exp_w))
    return elo


def main():
    df = load_matches()

    log.info("Building ELO warm-up (1968–2021)...")
    elo = build_elo(df)

    test_df = df[df["year"].isin(TEST_YEARS)].reset_index(drop=True)
    # Drop rows without serve stats (some older/challenger matches)
    test_df = test_df.dropna(subset=["w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
                                      "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon"])
    log.info(f"Test set: {len(test_df):,} matches with serve stats ({test_df['year'].min()}–{test_df['year'].max()})")

    log.info("Running in-play Markov simulation...")
    results = simulate_inplay(test_df, elo)

    # ── Console output ───────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  IN-PLAY MARKOV BACKTEST  (Live Score Simulation)")
    print("="*60)
    print(f"  Starting Bankroll : ${BANKROLL_INIT:,.2f}")
    print(f"  Ending Bankroll   : ${results['bankroll_final']:,.2f}")
    print(f"  Total P&L         : ${results['cum_pnl']:+,.2f}")
    print(f"  Total Bets        : {results['n_bets']:,}")
    print(f"  Win Rate          : {results['win_rate']:.1f}%")
    print(f"  Staking ROI       : {results['roi']:+.2f}%")
    print(f"  Avg Edge          : {results['avg_edge']*100:.2f}pp")
    print(f"  Avg Stake         : ${results['avg_stake']:.2f}")
    print()

    print("  ── Bet Timing (when in the match the bot bet) ──")
    for label, stat in sorted(results["timing"].items()):
        print(f"    {label:<18} : {stat['bets']:>5} bets  |  P&L ${stat['pnl']:+,.2f}")
    print()

    print("  ── Performance by Surface ──")
    for surf, stat in sorted(results["surface"].items()):
        wr = stat["wins"] / stat["bets"] * 100 if stat["bets"] > 0 else 0
        print(f"    {surf:<10} : {stat['bets']:>5} bets  |  {wr:.1f}% win rate  |  P&L ${stat['pnl']:+,.2f}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
