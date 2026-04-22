"""
wfo_inplay.py  —  Walk-Forward Optimisation with Live Score (In-Play) Backtesting
==================================================================================
Uses real ATP match data (Jeff Sackmann CSVs) including:
  - Actual score strings  (e.g. "6-3 3-6 7-5")
  - Real in-match serve statistics (1stIn, 1stWon, 2ndWon, svpt …)
  - Player ELO ratings (pre-match market proxy)
  - XGBoost pre-match model (trained on expanding window)

At each set boundary the Markov engine recomputes a live win probability using
the actual serve stats from that match.  Bets are placed on the first checkpoint
(pre-match, after S1, or after S2) where the edge exceeds the threshold.

Outputs
-------
  wfo_inplay_report.pdf  — 25-page PDF (in-play vs pre-match comparison)
  wfo_inplay_bets.csv    — every OOS bet with timing metadata
  wfo_inplay_params.json — consensus optimal parameters
"""

import os, sys, glob, json, logging, warnings, itertools
from datetime import datetime
sys.setrecursionlimit(10000)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss,
    roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markov_engine import LiveMatchState

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_DIR     = os.path.dirname(os.path.abspath(__file__))
ATP_DIR  = os.path.expanduser("~/Downloads/tennis_atp-master")
PDF_OUT  = os.path.join(_DIR, "wfo_inplay_report.pdf")
CSV_OUT  = os.path.join(_DIR, "wfo_inplay_bets.csv")
JSON_OUT = os.path.join(_DIR, "wfo_inplay_params.json")

# ── Constants ─────────────────────────────────────────────────────────────────
ELO_K         = 32
ELO_INIT      = 1500.0
BANKROLL_INIT = 500.0
MAX_BET_FRAC  = 0.20
WFO_TEST_START = 2015          # first OOS test year
WFO_MIN_TRAIN  = 5             # minimum training years before first fold

PARAM_GRID = {
    "min_edge":   [0.02, 0.04, 0.06, 0.08, 0.10],
    "kelly_frac": [0.05, 0.10, 0.20, 0.35, 0.50],
    "vig":        [0.05, 0.07, 0.09, 0.11],
}

SURFACE_MAP = {
    "Hard": "Hard", "Clay": "Clay", "Grass": "Grass",
    "Carpet": "Hard", "Indoor Hard": "Hard", "Outdoor Hard": "Hard",
}

MODEL_FEATURES = [
    "Surface_Hard", "Surface_Clay", "Surface_Grass", "Best_Of_Sets",
    "P1_Is_Right_Handed", "P1_Height_cm", "P1_Age", "P1_Rank", "P1_Rank_Points",
    "P2_Is_Right_Handed", "P2_Height_cm", "P2_Age", "P2_Rank", "P2_Rank_Points",
    "Height_Diff", "Age_Diff", "Rank_Diff", "Rank_Points_Diff",
    "Elo_Diff", "Surface_Height_Grass", "Age_Diff_Sq",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════

def load_atp_matches() -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(ATP_DIR, "atp_matches_[12]*.csv")))
    if not files:
        raise FileNotFoundError(f"No ATP CSVs in {ATP_DIR}")
    frames = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = year
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(combined):,} matches ({files[0][-8:-4]}–{files[-1][-8:-4]})")
    return combined


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Build model features + ELO + serve stats.  No lookahead."""
    df = raw.copy()
    df["surface_norm"] = df["surface"].map(SURFACE_MAP).fillna("Hard")
    df["winner_hand_enc"] = (df["winner_hand"] == "R").astype(int)
    df["loser_hand_enc"]  = (df["loser_hand"]  == "R").astype(int)

    for col in ["winner_ht","loser_ht","winner_age","loser_age",
                "winner_rank","loser_rank","winner_rank_points","loser_rank_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ["winner_rank","loser_rank","winner_rank_points","loser_rank_points"]:
        med = df.groupby("year")[col].transform("median")
        df[col] = df[col].fillna(med)

    for col in ["winner_ht","loser_ht","winner_age","loser_age"]:
        df[col] = df[col].fillna(df[col].median())

    # ELO — updated AFTER each match prediction (strict temporal order)
    elo: dict = {}
    elo_w_before = np.empty(len(df), dtype=float)
    elo_l_before = np.empty(len(df), dtype=float)
    for i, row in enumerate(df.itertuples(index=False)):
        wid, lid = str(row.winner_id), str(row.loser_id)
        ew, el = elo.get(wid, ELO_INIT), elo.get(lid, ELO_INIT)
        elo_w_before[i], elo_l_before[i] = ew, el
        exp_w = 1 / (1 + 10 ** ((el - ew) / 400))
        elo[wid] = ew + ELO_K * (1 - exp_w)
        elo[lid] = el + ELO_K * (0 - (1 - exp_w))

    df["elo_winner"] = elo_w_before
    df["elo_loser"]  = elo_l_before
    df["elo_prob_winner"] = 1 / (1 + 10 ** ((df["elo_loser"] - df["elo_winner"]) / 400))

    # Randomise P1/P2 to avoid winner-bias
    rng  = np.random.default_rng(42)
    swap = rng.random(len(df)) < 0.5

    def pick(wc, lc):
        w, l = df[wc].values, df[lc].values
        return np.where(swap, l, w), np.where(swap, w, l)

    p1_hand, p2_hand = pick("winner_hand_enc",   "loser_hand_enc")
    p1_ht,   p2_ht   = pick("winner_ht",         "loser_ht")
    p1_age,  p2_age  = pick("winner_age",        "loser_age")
    p1_rank, p2_rank = pick("winner_rank",       "loser_rank")
    p1_rpts, p2_rpts = pick("winner_rank_points","loser_rank_points")
    p1_elo,  p2_elo  = pick("elo_winner",        "elo_loser")

    y = np.where(swap, 0, 1)   # 1 = P1 wins

    out = pd.DataFrame({
        "year":           df["year"].values,
        "tourney_date":   df["tourney_date"].values,
        "surface_norm":   df["surface_norm"].values,
        # model features
        "Surface_Hard":   (df["surface_norm"] == "Hard").astype(int).values,
        "Surface_Clay":   (df["surface_norm"] == "Clay").astype(int).values,
        "Surface_Grass":  (df["surface_norm"] == "Grass").astype(int).values,
        "Best_Of_Sets":   pd.to_numeric(df["best_of"], errors="coerce").fillna(3).values,
        "P1_Is_Right_Handed": p1_hand,
        "P1_Height_cm":   p1_ht,
        "P1_Age":         p1_age,
        "P1_Rank":        p1_rank,
        "P1_Rank_Points": p1_rpts,
        "P2_Is_Right_Handed": p2_hand,
        "P2_Height_cm":   p2_ht,
        "P2_Age":         p2_age,
        "P2_Rank":        p2_rank,
        "P2_Rank_Points": p2_rpts,
        "Height_Diff":    p1_ht   - p2_ht,
        "Age_Diff":       p1_age  - p2_age,
        "Rank_Diff":      p1_rank - p2_rank,
        "Rank_Points_Diff": p1_rpts - p2_rpts,
        "Elo_Diff":       p1_elo  - p2_elo,
        "Surface_Height_Grass": (p1_ht - p2_ht) * (df["surface_norm"] == "Grass").astype(int).values,
        "Age_Diff_Sq":    np.sign(p1_age - p2_age) * ((p1_age - p2_age) / 10.0) ** 2,
        "elo_prob_p1":    np.where(swap, 1 - df["elo_prob_winner"].values, df["elo_prob_winner"].values),
        # serve stats (winner/loser perspective — needed for Markov)
        "w_svpt":   pd.to_numeric(df["w_svpt"],  errors="coerce").values,
        "w_1stIn":  pd.to_numeric(df["w_1stIn"], errors="coerce").values,
        "w_1stWon": pd.to_numeric(df["w_1stWon"],errors="coerce").values,
        "w_2ndWon": pd.to_numeric(df["w_2ndWon"],errors="coerce").values,
        "l_svpt":   pd.to_numeric(df["l_svpt"],  errors="coerce").values,
        "l_1stIn":  pd.to_numeric(df["l_1stIn"], errors="coerce").values,
        "l_1stWon": pd.to_numeric(df["l_1stWon"],errors="coerce").values,
        "l_2ndWon": pd.to_numeric(df["l_2ndWon"],errors="coerce").values,
        "score":    df["score"].values,
        "best_of":  pd.to_numeric(df["best_of"], errors="coerce").fillna(3).values,
        # whether P1 = winner in raw data
        "p1_is_winner": (~swap).astype(int),
        "target":   y,
    })
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SERVE PROBABILITY EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

def extract_serve_probs(row, p1_is_winner: bool) -> tuple | None:
    """
    Returns (p_serve_p1, p_return_p1) — serve/return win prob FROM P1's perspective.
    p_serve_p1  = P(P1 wins a point when P1 serves)
    p_return_p1 = P(P1 wins a point when P2 serves)
    """
    try:
        w_svpt  = float(row["w_svpt"]);  l_svpt  = float(row["l_svpt"])
        w_1stWon= float(row["w_1stWon"]);l_1stWon= float(row["l_1stWon"])
        w_2ndWon= float(row["w_2ndWon"]);l_2ndWon= float(row["l_2ndWon"])
        if w_svpt <= 0 or l_svpt <= 0:
            return None

        p_winner_serve = (w_1stWon + w_2ndWon) / w_svpt
        p_loser_serve  = (l_1stWon + l_2ndWon) / l_svpt

        if p1_is_winner:
            p_serve_p1  = p_winner_serve
            p_return_p1 = 1.0 - p_loser_serve
        else:
            p_serve_p1  = p_loser_serve
            p_return_p1 = 1.0 - p_winner_serve

        p_serve_p1  = float(np.clip(p_serve_p1,  0.40, 0.90))
        p_return_p1 = float(np.clip(p_return_p1, 0.10, 0.60))
        return p_serve_p1, p_return_p1
    except (ValueError, TypeError, ZeroDivisionError):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. SCORE PARSING
# ═══════════════════════════════════════════════════════════════════════════════

def parse_score(score_str, p1_is_winner: bool) -> list | None:
    """
    Parse '6-3 7-5' into list of (p1_games, p2_games) per set.
    Handles tiebreak annotations (7-6(4)), retirements → None.
    """
    if not isinstance(score_str, str):
        return None
    if any(x in score_str.upper() for x in ["RET","W/O","DEF","UNF","ABN"]):
        return None
    sets = []
    for s in score_str.strip().split():
        s_clean = s.split("(")[0]
        parts = s_clean.split("-")
        if len(parts) != 2:
            return None
        try:
            wg, lg = int(parts[0]), int(parts[1])
        except ValueError:
            return None
        if not (0 <= wg <= 7 and 0 <= lg <= 7):
            return None
        # Convert to P1/P2 perspective
        if p1_is_winner:
            sets.append((wg, lg))
        else:
            sets.append((lg, wg))
    return sets if len(sets) >= 2 else None


# ═══════════════════════════════════════════════════════════════════════════════
# 4. KELLY SIZING
# ═══════════════════════════════════════════════════════════════════════════════

KELLY_TIERS = [(0.70, 0.40), (0.60, 0.25), (0.55, 0.12), (0.00, 0.05)]

def kelly_bet(p: float, mkp_adj: float, bankroll: float, kelly_frac: float) -> float:
    if not np.isfinite(p) or not np.isfinite(mkp_adj):
        return 0.0
    edge = p - mkp_adj
    if edge <= 0 or mkp_adj >= 0.95:
        return 0.0
    f_star = edge / (1.0 - mkp_adj)
    tier = next((t for thresh, t in KELLY_TIERS if p >= thresh), 0.05)
    frac = min(f_star * tier * (kelly_frac / 0.25), MAX_BET_FRAC)
    return bankroll * frac


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PRE-MATCH BETTING SIMULATION (XGBoost only)
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_prematch(dates, model_probs, market_probs, outcomes, surfaces, p1_ranks,
                      bankroll_init=BANKROLL_INIT, kelly_frac=0.25, min_edge=0.04, vig=0.08):
    bankroll = bankroll_init
    records  = []
    for date, mp, mkp_raw, outcome, surf, rank in zip(
            dates, model_probs, market_probs, outcomes, surfaces, p1_ranks):
        if mp - mkp_raw < min_edge or not (0.05 < mkp_raw < 0.95):
            continue
        mkp_adj  = min(mkp_raw + vig / 2, 0.95)
        if mp - mkp_adj <= 0:
            continue
        bet = kelly_bet(mp, mkp_adj, bankroll, kelly_frac)
        if bet < 0.01:
            continue
        profit   = bet * (1 - mkp_adj) / mkp_adj if outcome == 1 else -bet
        bankroll = max(bankroll + profit, 0.01)
        records.append({"date": date, "model_prob": round(mp, 4),
                        "market_prob": round(mkp_raw, 4),
                        "edge": round(mp - mkp_raw, 4),
                        "bet_size": round(bet, 4), "outcome": int(outcome),
                        "profit": round(profit, 4), "bankroll": round(bankroll, 4),
                        "surface": surf, "p1_rank": rank, "timing": "pre-match",
                        "min_edge": min_edge, "kelly_frac": kelly_frac, "vig": vig})
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. IN-PLAY BETTING SIMULATION (Markov engine on real scores)
# ═══════════════════════════════════════════════════════════════════════════════

from markov_engine import (
    _game_win_prob, _set_win_prob, _tiebreak_win_prob, _match_win_prob,
)

def _safe_markov(lms: LiveMatchState, target_sets: int) -> float | None:
    s1, s2 = lms.match_sets
    if s1 >= target_sets or s2 >= target_sets:
        return None
    try:
        return lms.win_probability()
    except RecursionError:
        return None


def _clear_markov_caches():
    _game_win_prob.cache_clear()
    _tiebreak_win_prob.cache_clear()
    _set_win_prob.cache_clear()
    _match_win_prob.cache_clear()


def precompute_signals(feat_df: pd.DataFrame, model_probs: np.ndarray) -> pd.DataFrame:
    """
    Run Markov once per match per checkpoint and cache results.
    Serve probs are rounded to 2 dp so lru_cache hits across similar matches
    (reduces unique (p_serve, p_return) pairs from ~N to ~100, ~100x speedup).
    """
    _clear_markov_caches()
    rows = []
    for idx in range(len(feat_df)):
        row     = feat_df.iloc[idx]
        mp_xgb  = float(model_probs[idx])
        mkp_raw = float(row["elo_prob_p1"])
        outcome = int(row["target"])
        surf    = str(row["surface_norm"])
        rank    = float(row["P1_Rank"])
        date    = row["tourney_date"]
        p1_win  = bool(row["p1_is_winner"])
        target_sets = int(row["best_of"]) // 2 + 1

        sets   = parse_score(row["score"], p1_win)
        sprobs = extract_serve_probs(row, p1_win) if sets is not None else None

        if sprobs is not None:
            # Round to 2 dp → lru_cache keys collapse across similar matches
            ps = round(sprobs[0], 2)
            pr = round(sprobs[1], 2)
            lms = LiveMatchState(ps, pr)
            markov_p = _safe_markov(lms, target_sets)
            pm_signal = (0.50 * mp_xgb + 0.50 * markov_p
                         if markov_p is not None and np.isfinite(markov_p)
                         else mp_xgb)
            pm_markov = markov_p if (markov_p is not None and np.isfinite(markov_p)) else mp_xgb
        else:
            ps = pr = None
            pm_signal = mp_xgb
            pm_markov = mp_xgb

        rows.append({"date": date, "xgb_prob": mp_xgb, "markov_prob": pm_markov,
                     "signal": pm_signal, "market_prob": mkp_raw, "outcome": outcome,
                     "surface": surf, "p1_rank": rank, "timing": "pre-match",
                     "set_when_bet": 0})

        if ps is not None and sets is not None:
            lms = LiveMatchState(ps, pr)
            p1_sets_won = p2_sets_won = 0
            for set_idx, (p1g, p2g) in enumerate(sets):
                p1_sets_won += 1 if p1g > p2g else 0
                p2_sets_won += 1 if p2g > p1g else 0
                lms.match_sets          = (p1_sets_won, p2_sets_won)
                lms.current_set_games   = (0, 0)
                lms.current_game_points = (0, 0)

                markov_p = _safe_markov(lms, target_sets)
                if markov_p is None or not np.isfinite(markov_p):
                    break

                w      = min(0.50 + 0.20 * (set_idx + 1), 0.90)
                signal = (1 - w) * mp_xgb + w * markov_p
                rows.append({"date": date, "xgb_prob": mp_xgb,
                             "markov_prob": markov_p, "signal": signal,
                             "market_prob": mkp_raw, "outcome": outcome,
                             "surface": surf, "p1_rank": rank,
                             "timing": f"after_set_{set_idx+1}",
                             "set_when_bet": set_idx + 1})

    _clear_markov_caches()   # free memory between folds
    return pd.DataFrame(rows)


def simulate_inplay_from_signals(signals_df: pd.DataFrame,
                                  bankroll_init: float = BANKROLL_INIT,
                                  kelly_frac: float = 0.25,
                                  min_edge: float = 0.04,
                                  vig: float = 0.08) -> pd.DataFrame:
    """
    Given precomputed signals, pick the FIRST checkpoint per match with
    sufficient edge and simulate Kelly betting.  Fast: no Markov recomputation.
    """
    if signals_df.empty:
        return pd.DataFrame()

    bankroll = bankroll_init
    records  = []
    # Process chronologically, one bet per match (identified by date+surface+rank)
    # Group by match identity: (date, xgb_prob, market_prob, outcome)
    match_key = signals_df[["date","xgb_prob","market_prob","outcome"]].apply(tuple, axis=1)
    signals_df = signals_df.copy()
    signals_df["_mkey"] = match_key

    for _, match_rows in signals_df.groupby("_mkey", sort=False):
        placed = False
        # Try checkpoints in order: pre-match first, then after_set_1, 2, …
        match_rows = match_rows.sort_values("set_when_bet")
        for _, r in match_rows.iterrows():
            if placed:
                break
            signal  = float(r["signal"])
            mkp_raw = float(r["market_prob"])
            mkp_adj = min(mkp_raw + vig / 2, 0.95)
            edge    = signal - mkp_adj
            if abs(edge) < min_edge or not np.isfinite(signal):
                continue
            outcome = int(r["outcome"])
            bet_prob = signal if signal > 0.5 else 1.0 - signal
            bet_mkp  = mkp_adj if signal > 0.5 else 1.0 - mkp_adj
            bet_wins = (signal > 0.5) == (outcome == 1)
            bet = kelly_bet(bet_prob, bet_mkp, bankroll, kelly_frac)
            if bet < 0.01:
                continue
            profit   = bet * (1 - bet_mkp) / bet_mkp if bet_wins else -bet
            bankroll = max(bankroll + profit, 0.01)
            placed   = True
            records.append({
                "date": r["date"], "xgb_prob": round(float(r["xgb_prob"]), 4),
                "markov_prob": round(float(r["markov_prob"]), 4),
                "signal": round(signal, 4),
                "market_prob": round(mkp_raw, 4), "edge": round(abs(edge), 4),
                "bet_size": round(bet, 4), "outcome": outcome,
                "profit": round(profit, 4), "bankroll": round(bankroll, 4),
                "surface": r["surface"], "p1_rank": r["p1_rank"],
                "timing": r["timing"], "set_when_bet": int(r["set_when_bet"]),
                "min_edge": min_edge, "kelly_frac": kelly_frac, "vig": vig,
            })

    return pd.DataFrame(records)


def simulate_inplay(feat_df: pd.DataFrame, model_probs: np.ndarray,
                    kelly_frac: float = 0.25, min_edge: float = 0.04,
                    vig: float = 0.08) -> pd.DataFrame:
    """Convenience wrapper: precompute + simulate (used for OOS evaluation)."""
    signals = precompute_signals(feat_df, model_probs)
    return simulate_inplay_from_signals(signals, BANKROLL_INIT, kelly_frac, min_edge, vig)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(bets_df: pd.DataFrame) -> dict:
    zeros = {k: 0.0 for k in ["n_bets","win_rate","roi","sharpe","sortino","calmar",
                                "max_drawdown","max_dd_days","total_profit",
                                "avg_edge","avg_bet","bankroll_final","kelly_growth"]}
    if bets_df.empty:
        return zeros
    n      = len(bets_df)
    wr     = (bets_df["outcome"] == 1).sum() / n
    stakes = bets_df["bet_size"].values
    profs  = bets_df["profit"].values
    roi    = profs.sum() / stakes.sum() * 100 if stakes.sum() > 0 else 0.0
    curve  = bets_df["bankroll"].values
    peak   = np.maximum.accumulate(curve)
    dd     = (curve - peak) / peak
    max_dd = float(dd.min())
    dd_start, max_dd_len = 0, 0
    for i, v in enumerate(dd):
        if v < 0:
            if dd_start == 0 or dd[dd_start] == 0: dd_start = i
            max_dd_len = max(max_dd_len, i - dd_start)
        else:
            dd_start = 0
    rets    = profs / np.maximum(stakes, 1e-9)
    sharpe  = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    down    = rets[rets < 0]
    sortino = float(rets.mean() / down.std()) if len(down) > 1 and down.std() > 0 else 0.0
    calmar  = roi / (abs(max_dd) * 100 + 1e-9)
    kg      = float(np.exp(np.log1p(rets).mean())) - 1.0
    return {"n_bets": n, "win_rate": round(wr*100,2), "roi": round(roi,3),
            "sharpe": round(sharpe,4), "sortino": round(sortino,4), "calmar": round(calmar,4),
            "max_drawdown": round(max_dd*100,3), "max_dd_days": round(max_dd_len/2,1),
            "total_profit": round(float(profs.sum()),4), "avg_edge": round(float(bets_df["edge"].mean()),4),
            "avg_bet": round(float(stakes.mean()),4), "bankroll_final": round(float(curve[-1]),4),
            "kelly_growth": round(kg*100,4)}


def grid_search_pm(train_data: dict, objective: str = "sharpe") -> dict:
    keys, combos = list(PARAM_GRID.keys()), list(itertools.product(*PARAM_GRID.values()))
    best_score, best_params = -np.inf, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        bets   = simulate_prematch(**train_data, **params)
        m      = compute_metrics(bets)
        score  = m.get(objective, 0.0) if m["n_bets"] >= 15 else -99.0
        if score > best_score:
            best_score, best_params = score, params
    best_params["is_score"] = round(best_score, 4)
    return best_params


def grid_search_ip(signals_df: pd.DataFrame, objective: str = "sharpe") -> dict:
    """Fast grid search: Markov already precomputed, just vary min_edge/kelly/vig."""
    keys, combos = list(PARAM_GRID.keys()), list(itertools.product(*PARAM_GRID.values()))
    best_score, best_params = -np.inf, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        bets   = simulate_inplay_from_signals(signals_df, BANKROLL_INIT, **params)
        m      = compute_metrics(bets)
        score  = m.get(objective, 0.0) if m["n_bets"] >= 15 else -99.0
        if score > best_score:
            best_score, best_params = score, params
    best_params["is_score"] = round(best_score, 4)
    return best_params


# ═══════════════════════════════════════════════════════════════════════════════
# 8. XGBoost training
# ═══════════════════════════════════════════════════════════════════════════════

def train_xgb(X_tr, y_tr, X_te):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        verbosity=0, random_state=42, n_jobs=-1,
    )
    model.fit(X_tr, y_tr, verbose=False)
    return model, model.predict_proba(X_te)[:, 1]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. WALK-FORWARD FOLDS
# ═══════════════════════════════════════════════════════════════════════════════

def expanding_folds(years, min_train=WFO_MIN_TRAIN, test_w=1):
    ys = sorted(set(years))
    ys = [y for y in ys if y >= 2000]
    for i in range(min_train, len(ys) - test_w + 1):
        if ys[i] < WFO_TEST_START:
            continue
        yield ys[:i], ys[i:i+test_w], f"EXP_{ys[0]}-{ys[i-1]}→{ys[i]}"


def rolling_folds(years, train_w=6, test_w=1):
    ys = sorted(set(years))
    ys = [y for y in ys if y >= 2000]
    for i in range(train_w, len(ys) - test_w + 1):
        if ys[i] < WFO_TEST_START:
            continue
        yield ys[i-train_w:i], ys[i:i+test_w], f"ROLL_{ys[i-train_w]}-{ys[i-1]}→{ys[i]}"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. MAIN WFO RUNNER
# ═══════════════════════════════════════════════════════════════════════════════

class WFOInPlay:
    def __init__(self, feat_df: pd.DataFrame):
        self.df    = feat_df.reset_index(drop=True)
        self.years = feat_df["year"].values
        self.fold_results_pm  = []   # pre-match fold summaries
        self.fold_results_ip  = []   # in-play fold summaries
        self.all_pm_bets      = []
        self.all_ip_bets      = []
        self.fold_probs       = {}   # label → (y_true, y_prob)

    def run(self, scheme="both"):
        folds = []
        if scheme in ("expanding", "both"):
            folds += list(expanding_folds(self.years))
        if scheme in ("rolling", "both"):
            folds += list(rolling_folds(self.years))

        log.info(f"Running {len(folds)} WFO folds ({scheme}) — pre-match + in-play…")

        for train_ys, test_ys, label in folds:
            tr = np.isin(self.years, train_ys)
            te = np.isin(self.years, test_ys)
            if tr.sum() < 200 or te.sum() < 30:
                continue

            X_tr = self.df.loc[tr, MODEL_FEATURES].fillna(0).values
            y_tr = self.df.loc[tr, "target"].values
            X_te = self.df.loc[te, MODEL_FEATURES].fillna(0).values
            y_te = self.df.loc[te, "target"].values

            log.info(f"  [{label}] XGB training on {tr.sum():,} rows…")
            model, oos_probs = train_xgb(X_tr, y_tr, X_te)
            oos_probs = np.clip(oos_probs, 0.01, 0.99)
            is_probs  = np.clip(model.predict_proba(X_tr)[:, 1], 0.01, 0.99)

            # Feature importance for last fold
            self._last_model   = model
            self._last_feats   = MODEL_FEATURES

            self.fold_probs[label] = (y_te, oos_probs)

            te_df = self.df[te].reset_index(drop=True)
            tr_df = self.df[tr].reset_index(drop=True)

            # ── Pre-match IS grid search ──────────────────────────────
            pm_train_data = dict(
                dates=tr_df["tourney_date"].values,
                model_probs=is_probs, market_probs=tr_df["elo_prob_p1"].values,
                outcomes=y_tr, surfaces=tr_df["surface_norm"].values,
                p1_ranks=tr_df["P1_Rank"].values, bankroll_init=BANKROLL_INIT,
            )
            best_pm = grid_search_pm(pm_train_data)
            eval_pm = {k: best_pm[k] for k in ["min_edge","kelly_frac","vig"]}

            # ── In-play IS grid search (precompute Markov once, sweep params) ──
            # Use tail-2000 subsample for IS grid search — sufficient to rank combos, ~30s vs 8min
            IS_SAMPLE = 2000
            tr_sample = tr_df.tail(IS_SAMPLE).reset_index(drop=True)
            is_probs_sample = is_probs[-IS_SAMPLE:]
            log.info(f"  [{label}] precomputing IS Markov signals (sample={IS_SAMPLE})…")
            is_signals = precompute_signals(tr_sample, is_probs_sample)
            best_ip    = grid_search_ip(is_signals)
            eval_ip    = {k: best_ip[k] for k in ["min_edge","kelly_frac","vig"]}

            # ── OOS evaluation ────────────────────────────────────────
            pm_oos = simulate_prematch(
                dates=te_df["tourney_date"].values, model_probs=oos_probs,
                market_probs=te_df["elo_prob_p1"].values, outcomes=y_te,
                surfaces=te_df["surface_norm"].values, p1_ranks=te_df["P1_Rank"].values,
                bankroll_init=BANKROLL_INIT, **eval_pm)
            pm_oos["fold"]   = label
            pm_oos["scheme"] = "expanding" if label.startswith("EXP") else "rolling"

            log.info(f"  [{label}] precomputing OOS Markov signals…")
            oos_signals = precompute_signals(te_df, oos_probs)
            ip_oos = simulate_inplay_from_signals(oos_signals, BANKROLL_INIT, **eval_ip)
            ip_oos["fold"]   = label
            ip_oos["scheme"] = "expanding" if label.startswith("EXP") else "rolling"

            pm_m = compute_metrics(pm_oos)
            ip_m = compute_metrics(ip_oos)

            oos_acc = accuracy_score(y_te, (oos_probs >= 0.5).astype(int))
            oos_auc = roc_auc_score(y_te, oos_probs) if len(np.unique(y_te)) > 1 else 0.5

            row_pm = {"fold": label, "scheme": pm_oos["scheme"].iloc[0] if not pm_oos.empty else "?",
                      "n_train": int(tr.sum()), "n_test": int(te.sum()),
                      "oos_accuracy": round(oos_acc*100,2), "oos_auc": round(oos_auc,4),
                      **{f"best_{k}": v for k, v in best_pm.items()},
                      **{f"oos_{k}": v for k, v in pm_m.items()}}
            row_ip = {**row_pm,
                      **{f"best_{k}": v for k, v in best_ip.items()},
                      **{f"oos_{k}": v for k, v in ip_m.items()}}

            self.fold_results_pm.append(row_pm)
            self.fold_results_ip.append(row_ip)
            if not pm_oos.empty: self.all_pm_bets.append(pm_oos)
            if not ip_oos.empty: self.all_ip_bets.append(ip_oos)

            log.info(f"  [{label}]  PM: roi={pm_m['roi']:+.1f}% sharpe={pm_m['sharpe']:.3f} n={pm_m['n_bets']}"
                     f"  |  IP: roi={ip_m['roi']:+.1f}% sharpe={ip_m['sharpe']:.3f} n={ip_m['n_bets']}")

        pm_bets = pd.concat(self.all_pm_bets, ignore_index=True) if self.all_pm_bets else pd.DataFrame()
        ip_bets = pd.concat(self.all_ip_bets, ignore_index=True) if self.all_ip_bets else pd.DataFrame()
        return pm_bets, ip_bets

    def fold_dfs(self):
        return pd.DataFrame(self.fold_results_pm), pd.DataFrame(self.fold_results_ip)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. PDF GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

C = ["#2C7BB6","#D7191C","#1A9641","#FDAE61","#7B3294","#4DAC26","#ABD9E9"]

def _savefig(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

def _equity_on_ax(ax, bets_df, label, color, bankroll_init=BANKROLL_INIT):
    if bets_df.empty: return
    df = bets_df.copy()
    df["date2"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date2"]).sort_values("date2")
    curve = bankroll_init + df["profit"].cumsum()
    ax.plot(df["date2"], curve, lw=1.4, color=color, label=label)
    ax.fill_between(df["date2"], bankroll_init, curve,
                    where=curve >= bankroll_init, alpha=0.12, color=color)
    ax.fill_between(df["date2"], bankroll_init, curve,
                    where=curve < bankroll_init, alpha=0.12, color=C[1])

def generate_pdf(feat_df, pm_bets, ip_bets, fold_pm, fold_ip, wfo_obj, pdf_path=PDF_OUT):
    log.info(f"Generating PDF → {pdf_path}")
    sns.set_style("whitegrid")

    # Aggregate model metrics
    all_y, all_p = [], []
    for (yt, yp) in wfo_obj.fold_probs.values():
        all_y.append(yt); all_p.append(yp)
    agg_y = np.concatenate(all_y) if all_y else np.array([])
    agg_p = np.concatenate(all_p) if all_p else np.array([])
    if len(agg_y) > 0:
        agg_acc   = accuracy_score(agg_y, (agg_p >= 0.5).astype(int))
        agg_auc   = roc_auc_score(agg_y, agg_p)
        agg_brier = brier_score_loss(agg_y, agg_p)
    else:
        agg_acc = agg_auc = agg_brier = 0.0

    pm_m = compute_metrics(pm_bets)
    ip_m = compute_metrics(ip_bets)

    with PdfPages(pdf_path) as pdf:

        # ── PAGE 1 · Cover / Executive Summary ───────────────────────────
        fig = plt.figure(figsize=(13, 9))
        fig.patch.set_facecolor("#0D1117")
        ax  = fig.add_axes([0,0,1,1])
        ax.set_axis_off(); ax.set_facecolor("#0D1117")
        ax.text(0.5, 0.97, "Walk-Forward In-Play Backtest Report",
                color="white", fontsize=23, fontweight="bold",
                ha="center", transform=ax.transAxes)
        ax.text(0.5, 0.92,
                f"ATP 1968–2024 · Jeff Sackmann CSVs · Markov Engine + XGBoost | {datetime.now():%Y-%m-%d %H:%M}",
                color="#888", fontsize=10, ha="center", transform=ax.transAxes)

        kpis = [
            ("Dataset matches",      f"{len(feat_df):,}",             "1968–2024"),
            ("WFO folds",            f"{len(fold_pm)}",               "expanding + rolling"),
            ("OOS model accuracy",   f"{agg_acc*100:.2f}%",           "all OOS folds"),
            ("OOS AUC",              f"{agg_auc:.4f}",                "ROC area"),
            ("Brier score",          f"{agg_brier:.4f}",              "lower = better calibration"),
            ("── PRE-MATCH ──",      "──────────",                    "──────────────────────"),
            ("PM bets",              f"{pm_m['n_bets']:,}",            "after edge/vig filter"),
            ("PM win rate",          f"{pm_m['win_rate']:.2f}%",      ""),
            ("PM ROI",               f"{pm_m['roi']:+.2f}%",          "profit ÷ staked"),
            ("PM Sharpe",            f"{pm_m['sharpe']:.4f}",         ""),
            ("PM max drawdown",      f"{pm_m['max_drawdown']:.2f}%",  ""),
            ("PM bankroll final",    f"${pm_m['bankroll_final']:,.2f}",f"start ${BANKROLL_INIT:.0f}"),
            ("── IN-PLAY ──",        "──────────",                    "──────────────────────"),
            ("IP bets",              f"{ip_m['n_bets']:,}",            "Markov-gated bets"),
            ("IP win rate",          f"{ip_m['win_rate']:.2f}%",      ""),
            ("IP ROI",               f"{ip_m['roi']:+.2f}%",          "profit ÷ staked"),
            ("IP Sharpe",            f"{ip_m['sharpe']:.4f}",         ""),
            ("IP Sortino",           f"{ip_m['sortino']:.4f}",        ""),
            ("IP Calmar",            f"{ip_m['calmar']:.4f}",         ""),
            ("IP max drawdown",      f"{ip_m['max_drawdown']:.2f}%",  ""),
            ("IP total profit",      f"${ip_m['total_profit']:+,.2f}",""),
            ("IP bankroll final",    f"${ip_m['bankroll_final']:,.2f}",f"start ${BANKROLL_INIT:.0f}"),
        ]
        tbl = ax.table(cellText=[[k,v,n_] for k,v,n_ in kpis],
                       colLabels=["Metric","Value","Note"],
                       cellLoc="center", loc="center",
                       bbox=[0.03, 0.05, 0.94, 0.82])
        tbl.auto_set_font_size(False); tbl.set_fontsize(9.5)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#333333")
            if r == 0:
                cell.set_facecolor("#1F6FEB"); cell.set_text_props(color="white", fontweight="bold")
            elif kpis[r-1][0].startswith("──"):
                cell.set_facecolor("#2C3E50"); cell.set_text_props(color="#AAAAAA")
            elif r % 2 == 0:
                cell.set_facecolor("#161B22"); cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#0D1117"); cell.set_text_props(color="white")
        _savefig(pdf, fig)

        # ── PAGE 2 · Dual Equity Curve — PM vs IP ────────────────────────
        fig, ax = plt.subplots(figsize=(14, 5))
        _equity_on_ax(ax, pm_bets, "Pre-Match (XGBoost)", C[0])
        _equity_on_ax(ax, ip_bets, "In-Play  (Markov+XGB)", C[2])
        ax.axhline(BANKROLL_INIT, color="grey", linestyle="--", lw=0.8)
        ax.set_title("OOS Equity Curves — Pre-Match vs In-Play", fontsize=13)
        ax.set_ylabel("Bankroll ($)")
        ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
        ax.legend()
        _savefig(pdf, fig)

        # ── PAGE 3 · Drawdown Comparison ─────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Drawdown Analysis — Pre-Match vs In-Play", fontsize=13)
        for ax, bets, label, color in [(axes[0],pm_bets,"Pre-Match",C[0]),
                                        (axes[1],ip_bets,"In-Play",C[2])]:
            if bets.empty: continue
            df = bets.copy()
            df["date2"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
            df = df.dropna(subset=["date2"]).sort_values("date2")
            curve = BANKROLL_INIT + df["profit"].cumsum()
            peak  = curve.cummax()
            dd    = (curve - peak) / peak * 100
            ax.fill_between(df["date2"], dd, 0, alpha=0.5, color=C[1])
            ax.plot(df["date2"], dd, lw=0.8, color=C[1])
            ax.axhline(0, color="black", lw=0.7)
            ax.set_title(f"{label} Drawdown")
            ax.set_ylabel("Drawdown (%)")
        plt.tight_layout()
        _savefig(pdf, fig)

        # ── PAGE 4 · Fold Stability — IS vs OOS Sharpe (both methods) ────
        for fold_df, label_str, color_is, color_oos in [
            (fold_pm, "Pre-Match", C[0], C[1]),
            (fold_ip, "In-Play",   C[2], C[3]),
        ]:
            if fold_df.empty: continue
            fig, axes = plt.subplots(2, 1, figsize=(14, 8))
            fig.suptitle(f"Fold Stability — {label_str} (IS vs OOS Sharpe)", fontsize=13)
            for scheme, ax in [("expanding", axes[0]), ("rolling", axes[1])]:
                sub = fold_df[fold_df["scheme"] == scheme].reset_index(drop=True)
                if sub.empty: continue
                x, w = np.arange(len(sub)), 0.35
                ax.bar(x-w/2, sub.get("is_sharpe",[0]*len(sub)), w,
                       label="IS Sharpe", color=color_is, alpha=0.8)
                ax.bar(x+w/2, sub.get("oos_sharpe",[0]*len(sub)), w,
                       label="OOS Sharpe", color=color_oos, alpha=0.8)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["fold"], rotation=35, ha="right", fontsize=7)
                ax.set_title(f"{scheme.capitalize()} folds")
                ax.set_ylabel("Sharpe")
                ax.legend()
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 6 · Side-by-side fold ROI bars ──────────────────────────
        if not fold_pm.empty and not fold_ip.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("OOS ROI per Fold — Pre-Match vs In-Play", fontsize=13)
            for ax, fold_df, label_str, color in [
                (axes[0], fold_pm, "Pre-Match", C[0]),
                (axes[1], fold_ip, "In-Play",   C[2]),
            ]:
                roi_vals = fold_df.get("oos_roi", pd.Series([0]*len(fold_df)))
                bar_colors = [C[2] if v >= 0 else C[1] for v in roi_vals]
                ax.bar(range(len(fold_df)), roi_vals, color=bar_colors, alpha=0.85)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(range(len(fold_df)))
                ax.set_xticklabels(fold_df["fold"], rotation=35, ha="right", fontsize=6)
                ax.set_title(f"{label_str} — OOS ROI (%)")
                ax.set_ylabel("ROI %")
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 7 · Bet timing breakdown (in-play only) ─────────────────
        if not ip_bets.empty and "timing" in ip_bets.columns:
            fig, axes = plt.subplots(1, 3, figsize=(14, 5))
            fig.suptitle("In-Play Bet Timing Analysis", fontsize=13)

            timing_grp = ip_bets.groupby("timing").agg(
                n=("outcome","count"),
                win_rate=("outcome","mean"),
                roi=("profit", lambda x: x.sum() / ip_bets.loc[x.index,"bet_size"].sum() * 100),
                pnl=("profit","sum"),
            ).reset_index()
            timing_order = ["pre-match","after_set_1","after_set_2","after_set_3","after_set_4"]
            timing_grp = timing_grp.set_index("timing").reindex(
                [t for t in timing_order if t in timing_grp["timing"].values]).reset_index()

            ax = axes[0]
            ax.bar(timing_grp["timing"], timing_grp["n"], color=C[0], alpha=0.85)
            ax.set_title("# Bets by Timing")
            ax.set_ylabel("Count")
            ax.tick_params(axis="x", rotation=30)

            ax = axes[1]
            colors_wr = [C[2] if v >= 0.5 else C[1] for v in timing_grp["win_rate"]]
            ax.bar(timing_grp["timing"], timing_grp["win_rate"]*100, color=colors_wr, alpha=0.85)
            ax.axhline(50, color="black", linestyle="--", lw=0.8)
            ax.set_title("Win Rate by Timing")
            ax.set_ylabel("Win Rate (%)")
            ax.tick_params(axis="x", rotation=30)

            ax = axes[2]
            colors_roi = [C[2] if v >= 0 else C[1] for v in timing_grp["roi"]]
            ax.bar(timing_grp["timing"], timing_grp["roi"], color=colors_roi, alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_title("ROI (%) by Timing")
            ax.set_ylabel("ROI %")
            ax.tick_params(axis="x", rotation=30)

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 8 · Markov prob vs XGBoost prob scatter ──────────────────
        if not ip_bets.empty and "markov_prob" in ip_bets.columns:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Markov vs XGBoost Signal Analysis", fontsize=13)

            ax = axes[0]
            scatter = ax.scatter(ip_bets["xgb_prob"], ip_bets["markov_prob"],
                                  c=ip_bets["outcome"], cmap="RdYlGn",
                                  alpha=0.25, s=10, vmin=0, vmax=1)
            ax.plot([0,1],[0,1],"--", color="grey", lw=0.8, label="Equal line")
            ax.set_xlabel("XGBoost Pre-Match Prob")
            ax.set_ylabel("Markov In-Play Prob")
            ax.set_title("XGB vs Markov Probability (green=P1 won)")
            ax.legend()
            plt.colorbar(scatter, ax=ax, label="P1 Outcome")

            ax = axes[1]
            ax.hist(ip_bets["markov_prob"] - ip_bets["xgb_prob"],
                    bins=35, color=C[4], edgecolor="white")
            ax.axvline(0, color="black", lw=0.8)
            ax.set_title("Markov − XGBoost Probability Difference")
            ax.set_xlabel("Prob Difference (positive = Markov more bullish on P1)")
            ax.set_ylabel("Count")

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 9 · Monthly P&L heatmaps (PM + IP) ──────────────────────
        for bets, label_str in [(pm_bets,"Pre-Match"), (ip_bets,"In-Play")]:
            if bets.empty: continue
            fig, ax = plt.subplots(figsize=(14, 5))
            df_mo = bets.copy()
            df_mo["date2"] = pd.to_datetime(df_mo["date"].astype(str), format="%Y%m%d", errors="coerce")
            df_mo = df_mo.dropna(subset=["date2"])
            df_mo["yr"] = df_mo["date2"].dt.year
            df_mo["mo"] = df_mo["date2"].dt.month
            pivot = df_mo.groupby(["yr","mo"])["profit"].sum().unstack("mo")
            pivot.columns = [f"M{c:02d}" for c in pivot.columns]
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".0f",
                        center=0, linewidths=0.3)
            ax.set_title(f"Monthly P&L ($) — {label_str}")
            ax.set_xlabel("Month"); ax.set_ylabel("Year")
            _savefig(pdf, fig)

        # ── PAGE 11 · Calibration + ROC ──────────────────────────────────
        if len(agg_y) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Model Calibration & Discrimination (All OOS Folds)", fontsize=13)
            ax = axes[0]
            pt, pp = calibration_curve(agg_y, agg_p, n_bins=12)
            ax.plot(pp, pt, "o-", color=C[0], lw=2, label="Model")
            ax.plot([0,1],[0,1],"--", color="grey", label="Perfect")
            ax.fill_between(pp, pt, pp, alpha=0.15, color=C[1])
            ax.set_title("Reliability Diagram")
            ax.set_xlabel("Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.text(0.05, 0.92, f"Brier={agg_brier:.4f}", transform=ax.transAxes)
            ax.legend()
            ax = axes[1]
            fpr, tpr, _ = roc_curve(agg_y, agg_p)
            ax.plot(fpr, tpr, color=C[0], lw=2, label=f"AUC={agg_auc:.4f}")
            ax.plot([0,1],[0,1],"--", color="grey")
            ax.fill_between(fpr, tpr, alpha=0.12, color=C[0])
            ax.set_title("ROC Curve"); ax.legend(loc="lower right")
            _savefig(pdf, fig)

        # ── PAGE 12 · Confidence bin → win rate heatmap ──────────────────
        if len(agg_y) > 0:
            fig, ax = plt.subplots(figsize=(11, 3))
            bins   = np.arange(0.45, 1.01, 0.05)
            labels = [f"{b:.2f}" for b in bins[:-1]]
            cats   = pd.cut(agg_p, bins=bins, labels=labels)
            cal_df = pd.DataFrame({"bin": cats, "outcome": agg_y}).dropna(subset=["bin"])
            wr_df  = cal_df.groupby("bin", observed=True)["outcome"].agg(["mean","count"]).reset_index()
            pivot  = wr_df.set_index("bin")[["mean"]].T
            annot  = [[f"{v:.2f}\n(n={n_})" for v,n_ in zip(wr_df["mean"],wr_df["count"])]]
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=annot, fmt="",
                        vmin=0.3, vmax=0.9, linewidths=0.4)
            ax.set_title("Confidence Bin → Actual Win Rate")
            ax.set_xlabel("Model Confidence"); ax.set_yticks([])
            _savefig(pdf, fig)

        # ── PAGE 13 · Parameter heatmaps (Sharpe surface — IP) ───────────
        if not ip_bets.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("In-Play Parameter Sensitivity (OOS data)", fontsize=13)
            sub_d = dict(dates=ip_bets["date"].values,
                         model_probs=ip_bets.get("signal", ip_bets["xgb_prob"]).values,
                         market_probs=ip_bets["market_prob"].values,
                         outcomes=ip_bets["outcome"].values,
                         surfaces=ip_bets["surface"].values,
                         p1_ranks=ip_bets["p1_rank"].values,
                         bankroll_init=BANKROLL_INIT)
            for metric, ax, title in [("sharpe",axes[0],"Sharpe Surface"),
                                       ("roi",   axes[1],"ROI Surface")]:
                mes = PARAM_GRID["min_edge"]
                kfs = PARAM_GRID["kelly_frac"]
                mat = np.full((len(mes),len(kfs)), np.nan)
                for i,me in enumerate(mes):
                    for j,kf in enumerate(kfs):
                        b = simulate_prematch(**sub_d, min_edge=me, kelly_frac=kf, vig=0.07)
                        m_ = compute_metrics(b)
                        if m_["n_bets"] >= 15:
                            mat[i,j] = m_.get(metric,0.0)
                sns.heatmap(pd.DataFrame(mat,index=mes,columns=kfs),
                            ax=ax, cmap="RdYlGn", annot=True, fmt=".3f", linewidths=0.4)
                ax.set_xlabel("Kelly Fraction"); ax.set_ylabel("Min Edge")
                ax.set_title(title)
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 14 · Edge × Surface ROI (both strategies) ───────────────
        for bets, label_str in [(pm_bets,"Pre-Match"),(ip_bets,"In-Play")]:
            if bets.empty: continue
            fig, ax = plt.subplots(figsize=(9,5))
            df_es = bets.copy()
            df_es["edge_bin"] = pd.cut(df_es["edge"],
                                        bins=[0,.04,.06,.08,.10,.99],
                                        labels=["<4%","4-6%","6-8%","8-10%",">10%"])
            df_es["roi_bet"] = df_es["profit"] / df_es["bet_size"].clip(1e-6) * 100
            pivot = df_es.groupby(["edge_bin","surface"], observed=True)["roi_bet"] \
                         .mean().unstack("surface").fillna(0)
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".1f",
                        center=0, linewidths=0.4)
            ax.set_title(f"ROI (%) — Edge Bin × Surface  [{label_str}]")
            _savefig(pdf, fig)

        # ── PAGE 16 · Win rate by surface (both) ─────────────────────────
        if not pm_bets.empty and not ip_bets.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Win Rate & ROI by Surface", fontsize=13)
            for ax, bets, label_str in [(axes[0],pm_bets,"Pre-Match"),
                                         (axes[1],ip_bets,"In-Play")]:
                if bets.empty: continue
                sg = bets.groupby("surface").agg(
                    wr=("outcome","mean"),
                    n=("outcome","count"),
                    roi=("profit", lambda x: x.sum()/bets.loc[x.index,"bet_size"].sum()*100),
                ).reset_index()
                bars = ax.bar(sg["surface"], sg["wr"]*100,
                              color=[C[0],C[2],C[3]][:len(sg)], alpha=0.85)
                for bar, row in zip(bars, sg.itertuples()):
                    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                            f"{bar.get_height():.1f}%\n(n={row.n})\nROI:{row.roi:.1f}%",
                            ha="center", fontsize=9)
                ax.set_ylim(0, 85)
                ax.set_title(f"{label_str} — Win Rate by Surface")
                ax.set_ylabel("Win Rate (%)")
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 17 · Feature importance ─────────────────────────────────
        if hasattr(wfo_obj, "_last_model"):
            fig, ax = plt.subplots(figsize=(11,7))
            imp = wfo_obj._last_model.feature_importances_
            imp_s = pd.Series(imp, index=MODEL_FEATURES).sort_values()
            imp_s.plot(kind="barh", ax=ax, color=C[0], alpha=0.85)
            ax.set_title("XGBoost Feature Importance (last fold — gain)")
            ax.set_xlabel("Importance")
            _savefig(pdf, fig)

        # ── PAGE 18 · Feature correlation ────────────────────────────────
        corr_cols = [c for c in MODEL_FEATURES + ["target"] if c in feat_df.columns]
        fig, ax   = plt.subplots(figsize=(13,11))
        corr = feat_df[corr_cols].corr(method="spearman")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, ax=ax, mask=mask, cmap="coolwarm",
                    annot=True, fmt=".2f", linewidths=0.2,
                    vmin=-1, vmax=1, square=True, cbar_kws={"shrink":0.6},
                    annot_kws={"size":6})
        ax.set_title("Feature Correlation Matrix (Spearman)")
        plt.tight_layout()
        _savefig(pdf, fig)

        # ── PAGE 19 · Accuracy by year × surface ─────────────────────────
        pred_rows = []
        for label, (yt, yp) in wfo_obj.fold_probs.items():
            year = int(label.split("→")[1])
            sub  = feat_df[feat_df["year"] == year].head(len(yt)).copy()
            sub  = sub.reset_index(drop=True)
            sub  = sub.iloc[:len(yt)]
            sub["y_true"]  = yt
            sub["y_prob"]  = yp
            sub["y_pred"]  = (yp >= 0.5).astype(int)
            sub["correct"] = (sub["y_pred"] == sub["y_true"]).astype(int)
            pred_rows.append(sub)

        if pred_rows:
            pred_df = pd.concat(pred_rows, ignore_index=True)
            pivot   = pred_df.groupby(["year","surface_norm"])["correct"] \
                             .mean().unstack("surface_norm")
            pivot   = pivot[[c for c in ["Hard","Clay","Grass"] if c in pivot.columns]]
            fig, ax = plt.subplots(figsize=(10,8))
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                        vmin=0.45, vmax=0.72, linewidths=0.4)
            ax.set_title("Model Accuracy — Year × Surface (OOS)")
            _savefig(pdf, fig)

        # ── PAGE 20 · Confusion matrix + Rank differential accuracy ───────
        if pred_rows and len(pred_df) > 0:
            fig, axes = plt.subplots(1,2,figsize=(13,5))
            ax = axes[0]
            cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["P1 Loses","P1 Wins"])
            ax.set_yticklabels(["P1 Loses","P1 Wins"])
            for i in range(2):
                for j in range(2):
                    ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center",
                            color="white" if cm[i,j]>cm.max()/2 else "black", fontsize=14)
            ax.set_title("Confusion Matrix"); plt.colorbar(im, ax=ax)

            ax = axes[1]
            rd = pred_df["P1_Rank"] - pred_df["P2_Rank"]
            pred_df["rank_bucket"] = pd.cut(rd,
                bins=[-np.inf,-100,-20,0,20,100,np.inf],
                labels=["P1 much better","P1 better","P1 slight",
                        "P2 slight","P2 better","P2 much better"])
            ba = pred_df.groupby("rank_bucket",observed=True)["correct"].mean()*100
            bn = pred_df.groupby("rank_bucket",observed=True)["correct"].count()
            ba.plot(kind="bar", ax=ax, color=C[0], alpha=0.85, rot=30)
            for i,(a,n_) in enumerate(zip(ba,bn)):
                ax.text(i, a+0.5, f"{a:.0f}%\nn={n_}", ha="center", fontsize=8)
            ax.set_ylim(40,90); ax.set_title("Accuracy by Rank Differential")
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 21 · Fold KPI table (In-Play) ───────────────────────────
        for fold_df, label_str in [(fold_ip,"In-Play"),(fold_pm,"Pre-Match")]:
            if fold_df.empty: continue
            show = ["fold","n_test","oos_accuracy","oos_auc",
                    "oos_n_bets","oos_win_rate","oos_roi","oos_sharpe",
                    "oos_max_drawdown","best_min_edge","best_kelly_frac","best_vig"]
            show = [c for c in show if c in fold_df.columns]
            chunk_size = 22
            for cs in range(0, len(fold_df), chunk_size):
                chunk = fold_df[show].iloc[cs:cs+chunk_size]
                fig, ax = plt.subplots(figsize=(16, max(3, len(chunk)*0.55+2)))
                ax.axis("off")
                tbl = ax.table(cellText=chunk.round(3).astype(str).values.tolist(),
                               colLabels=[c.replace("_"," ").replace("oos ","OOS ").replace("best ","").title()
                                          for c in show],
                               loc="center", cellLoc="center")
                tbl.auto_set_font_size(False); tbl.set_fontsize(7)
                tbl.scale(1.2, 1.6)
                ax.set_title(f"Fold Summary — {label_str}", fontsize=12, pad=12)
                _savefig(pdf, fig)

        # ── PAGE 23 · Methodology ─────────────────────────────────────────
        fig = plt.figure(figsize=(13,9))
        ax  = fig.add_axes([0.04,0.02,0.92,0.95]); ax.set_axis_off()
        text = (
            "Methodology — In-Play Walk-Forward Backtest\n"
            "═══════════════════════════════════════════════════════════════════\n\n"
            "DATA\n"
            f"  · Jeff Sackmann ATP match CSVs, 1968–2024  ({len(feat_df):,} matches)\n"
            "  · Serve stats (w_1stIn, w_1stWon, w_2ndWon, l_svpt …) used for Markov\n"
            "  · Score strings parsed into per-set (p1_games, p2_games) pairs\n"
            "  · ELO ratings updated chronologically — no lookahead\n\n"
            "WALK-FORWARD SCHEME\n"
            f"  · Test period begins {WFO_TEST_START} (no OOS results before this year)\n"
            "  · Expanding window: train all years ≤ T, test year T+1\n"
            "  · Rolling window  : train most recent 6 years, test next year\n"
            "  · Parameter grid searched IN-SAMPLE (Sharpe objective) per fold\n\n"
            "PRE-MATCH MODEL (XGBoost)\n"
            "  · 21 features: surface dummies, best-of, handedness, height,\n"
            "    age, rank, rank-points, ELO diff and derived differentials\n"
            "  · Trained fresh on each fold's training window\n"
            "  · n_estimators=300, max_depth=4, lr=0.05, subsample=0.8\n\n"
            "IN-PLAY MARKOV ENGINE\n"
            "  · Serve prob derivation from match serve stats:\n"
            "    p_serve  = (1stWon + 2ndWon) / svpt  (clamped 0.40–0.90)\n"
            "    p_return = 1 − opponent p_serve       (clamped 0.10–0.60)\n"
            "  · LiveMatchState updates after each set is fed in\n"
            "  · Signal blend (weight increases with sets played):\n"
            "    pre-match: 50% XGB + 50% Markov\n"
            "    after S1:  30% XGB + 70% Markov\n"
            "    after S2+: 10% XGB + 90% Markov\n"
            "  · Bet placed on FIRST checkpoint where |signal − market| > min_edge\n"
            "  · Only ONE bet per match (prevents multiple exposure)\n\n"
            "BETTING MECHANICS\n"
            "  · Market = ELO-implied probability (pre-match; not adjusted in-play)\n"
            "  · Vig applied: mkp_adj = mkp_raw + vig/2  (capped at 0.95)\n"
            "  · Confidence-tiered fractional Kelly:\n"
            "    signal ≥ 0.70 → tier 0.40  |  ≥ 0.60 → 0.25\n"
            "    ≥ 0.55 → 0.12              |  < 0.55 → 0.05\n"
            "  · Position cap: 20% of bankroll per bet\n"
            "  · Starting bankroll: $500\n\n"
            "WHY IN-PLAY IMPROVES ACCURACY\n"
            "  · After set 1 you know whether the pre-match favourite actually won it\n"
            "  · Markov recalculates the true remaining match win probability given\n"
            "    the real score — e.g. being up a set from 40% pre-match → 65% live\n"
            "  · The blend allows the model to CONFIRM or CONTRADICT the pre-match\n"
            "    signal, filtering out matches where XGBoost was overconfident\n"
        )
        ax.text(0.0, 1.0, text, va="top", ha="left",
                fontsize=9, fontfamily="monospace",
                transform=ax.transAxes, color="#111111")
        _savefig(pdf, fig)

    log.info(f"PDF saved → {pdf_path}  ({os.path.getsize(pdf_path)//1024}KB)")


# ═══════════════════════════════════════════════════════════════════════════════
# 12. ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    log.info("Loading ATP match data…")
    raw = load_atp_matches()

    log.info("Building features + ELO (may take ~60s)…")
    feat_df = build_features(raw)
    log.info(f"Feature rows: {len(feat_df):,}  years: {feat_df['year'].min()}–{feat_df['year'].max()}")

    wfo = WFOInPlay(feat_df)
    pm_bets, ip_bets = wfo.run(scheme="both")
    fold_pm, fold_ip = wfo.fold_dfs()

    # Save bets CSV
    if not ip_bets.empty:
        ip_bets.to_csv(CSV_OUT, index=False)
        log.info(f"In-play bets CSV: {CSV_OUT}  ({len(ip_bets):,} rows)")

    # Save params JSON
    param_cols = ["best_min_edge","best_kelly_frac","best_vig"]
    if not fold_ip.empty:
        consensus = {}
        for col in param_cols:
            if col in fold_ip.columns:
                consensus[col.replace("best_","")] = float(fold_ip[col].mode().iloc[0])
        with open(JSON_OUT, "w") as f:
            json.dump({"generated": datetime.now().isoformat(),
                       "consensus_params": consensus,
                       "fold_params": fold_ip[["fold"]+[c for c in param_cols if c in fold_ip.columns]]
                                      .to_dict(orient="records")}, f, indent=2)
        log.info(f"Consensus in-play params: {consensus}")

    generate_pdf(feat_df, pm_bets, ip_bets, fold_pm, fold_ip, wfo, PDF_OUT)

    # Console summary
    pm_m = compute_metrics(pm_bets)
    ip_m = compute_metrics(ip_bets)
    print("\n" + "="*70)
    print("PRE-MATCH (XGBoost only)")
    print(f"  bets={pm_m['n_bets']:,}  win={pm_m['win_rate']:.1f}%  "
          f"roi={pm_m['roi']:+.2f}%  sharpe={pm_m['sharpe']:.4f}  "
          f"maxDD={pm_m['max_drawdown']:.1f}%  bankroll=${pm_m['bankroll_final']:,.2f}")
    print("IN-PLAY (Markov + XGBoost blend)")
    print(f"  bets={ip_m['n_bets']:,}  win={ip_m['win_rate']:.1f}%  "
          f"roi={ip_m['roi']:+.2f}%  sharpe={ip_m['sharpe']:.4f}  "
          f"maxDD={ip_m['max_drawdown']:.1f}%  bankroll=${ip_m['bankroll_final']:,.2f}")
    if not fold_ip.empty and "oos_sharpe" in fold_ip.columns:
        top = fold_ip.nlargest(5,"oos_sharpe")[
            ["fold","oos_sharpe","oos_roi","oos_n_bets","best_min_edge","best_kelly_frac"]]
        print("\nTOP 5 IN-PLAY FOLDS BY OOS SHARPE")
        print(top.to_string(index=False))
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
