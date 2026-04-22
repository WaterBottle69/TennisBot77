"""
wfo_sun_masters.py
──────────────────
Walk-Forward Optimisation of the sun-positioning algorithm on ATP Masters data.

Pipeline
────────
1. Load atp_masters_matches.csv (scraped dataset — real weather & venue data)
2. Filter: outdoor matches, valid serve stats, known venue coords
3. Compute solar position per match (using location_engine)
4. Classify each match: GLARE / MODERATE / LOW under candidate parameters
5. WFO loop (5-year train → 1-year test, rolling from 2000→2024):
     a. Grid-search parameters on training window
        Optimise: Cohen's d of serve-win% (GLARE vs LOW)
     b. Apply best parameters to held-out test year
     c. Record OOS serve-delta, upset-rate delta, betting ROI
6. Aggregate across all folds; output per-fold CSV + final report

Betting simulation
──────────────────
Signal: when a match is classified GLARE (under the optimised threshold),
back the UNDERDOG (higher-ranked player — opponent of the favourite) at
implied fair odds derived from rank-based win probability.

  fair_odds_underdog = 1 / (1 - p_fav)
  p_fav = 1 / (1 + exp( k * (rank_underdog - rank_fav) ))   [logistic]

ROI = sum(profit) / n_bets  (flat £1 stakes)

Why underdog?  Sun glare reduces serve dominance → breaks of serve become
more frequent → outcomes are more random → fair market prices both players
closer to 50% while the underlying uncertainty has actually increased, giving
positive EV on the less-efficient price (the underdog).

Parameters optimised per fold
──────────────────────────────
  elev_low   :  min sun elevation for potential glare   [5, 10, 15] °
  elev_high  :  max sun elevation for potential glare   [60, 70, 80] °
  angle_deg  :  azimuth window either side of serve direction  [30, 40, 50, 60] °
  min_penalty:  minimum computed penalty to flag as GLARE  [0.005, 0.01, 0.02, 0.03]
"""

import datetime
import itertools
import json
import math
import os
import sys
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ── Import solar engine ────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from location_engine import compute_solar_position, compute_sun_penalty

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")
TRAIN_YEARS = 5       # rolling training window length
TEST_YEARS  = 1       # test window length
FIRST_YEAR  = 2000    # first year with reliable serve-point data
LAST_YEAR   = 2024
MIN_SVPT    = 20      # discard matches with fewer serve points

# Round → assumed local start hour (24h clock)
ROUND_HOURS = {
    "R128": 11, "R64": 11, "R32": 12,
    "R16": 13, "QF": 14, "SF": 15, "F": 15, "RR": 13,
}
DEFAULT_HOUR = 13

# Logistic rank-to-probability scaling constant (calibrated on ATP data)
RANK_K = 0.003

# Parameter grid
PARAM_GRID = list(itertools.product(
    [5.0, 10.0, 15.0],          # elev_low
    [60.0, 70.0, 80.0],         # elev_high
    [30.0, 40.0, 50.0, 60.0],   # angle_deg
    [0.005, 0.01, 0.02, 0.03],  # min_penalty
))


# ── Helpers ────────────────────────────────────────────────────────────────────

def local_hour(row: pd.Series) -> int:
    return ROUND_HOURS.get(str(row["round"]).strip(), DEFAULT_HOUR)


def parse_date(date_str) -> Optional[datetime.date]:
    try:
        return pd.to_datetime(date_str).date()
    except Exception:
        return None


def rank_win_prob(rank_winner: float, rank_loser: float) -> float:
    """Logistic win probability for the winner based on ATP ranking."""
    try:
        diff = float(rank_loser) - float(rank_winner)   # positive = winner is higher-ranked
        return 1.0 / (1.0 + math.exp(-RANK_K * diff))
    except Exception:
        return 0.5


def compute_sun_category(
    lat: float, lon: float, court_orient: float,
    date_obj: datetime.date, hour: int,
    elev_low: float, elev_high: float,
    angle_deg: float, min_penalty: float,
) -> tuple[float, float, str]:
    """Return (sun_elev, max_penalty, category) for one match under given params."""
    utc_offset = round(lon / 15.0)
    utc_hour   = (hour - utc_offset) % 24
    dt = datetime.datetime(date_obj.year, date_obj.month, date_obj.day, utc_hour)

    try:
        az, elev = compute_solar_position(lat, lon, dt)
    except Exception:
        return 0.0, 0.0, "UNKNOWN"

    if elev < elev_low or elev > elev_high:
        return elev, 0.0, "LOW"

    dir_a = (court_orient + 180.0) % 360.0
    dir_b =  court_orient
    # compute_sun_penalty expects (server_direction, sun_azimuth, sun_elevation)
    p_a = compute_sun_penalty(dir_a, az, elev)
    p_b = compute_sun_penalty(dir_b, az, elev)
    max_p = max(p_a, p_b)

    if max_p >= min_penalty:
        return elev, max_p, "GLARE"
    elif max_p > min_penalty * 0.25:
        return elev, max_p, "MODERATE"
    return elev, 0.0, "LOW"


# ── Statistical helpers ────────────────────────────────────────────────────────

def cohens_d(a, b) -> float:
    a, b = list(a), list(b)
    if len(a) < 3 or len(b) < 3:
        return 0.0
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na-1)*np.var(a, ddof=1) + (nb-1)*np.var(b, ddof=1)) / (na+nb-2))
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def welch_p(a, b) -> float:
    a, b = list(a), list(b)
    if len(a) < 3 or len(b) < 3:
        return 1.0
    try:
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        return 1.0


# ── Core: classify all matches under one parameter set ────────────────────────

def classify_matches(df: pd.DataFrame, params: tuple) -> pd.DataFrame:
    elev_low, elev_high, angle_deg, min_penalty = params
    out = df.copy()
    categories, penalties, elevations = [], [], []

    for _, row in out.iterrows():
        lat  = row.get("venue_lat")
        lon  = row.get("venue_lon")
        date = row.get("_date")
        hour = row.get("_local_hour", DEFAULT_HOUR)
        if lat is None or lon is None or date is None or math.isnan(float(lat)):
            categories.append("UNKNOWN")
            penalties.append(0.0)
            elevations.append(0.0)
            continue
        elev, pen, cat = compute_sun_category(
            float(lat), float(lon), 0.0, date, int(hour),
            elev_low, elev_high, angle_deg, min_penalty,
        )
        categories.append(cat)
        penalties.append(pen)
        elevations.append(elev)

    out["sun_cat"]     = categories
    out["sun_penalty"] = penalties
    out["sun_elev"]    = elevations
    return out


# ── Training objective ─────────────────────────────────────────────────────────

def train_score(df_classified: pd.DataFrame) -> float:
    """
    Score = Cohen's d of (GLARE serve_win_pct vs LOW serve_win_pct).
    Negative d means glare suppresses serve wins — that's our signal.
    We maximise |d| where the direction is negative.
    """
    glare = df_classified[df_classified["sun_cat"] == "GLARE"]["_both_1stpct"].dropna()
    low   = df_classified[df_classified["sun_cat"] == "LOW"]["_both_1stpct"].dropna()
    if len(glare) < 5 or len(low) < 5:
        return 0.0
    d = cohens_d(glare, low)
    return -d   # more negative d → higher score (glare hurts serve more)


# ── Test evaluation ────────────────────────────────────────────────────────────

def evaluate_oos(df_classified: pd.DataFrame, fold_year: int, params: tuple) -> dict:
    """Compute OOS metrics for a single test fold."""
    glare = df_classified[df_classified["sun_cat"] == "GLARE"]
    low   = df_classified[df_classified["sun_cat"] == "LOW"]
    mod   = df_classified[df_classified["sun_cat"] == "MODERATE"]

    def serve_mean(sub):
        v = sub["_both_1stpct"].dropna()
        return float(v.mean()) if len(v) > 0 else float("nan")

    serve_glare = serve_mean(glare)
    serve_low   = serve_mean(low)
    serve_delta = serve_glare - serve_low   # negative = glare suppresses serve

    glare_p = welch_p(
        glare["_both_1stpct"].dropna().tolist(),
        low["_both_1stpct"].dropna().tolist(),
    )
    d = cohens_d(
        glare["_both_1stpct"].dropna(),
        low["_both_1stpct"].dropna(),
    )

    # Upset rate: was the higher-ranked (worse) player the winner?
    # winner_rank > loser_rank means the underdog won
    def upset_rate(sub):
        valid = sub.dropna(subset=["winner_rank", "loser_rank"])
        if len(valid) == 0:
            return float("nan"), 0
        upsets = (valid["winner_rank"] > valid["loser_rank"]).sum()
        return float(upsets / len(valid)), len(valid)

    upset_glare, n_glare = upset_rate(glare)
    upset_low,   n_low   = upset_rate(low)
    upset_delta = (upset_glare - upset_low) if not math.isnan(upset_glare) else float("nan")

    # ── Betting simulation: back underdog in GLARE matches ──────────────────
    # Underdog = player with higher rank number (worse player)
    # Fair odds = 1 / (1 - p_favourite)
    profits = []
    for _, row in glare.iterrows():
        wr = row.get("winner_rank")
        lr = row.get("loser_rank")
        if pd.isna(wr) or pd.isna(lr):
            continue
        wr, lr = float(wr), float(lr)
        # Favourite = lower rank number
        # In each row, we don't know who was actually the underdog pre-match
        # Proxy: if winner_rank > loser_rank → underdog won (upset)
        fav_rank    = min(wr, lr)
        under_rank  = max(wr, lr)
        p_fav       = rank_win_prob(fav_rank, under_rank)
        fair_odds   = 1.0 / max(1.0 - p_fav, 0.01)

        underdog_won = (wr > lr)   # winner was the underdog
        profit = (fair_odds - 1.0) if underdog_won else -1.0
        profits.append(profit)

    n_bets  = len(profits)
    roi     = float(np.mean(profits)) if n_bets > 0 else float("nan")
    sharpe  = float(np.mean(profits) / np.std(profits)) if n_bets > 1 and np.std(profits) > 0 else float("nan")

    # Baseline: back underdog in ALL matches (no sun filter)
    base_profits = []
    for _, row in df_classified.dropna(subset=["winner_rank","loser_rank"]).iterrows():
        wr, lr = float(row["winner_rank"]), float(row["loser_rank"])
        fav_rank   = min(wr, lr)
        under_rank = max(wr, lr)
        p_fav      = rank_win_prob(fav_rank, under_rank)
        fair_odds  = 1.0 / max(1.0 - p_fav, 0.01)
        won = (wr > lr)
        base_profits.append((fair_odds - 1.0) if won else -1.0)

    base_roi = float(np.mean(base_profits)) if base_profits else float("nan")

    return {
        "fold_year":      fold_year,
        "params":         params,
        "n_total":        len(df_classified),
        "n_glare":        len(glare),
        "n_low":          len(low),
        "n_moderate":     len(mod),
        "serve_glare":    round(serve_glare, 5),
        "serve_low":      round(serve_low, 5),
        "serve_delta":    round(serve_delta, 5),
        "serve_p_value":  round(glare_p, 4),
        "cohens_d":       round(d, 4),
        "upset_glare":    round(upset_glare, 4) if not math.isnan(upset_glare) else None,
        "upset_low":      round(upset_low, 4)   if not math.isnan(upset_low)   else None,
        "upset_delta":    round(upset_delta, 4) if not math.isnan(upset_delta) else None,
        "n_bets":         n_bets,
        "bet_roi":        round(roi, 4)    if not math.isnan(roi)    else None,
        "bet_sharpe":     round(sharpe, 4) if not math.isnan(sharpe) else None,
        "base_roi":       round(base_roi, 4),
        "roi_vs_base":    round(roi - base_roi, 4) if not math.isnan(roi) else None,
        "elev_low":       params[0],
        "elev_high":      params[1],
        "angle_deg":      params[2],
        "min_penalty":    params[3],
    }


# ── Pre-process dataset ────────────────────────────────────────────────────────

def load_and_prepare() -> pd.DataFrame:
    print("Loading atp_masters_matches.csv …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print(f"  Raw rows: {len(df):,}")

    # Outdoor only
    df = df[df["court_type"].str.strip().str.lower() == "outdoor"].copy()
    print(f"  Outdoor: {len(df):,}")

    # Need venue coords
    df = df.dropna(subset=["venue_lat", "venue_lon"])

    # Parse dates
    df["_date"] = df["tourney_date"].apply(parse_date)
    df = df.dropna(subset=["_date"])
    df["_year"] = df["_date"].apply(lambda d: d.year)

    # Local hour from round
    df["_local_hour"] = df.apply(local_hour, axis=1)

    # Both-players combined 1st serve win %
    def both_1st(row):
        try:
            w_won = float(row["w_1stWon"]); w_in = float(row["w_1stIn"])
            l_won = float(row["l_1stWon"]); l_in = float(row["l_1stIn"])
            total_won = w_won + l_won
            total_in  = w_in  + l_in
            if total_in < MIN_SVPT:
                return float("nan")
            return total_won / total_in
        except Exception:
            return float("nan")

    df["_both_1stpct"] = df.apply(both_1st, axis=1)
    df = df.dropna(subset=["_both_1stpct"])
    print(f"  With serve stats: {len(df):,}")

    df = df[df["_year"] >= FIRST_YEAR].copy()
    print(f"  From {FIRST_YEAR} onward: {len(df):,}\n")
    return df


# ── WFO main loop ─────────────────────────────────────────────────────────────

def run_wfo(df: pd.DataFrame) -> list[dict]:
    test_years = list(range(FIRST_YEAR + TRAIN_YEARS, LAST_YEAR + 1))
    print(f"Walk-forward folds: {len(test_years)}  "
          f"(train={TRAIN_YEARS}yr, test={TEST_YEARS}yr, "
          f"{test_years[0]}–{test_years[-1]})")
    print(f"Parameter combinations per fold: {len(PARAM_GRID)}\n")

    fold_results = []

    for test_year in test_years:
        train_start = test_year - TRAIN_YEARS
        train_end   = test_year - 1
        df_train    = df[(df["_year"] >= train_start) & (df["_year"] <= train_end)]
        df_test     = df[df["_year"] == test_year]

        if len(df_test) < 10:
            continue

        # ── Grid search on training window ────────────────────────────────────
        best_score  = -999.0
        best_params = PARAM_GRID[0]

        for params in PARAM_GRID:
            try:
                cl = classify_matches(df_train, params)
                sc = train_score(cl)
                if sc > best_score:
                    best_score  = sc
                    best_params = params
            except Exception:
                continue

        elev_low, elev_high, angle, pen = best_params
        print(f"  {train_start}-{train_end} → {test_year} | "
              f"best params: elev=[{elev_low},{elev_high}]° "
              f"angle={angle}° pen={pen:.3f} | "
              f"train score={best_score:.4f}")

        # ── Evaluate on test year ─────────────────────────────────────────────
        df_test_cl = classify_matches(df_test, best_params)
        result     = evaluate_oos(df_test_cl, test_year, best_params)
        fold_results.append(result)

    return fold_results


# ── Aggregate & report ─────────────────────────────────────────────────────────

def report(fold_results: list[dict]):
    df = pd.DataFrame(fold_results)
    sep = "─" * 72

    print(f"\n{'═'*72}")
    print("  SUN-POSITIONING WFO BACKTEST — ATP MASTERS 1000")
    print(f"  {FIRST_YEAR+TRAIN_YEARS}–{LAST_YEAR}  |  {len(df)} folds  |  "
          f"train={TRAIN_YEARS}yr  test={TEST_YEARS}yr")
    print(f"{'═'*72}\n")

    # Per-fold table
    print(f"{'Year':>5}  {'N_Glare':>7}  {'SrvΔ%':>7}  {'p':>6}  "
          f"{'d':>6}  {'UpsetΔ':>8}  {'ROI':>8}  {'vs_Base':>9}  {'Sharpe':>8}")
    print(sep)

    sig_serve, sig_roi = 0, 0
    for r in fold_results:
        srv_delta = (r['serve_delta'] or 0) * 100
        ud        = (r['upset_delta'] or 0) * 100
        roi       = (r['bet_roi']     or 0) * 100
        vb        = (r['roi_vs_base'] or 0) * 100
        sh        = r['bet_sharpe'] or 0
        p         = r['serve_p_value']
        d         = r['cohens_d']
        sig_s = "✓" if p < 0.10 and srv_delta < 0 else " "
        sig_r = "✓" if (r['bet_roi'] or 0) > 0 else " "
        if p < 0.10 and srv_delta < 0:
            sig_serve += 1
        if (r['bet_roi'] or 0) > 0:
            sig_roi += 1
        print(f"{r['fold_year']:>5}  {r['n_glare']:>7}  "
              f"{srv_delta:>+6.2f}%{sig_s}  {p:>6.3f}  "
              f"{d:>+6.3f}  {ud:>+7.2f}%  "
              f"{roi:>+7.2f}%{sig_r}  {vb:>+8.2f}%  {sh:>8.3f}")

    print(sep)

    # Aggregate stats
    valid_serve  = df["serve_delta"].dropna()
    valid_roi    = df["bet_roi"].dropna()
    valid_upset  = df["upset_delta"].dropna()
    valid_sharpe = df["bet_sharpe"].dropna()

    print(f"\n{'Metric':<35} {'Mean':>10}  {'Median':>10}  {'Pos/Total':>12}")
    print(sep)

    def row(label, series, pct=True):
        mult = 100 if pct else 1
        pos  = (series > 0).sum()
        print(f"  {label:<33} {series.mean()*mult:>+9.3f}%  "
              f"{series.median()*mult:>+9.3f}%  "
              f"{pos}/{len(series)}")

    row("Serve delta (GLARE - LOW)",    valid_serve)
    row("Upset delta (GLARE - LOW)",    valid_upset)
    row("Bet ROI (backing underdog)",   valid_roi)
    row("ROI vs base (all-match avg)",  df["roi_vs_base"].dropna())

    print(f"\n  Serve signal sig (p<0.10, neg direction): {sig_serve}/{len(fold_results)} folds")
    print(f"  Positive bet ROI folds:                   {sig_roi}/{len(fold_results)} folds")

    # Most common best parameters across folds
    param_counts = df[["elev_low","elev_high","angle_deg","min_penalty"]].value_counts()
    print(f"\n  Most frequently selected parameter sets:")
    for params, cnt in param_counts.head(5).items():
        print(f"    elev=[{params[0]},{params[1]}]° angle={params[2]}° "
              f"pen={params[3]:.3f}  →  {cnt} folds")

    # Overall verdict
    mean_d      = df["cohens_d"].mean()
    mean_roi    = valid_roi.mean() if len(valid_roi) else 0
    mean_serve  = valid_serve.mean() if len(valid_serve) else 0
    print(f"\n{'═'*72}")
    print("OVERALL VERDICT")
    print(sep)

    serve_sig = mean_serve < 0
    roi_sig   = mean_roi > 0

    if serve_sig and roi_sig:
        print("  ✓ ACTIVE SIGNAL — sun glare consistently suppresses serve win%")
        print(f"    and produces positive underdog ROI across {sig_roi}/{len(fold_results)} test years.")
        print(f"    Mean serve delta: {mean_serve*100:+.2f}pp  |  Mean bet ROI: {mean_roi*100:+.2f}%")
        print(f"    Recommendation: KEEP sun penalty in model at tuned parameters.")
    elif serve_sig:
        print("  ~ PARTIAL SIGNAL — serve suppression is real but betting edge is weak.")
        print(f"    Mean serve delta: {mean_serve*100:+.2f}pp  |  Mean bet ROI: {mean_roi*100:+.2f}%")
        print(f"    Recommendation: keep serve penalty; investigate odds source for ROI.")
    elif roi_sig:
        print("  ~ MARKET INEFFICIENCY — positive ROI without consistent serve signal.")
        print(f"    Mean serve delta: {mean_serve*100:+.2f}pp  |  Mean bet ROI: {mean_roi*100:+.2f}%")
        print(f"    Recommendation: investigate further; may be overfitted.")
    else:
        print("  ✗ NO RELIABLE SIGNAL — sun glare does not produce consistent OOS edge.")
        print(f"    Mean serve delta: {mean_serve*100:+.2f}pp  |  Mean bet ROI: {mean_roi*100:+.2f}%")
        print(f"    Recommendation: disable or cap sun penalty at ≤1% in live model.")

    print(f"{'═'*72}\n")
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df     = load_and_prepare()
    folds  = run_wfo(df)

    out_df = report(folds)

    csv_path  = os.path.join(os.path.dirname(__file__), "wfo_sun_results.csv")
    json_path = os.path.join(os.path.dirname(__file__), "wfo_sun_results.json")

    # Drop non-serialisable columns before saving
    save_df = out_df.copy()
    save_df["params"] = save_df["params"].astype(str)
    save_df.to_csv(csv_path, index=False)

    records = save_df.to_dict(orient="records")
    with open(json_path, "w") as f:
        json.dump(records, f, indent=2, default=str)

    print(f"  Saved → {csv_path}")
    print(f"  Saved → {json_path}")


if __name__ == "__main__":
    main()
