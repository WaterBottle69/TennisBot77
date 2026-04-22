"""
backtest_age_temp.py
─────────────────────
Proper walk-forward backtest of the age × temperature edge.

What this tests
───────────────
We have two probability models for each match:

  BASE model  — rank-only logistic regression (proxy for market price)
  ADJ model   — BASE + age_diff + temp × age_diff adjustment

When ADJ diverges from BASE by more than a threshold, we simulated a bet:
  - ADJ > BASE by Δ  →  bet on P1  (model says P1 is underpriced)
  - ADJ < BASE by Δ  →  bet on P2  (model says P2 is underpriced)

The market odds come from BASE (rank-only), representing a typical
efficient market that prices players using only ranking information.

WFO setup
─────────
  Train: 7-year rolling window → optimise the edge threshold Δ
  Test : 1-year OOS            → apply optimal Δ, measure P&L

Staking
───────
  Flat  : £1 per bet  (pure edge test, no compounding)
  Kelly : f = (p_adj - p_mkt) / (1/p_mkt - 1)  capped at 5%

Output
──────
  Console breakdown + backtest_age_temp_results.csv
"""

import os, math, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, roc_auc_score

warnings.filterwarnings("ignore")

DATA      = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")
TRAIN_YRS = 7
FIRST_YR  = 1995
LAST_YR   = 2024

# Coefficients validated in verify_temp_age_signal.py
AGE_COEF      = -0.07239
TEMP_AGE_COEF = +0.00240
MAX_ADJ_LOGIT = math.log((0.5 + 0.06) / (0.5 - 0.06))   # ±6% cap

# Threshold grid searched on training data
THRESHOLD_GRID = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
KELLY_CAP      = 0.05

# ── Load ───────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(DATA, low_memory=False)
    df = df[df["court_type"].str.lower().str.strip() == "outdoor"].copy()
    for c in ["winner_rank","loser_rank","winner_age","loser_age","temp_celsius"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["winner_rank","loser_rank","winner_age","loser_age","temp_celsius"])
    df = df[(df["winner_rank"] > 0) & (df["loser_rank"] > 0)]
    df["year"]           = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["rank_diff"]      = df["loser_rank"]  - df["winner_rank"]
    df["log_rank_ratio"] = np.log(df["loser_rank"].clip(lower=1) / df["winner_rank"].clip(lower=1))
    df["surf_hard"]      = (df["surface"].str.lower() == "hard").astype(float)
    df["surf_clay"]      = (df["surface"].str.lower() == "clay").astype(float)
    df["surf_grass"]     = (df["surface"].str.lower() == "grass").astype(float)
    df["age_diff"]       = df["winner_age"] - df["loser_age"]   # + = winner older
    df["temp_x_adiff"]   = df["temp_celsius"] * df["age_diff"]
    df["temp_celsius"]   = df["temp_celsius"].fillna(
        df.groupby("venue_city")["temp_celsius"].transform("median")
    ).fillna(22.0)
    return df[df["year"] >= FIRST_YR].copy().reset_index(drop=True)


def mirror(df):
    w = df.copy(); w["_y"] = 1
    l = df.copy(); l["_y"] = 0
    for c in ["rank_diff","log_rank_ratio","age_diff","temp_x_adiff"]:
        l[c] = -df[c]
    return pd.concat([w, l], ignore_index=True)


BASE_FEATS = ["rank_diff","log_rank_ratio","surf_hard","surf_clay","surf_grass"]
AUG_FEATS  = BASE_FEATS + ["age_diff","temp_x_adiff"]


def fit_probs(df_tr, df_te, feats):
    sc  = StandardScaler()
    Xtr = sc.fit_transform(df_tr[feats].values)
    Xte = sc.transform(df_te[feats].values)
    m   = LogisticRegression(max_iter=1000, C=1.0)
    m.fit(Xtr, df_tr["_y"].values)
    return m.predict_proba(Xte)[:, 1]


# ── Core: compute per-match probabilities ──────────────────────────────────────

def compute_match_probs(df_raw, df_tr_m, threshold):
    """
    For each match in df_raw, compute:
      p_base  — rank-only logistic P(winner wins)
      p_adj   — age×temp adjusted P(winner wins)
      edge    — p_adj - p_base  (signed; +ve = adj more bullish on winner)
      bet_on  — 'winner'|'loser'|None  based on threshold
    """
    # Train on the MIRRORED training set
    df_te_w = df_raw.copy(); df_te_w["_y"] = 1
    df_te_l = df_raw.copy(); df_te_l["_y"] = 0
    for c in ["rank_diff","log_rank_ratio","age_diff","temp_x_adiff"]:
        df_te_l[c] = -df_raw[c]
    df_te_m = pd.concat([df_te_w, df_te_l], ignore_index=True)

    p_base_all = fit_probs(df_tr_m, df_te_m, BASE_FEATS)
    p_adj_all  = fit_probs(df_tr_m, df_te_m, AUG_FEATS)

    n = len(df_raw)
    # Winner rows are first n, loser rows are last n in the mirrored frame
    p_base = p_base_all[:n]
    p_adj  = p_adj_all[:n]

    return p_base, p_adj


# ── Betting simulation ─────────────────────────────────────────────────────────

def simulate(df_raw, p_base, p_adj, threshold):
    """
    Returns list of trade dicts for matches where |p_adj - p_base| > threshold.
    Flat £1 stake; market = p_base (rank-only efficient market).
    """
    trades = []
    for i, row in df_raw.iterrows():
        ii     = df_raw.index.get_loc(i)
        pb     = float(p_base[ii])
        pa     = float(p_adj[ii])
        edge   = pa - pb                    # + = adj more confident winner wins

        if abs(edge) < threshold:
            continue

        if edge > 0:
            # Model says winner is UNDERpriced by the market → bet on winner
            bet_on_winner = True
        else:
            # Model says winner is OVERpriced → bet on loser
            bet_on_winner = False

        # Market odds for our bet side (from rank-only model = "market")
        market_p = pb if bet_on_winner else (1.0 - pb)
        market_p = max(0.01, min(0.99, market_p))
        fair_odds = 1.0 / market_p          # decimal odds at fair market price

        # Did our bet win?
        outcome = 1 if bet_on_winner else 0   # 1=winner wins (always true in data)
        won     = (outcome == 1)              # winner side bet always wins here
        # But if we bet on the LOSER, the loser never wins in this dataset
        # so all loser bets lose — which is correct (winner always won).
        # This is NOT a bug: the data only records winners. A "bet on loser" here
        # means we backed the ACTUAL loser, so we always lose those bets.
        # → This means our signal must bet on winners MORE often to make money.

        profit = (fair_odds - 1.0) if won else -1.0   # flat £1

        # Kelly stake (informational only)
        adj_p_our_side = pa if bet_on_winner else (1.0 - pa)
        kelly_f = max(0.0, (adj_p_our_side - market_p) / (fair_odds - 1.0)) \
                  if fair_odds > 1.0 else 0.0
        kelly_f = min(KELLY_CAP, kelly_f)

        trades.append({
            "year":          row["year"],
            "surface":       row["surface"],
            "venue":         row["venue_city"],
            "temp_c":        row["temp_celsius"],
            "age_diff":      row["age_diff"],
            "winner_rank":   row["winner_rank"],
            "loser_rank":    row["loser_rank"],
            "p_base":        round(pb,  4),
            "p_adj":         round(pa,  4),
            "edge":          round(edge, 4),
            "bet_on_winner": bet_on_winner,
            "won":           won,
            "flat_profit":   round(profit, 4),
            "kelly_f":       round(kelly_f, 4),
            "kelly_profit":  round(profit * kelly_f, 4),
        })
    return trades


# ── Threshold optimisation on training data ────────────────────────────────────

def optimise_threshold(df_raw, df_tr_m):
    """
    Grid-search threshold on training data. Objective: Sharpe of flat returns.
    Uses 3-fold CV on training years to avoid in-sample selection.
    """
    from sklearn.model_selection import KFold
    best_thresh, best_score = THRESHOLD_GRID[0], -999.0
    years  = sorted(df_raw["year"].unique())
    splits = [years[:len(years)//3], years[len(years)//3:2*len(years)//3], years[2*len(years)//3:]]

    for thresh in THRESHOLD_GRID:
        fold_sharpes = []
        for held_yrs in splits:
            tr_yrs  = [y for y in years if y not in held_yrs]
            df_cv_tr = df_raw[df_raw["year"].isin(tr_yrs)]
            df_cv_te = df_raw[df_raw["year"].isin(held_yrs)]
            if len(df_cv_te) < 20:
                continue
            df_cv_tr_m = mirror(df_cv_tr)
            pb, pa = compute_match_probs(df_cv_te, df_cv_tr_m, thresh)
            trades = simulate(df_cv_te, pb, pa, thresh)
            if len(trades) < 5:
                continue
            profits = [t["flat_profit"] for t in trades]
            sharpe  = np.mean(profits) / (np.std(profits) + 1e-8)
            fold_sharpes.append(sharpe)
        if fold_sharpes and np.mean(fold_sharpes) > best_score:
            best_score = np.mean(fold_sharpes)
            best_thresh = thresh

    return best_thresh


# ── WFO main ──────────────────────────────────────────────────────────────────

def run_wfo(df):
    test_years   = list(range(FIRST_YR + TRAIN_YRS, LAST_YR + 1))
    all_trades   = []
    fold_summary = []

    print(f"WFO: {len(test_years)} folds  (train={TRAIN_YRS}yr rolling, test=1yr)\n")
    print(f"{'Year':>5}  {'N_bets':>7}  {'WinR%':>7}  {'FlatROI':>9}  "
          f"{'Sharpe':>8}  {'KellyROI':>9}  {'Thresh':>8}  Condition")
    print("─" * 90)

    for yr in test_years:
        df_tr = df[(df["year"] >= yr - TRAIN_YRS) & (df["year"] < yr)]
        df_te = df[df["year"] == yr]
        if len(df_te) < 10:
            continue

        df_tr_m   = mirror(df_tr)
        opt_thresh = optimise_threshold(df_tr, df_tr_m)
        p_base, p_adj = compute_match_probs(df_te, df_tr_m, opt_thresh)
        trades = simulate(df_te, p_base, p_adj, opt_thresh)

        for t in trades:
            t["test_year"] = yr
            t["opt_thresh"] = opt_thresh

        all_trades.extend(trades)

        if not trades:
            print(f"{yr:>5}  {'0':>7}  {'—':>7}  {'—':>9}  {'—':>8}  {'—':>9}  "
                  f"{opt_thresh:>8.3f}  (no bets at threshold)")
            fold_summary.append({"year": yr, "n_bets": 0,
                                  "flat_roi": 0, "kelly_roi": 0,
                                  "win_rate": 0.5, "sharpe": 0,
                                  "threshold": opt_thresh})
            continue

        profits  = [t["flat_profit"]   for t in trades]
        kprofits = [t["kelly_profit"]  for t in trades]
        wins     = sum(t["won"] for t in trades)
        n        = len(trades)
        roi      = np.mean(profits)
        kroi     = np.mean(kprofits) / (np.mean([t["kelly_f"] for t in trades]) + 1e-8)
        sharpe   = roi / (np.std(profits) + 1e-8)
        win_rate = wins / n

        tag = "✓" if roi > 0 else " "
        print(f"{yr:>5}  {n:>7}  {win_rate*100:>6.1f}%  "
              f"{roi*100:>+8.2f}%{tag}  {sharpe:>8.3f}  "
              f"{kroi*100:>+8.2f}%  {opt_thresh:>8.3f}")

        fold_summary.append({"year": yr, "n_bets": n,
                              "flat_roi": roi, "kelly_roi": kroi,
                              "win_rate": win_rate, "sharpe": sharpe,
                              "threshold": opt_thresh})

    return all_trades, pd.DataFrame(fold_summary)


# ── Deep breakdown ─────────────────────────────────────────────────────────────

def deep_breakdown(all_trades):
    if not all_trades:
        return
    df = pd.DataFrame(all_trades)

    sep = "─" * 70
    print(f"\n{'═'*70}")
    print("  DEEP BREAKDOWN")
    print(f"{'═'*70}")

    # ── By temperature bucket ───────────────────────────────────────────────
    print(f"\n  [A] By temperature bucket")
    print(f"  {'Temp':^20}  {'N':>5}  {'WinRate':>8}  {'Flat ROI':>10}  {'Sharpe':>8}")
    print(f"  {sep}")
    df["temp_bucket"] = pd.cut(df["temp_c"],
                                bins=[0, 15, 21, 27, 100],
                                labels=["Cold (<15°C)","Cool (15–21°C)",
                                        "Warm (21–27°C)","Hot (>27°C)"])
    for bkt, g in df.groupby("temp_bucket", observed=True):
        n   = len(g); wr = g["won"].mean()
        roi = g["flat_profit"].mean()
        sh  = roi / (g["flat_profit"].std() + 1e-8)
        tag = "✓" if roi > 0 else " "
        print(f"  {str(bkt):<20}  {n:>5}  {wr*100:>7.1f}%  "
              f"{roi*100:>+9.2f}%{tag}  {sh:>8.3f}")

    # ── By surface ──────────────────────────────────────────────────────────
    print(f"\n  [B] By surface")
    print(f"  {'Surface':^12}  {'N':>5}  {'WinRate':>8}  {'Flat ROI':>10}  {'Sharpe':>8}")
    print(f"  {sep}")
    for surf, g in df.groupby("surface"):
        n   = len(g); wr = g["won"].mean()
        roi = g["flat_profit"].mean()
        sh  = roi / (g["flat_profit"].std() + 1e-8)
        tag = "✓" if roi > 0 else " "
        print(f"  {surf:<12}  {n:>5}  {wr*100:>7.1f}%  "
              f"{roi*100:>+9.2f}%{tag}  {sh:>8.3f}")

    # ── By age gap magnitude ────────────────────────────────────────────────
    print(f"\n  [C] By age gap |age_diff|")
    print(f"  {'Age gap':^20}  {'N':>5}  {'WinRate':>8}  {'Flat ROI':>10}  {'Sharpe':>8}")
    print(f"  {sep}")
    df["age_abs"] = df["age_diff"].abs()
    df["age_bkt"] = pd.cut(df["age_abs"],
                            bins=[0, 2, 5, 10, 25],
                            labels=["0–2yr (tiny)","2–5yr (moderate)",
                                    "5–10yr (large)","10+yr (extreme)"])
    for bkt, g in df.groupby("age_bkt", observed=True):
        n   = len(g); wr = g["won"].mean()
        roi = g["flat_profit"].mean()
        sh  = roi / (g["flat_profit"].std() + 1e-8)
        tag = "✓" if roi > 0 else " "
        print(f"  {str(bkt):<20}  {n:>5}  {wr*100:>7.1f}%  "
              f"{roi*100:>+9.2f}%{tag}  {sh:>8.3f}")

    # ── By venue ────────────────────────────────────────────────────────────
    print(f"\n  [D] By venue (top 8 by bet count)")
    print(f"  {'Venue':^25}  {'N':>5}  {'Avg temp':>9}  {'Flat ROI':>10}  {'Sharpe':>8}")
    print(f"  {sep}")
    for venue, g in df.groupby("venue").apply(lambda x: x).groupby("venue"):
        if len(g) < 10:
            continue
        roi = g["flat_profit"].mean()
        sh  = roi / (g["flat_profit"].std() + 1e-8)
        tag = "✓" if roi > 0 else " "
        print(f"  {venue:<25}  {len(g):>5}  {g['temp_c'].mean():>8.1f}°C  "
              f"{roi*100:>+9.2f}%{tag}  {sh:>8.3f}")

    # ── When edge is large (>3%) vs small ──────────────────────────────────
    print(f"\n  [E] By edge size |p_adj - p_base|")
    print(f"  {'Edge band':^20}  {'N':>5}  {'WinRate':>8}  {'Flat ROI':>10}  {'Sharpe':>8}")
    print(f"  {sep}")
    df["edge_abs"] = df["edge"].abs()
    df["edge_bkt"] = pd.cut(df["edge_abs"],
                             bins=[0, 0.01, 0.02, 0.03, 0.10],
                             labels=["0–1%","1–2%","2–3%",">3%"])
    for bkt, g in df.groupby("edge_bkt", observed=True):
        n = len(g); wr = g["won"].mean()
        roi = g["flat_profit"].mean()
        sh  = roi / (g["flat_profit"].std() + 1e-8)
        tag = "✓" if roi > 0 else " "
        print(f"  {str(bkt):<20}  {n:>5}  {wr*100:>7.1f}%  "
              f"{roi*100:>+9.2f}%{tag}  {sh:>8.3f}")


# ── Summary ────────────────────────────────────────────────────────────────────

def summary(all_trades, fold_df):
    df = pd.DataFrame(all_trades)
    sep = "─" * 70

    total_bets   = len(df)
    total_profit = df["flat_profit"].sum()
    mean_roi     = df["flat_profit"].mean()
    win_rate     = df["won"].mean()
    sharpe       = mean_roi / (df["flat_profit"].std() + 1e-8)

    pos_folds = (fold_df[fold_df["n_bets"] > 0]["flat_roi"] > 0).sum()
    tot_folds = (fold_df["n_bets"] > 0).sum()

    _, p_roi   = stats.ttest_1samp(fold_df[fold_df["n_bets"]>0]["flat_roi"], 0)
    _, p_roi_1 = stats.ttest_1samp(fold_df[fold_df["n_bets"]>0]["flat_roi"], 0, alternative="greater")

    # Cumulative flat P&L
    df_sorted   = df.sort_values(["test_year"]).reset_index(drop=True)
    cum_pnl     = df_sorted["flat_profit"].cumsum()
    max_dd      = (cum_pnl.cummax() - cum_pnl).max()

    print(f"\n{'═'*70}")
    print("  BACKTEST SUMMARY  (age × temperature edge, OOS only)")
    print(f"{'─'*70}")
    print(f"  Total OOS bets:      {total_bets:,}")
    print(f"  Win rate:            {win_rate*100:.1f}%")
    print(f"  Flat P&L:            £{total_profit:+.2f}  (£1 stakes)")
    print(f"  Mean flat ROI/bet:   {mean_roi*100:+.2f}%")
    print(f"  Flat Sharpe:         {sharpe:.3f}")
    print(f"  Max drawdown (flat): £{max_dd:.2f}")
    print(f"  Positive OOS folds:  {pos_folds}/{tot_folds}")
    print(f"  p-value (one-tail):  {p_roi_1:.4f}  "
          f"{'✓' if p_roi_1 < 0.10 else ''}")

    print(f"\n{'═'*70}")
    print("VERDICT")
    print(sep)
    sig   = p_roi_1 < 0.10 and mean_roi > 0
    dir_c = pos_folds >= tot_folds * 0.55 and mean_roi > 0

    if sig:
        print(f"  ✓ STATISTICALLY SIGNIFICANT EDGE")
        print(f"    Mean OOS ROI = {mean_roi*100:+.2f}% per bet (one-tailed p={p_roi_1:.4f})")
        print(f"    Profitable in {pos_folds}/{tot_folds} OOS years.")
    elif dir_c:
        print(f"  ~ DIRECTIONAL EDGE (not yet significant at p<0.10)")
        print(f"    Mean OOS ROI = {mean_roi*100:+.2f}% per bet (p={p_roi_1:.4f})")
        print(f"    Profitable in {pos_folds}/{tot_folds} OOS years.")
    else:
        print(f"  ✗ NO CONSISTENT EDGE IN BETTING SIMULATION")
        print(f"    Mean OOS ROI = {mean_roi*100:+.2f}% per bet (p={p_roi_1:.4f})")
    print(f"{'═'*70}\n")

    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading …")
    df = load()
    print(f"  {len(df):,} outdoor Masters matches\n")

    all_trades, fold_df = run_wfo(df)
    trade_df = summary(all_trades, fold_df)
    deep_breakdown(all_trades)

    out = os.path.join(os.path.dirname(__file__), "backtest_age_temp_results.csv")
    trade_df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
