"""
wfo_edge_search.py
──────────────────
Systematic edge search on ATP Masters 1000 data (1991–2024).
Walk-forward optimised to avoid overfitting.

Each candidate signal is tested for INCREMENTAL lift over a rank-only baseline.
The WFO selects the best signals on training data and evaluates them OOS.

Candidate signals (all pre-match observable)
─────────────────────────────────────────────
  1.  altitude_m                        — venue altitude raw
  2.  altitude × height_diff            — tall servers benefit more at altitude
  3.  altitude × rank_diff              — rank matters more/less at altitude?
  4.  wind_speed_kmh                    — wind reduces serve edge
  5.  wind × rank_diff                  — wind equalises; underdogs cover more
  6.  wind × height_diff                — tall servers hurt more by wind
  7.  temp_celsius                      — heat as a physical stressor
  8.  temp × age_diff                   — older players struggle more in heat
  9.  humidity_pct                      — high humidity slows ball, helps baseliners
 10.  humidity × surface_hard           — humidity effect strongest on hard courts
 11.  age_diff                          — experience/physical prime
 12.  height_diff                       — serve weapon proxy
 13.  left_vs_right                     — left-hander matchup asymmetry
 14.  round_weight                      — QF/SF/F vs early rounds
 15.  high_altitude_flag × height_diff  — threshold interaction (alt > 400m)
 16.  extreme_wind_flag × rank_diff     — threshold: wind > 18 km/h

Walk-forward setup
──────────────────
  Training window  : 7 years (more data = more reliable feature selection)
  Test window      : 1 year
  Roll by          : 1 year
  Folds            : 1998–2024 → 20 test folds

Feature selection within each training fold
────────────────────────────────────────────
  For each candidate, compute its OOS-safe importance via 5-fold CV
  on the training data. Select features with CV Brier lift > 0 in ≥3/5 CV folds.
  Then train the final augmented model on the full training window.

Output
──────
  Console report + wfo_edge_results.csv
"""

import os, sys, warnings, math
import numpy as np
import pandas as pd
from itertools import combinations
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")
TRAIN_YEARS = 7
FIRST_YEAR  = 1995
LAST_YEAR   = 2024
MIN_RANK    = 1
MAX_RANK    = 2000

# ── Load & engineer ────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # Outdoor only, known ranks
    df = df[df["court_type"].str.lower().str.strip() == "outdoor"].copy()
    df = df.dropna(subset=["winner_rank","loser_rank","venue_lat","venue_lon"])
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce")
    df = df.dropna(subset=["winner_rank","loser_rank"])
    df = df[(df["winner_rank"] > 0) & (df["loser_rank"] > 0)]

    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # ── Rank baseline ──────────────────────────────────────────────────────────
    # Positive = winner was the favourite (lower rank number = better)
    df["rank_diff"]      = df["loser_rank"]  - df["winner_rank"]
    df["log_rank_ratio"] = np.log(
        df["loser_rank"].clip(lower=1) / df["winner_rank"].clip(lower=1)
    )

    # ── Surface dummies ────────────────────────────────────────────────────────
    df["surf_hard"]  = (df["surface"].str.lower() == "hard").astype(float)
    df["surf_clay"]  = (df["surface"].str.lower() == "clay").astype(float)
    df["surf_grass"] = (df["surface"].str.lower() == "grass").astype(float)

    # ── Physical differences (winner − loser) ─────────────────────────────────
    df["height_diff"] = pd.to_numeric(df["winner_ht"],  errors="coerce") \
                      - pd.to_numeric(df["loser_ht"],   errors="coerce")
    df["age_diff"]    = pd.to_numeric(df["winner_age"], errors="coerce") \
                      - pd.to_numeric(df["loser_age"],  errors="coerce")

    # Left-vs-right: 1 if winner is left-handed and loser is right-handed
    df["winner_left"] = (df["winner_hand"].str.upper() == "L").astype(float)
    df["loser_left"]  = (df["loser_hand"].str.upper()  == "L").astype(float)
    df["left_vs_right"] = (df["winner_left"] - df["loser_left"])  # +1 L beats R, -1 R beats L

    # ── Round weight ──────────────────────────────────────────────────────────
    round_map = {"R128":1,"R64":1,"R32":2,"R16":3,"QF":4,"SF":5,"F":6,"RR":2}
    df["round_weight"] = df["round"].map(round_map).fillna(2).astype(float)

    # ── Environmental ─────────────────────────────────────────────────────────
    for col in ["altitude_m","wind_speed_kmh","temp_celsius","humidity_pct"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Fill missing env with venue median (since we have the scraped data)
    for col in ["altitude_m","wind_speed_kmh","temp_celsius","humidity_pct"]:
        medians = df.groupby("venue_city")[col].transform("median")
        df[col] = df[col].fillna(medians).fillna(df[col].median())

    # ── Candidate feature engineering ─────────────────────────────────────────
    alt  = df["altitude_m"]
    wind = df["wind_speed_kmh"]
    temp = df["temp_celsius"]
    hum  = df["humidity_pct"]
    hdif = df["height_diff"].fillna(0)
    adif = df["age_diff"].fillna(0)
    rdif = df["rank_diff"]
    lrr  = df["log_rank_ratio"]

    df["feat_altitude"]          = alt / 100.0          # scale
    df["feat_alt_x_hdiff"]       = (alt / 100.0) * hdif
    df["feat_alt_x_rdiff"]       = (alt / 100.0) * rdif
    df["feat_wind"]              = wind
    df["feat_wind_x_rdiff"]      = wind * rdif
    df["feat_wind_x_hdiff"]      = wind * hdif
    df["feat_temp"]              = temp
    df["feat_temp_x_adiff"]      = temp * adif
    df["feat_humidity"]          = hum
    df["feat_hum_x_hard"]        = hum * df["surf_hard"]
    df["feat_age_diff"]          = adif
    df["feat_height_diff"]       = hdif
    df["feat_left_vs_right"]     = df["left_vs_right"]
    df["feat_round_weight"]      = df["round_weight"]
    df["feat_hi_alt_x_hdiff"]    = (alt > 400).astype(float) * hdif
    df["feat_hi_wind_x_rdiff"]   = (wind > 18).astype(float) * rdif

    # ── Mirror rows for binary classification ─────────────────────────────────
    feat_cols = [c for c in df.columns if c.startswith("feat_")]

    # Winner rows → outcome 1
    w = df.copy(); w["_outcome"] = 1

    # Loser rows → flip signed features, outcome 0
    l = df.copy(); l["_outcome"] = 0
    l["rank_diff"]      = -rdif
    l["log_rank_ratio"] = -lrr
    l["feat_alt_x_rdiff"]     = (alt / 100.0) * (-rdif)
    l["feat_wind_x_rdiff"]    = wind * (-rdif)
    l["feat_wind_x_hdiff"]    = wind * (-hdif)
    l["feat_temp_x_adiff"]    = temp * (-adif)
    l["feat_hum_x_hard"]      = hum * df["surf_hard"]   # symmetric
    l["feat_age_diff"]        = -adif
    l["feat_height_diff"]     = -hdif
    l["feat_left_vs_right"]   = -df["left_vs_right"]
    l["feat_hi_alt_x_hdiff"]  = (alt > 400).astype(float) * (-hdif)
    l["feat_hi_wind_x_rdiff"] = (wind > 18).astype(float) * (-rdif)
    # altitude, wind, temp, humidity, round are symmetric → unchanged

    df = pd.concat([w, l], ignore_index=True)
    df = df[df["year"] >= FIRST_YEAR].copy()

    print(f"Rows (mirrored): {len(df):,}  |  Years: {df['year'].min()}–{df['year'].max()}")
    return df


# ── Model helpers ──────────────────────────────────────────────────────────────

BASE_FEATS = ["rank_diff", "log_rank_ratio", "surf_hard", "surf_clay", "surf_grass"]
CANDIDATE_FEATS = [
    "feat_altitude",
    "feat_alt_x_hdiff",
    "feat_alt_x_rdiff",
    "feat_wind",
    "feat_wind_x_rdiff",
    "feat_wind_x_hdiff",
    "feat_temp",
    "feat_temp_x_adiff",
    "feat_humidity",
    "feat_hum_x_hard",
    "feat_age_diff",
    "feat_height_diff",
    "feat_left_vs_right",
    "feat_round_weight",
    "feat_hi_alt_x_hdiff",
    "feat_hi_wind_x_rdiff",
]


def make_model():
    return LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0)


def fit_eval(X_tr, y_tr, X_te, y_te):
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_tr)
    Xte = sc.transform(X_te)
    m   = make_model()
    m.fit(Xtr, y_tr)
    p   = m.predict_proba(Xte)[:, 1]
    return brier_score_loss(y_te, p), roc_auc_score(y_te, p), p


def cv_brier_lift(X_tr, y_tr, base_cols, extra_col, n_splits=5):
    """
    5-fold CV on training data: does adding extra_col improve Brier score?
    Returns (mean_lift, fraction_of_folds_positive).
    """
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    lifts  = []
    X      = X_tr
    y      = y_tr
    base_idx = [list(X_tr.columns).index(c) for c in base_cols if c in X_tr.columns]
    aug_idx  = base_idx + ([list(X_tr.columns).index(extra_col)]
                           if extra_col in X_tr.columns else [])

    for tr_idx, val_idx in kf.split(X):
        Xtr_b = X.iloc[tr_idx, base_idx].values
        Xval_b= X.iloc[val_idx, base_idx].values
        Xtr_a = X.iloc[tr_idx, aug_idx].values
        Xval_a= X.iloc[val_idx, aug_idx].values
        ytr   = y.iloc[tr_idx].values
        yval  = y.iloc[val_idx].values

        b_base, _, _ = fit_eval(pd.DataFrame(Xtr_b), ytr, pd.DataFrame(Xval_b), yval)
        b_aug,  _, _ = fit_eval(pd.DataFrame(Xtr_a), ytr, pd.DataFrame(Xval_a), yval)
        lifts.append(b_base - b_aug)

    return float(np.mean(lifts)), sum(l > 0 for l in lifts) / len(lifts)


# ── WFO ───────────────────────────────────────────────────────────────────────

def run_wfo(df):
    test_years = list(range(FIRST_YEAR + TRAIN_YEARS, LAST_YEAR + 1))
    all_cols   = BASE_FEATS + CANDIDATE_FEATS
    df_work    = df[all_cols + ["year","_outcome"]].copy().reset_index(drop=True)

    fold_rows = []
    signal_counts = {f: 0 for f in CANDIDATE_FEATS}  # times selected OOS

    print(f"\nWalk-forward: {len(test_years)} folds  "
          f"(train={TRAIN_YEARS}yr rolling, test=1yr)\n")

    header = (f"{'Year':>5}  {'N':>5}  {'Base_BS':>8}  {'Aug_BS':>8}  "
              f"{'Lift':>8}  {'Base_AUC':>9}  {'Aug_AUC':>9}  "
              f"{'AUC_Lift':>9}  Selected features")
    print(header)
    print("─" * 130)

    for test_year in test_years:
        tr_mask  = (df_work["year"] >= test_year - TRAIN_YEARS) & (df_work["year"] < test_year)
        te_mask  = df_work["year"] == test_year
        df_tr    = df_work[tr_mask]
        df_te    = df_work[te_mask]

        if len(df_te) < 40:
            continue

        y_tr = df_tr["_outcome"]
        y_te = df_te["_outcome"].values

        # ── Feature selection via CV on training data ──────────────────────────
        selected = []
        cv_details = {}
        for feat in CANDIDATE_FEATS:
            mean_lift, pos_frac = cv_brier_lift(df_tr[all_cols], y_tr, BASE_FEATS, feat)
            cv_details[feat] = (mean_lift, pos_frac)
            # Select if positive in ≥ 3/5 CV folds AND mean lift > small threshold
            if pos_frac >= 0.6 and mean_lift > 1e-5:
                selected.append(feat)

        # ── Train models on full training window ───────────────────────────────
        aug_feats = BASE_FEATS + selected
        X_tr_base = df_tr[BASE_FEATS]
        X_te_base = df_te[BASE_FEATS]
        X_tr_aug  = df_tr[aug_feats]
        X_te_aug  = df_te[aug_feats]

        bs_base, auc_base, _ = fit_eval(X_tr_base, y_tr, X_te_base, y_te)
        bs_aug,  auc_aug,  p_aug = fit_eval(X_tr_aug, y_tr, X_te_aug, y_te)

        lift_bs  = bs_base  - bs_aug
        lift_auc = auc_aug  - auc_base

        for f in selected:
            signal_counts[f] += 1

        tag_bs  = "✓" if lift_bs  > 0 else "✗"
        tag_auc = "✓" if lift_auc > 0 else "✗"
        sel_str = ", ".join(f.replace("feat_","") for f in selected) if selected else "(none)"

        n_te = len(df_te) // 2  # actual matches (before mirroring)
        print(f"{test_year:>5}  {n_te:>5}  {bs_base:>8.5f}  {bs_aug:>8.5f}  "
              f"{lift_bs:>+7.5f}{tag_bs}  {auc_base:>9.5f}  {auc_aug:>9.5f}  "
              f"{lift_auc:>+8.5f}{tag_auc}  {sel_str}")

        fold_rows.append({
            "year":         test_year,
            "n_matches":    n_te,
            "base_brier":   round(bs_base,  6),
            "aug_brier":    round(bs_aug,   6),
            "brier_lift":   round(lift_bs,  6),
            "base_auc":     round(auc_base, 6),
            "aug_auc":      round(auc_aug,  6),
            "auc_lift":     round(lift_auc, 6),
            "n_selected":   len(selected),
            "selected_feats": "|".join(f.replace("feat_","") for f in selected),
            **{f"sel_{f.replace('feat_','')}": int(f in selected) for f in CANDIDATE_FEATS},
        })

    return pd.DataFrame(fold_rows), signal_counts


# ── Deep-dive on best signals ─────────────────────────────────────────────────

def deep_dive(df_raw, signal_counts):
    """For the top 3 most-selected signals, show detailed effect analysis."""
    top3 = sorted(signal_counts.items(), key=lambda x: -x[1])[:5]
    print(f"\n{'═'*80}")
    print("DEEP DIVE — TOP SIGNALS")
    print("═"*80)
    # Use full outdoor dataset for effect analysis (no mirroring, just winner rows)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["court_type"].str.lower().str.strip() == "outdoor"].copy()
    df = df.dropna(subset=["winner_rank","loser_rank"])
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce")
    df = df.dropna(subset=["winner_rank","loser_rank"])
    for c in ["altitude_m","wind_speed_kmh","temp_celsius","humidity_pct",
              "winner_ht","loser_ht","winner_age","loser_age"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["height_diff"] = df["winner_ht"] - df["loser_ht"]
    df["age_diff"]    = df["winner_age"] - df["loser_age"]
    df["rank_diff"]   = df["loser_rank"] - df["winner_rank"]
    df["fav_won"]     = (df["rank_diff"] > 0).astype(int)   # favourite won = rank_diff > 0
    df = df[pd.to_numeric(df["year"], errors="coerce") >= FIRST_YEAR]

    for feat_name, count in top3:
        if count == 0:
            continue
        clean = feat_name.replace("feat_","")
        print(f"\n  Signal: {clean}  (selected in {count}/{LAST_YEAR-FIRST_YEAR-TRAIN_YEARS+1} OOS folds)")
        print(f"  {'─'*60}")

        if clean == "altitude":
            buckets = pd.qcut(df["altitude_m"].fillna(0), q=3,
                              labels=["Low (<50m)","Mid (50–200m)","High (>200m)"])
            g = df.groupby(buckets, observed=True)["fav_won"].agg(["mean","count"])
            for label, row in g.iterrows():
                print(f"    {label}: fav wins {row['mean']*100:.1f}%  (n={int(row['count']):,})")

        elif clean == "alt_x_hdiff":
            alt_hi = df["altitude_m"].fillna(0) > 400
            hdiff_hi = df["height_diff"].fillna(0) > 3
            for alt_l, alt_v in [("Low alt",~alt_hi), ("High alt",alt_hi)]:
                for hd_l, hd_v in [("shorter/equal winner",~hdiff_hi), ("taller winner",hdiff_hi)]:
                    sub = df[alt_v & hd_v]
                    if len(sub) > 20:
                        print(f"    {alt_l} + {hd_l}: fav wins {sub['fav_won'].mean()*100:.1f}%  "
                              f"upset rate {(1-sub['fav_won'].mean())*100:.1f}%  (n={len(sub):,})")

        elif clean == "wind":
            buckets = pd.qcut(df["wind_speed_kmh"].fillna(df["wind_speed_kmh"].median()), q=4,
                              labels=["Q1 calm","Q2","Q3","Q4 windy"])
            g = df.groupby(buckets, observed=True)["fav_won"].agg(["mean","count"])
            for label, row in g.iterrows():
                print(f"    {label}: fav wins {row['mean']*100:.1f}%  (n={int(row['count']):,})")

        elif clean == "wind_x_rdiff":
            wind_hi = df["wind_speed_kmh"].fillna(0) > 18
            for wl, wv in [("Low wind",~wind_hi), ("High wind",wind_hi)]:
                sub = df[wv]
                if len(sub) > 20:
                    print(f"    {wl}: fav wins {sub['fav_won'].mean()*100:.1f}%  "
                          f"(n={len(sub):,})")

        elif clean == "age_diff":
            age_q = pd.qcut(df["age_diff"].fillna(0), q=3,
                            labels=["Younger winner","Similar age","Older winner"])
            g = df.groupby(age_q, observed=True)["fav_won"].agg(["mean","count"])
            for label, row in g.iterrows():
                print(f"    {label}: is-fav wins {row['mean']*100:.1f}%  (n={int(row['count']):,})")

        elif clean == "height_diff":
            hdiff_q = pd.qcut(df["height_diff"].fillna(0), q=3,
                              labels=["Shorter winner","Similar ht","Taller winner"])
            g = df.groupby(hdiff_q, observed=True)["fav_won"].agg(["mean","count"])
            for label, row in g.iterrows():
                print(f"    {label}: is-fav wins {row['mean']*100:.1f}%  (n={int(row['count']):,})")

        elif clean in ("hi_alt_x_hdiff","alt_x_rdiff","temp","temp_x_adiff",
                       "humidity","hum_x_hard","wind_x_hdiff","hi_wind_x_rdiff",
                       "left_vs_right","round_weight"):
            print(f"    (see wfo_edge_results.csv for fold-by-fold detail)")


# ── Report ─────────────────────────────────────────────────────────────────────

def report(fold_df, signal_counts, n_folds):
    pos_brier = (fold_df["brier_lift"] > 0).sum()
    pos_auc   = (fold_df["auc_lift"]   > 0).sum()
    mean_bl   = fold_df["brier_lift"].mean()
    mean_al   = fold_df["auc_lift"].mean()
    _, p_b    = stats.ttest_1samp(fold_df["brier_lift"].dropna(), 0)
    _, p_a    = stats.ttest_1samp(fold_df["auc_lift"].dropna(),   0)

    print(f"\n{'═'*80}")
    print("EDGE SEARCH SUMMARY")
    print(f"{'─'*80}")
    print(f"  Metric          Mean lift    Pos/Total   p-value")
    print(f"{'─'*80}")
    print(f"  Brier score   {mean_bl:>+10.6f}   "
          f"{pos_brier:>2}/{len(fold_df)}        p={p_b:.4f}"
          f"  {'✓' if p_b < 0.10 and mean_bl > 0 else ''}")
    print(f"  AUC           {mean_al:>+10.6f}   "
          f"{pos_auc:>2}/{len(fold_df)}        p={p_a:.4f}"
          f"  {'✓' if p_a < 0.10 and mean_al > 0 else ''}")

    print(f"\n  Signal frequency across {n_folds} OOS folds (selected by CV):")
    print(f"  {'Signal':<25}  {'Times selected':>15}  {'Selection rate':>15}")
    print(f"  {'─'*60}")
    for feat, cnt in sorted(signal_counts.items(), key=lambda x: -x[1]):
        clean = feat.replace("feat_","")
        rate  = cnt / n_folds
        bar   = "█" * int(rate * 20)
        print(f"  {clean:<25}  {cnt:>8}/{n_folds:<6}   {rate*100:>6.1f}%  {bar}")

    print(f"\n{'═'*80}")
    # Verdict
    top_signal = max(signal_counts, key=signal_counts.get)
    top_rate   = signal_counts[top_signal] / n_folds

    if mean_bl > 0 and p_b < 0.10:
        print("VERDICT: ✓ GENUINE EDGE FOUND")
        print(f"  Adding environmental/physical features to rank baseline")
        print(f"  improves Brier score in {pos_brier}/{len(fold_df)} OOS folds (p={p_b:.4f}).")
    elif pos_brier >= len(fold_df) * 0.55:
        print("VERDICT: ~ MARGINAL EDGE (directionally consistent, not quite significant)")
        print(f"  Augmented model beats baseline in {pos_brier}/{len(fold_df)} OOS folds.")
    else:
        print("VERDICT: ✗ NO ROBUST EDGE ABOVE RANK BASELINE")

    if top_rate >= 0.5:
        print(f"\n  Strongest signal: '{top_signal.replace('feat_','')}' "
              f"(selected {signal_counts[top_signal]}/{n_folds} folds = {top_rate*100:.0f}%)")

    print(f"{'═'*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading and engineering features …")
    df = load()

    fold_df, signal_counts = run_wfo(df)
    n_folds = len(fold_df)

    report(fold_df, signal_counts, n_folds)
    deep_dive(df, signal_counts)

    out = os.path.join(os.path.dirname(__file__), "wfo_edge_results.csv")
    fold_df.to_csv(out, index=False)
    print(f"\n  Saved → {out}")


if __name__ == "__main__":
    main()
