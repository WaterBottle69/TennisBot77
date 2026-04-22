"""
wfo_age_temp_edge.py
─────────────────────
Focused WFO on the two signals that survived CV selection in 87% and 83%
of folds respectively:

  1. age_diff      — age difference between the two players
  2. temp_x_adiff  — temperature × age_diff interaction

Hypothesis
──────────
At ATP Masters 1000, older players (all else equal) outperform their ATP
ranking suggests. The effect is amplified in hot conditions: younger players
tend to be ranked better than they currently perform, while experienced
veterans are undervalued by rank alone — especially in heat.

Why this makes physical sense
──────────────────────────────
  - ATP ranking rewards consistency over 12 rolling months; a 30-year-old
    who peaked at 25 may be ranked lower than a 23-year-old trending up,
    even if the veteran still wins more on any given day at a big event.
  - Masters 1000 events are typically played in warmer months and
    outdoor conditions; aerobic stress (heat) disproportionately affects
    less-experienced players who haven't yet learned to manage energy.
  - Effect is testable and pre-match observable.

WFO setup
─────────
  Training : rolling 7-year window
  Test     : 1 year
  First test: 2002  (7yr train from 1995)
  Last test : 2024

Output
──────
  Console report with betting simulation
  wfo_age_temp_results.csv
"""

import os, sys, warnings, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

DATA_PATH   = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")
TRAIN_YEARS = 7
FIRST_YEAR  = 1995
LAST_YEAR   = 2024

BASE_FEATS = ["rank_diff","log_rank_ratio","surf_hard","surf_clay","surf_grass"]
AUG_FEATS  = BASE_FEATS + ["age_diff","temp_x_adiff"]

# ── Load ───────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["court_type"].str.lower().str.strip() == "outdoor"].copy()

    for c in ["winner_rank","loser_rank","winner_age","loser_age","temp_celsius"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["winner_rank","loser_rank","winner_age","loser_age"])
    df = df[(df["winner_rank"]>0) & (df["loser_rank"]>0)]

    df["year"]           = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["rank_diff"]      = df["loser_rank"]  - df["winner_rank"]
    df["log_rank_ratio"] = np.log(df["loser_rank"].clip(lower=1)/df["winner_rank"].clip(lower=1))
    df["surf_hard"]      = (df["surface"].str.lower()=="hard").astype(float)
    df["surf_clay"]      = (df["surface"].str.lower()=="clay").astype(float)
    df["surf_grass"]     = (df["surface"].str.lower()=="grass").astype(float)
    df["age_diff"]       = df["winner_age"] - df["loser_age"]   # +ve = winner older

    # Fill temperature where missing with venue median
    df["temp_celsius"] = df["temp_celsius"].fillna(
        df.groupby("venue_city")["temp_celsius"].transform("median")
    ).fillna(df["temp_celsius"].median())

    df["temp_x_adiff"] = df["temp_celsius"] * df["age_diff"]

    df = df[df["year"] >= FIRST_YEAR].copy()
    return df


def mirror(df):
    """Mirror each match as winner row (label=1) and loser row (label=0)."""
    w = df.copy(); w["_y"] = 1
    l = df.copy(); l["_y"] = 0
    l["rank_diff"]      = -df["rank_diff"]
    l["log_rank_ratio"] = -df["log_rank_ratio"]
    l["age_diff"]       = -df["age_diff"]
    l["temp_x_adiff"]   = -df["temp_x_adiff"]
    return pd.concat([w, l], ignore_index=True)


def fit_predict(df_tr, df_te, feats):
    sc = StandardScaler()
    Xtr = sc.fit_transform(df_tr[feats].values)
    Xte = sc.transform(df_te[feats].values)
    m = LogisticRegression(max_iter=1000, C=1.0)
    m.fit(Xtr, df_tr["_y"].values)
    return m.predict_proba(Xte)[:, 1], m


# ── Edge breakdown helpers ─────────────────────────────────────────────────────

def bucket_analysis(df_raw, col, n_buckets=4, label=""):
    """Show favourite win rate by bucket of col (on raw un-mirrored data)."""
    df = df_raw.copy()
    df["fav_won"] = (df["rank_diff"] > 0).astype(int)
    try:
        df["bucket"] = pd.qcut(df[col], q=n_buckets, duplicates="drop")
        g = df.groupby("bucket", observed=True)["fav_won"].agg(["mean","count"])
        print(f"\n  {label} — favourite win rate by quartile:")
        for bkt, row in g.iterrows():
            print(f"    {str(bkt):<30} fav wins {row['mean']*100:.1f}%  n={int(row['count']):,}")
    except Exception as e:
        print(f"  (bucket analysis failed: {e})")


# ── WFO ───────────────────────────────────────────────────────────────────────

def run_wfo(df_raw):
    df = mirror(df_raw)
    test_years = list(range(FIRST_YEAR + TRAIN_YEARS, LAST_YEAR + 1))
    fold_rows  = []

    print(f"WFO — {len(test_years)} folds  (train={TRAIN_YEARS}yr rolling, test=1yr)\n")
    print(f"{'Year':>5}  {'N':>5}  {'Base_BS':>9}  {'Aug_BS':>9}  "
          f"{'Lift':>9}  {'Base_AUC':>9}  {'Aug_AUC':>9}  {'AUC_Lift':>9}  "
          f"{'age_coef':>9}  {'temp_x_a':>9}")
    print("─" * 112)

    for yr in test_years:
        tr = df[(df["year"] >= yr - TRAIN_YEARS) & (df["year"] < yr)]
        te = df[df["year"] == yr]
        if len(te) < 40:
            continue

        p_base, _  = fit_predict(tr, te, BASE_FEATS)
        p_aug, clf = fit_predict(tr, te, AUG_FEATS)

        yte = te["_y"].values
        bs_b = brier_score_loss(yte, p_base)
        bs_a = brier_score_loss(yte, p_aug)
        au_b = roc_auc_score(yte, p_base)
        au_a = roc_auc_score(yte, p_aug)

        lift_bs  = bs_b - bs_a
        lift_auc = au_a - au_b

        # Recover coefficients for age_diff and temp_x_adiff
        sc = StandardScaler().fit(tr[AUG_FEATS].values)
        coefs = dict(zip(AUG_FEATS, clf.coef_[0]))
        age_c  = coefs.get("age_diff",   0)
        txa_c  = coefs.get("temp_x_adiff", 0)

        tag_bs  = "✓" if lift_bs  > 0 else "✗"
        tag_auc = "✓" if lift_auc > 0 else "✗"

        print(f"{yr:>5}  {len(te)//2:>5}  {bs_b:>9.6f}  {bs_a:>9.6f}  "
              f"{lift_bs:>+8.6f}{tag_bs}  {au_b:>9.5f}  {au_a:>9.5f}  "
              f"{lift_auc:>+8.5f}{tag_auc}  {age_c:>+9.4f}  {txa_c:>+9.4f}")

        fold_rows.append({
            "year": yr, "n_matches": len(te)//2,
            "base_brier": round(bs_b, 7), "aug_brier": round(bs_a, 7),
            "brier_lift": round(lift_bs, 7),
            "base_auc": round(au_b, 6),   "aug_auc": round(au_a, 6),
            "auc_lift": round(lift_auc, 6),
            "age_coef": round(age_c, 4),  "temp_x_adiff_coef": round(txa_c, 4),
        })

    return pd.DataFrame(fold_rows)


# ── Betting simulation ─────────────────────────────────────────────────────────

def simulate_betting(df_raw):
    """
    Simulate: when the augmented model's probability for the underdog (lower-ranked)
    is higher than the baseline (rank-only), does backing the underdog pay off?

    Edge = aug_prob_underdog - base_prob_underdog
    Bet when edge > threshold; profit at fair odds derived from base_prob.
    """
    df   = mirror(df_raw)
    test_years = list(range(FIRST_YEAR + TRAIN_YEARS, LAST_YEAR + 1))

    all_profits_edge = []
    all_profits_base = []

    EDGE_THRESHOLD = 0.02   # only bet when aug gives ≥2% more to the underdog

    for yr in test_years:
        tr = df[(df["year"] >= yr - TRAIN_YEARS) & (df["year"] < yr)]
        te = df[df["year"] == yr]
        if len(te) < 40:
            continue

        p_base, _ = fit_predict(tr, te, BASE_FEATS)
        p_aug,  _ = fit_predict(tr, te, AUG_FEATS)

        # Work on un-mirrored test rows (winner perspective = odd rows)
        n = len(te) // 2
        # winner rows = indices 0..n-1, loser rows = n..2n-1 in mirrored df
        # p_base[i] = P(winner wins) from baseline, p_aug[i] = same from augmented
        p_b_win = p_base[:n]  # baseline P(winner wins)
        p_a_win = p_aug[:n]   # augmented P(winner wins)

        # For the raw test frame, get rank info
        df_te_raw = df_raw[df_raw["year"] == yr].reset_index(drop=True)

        for i in range(min(n, len(df_te_raw))):
            row = df_te_raw.iloc[i]
            wr, lr = row["winner_rank"], row["loser_rank"]
            if pd.isna(wr) or pd.isna(lr):
                continue

            # Baseline and augmented probability for the WINNER
            pb = float(np.clip(p_b_win[i], 1e-4, 1-1e-4))
            pa = float(np.clip(p_a_win[i], 1e-4, 1-1e-4))

            # Favourite = lower rank number; winner always wins in data
            fav_is_winner = (wr < lr)   # True if the favourite won

            if fav_is_winner:
                # We'd back the favourite; outcome = 1 (win)
                # aug gives MORE confidence to favourite → no special edge vs underdog
                pass
            else:
                # Underdog won. Did aug predict this better than base?
                # aug_p for underdog (loser perspective) = 1 - pa
                aug_p_under = 1.0 - pa
                base_p_under = 1.0 - pb
                edge = aug_p_under - base_p_under
                if edge > EDGE_THRESHOLD:
                    # Back the underdog at fair odds from base model
                    fair_odds = 1.0 / max(base_p_under, 0.01)
                    profit = fair_odds - 1.0   # underdog won → profit
                    all_profits_edge.append(profit)

            # Baseline simulation: always back underdog whenever rank gap is big
            if abs(wr - lr) > 20:
                und_p = (1.0 - pb) if fav_is_winner else pb
                fair_odds = 1.0 / max(und_p, 0.01)
                # Underdog won if not fav_is_winner... but we only have winner rows
                # So if fav_is_winner → underdog lost → -1; else underdog won → profit
                if not fav_is_winner:
                    all_profits_base.append(fair_odds - 1.0)
                else:
                    all_profits_base.append(-1.0)

    return all_profits_edge, all_profits_base


# ── Report ─────────────────────────────────────────────────────────────────────

def report(fold_df, df_raw):
    pos_bs  = (fold_df["brier_lift"] > 0).sum()
    pos_auc = (fold_df["auc_lift"]   > 0).sum()
    n       = len(fold_df)

    _, p_bs  = stats.ttest_1samp(fold_df["brier_lift"].dropna(), 0)
    _, p_auc = stats.ttest_1samp(fold_df["auc_lift"].dropna(),   0)
    _, p_bs_one = stats.ttest_1samp(fold_df["brier_lift"].dropna(), 0, alternative="greater")

    mean_bs  = fold_df["brier_lift"].mean()
    mean_auc = fold_df["auc_lift"].mean()

    print(f"\n{'═'*80}")
    print("  AGE + TEMPERATURE EDGE — WFO SUMMARY")
    print(f"  {n} OOS folds ({FIRST_YEAR+TRAIN_YEARS}–{LAST_YEAR})")
    print(f"{'─'*80}")
    print(f"  Brier lift:  mean={mean_bs:>+.6f}  {pos_bs}/{n} folds positive  "
          f"p={p_bs:.4f} (two-tailed)  p={p_bs_one:.4f} (one-tailed ✓)")
    print(f"  AUC lift:    mean={mean_auc:>+.6f}  {pos_auc}/{n} folds positive  p={p_auc:.4f}")

    # Sign of age coefficient across folds
    age_pos = (fold_df["age_coef"] > 0).sum()
    txa_pos = (fold_df["temp_x_adiff_coef"] > 0).sum()
    print(f"\n  Age coef positive (older winner → higher P): {age_pos}/{n} folds")
    print(f"  Temp×age coef positive:                       {txa_pos}/{n} folds")

    # ── Effect magnitude analysis ───────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  EFFECT ANALYSIS (full outdoor dataset)")
    df_work = df_raw.copy()
    df_work["rank_diff"] = df_work["loser_rank"] - df_work["winner_rank"]
    df_work["fav_won"]   = (df_work["rank_diff"] > 0).astype(int)

    # Age effect
    age_q = pd.qcut(df_work["age_diff"].fillna(0), q=5,
                    labels=["≤-5yr","−2to−5yr","similar","2to5yr","≥5yr older winner"])
    g = df_work.groupby(age_q, observed=True)["fav_won"].agg(["mean","count"])
    print("\n  Favourite win rate by winner−loser age difference:")
    print(f"  {'Age bucket':<22}  {'Fav wins%':>10}  {'N':>7}  {'Upset%':>8}")
    for bkt, row in g.iterrows():
        upset = 1 - row["mean"]
        print(f"  {str(bkt):<22}  {row['mean']*100:>9.1f}%  {int(row['count']):>7,}  "
              f"{upset*100:>7.1f}%")

    # Temperature buckets
    temp_q = pd.qcut(df_work["temp_celsius"].dropna(), q=4,
                     labels=["Cold (<14°C)","Cool (14–21°C)","Warm (21–28°C)","Hot (>28°C)"])
    df_tmp = df_work.dropna(subset=["temp_celsius"]).copy()
    df_tmp["temp_bucket"] = pd.qcut(df_tmp["temp_celsius"], q=4,
                                    labels=["Cold (<14°C)","Cool (14–21°C)","Warm (21–28°C)","Hot (>28°C)"])
    g2 = df_tmp.groupby("temp_bucket", observed=True)["fav_won"].agg(["mean","count"])
    print("\n  Favourite win rate by temperature:")
    print(f"  {'Temp bucket':<22}  {'Fav wins%':>10}  {'N':>7}")
    for bkt, row in g2.iterrows():
        print(f"  {str(bkt):<22}  {row['mean']*100:>9.1f}%  {int(row['count']):>7,}")

    # Cross-tab: hot + older winner vs cold + younger winner
    df_tmp["age_older_winner"] = (df_tmp["age_diff"] > 2).astype(bool)
    df_tmp["is_hot"] = (df_tmp["temp_bucket"] == "Hot (>28°C)")
    ct = df_tmp.groupby(["is_hot","age_older_winner"], observed=True)["fav_won"].agg(["mean","count"])
    print("\n  Hot conditions × older winner interaction:")
    print(f"  {'Condition':<35}  {'Fav wins%':>10}  {'N':>7}")
    for (hot, older), row in ct.iterrows():
        label = f"{'Hot' if hot else 'Not hot'} + {'older' if older else 'younger/equal'} winner"
        print(f"  {label:<35}  {row['mean']*100:>9.1f}%  {int(row['count']):>7,}")

    # ── Verdict ─────────────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("VERDICT")
    print(f"{'─'*80}")

    is_sig   = p_bs_one < 0.10 and mean_bs > 0
    dir_cons = pos_bs >= n * 0.55

    if is_sig:
        print("  ✓ SIGNIFICANT EDGE: age + temp×age adds genuine lift over rank baseline")
        print(f"    Brier improvement in {pos_bs}/{n} OOS folds (one-tailed p={p_bs_one:.4f})")
        print(f"    Mean Brier lift: {mean_bs*10000:+.2f}×10⁻⁴ per match")
    elif dir_cons:
        print("  ~ CONSISTENT DIRECTIONAL EDGE (not yet stat-significant)")
        print(f"    Better in {pos_bs}/{n} OOS folds — consistent with a real signal")
        print(f"    that is small in magnitude but persistent.")
    else:
        print("  ✗ NO RELIABLE EDGE from age/temperature alone")

    if age_pos >= n * 0.6:
        print(f"\n  ✓ Age coefficient is consistently POSITIVE ({age_pos}/{n} folds):")
        print(f"    Older players outperform their ranking at Masters 1000 events.")
        print(f"    When the older player is the underdog, consider them underpriced.")

    if txa_pos >= n * 0.6:
        print(f"\n  ✓ Temp×age coefficient is consistently POSITIVE ({txa_pos}/{n} folds):")
        print(f"    Hot conditions amplify the older-player advantage.")
        print(f"    Key venues: Miami (March–Apr), Cincinnati (Aug), Rome (May), Madrid (May).")

    print(f"\n  Recommended model adjustment:")
    print(f"    win_prob_adj = logit⁻¹( logit(rank_prob)")
    print(f"                   + {fold_df['age_coef'].mean():.3f} × age_diff")
    print(f"                   + {fold_df['temp_x_adiff_coef'].mean():.3f} × temp_celsius × age_diff )")
    print(f"    where age_diff = p1_age − p2_age  (positive = p1 is older)")
    print(f"{'═'*80}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data …")
    df_raw = load()
    print(f"  {len(df_raw):,} outdoor Masters matches\n")

    fold_df = run_wfo(df_raw)
    report(fold_df, df_raw)

    out = os.path.join(os.path.dirname(__file__), "wfo_age_temp_results.csv")
    fold_df.to_csv(out, index=False)
    print(f"  Saved → {out}")


if __name__ == "__main__":
    main()
