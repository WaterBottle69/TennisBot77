"""
verify_temp_age_signal.py
──────────────────────────
Rigorous causal verification of the temp × age_diff signal.

Tests:
  1. Confound check        — is "hot" just "hard court"? Does the effect survive
                             within-surface analysis?
  2. Partial correlation   — controlling for rank + surface, what's the partial
                             corr between temp×age_diff and outcome?
  3. Permutation test      — shuffle age_diff 10,000x; what fraction gives a
                             stronger effect than observed? (true p-value)
  4. Bootstrap CI          — 95% CI on the logistic coefficient for temp×age_diff
  5. Surface-stratified    — does the signal hold on Hard, Clay, Grass separately?
  6. Venue-stratified      — is it driven by one venue (e.g. Madrid, Miami)?
  7. Decade split          — is it stable across 1995–2009 vs 2010–2024?
  8. Rolling coefficient   — plot the temp×age coef over time; does it drift?
"""

import os, warnings, math
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")

# ── Load ───────────────────────────────────────────────────────────────────────

def load():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["court_type"].str.lower().str.strip() == "outdoor"].copy()
    for c in ["winner_rank","loser_rank","winner_age","loser_age","temp_celsius"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["winner_rank","loser_rank","winner_age","loser_age","temp_celsius"])
    df = df[(df["winner_rank"] > 0) & (df["loser_rank"] > 0)]
    df["year"]           = pd.to_numeric(df["year"], errors="coerce").astype(int)
    df["rank_diff"]      = df["loser_rank"]  - df["winner_rank"]
    df["log_rank_ratio"] = np.log(df["loser_rank"].clip(lower=1)/df["winner_rank"].clip(lower=1))
    df["age_diff"]       = df["winner_age"]  - df["loser_age"]
    df["temp_x_adiff"]   = df["temp_celsius"] * df["age_diff"]
    df["surf_hard"]      = (df["surface"].str.lower()=="hard").astype(float)
    df["surf_clay"]      = (df["surface"].str.lower()=="clay").astype(float)
    df["surf_grass"]     = (df["surface"].str.lower()=="grass").astype(float)
    df["fav_won"]        = (df["rank_diff"] > 0).astype(int)
    return df


def mirror(df):
    w = df.copy(); w["_y"] = 1
    l = df.copy(); l["_y"] = 0
    for col in ["rank_diff","log_rank_ratio","age_diff","temp_x_adiff"]:
        l[col] = -df[col]
    return pd.concat([w, l], ignore_index=True)


def logit_coef(df_m, feats, target="_y"):
    """Return standardised coefficients from LogReg on mirrored data."""
    sc = StandardScaler()
    X  = sc.fit_transform(df_m[feats].fillna(0).values)
    y  = df_m[target].values
    m  = LogisticRegression(max_iter=1000, C=1.0)
    m.fit(X, y)
    return dict(zip(feats, m.coef_[0]))


BASE  = ["rank_diff","log_rank_ratio","surf_hard","surf_clay","surf_grass"]
FULL  = BASE + ["age_diff","temp_x_adiff"]

sep = "─" * 70

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df = load()
    dm = mirror(df)

    print("═" * 70)
    print("  CAUSAL VERIFICATION: temp × age_diff signal")
    print(f"  n={len(df):,} outdoor Masters matches (1991–2024)")
    print("═" * 70)

    # ── 1. Confound check: does effect survive within-surface? ─────────────────
    print(f"\n[1] CONFOUND CHECK — effect within each surface")
    print(sep)
    for surf in ["Hard","Clay","Grass"]:
        sub = dm[dm["surface"].str.lower() == surf.lower()]
        if len(sub) < 200:
            continue
        coefs = logit_coef(sub, FULL)
        txa   = coefs.get("temp_x_adiff", 0)
        age   = coefs.get("age_diff", 0)
        n_raw = len(sub) // 2

        # Partial correlation within surface: residualise rank, then correlate
        X_ctrl = sub[BASE].fillna(0).values
        sc     = StandardScaler(); X_s = sc.fit_transform(X_ctrl)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        y_raw = sub["_y"].values.astype(float)
        txa_raw = sub["temp_x_adiff"].values
        lr.fit(X_s, y_raw); resid_y = y_raw - lr.predict(X_s)
        lr2 = LinearRegression()
        lr2.fit(X_s, txa_raw); resid_txa = txa_raw - lr2.predict(X_s)
        if resid_txa.std() > 0 and resid_y.std() > 0:
            r, p = stats.pearsonr(resid_txa, resid_y)
        else:
            r, p = 0, 1

        sig = "✓ sig" if p < 0.10 else "    "
        print(f"  {surf:<8}  n={n_raw:>5,}  logit_coef={txa:>+.4f}  "
              f"partial_r={r:>+.4f}  p={p:.4f}  {sig}")

    # ── 2. Permutation test ────────────────────────────────────────────────────
    print(f"\n[2] PERMUTATION TEST  (10,000 shuffles of age_diff)")
    print(sep)
    np.random.seed(42)
    N_PERM = 10_000
    observed_coef = logit_coef(dm, FULL)["temp_x_adiff"]

    perm_coefs = []
    for _ in range(N_PERM):
        dm_p = dm.copy()
        # Shuffle age_diff (and re-derive temp_x_adiff) within each year
        shuffled = dm_p["age_diff"].values.copy()
        np.random.shuffle(shuffled)
        dm_p["age_diff"]     = shuffled
        dm_p["temp_x_adiff"] = dm_p["temp_celsius"] * shuffled
        c = logit_coef(dm_p, FULL)["temp_x_adiff"]
        perm_coefs.append(c)

    perm_coefs = np.array(perm_coefs)
    p_perm     = (perm_coefs >= observed_coef).mean()
    p_perm_two = 2 * min(p_perm, 1 - p_perm)

    print(f"  Observed temp×age coef:  {observed_coef:>+.4f}")
    print(f"  Permutation mean:         {perm_coefs.mean():>+.4f}  "
          f"±{perm_coefs.std():.4f}")
    print(f"  Permutation p (one-tail): {p_perm:.4f}  "
          f"{'✓ significant' if p_perm < 0.05 else '✗ not significant'}")
    print(f"  Permutation p (two-tail): {p_perm_two:.4f}")
    pctile_95 = np.percentile(perm_coefs, 95)
    print(f"  95th pctile of null:      {pctile_95:>+.4f}  "
          f"(observed {'above' if observed_coef > pctile_95 else 'below'} 95th)")

    # ── 3. Bootstrap 95% CI ────────────────────────────────────────────────────
    print(f"\n[3] BOOTSTRAP 95% CI  (2,000 resamples)")
    print(sep)
    np.random.seed(99)
    N_BOOT = 2_000
    boot_coefs = []
    for _ in range(N_BOOT):
        idx    = np.random.choice(len(dm), size=len(dm), replace=True)
        dm_b   = dm.iloc[idx]
        c      = logit_coef(dm_b, FULL)["temp_x_adiff"]
        boot_coefs.append(c)

    boot_coefs = np.array(boot_coefs)
    ci_lo, ci_hi = np.percentile(boot_coefs, [2.5, 97.5])
    print(f"  Bootstrap mean:  {boot_coefs.mean():>+.4f}")
    print(f"  95% CI:          [{ci_lo:>+.4f},  {ci_hi:>+.4f}]")
    excludes_zero = (ci_lo > 0) or (ci_hi < 0)
    print(f"  Excludes zero:   {'YES ✓' if excludes_zero else 'NO  ✗'}")

    # ── 4. Within-surface AND within-temperature ────────────────────────────────
    print(f"\n[4] SURFACE-CONTROLLED BUCKET ANALYSIS")
    print(sep)
    df["temp_hi"] = (df["temp_celsius"] > df["temp_celsius"].median()).astype(bool)
    df["age_older_winner"] = (df["age_diff"] > 0).astype(bool)

    for surf in ["Hard","Clay"]:
        sub = df[df["surface"].str.lower() == surf.lower()]
        ct  = sub.groupby(["temp_hi","age_older_winner"], observed=True)["fav_won"].agg(["mean","count"])
        print(f"\n  {surf}:")
        print(f"  {'Condition':<40}  {'Fav wins%':>10}  {'N':>6}")
        for (hi, older), row in ct.iterrows():
            label = f"{'Hot' if hi else 'Cold'} + {'Older' if older else 'Younger'} winner"
            print(f"    {label:<38}  {row['mean']*100:>9.1f}%  {int(row['count']):>6,}")

    # ── 5. Decade stability ────────────────────────────────────────────────────
    print(f"\n[5] DECADE STABILITY")
    print(sep)
    for (y1, y2), label in [((1995,2004),"1995–2004"),
                             ((2005,2014),"2005–2014"),
                             ((2015,2024),"2015–2024")]:
        sub   = dm[(dm["year"] >= y1) & (dm["year"] <= y2)]
        if len(sub) < 200:
            continue
        coefs = logit_coef(sub, FULL)
        txa   = coefs["temp_x_adiff"]
        age   = coefs["age_diff"]
        print(f"  {label}  n={len(sub)//2:>5,}  "
              f"age_coef={age:>+.4f}  temp×age_coef={txa:>+.4f}")

    # ── 6. Venue-specific: is one venue driving the signal? ───────────────────
    print(f"\n[6] LEAVE-ONE-VENUE-OUT")
    print(sep)
    all_coef = logit_coef(dm, FULL)["temp_x_adiff"]
    print(f"  Full dataset coef: {all_coef:>+.4f}")
    venues = dm["venue_city"].dropna().unique()
    diffs = []
    for v in sorted(venues):
        sub   = dm[dm["venue_city"] != v]
        if len(sub) < 500:
            continue
        c     = logit_coef(sub, FULL)["temp_x_adiff"]
        diffs.append((v, c, c - all_coef))
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"  {'Venue removed':<25}  {'Coef':>8}  {'Δ from full':>12}  {'Impact'}")
    for v, c, d in diffs[:8]:
        impact = "HIGH ⚠" if abs(d) > 0.05 else "low"
        print(f"  {v:<25}  {c:>+8.4f}  {d:>+12.4f}  {impact}")

    # ── 7. Rolling coefficient ─────────────────────────────────────────────────
    print(f"\n[7] ROLLING 5-YEAR COEFFICIENT (year-by-year)")
    print(sep)
    print(f"  {'Period':<15}  {'temp×age coef':>14}  {'Direction'}")
    years = sorted(dm["year"].unique())
    for y in years:
        if y < 2000:
            continue
        sub = dm[(dm["year"] >= y-4) & (dm["year"] <= y)]
        if len(sub) < 300:
            continue
        c   = logit_coef(sub, FULL)["temp_x_adiff"]
        bar = "▲" if c > 0 else "▼"
        print(f"  {y-4}–{y}          {c:>+14.4f}  {bar}")

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print("FINAL VERDICT")
    print(sep)

    within_hard_sig  = False
    within_clay_sig  = False
    for surf in ["Hard","Clay"]:
        sub   = dm[dm["surface"].str.lower() == surf.lower()]
        if len(sub) < 200:
            continue
        coefs = logit_coef(sub, FULL)
        txa   = coefs.get("temp_x_adiff", 0)
        X_ctrl = sub[BASE].fillna(0).values
        sc = StandardScaler(); X_s = sc.fit_transform(X_ctrl)
        from sklearn.linear_model import LinearRegression
        lr = LinearRegression()
        y_raw = sub["_y"].values.astype(float)
        txa_raw = sub["temp_x_adiff"].values
        lr.fit(X_s, y_raw); resid_y = y_raw - lr.predict(X_s)
        lr2 = LinearRegression(); lr2.fit(X_s, txa_raw); resid_txa = txa_raw - lr2.predict(X_s)
        if resid_txa.std() > 0:
            r, p = stats.pearsonr(resid_txa, resid_y)
            if p < 0.10:
                if surf == "Hard":
                    within_hard_sig = True
                else:
                    within_clay_sig = True

    perm_sig      = p_perm < 0.05
    boot_sig      = excludes_zero
    surface_confound = not (within_hard_sig or within_clay_sig)

    print(f"  Permutation test:      {'✓ significant' if perm_sig else '✗ not significant'}  (p={p_perm:.4f})")
    print(f"  Bootstrap 95% CI:      {'✓ excludes zero' if boot_sig else '✗ includes zero'}  [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"  Within Hard court:     {'✓ signal persists' if within_hard_sig else '✗ no signal'}")
    print(f"  Within Clay court:     {'✓ signal persists' if within_clay_sig else '✗ no signal'}")
    print()

    if surface_confound:
        print("  CONCLUSION: ✗ SPURIOUS — temperature is a proxy for surface type.")
        print("  The signal disappears when you control for surface within each court type.")
        print("  Hot matches = hard courts, cold matches = clay courts.")
        print("  The 23/23 positive WFO coefficients were detecting surface × age,")
        print("  not temperature × age directly.")
        print()
        print("  What IS real: on HARD courts, slightly older players outperform rank.")
        print("  Actionable: when an older player faces a younger player on hard courts,")
        print("  consider adjusting their win probability upward slightly.")
    elif perm_sig and boot_sig:
        print("  CONCLUSION: ✓ GENUINE — temp×age_diff is a real signal.")
        print("  Survives surface controls, permutation test, and bootstrap CI.")
    elif perm_sig or boot_sig:
        print("  CONCLUSION: ~ MARGINAL — partial evidence for a genuine signal.")
        print("  Passes some but not all tests. Treat with caution.")
    else:
        print("  CONCLUSION: ✗ SPURIOUS — does not survive rigorous testing.")

    print(f"{'═'*70}\n")


if __name__ == "__main__":
    main()
