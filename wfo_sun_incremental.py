"""
wfo_sun_incremental.py
──────────────────────
Tests whether the sun-positioning signal adds INCREMENTAL alpha on top of
the rank/surface baseline that already exists in the model.

This is NOT a standalone edge test. The question is:

  "Given I already know player rankings and surface, does knowing the
   sun conditions at match time improve my win-probability estimate?"

Method (Walk-Forward, 5-yr train / 1-yr test, 2005–2024)
─────────────────────────────────────────────────────────
For each fold:

  Baseline model   : LogReg(rank_diff, log_rank_ratio, surface)
  Augmented model  : LogReg(same + sun_score, sun_score × rank_diff,
                            sun_score × log_rank_ratio)

  where sun_score = max(penalty_end_A, penalty_end_B) for that match,
  computed with the corrected NOAA solar formula + optimised parameters
  from wfo_sun_masters.py (elev=[5,60]°, angle=30°, min_pen=0.01).

Metrics per fold (OOS test year):
  - Brier score (lower = better calibration)
  - Log-loss    (lower = better)
  - AUC         (higher = better rank-order)
  - Brier lift  = baseline_brier - augmented_brier  (positive = sun helps)
  - Calibration on GLARE vs LOW subsets
  - Expected value gain: if you use model_prob + sun adjustment,
    does your edge vs a rank-only market increase?

Output: wfo_sun_incremental.csv + printed report
"""

import datetime, math, os, sys, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from location_engine import compute_solar_position, compute_sun_penalty

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "atp_masters_matches.csv")
TRAIN_YEARS = 5
FIRST_YEAR  = 2000
LAST_YEAR   = 2024
MIN_SVPT    = 20

# Best parameters from wfo_sun_masters.py (most frequent across all folds)
ELEV_LOW    = 5.0
ELEV_HIGH   = 60.0
ANGLE_DEG   = 30.0
MIN_PENALTY = 0.01

ROUND_HOURS = {
    "R128": 11, "R64": 11, "R32": 12,
    "R16": 13, "QF": 14, "SF": 15, "F": 15, "RR": 13,
}
DEFAULT_HOUR = 13


# ── Solar penalty for a full match (max over both ends) ───────────────────────

def match_sun_score(lat, lon, date_obj, local_hour, court_orient=0.0):
    """
    Returns (sun_score, sun_elev, sun_cat) for a match.
    sun_score = max penalty across both serving ends.
    """
    if lat is None or lon is None or math.isnan(float(lat)):
        return 0.0, 0.0, "UNKNOWN"

    utc_offset = round(float(lon) / 15.0)
    utc_hour   = (local_hour - utc_offset) % 24
    dt = datetime.datetime(date_obj.year, date_obj.month, date_obj.day, utc_hour)

    try:
        az, elev = compute_solar_position(float(lat), float(lon), dt)
    except Exception:
        return 0.0, 0.0, "UNKNOWN"

    if elev < ELEV_LOW or elev > ELEV_HIGH:
        return 0.0, elev, "LOW"

    dir_a = (court_orient + 180.0) % 360.0
    dir_b = court_orient
    p_a   = compute_sun_penalty(dir_a, az, elev)
    p_b   = compute_sun_penalty(dir_b, az, elev)
    score = max(p_a, p_b)

    if score >= MIN_PENALTY:
        cat = "GLARE"
    elif score > MIN_PENALTY * 0.25:
        cat = "MODERATE"
    else:
        cat = "LOW"

    return score, elev, cat


# ── Pre-process ───────────────────────────────────────────────────────────────

def load_and_prepare():
    print("Loading atp_masters_matches.csv …")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[df["court_type"].str.strip().str.lower() == "outdoor"].copy()
    df = df.dropna(subset=["venue_lat", "venue_lon", "winner_rank", "loser_rank"])

    df["_date"] = pd.to_datetime(df["tourney_date"], errors="coerce").dt.date
    df = df.dropna(subset=["_date"])
    df["_year"]  = df["_date"].apply(lambda d: d.year)
    df["_local_hour"] = df["round"].map(ROUND_HOURS).fillna(DEFAULT_HOUR).astype(int)

    # Mirror each match as two rows: winner perspective (label=1) + loser (label=0)
    # Features from winner POV: rank_diff = loser_rank - winner_rank (+ve = winner favoured)
    surf_hard  = (df["surface"].str.lower() == "hard").astype(float)
    surf_clay  = (df["surface"].str.lower() == "clay").astype(float)
    surf_grass = (df["surface"].str.lower() == "grass").astype(float)

    winner_rows = df.copy()
    winner_rows["_rank_diff"]      = df["loser_rank"]  - df["winner_rank"]
    winner_rows["_log_rank_ratio"] = np.log(df["loser_rank"].clip(lower=1) / df["winner_rank"].clip(lower=1))
    winner_rows["_surface_hard"]   = surf_hard
    winner_rows["_surface_clay"]   = surf_clay
    winner_rows["_surface_grass"]  = surf_grass
    winner_rows["_outcome"]        = 1

    loser_rows = df.copy()
    loser_rows["_rank_diff"]       = df["winner_rank"] - df["loser_rank"]   # flipped
    loser_rows["_log_rank_ratio"]  = np.log(df["winner_rank"].clip(lower=1) / df["loser_rank"].clip(lower=1))
    loser_rows["_surface_hard"]    = surf_hard
    loser_rows["_surface_clay"]    = surf_clay
    loser_rows["_surface_grass"]   = surf_grass
    loser_rows["_outcome"]         = 0

    df = pd.concat([winner_rows, loser_rows], ignore_index=True)

    # Sun features
    print("  Computing solar positions …")
    scores, elevs, cats = [], [], []
    for _, row in df.iterrows():
        sc, el, ct = match_sun_score(
            row["venue_lat"], row["venue_lon"],
            row["_date"], row["_local_hour"]
        )
        scores.append(sc)
        elevs.append(el)
        cats.append(ct)
    df["_sun_score"] = scores
    df["_sun_elev"]  = elevs
    df["_sun_cat"]   = cats

    # Interaction terms: sun × rank signal
    df["_sun_x_rdiff"]  = df["_sun_score"] * df["_rank_diff"]
    df["_sun_x_logrr"]  = df["_sun_score"] * df["_log_rank_ratio"]

    df = df[df["_year"] >= FIRST_YEAR].copy()
    print(f"  Ready: {len(df):,} outdoor Masters matches ({FIRST_YEAR}–{LAST_YEAR})")
    print(f"  GLARE: {(df['_sun_cat']=='GLARE').sum():,}  "
          f"MODERATE: {(df['_sun_cat']=='MODERATE').sum():,}  "
          f"LOW: {(df['_sun_cat']=='LOW').sum():,}\n")
    return df


# ── Model helpers ─────────────────────────────────────────────────────────────

BASE_FEATS = ["_rank_diff", "_log_rank_ratio",
              "_surface_hard", "_surface_clay", "_surface_grass"]
AUG_FEATS  = BASE_FEATS + ["_sun_score", "_sun_x_rdiff", "_sun_x_logrr"]


def fit_predict(X_tr, y_tr, X_te):
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)
    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_tr_s, y_tr)
    return clf.predict_proba(X_te_s)[:, 1]


def metrics(y_true, y_pred, label):
    bs  = brier_score_loss(y_true, y_pred)
    ll  = log_loss(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred) if len(set(y_true)) > 1 else 0.5
    return {"label": label, "brier": bs, "logloss": ll, "auc": auc}


# ── Calibration on GLARE subset ───────────────────────────────────────────────

def calibration_slope(y_true, y_pred):
    """Logistic calibration slope. 1.0 = perfectly calibrated, <1 = over-confident."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.clip(np.array(y_pred, dtype=float), 1e-6, 1-1e-6)
    log_odds = np.log(y_pred / (1 - y_pred))
    try:
        slope, _, _, p, _ = stats.linregress(log_odds, y_true)
        return float(slope), float(p)
    except Exception:
        return float("nan"), float("nan")


# ── WFO loop ──────────────────────────────────────────────────────────────────

def run_wfo(df):
    test_years  = list(range(FIRST_YEAR + TRAIN_YEARS, LAST_YEAR + 1))
    fold_results = []

    print(f"Walk-forward: {len(test_years)} folds  "
          f"(train={TRAIN_YEARS}yr rolling, test=1yr)\n")
    print(f"{'Year':>5}  {'N':>5}  {'N_Glare':>7}  "
          f"{'Base_BS':>9}  {'Aug_BS':>9}  {'BS_Lift':>9}  "
          f"{'Base_AUC':>9}  {'Aug_AUC':>9}  {'AUC_Lift':>9}  "
          f"{'Glare_calib_base':>18}  {'Glare_calib_aug':>17}")
    print("─" * 130)

    for test_year in test_years:
        tr_start = test_year - TRAIN_YEARS
        df_tr = df[(df["_year"] >= tr_start) & (df["_year"] < test_year)]
        df_te = df[df["_year"] == test_year]

        if len(df_te) < 20:
            continue

        y_tr = df_tr["_outcome"].values
        y_te = df_te["_outcome"].values

        X_tr_base = df_tr[BASE_FEATS].values
        X_te_base = df_te[BASE_FEATS].values
        X_tr_aug  = df_tr[AUG_FEATS].values
        X_te_aug  = df_te[AUG_FEATS].values

        p_base = fit_predict(X_tr_base, y_tr, X_te_base)
        p_aug  = fit_predict(X_tr_aug,  y_tr, X_te_aug)

        m_base = metrics(y_te, p_base, "base")
        m_aug  = metrics(y_te, p_aug,  "aug")

        bs_lift  = m_base["brier"] - m_aug["brier"]    # positive = aug better
        auc_lift = m_aug["auc"]    - m_base["auc"]

        # Calibration on GLARE subset only
        glare_mask = (df_te["_sun_cat"] == "GLARE").values
        if glare_mask.sum() >= 10:
            cs_base, _ = calibration_slope(y_te[glare_mask], p_base[glare_mask])
            cs_aug,  _ = calibration_slope(y_te[glare_mask], p_aug[glare_mask])
        else:
            cs_base = cs_aug = float("nan")

        fold_results.append({
            "year":           test_year,
            "n_total":        len(df_te),
            "n_glare":        int(glare_mask.sum()),
            "base_brier":     round(m_base["brier"],   5),
            "aug_brier":      round(m_aug["brier"],    5),
            "brier_lift":     round(bs_lift,            5),
            "base_logloss":   round(m_base["logloss"],  5),
            "aug_logloss":    round(m_aug["logloss"],   5),
            "logloss_lift":   round(m_base["logloss"] - m_aug["logloss"], 5),
            "base_auc":       round(m_base["auc"],      5),
            "aug_auc":        round(m_aug["auc"],       5),
            "auc_lift":       round(auc_lift,            5),
            "glare_calib_base": round(cs_base, 4) if not math.isnan(cs_base) else None,
            "glare_calib_aug":  round(cs_aug,  4) if not math.isnan(cs_aug)  else None,
        })

        bs_tag  = "✓" if bs_lift  > 0 else " "
        auc_tag = "✓" if auc_lift > 0 else " "
        calib_base_str = f"{cs_base:+.3f}" if not math.isnan(cs_base) else "  n/a"
        calib_aug_str  = f"{cs_aug:+.3f}"  if not math.isnan(cs_aug)  else "  n/a"

        print(f"{test_year:>5}  {len(df_te):>5}  {glare_mask.sum():>7}  "
              f"{m_base['brier']:>9.5f}  {m_aug['brier']:>9.5f}  "
              f"{bs_lift:>+8.5f}{bs_tag}  "
              f"{m_base['auc']:>9.5f}  {m_aug['auc']:>9.5f}  "
              f"{auc_lift:>+8.5f}{auc_tag}  "
              f"{calib_base_str:>18}  {calib_aug_str:>17}")

    return fold_results


# ── Summary & report ──────────────────────────────────────────────────────────

def report(fold_results):
    df = pd.DataFrame(fold_results)
    sep = "─" * 80

    pos_brier  = (df["brier_lift"] > 0).sum()
    pos_auc    = (df["auc_lift"]   > 0).sum()
    mean_brier = df["brier_lift"].mean()
    mean_auc   = df["auc_lift"].mean()

    # Paired t-test across folds: is the mean lift significantly > 0?
    _, p_brier = stats.ttest_1samp(df["brier_lift"].dropna(), 0)
    _, p_auc   = stats.ttest_1samp(df["auc_lift"].dropna(),   0)

    print(f"\n{'═'*80}")
    print("  INCREMENTAL ALPHA TEST — SUN ON TOP OF RANK/SURFACE MODEL")
    print(f"  {len(df)} OOS folds  |  {FIRST_YEAR+TRAIN_YEARS}–{LAST_YEAR}")
    print(f"{'═'*80}\n")

    print(f"  Metric              Mean lift   Pos folds   p-value (paired t)")
    print(sep)
    print(f"  Brier score       {mean_brier:>+10.5f}   "
          f"{pos_brier:>2}/{len(df)}       p={p_brier:.3f}"
          f"  {'✓ significant' if p_brier < 0.10 else ''}")
    print(f"  AUC               {mean_auc:>+10.5f}   "
          f"{pos_auc:>2}/{len(df)}       p={p_auc:.3f}"
          f"  {'✓ significant' if p_auc < 0.10 else ''}")
    print(f"  Log-loss          {df['logloss_lift'].mean():>+10.5f}   "
          f"{(df['logloss_lift']>0).sum():>2}/{len(df)}")

    # Calibration analysis: glare matches are miscalibrated by baseline?
    calib_df = df.dropna(subset=["glare_calib_base","glare_calib_aug"])
    if len(calib_df) > 0:
        mean_cb = calib_df["glare_calib_base"].mean()
        mean_ca = calib_df["glare_calib_aug"].mean()
        print(f"\n  Calibration slope on GLARE matches  (1.0 = perfect):")
        print(f"    Baseline model:   {mean_cb:>+.3f}  "
              f"({'over-confident' if mean_cb < 0.95 else 'well calibrated'})")
        print(f"    Augmented model:  {mean_ca:>+.3f}  "
              f"({'over-confident' if mean_ca < 0.95 else 'well calibrated'})")
        print(f"    Calibration gain: {mean_ca - mean_cb:>+.3f}  "
              f"({'✓ sun corrects calibration' if mean_ca > mean_cb else 'no improvement'})")

    print(f"\n{sep}")
    print("VERDICT")
    print(sep)

    brier_sig   = p_brier < 0.10 and mean_brier > 0
    auc_sig     = p_auc   < 0.10 and mean_auc   > 0
    brier_dir   = mean_brier > 0
    calib_gain  = (len(calib_df) > 0 and
                   calib_df["glare_calib_aug"].mean() > calib_df["glare_calib_base"].mean())

    if brier_sig or auc_sig:
        print("  ✓ SUN ADDS SIGNIFICANT INCREMENTAL ALPHA")
        print(f"    The augmented model (rank + surface + sun) outperforms the baseline")
        print(f"    on Brier score in {pos_brier}/{len(df)} OOS years (p={p_brier:.3f})")
        print(f"    and AUC in {pos_auc}/{len(df)} OOS years (p={p_auc:.3f}).")
        print(f"    Recommendation: include sun_score and sun×rank interaction")
        print(f"    as features in your XGBoost / model pipeline.")

    elif brier_dir and pos_brier >= len(df) * 0.6:
        print("  ~ WEAK BUT CONSISTENT INCREMENTAL SIGNAL")
        print(f"    Sun improves Brier score in {pos_brier}/{len(df)} OOS years,")
        print(f"    mean lift = {mean_brier*1e4:+.2f}×10⁻⁴ Brier points.")
        print(f"    Not statistically significant across folds (p={p_brier:.3f}),")
        print(f"    but the direction is consistent.")
        if calib_gain:
            print(f"    ✓ Calibration on GLARE matches does improve (+{(calib_df['glare_calib_aug'].mean()-calib_df['glare_calib_base'].mean()):.3f}).")
            print(f"    Recommendation: use sun as a CALIBRATION adjustment on outdoor")
            print(f"    hard/clay matches, not as a full standalone feature.")
        else:
            print(f"    Recommendation: keep sun penalty small (≤1%); monitor over")
            print(f"    more data before promoting to a primary feature.")

    else:
        print("  ✗ SUN DOES NOT ADD INCREMENTAL ALPHA")
        print(f"    The augmented model is not reliably better than rank+surface alone.")
        print(f"    Brier improvement in only {pos_brier}/{len(df)} OOS years (p={p_brier:.3f}).")
        print(f"    Recommendation: remove sun penalty from the model to avoid")
        print(f"    overfitting noise into your predictions.")

    print(f"{'═'*80}\n")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df    = load_and_prepare()
    folds = run_wfo(df)
    out   = report(folds)

    out_path = os.path.join(os.path.dirname(__file__), "wfo_sun_incremental.csv")
    out.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")


if __name__ == "__main__":
    main()
