"""
backtest.py — Walk-forward backtesting for the tennis mentality score pipeline.

Includes:
  - run_walk_forward: year-by-year Logistic Regression comparison (Elo vs Elo+Mental)
  - run_ablation: feature importance via leave-one-out re-scoring
  - run_permutation_test: statistical significance of mentality_diff
  - run_stratified: results by surface and tournament level
"""

import logging
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler

from score import build_mentality_scores

logger = logging.getLogger(__name__)

TEST_YEARS   = [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024]
TRAIN_START  = 2012
MIN_TEST_ROWS = 30

FEATURE_WEIGHTS_FULL = {
    "deciding_set_win_pct": 0.353,
    "tb_win_pct":           0.294,
    "bp_save_pct":          0.353,
}


def _fit_predict(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
) -> np.ndarray:
    """Fit StandardScaler + LogisticRegression, return test probabilities."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = LogisticRegression(max_iter=500)
        lr.fit(X_tr_s, y_train)
    return lr.predict_proba(X_te_s)[:, 1]


def run_walk_forward(matches_scored_df: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward backtest comparing Elo-only vs Elo+Mentality models.

    Parameters
    ----------
    matches_scored_df : pd.DataFrame
        Must contain columns: tourney_date, elo_diff, mentality_diff.
        The label is always 1 (winner perspective — winner always won).

    Returns
    -------
    pd.DataFrame with columns:
        year, n_test, brier_baseline, brier_full, improvement, acc_elo, acc_full
    """
    df = matches_scored_df.copy()
    df["year"] = pd.to_datetime(df["tourney_date"]).dt.year

    # Label: always 1 since rows represent winner's perspective
    df["label"] = 1

    rows = []
    for test_year in TEST_YEARS:
        try:
            train_mask = (df["year"] >= TRAIN_START) & (df["year"] < test_year)
            test_mask  = df["year"] == test_year

            train = df[train_mask].dropna(subset=["elo_diff", "mentality_diff"])
            test  = df[test_mask].dropna(subset=["elo_diff", "mentality_diff"])

            if len(test) < MIN_TEST_ROWS:
                logger.info("Skipping year %s: only %d test rows", test_year, len(test))
                continue
            if len(train) < MIN_TEST_ROWS:
                logger.info("Skipping year %s: only %d train rows", test_year, len(train))
                continue

            y_train = train["label"].values
            y_test  = test["label"].values

            # Baseline: Elo only
            X_tr_base = train[["elo_diff"]].values
            X_te_base = test[["elo_diff"]].values
            prob_base = _fit_predict(X_tr_base, y_train, X_te_base)

            # Full: Elo + mentality_diff
            X_tr_full = train[["elo_diff", "mentality_diff"]].values
            X_te_full = test[["elo_diff", "mentality_diff"]].values
            prob_full = _fit_predict(X_tr_full, y_train, X_te_full)

            brier_base = brier_score_loss(y_test, prob_base)
            brier_full = brier_score_loss(y_test, prob_full)
            improvement = brier_base - brier_full

            acc_elo  = float(np.mean((prob_base > 0.5) == y_test))
            acc_full = float(np.mean((prob_full > 0.5) == y_test))

            rows.append({
                "year":            test_year,
                "n_test":          len(test),
                "brier_baseline":  brier_base,
                "brier_full":      brier_full,
                "improvement":     improvement,
                "acc_elo":         acc_elo,
                "acc_full":        acc_full,
            })
        except Exception as exc:
            logger.warning("walk_forward failed for year %s: %s", test_year, exc)

    return pd.DataFrame(rows)


def _rebuild_mentality(
    matches_scored_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
    weights: Dict[str, float],
) -> pd.DataFrame:
    """Rebuild mentality scores using custom feature weights.

    Parameters
    ----------
    matches_scored_df : pd.DataFrame
        Matches with tourney_date and elo_diff.
    player_features_df : pd.DataFrame
        Player rolling features.
    weights : dict
        Mapping feature_name -> weight (must sum to 1.0).

    Returns
    -------
    matches_scored_df copy with updated mentality_diff column.
    """
    # Temporarily monkey-patch score module weights
    import score as score_module
    original_weights = dict(score_module.FEATURE_WEIGHTS)
    score_module.FEATURE_WEIGHTS = weights
    try:
        # Strip existing mentality columns
        base_cols = [c for c in matches_scored_df.columns
                     if c not in ("winner_mentality", "loser_mentality", "mentality_diff")]
        base_df = matches_scored_df[base_cols].copy()
        rescored = build_mentality_scores(base_df, player_features_df)
        rescored["mentality_diff"] = rescored["winner_mentality"] - rescored["loser_mentality"]
    finally:
        score_module.FEATURE_WEIGHTS = original_weights
    return rescored


def run_ablation(
    matches_scored_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Feature importance via leave-one-out ablation.

    For each feature, rebuild mentality scores without that feature,
    run walk_forward, and measure the brier cost of its removal.

    Returns
    -------
    pd.DataFrame with columns: feature, brier_ablated, importance, rank
    """
    # Baseline mean brier from full model
    wf_full = run_walk_forward(matches_scored_df)
    if wf_full.empty:
        return pd.DataFrame(columns=["feature", "brier_ablated", "importance", "rank"])
    mean_brier_full = wf_full["brier_full"].mean()

    all_features = list(FEATURE_WEIGHTS_FULL.keys())
    rows = []

    for ablated_feat in all_features:
        try:
            # Build weights without the ablated feature and renormalize
            remaining = {f: w for f, w in FEATURE_WEIGHTS_FULL.items() if f != ablated_feat}
            total = sum(remaining.values())
            if total <= 0:
                continue
            normalized = {f: w / total for f, w in remaining.items()}
            # Set ablated feature weight to 0 (will be included but zero-weighted)
            normalized[ablated_feat] = 0.0

            rescored = _rebuild_mentality(matches_scored_df, player_features_df, normalized)
            wf_abl = run_walk_forward(rescored)
            if wf_abl.empty:
                continue
            mean_brier_abl = wf_abl["brier_full"].mean()
            importance = mean_brier_abl - mean_brier_full  # positive = removing hurt
            rows.append({
                "feature":       ablated_feat,
                "brier_ablated": mean_brier_abl,
                "importance":    importance,
            })
        except Exception as exc:
            logger.warning("Ablation failed for feature %s: %s", ablated_feat, exc)

    if not rows:
        return pd.DataFrame(columns=["feature", "brier_ablated", "importance", "rank"])

    abl_df = pd.DataFrame(rows)
    abl_df = abl_df.sort_values("importance", ascending=False).reset_index(drop=True)
    abl_df["rank"] = abl_df.index + 1
    return abl_df


def run_permutation_test(
    matches_scored_df: pd.DataFrame,
    n_shuffles: int = 500,
) -> dict:
    """Permutation test: shuffle mentality_diff to assess statistical significance.

    Parameters
    ----------
    matches_scored_df : pd.DataFrame
    n_shuffles : int
        Number of shuffles (default 500).

    Returns
    -------
    dict with keys: real_brier, null_mean, null_std, p_value
    """
    # Real result
    wf_real = run_walk_forward(matches_scored_df)
    if wf_real.empty:
        return {"real_brier": np.nan, "null_mean": np.nan, "null_std": np.nan, "p_value": np.nan}

    real_brier = wf_real["brier_full"].mean()

    # Permutation loop — shuffle mentality_diff each time
    null_briers = []
    rng = np.random.default_rng(seed=42)
    mentality_vals = matches_scored_df["mentality_diff"].values.copy()

    for _ in range(n_shuffles):
        try:
            shuffled = matches_scored_df.copy()
            shuffled["mentality_diff"] = rng.permutation(mentality_vals)
            wf_shuf = run_walk_forward(shuffled)
            if not wf_shuf.empty:
                null_briers.append(wf_shuf["brier_full"].mean())
        except Exception as exc:
            logger.debug("Permutation shuffle failed: %s", exc)

    if not null_briers:
        return {
            "real_brier": real_brier,
            "null_mean":  np.nan,
            "null_std":   np.nan,
            "p_value":    np.nan,
        }

    null_arr = np.array(null_briers)
    null_mean = float(np.mean(null_arr))
    null_std  = float(np.std(null_arr))
    # p_value: fraction of shuffled mean brier <= real mean brier
    p_value = float(np.mean(null_arr <= real_brier))

    return {
        "real_brier": real_brier,
        "null_mean":  null_mean,
        "null_std":   null_std,
        "p_value":    p_value,
    }


def _level_group(level) -> str:
    """Map tourney_level to G, M, or A."""
    if level == "G":
        return "G"
    if level == "M":
        return "M"
    return "A"


def run_stratified(matches_scored_df: pd.DataFrame) -> pd.DataFrame:
    """Run walk_forward separately by surface and tournament level.

    Surfaces: Hard, Clay, Grass
    Levels:   G, M, A (everything else mapped to A)

    Returns
    -------
    pd.DataFrame with columns: stratum, brier_baseline, brier_full, improvement
    """
    df = matches_scored_df.copy()

    # Normalize level column
    if "tourney_level" in df.columns:
        df["_level_group"] = df["tourney_level"].apply(_level_group)
    else:
        df["_level_group"] = "A"

    if "surface" not in df.columns:
        df["surface"] = "Hard"

    strata = []

    # Surface strata
    for surf in ["Hard", "Clay", "Grass"]:
        subset = df[df["surface"] == surf]
        try:
            wf = run_walk_forward(subset)
            if wf.empty:
                continue
            strata.append({
                "stratum":         f"{surf} court",
                "brier_baseline":  wf["brier_baseline"].mean(),
                "brier_full":      wf["brier_full"].mean(),
                "improvement":     wf["improvement"].mean(),
            })
        except Exception as exc:
            logger.warning("Stratified failed for surface %s: %s", surf, exc)

    # Level strata
    level_labels = {"G": "Grand Slams", "M": "Masters", "A": "Other"}
    for lv, label in level_labels.items():
        subset = df[df["_level_group"] == lv]
        try:
            wf = run_walk_forward(subset)
            if wf.empty:
                continue
            strata.append({
                "stratum":         label,
                "brier_baseline":  wf["brier_baseline"].mean(),
                "brier_full":      wf["brier_full"].mean(),
                "improvement":     wf["improvement"].mean(),
            })
        except Exception as exc:
            logger.warning("Stratified failed for level %s: %s", lv, exc)

    return pd.DataFrame(strata)
