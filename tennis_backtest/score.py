"""
score.py — Build per-match mentality scores from rolling player features.

Weights (reweighted to sum to 1.0 since Feature 4 is absent):
  deciding_set_win_pct : 0.353
  tb_win_pct           : 0.294
  bp_save_pct          : 0.353
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_WEIGHTS = {
    "deciding_set_win_pct": 0.353,
    "tb_win_pct":           0.294,
    "bp_save_pct":          0.353,
}

FILL_VALUE = 0.5  # Fill for players with < 10 prior matches


def _year_percentile_ranks(
    features_df: pd.DataFrame,
    matches_df: pd.DataFrame,
    feature_names: list,
) -> pd.DataFrame:
    """Compute year-level percentile rank for each feature.

    For each year, rank all player values for that year among all active players.
    'Active' means the player appeared in at least one match that year.

    Returns a copy of features_df with added columns: {feat}_pct for each feat.
    """
    # Add year to features_df from matches_df
    date_map = matches_df["tourney_date"].rename("match_date")
    match_year = matches_df[["tourney_date"]].copy()
    match_year["match_year"] = match_year["tourney_date"].dt.year
    match_year["match_idx"] = matches_df.index

    feat = features_df.copy()
    feat = feat.merge(
        match_year[["match_idx", "match_year"]],
        on="match_idx",
        how="left",
    )

    for fname in feature_names:
        pct_col = f"{fname}_pct"
        # Fill NaN feature values with FILL_VALUE before ranking
        feat[fname] = feat[fname].fillna(FILL_VALUE)
        # For each year, compute percentile rank (0..1)
        feat[pct_col] = feat.groupby("match_year")[fname].rank(pct=True)
        # Fill any remaining NaN percentiles with 0.5
        feat[pct_col] = feat[pct_col].fillna(0.5)

    return feat


def build_mentality_scores(
    matches_df: pd.DataFrame,
    player_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach winner/loser mentality scores to each match row.

    Steps:
    1. Look up winner and loser feature values per match.
    2. Compute year-level percentile rank for each feature.
    3. Compute weighted-average mentality score (0–100).
    4. Attach winner_mentality, loser_mentality, mentality_diff.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Full ATP matches DataFrame (already indexed 0..N-1).
    player_features_df : pd.DataFrame
        Output of features.compute_player_features().

    Returns
    -------
    pd.DataFrame (copy of matches_df) with added columns:
        winner_mentality, loser_mentality, mentality_diff
    """
    feature_names = list(FEATURE_WEIGHTS.keys())

    if player_features_df.empty:
        out = matches_df.copy()
        out["winner_mentality"] = np.nan
        out["loser_mentality"] = np.nan
        out["mentality_diff"] = np.nan
        return out

    # Ensure match_idx aligns with DataFrame index
    mdf = matches_df.copy().reset_index(drop=True)
    mdf["match_idx"] = mdf.index

    # Compute percentile ranks for all player-match rows
    feat_ranked = _year_percentile_ranks(player_features_df, mdf, feature_names)

    def _weighted_score(row: pd.Series) -> float:
        total = 0.0
        for fname, weight in FEATURE_WEIGHTS.items():
            pct_col = f"{fname}_pct"
            val = row.get(pct_col, 0.5)
            if pd.isna(val):
                val = 0.5
            total += val * weight
        return total * 100.0

    feat_ranked["mentality_score"] = feat_ranked.apply(_weighted_score, axis=1)

    # Split by winner/loser using the original winner_id/loser_id columns
    winner_ids = mdf.set_index("match_idx")["winner_id"].to_dict()
    loser_ids  = mdf.set_index("match_idx")["loser_id"].to_dict()

    # Build match_idx -> player_id -> score lookup
    score_lookup = (
        feat_ranked[["match_idx", "player_id", "mentality_score"]]
        .set_index(["match_idx", "player_id"])["mentality_score"]
    )

    def _lookup(match_idx, player_id):
        try:
            pid = int(player_id) if not pd.isna(player_id) else None
            if pid is None:
                return np.nan
            return score_lookup.loc[(match_idx, pid)]
        except (KeyError, TypeError, ValueError):
            return np.nan

    mdf["winner_mentality"] = [
        _lookup(idx, winner_ids.get(idx)) for idx in mdf["match_idx"]
    ]
    mdf["loser_mentality"] = [
        _lookup(idx, loser_ids.get(idx)) for idx in mdf["match_idx"]
    ]
    mdf["mentality_diff"] = mdf["winner_mentality"] - mdf["loser_mentality"]

    # Drop helper column
    mdf = mdf.drop(columns=["match_idx"])
    return mdf
