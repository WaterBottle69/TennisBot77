"""
features.py — Rolling feature computation for the tennis mentality score pipeline.

All features use strict no-lookahead rolling windows (shift by 1 match per player).
Feature 4 (challenge_success_rate) is always set to 0.5 as challenge data is unavailable.
"""

import logging
import re
from typing import List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum number of prior matches before a feature is meaningful
MIN_PERIODS = 10

# Rolling window sizes in days
DECIDING_SET_WINDOW_DAYS = 1095  # ~3 years
TB_WINDOW_DAYS = 730            # ~2 years
BP_WINDOW_DAYS = 730            # ~2 years


def _parse_sets(score: str) -> List[str]:
    """Split a score string by spaces and filter out non-set tokens.

    Non-set tokens include tiebreak annotations like "(7)" or "(10)".
    """
    if not isinstance(score, str):
        return []
    tokens = score.split()
    # Keep tokens that look like a set score: digits-digits, possibly with (n) suffix
    # e.g. "6-4", "7-6(3)", "6-7(5)"
    set_pattern = re.compile(r"^\d+-\d+(\(\d+\))?$")
    return [t for t in tokens if set_pattern.match(t)]


def _is_deciding_set_match(score: str, best_of) -> bool:
    """Return True if the match went to the deciding set.

    best_of=3  → deciding set = 3rd set played
    best_of=5  → deciding set = 5th set played
    """
    try:
        bo = int(best_of)
    except (TypeError, ValueError):
        bo = 3

    sets = _parse_sets(score)
    n_sets = len(sets)
    if bo == 5:
        return n_sets == 5
    # Default to best_of=3
    return n_sets == 3


def _has_tiebreak(score: str) -> bool:
    """Return True if any set in the score was decided by a tiebreak (7-6 or 6-7)."""
    sets = _parse_sets(score)
    for s in sets:
        base = s.split("(")[0]
        if base in ("7-6", "6-7"):
            return True
    return False


def compute_player_features(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling per-player mentality features with strict no-lookahead.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Full match DataFrame sorted by tourney_date ascending.
        Required columns: winner_id, loser_id, tourney_date, score, best_of,
                          w_bpSaved, w_bpFaced, l_bpSaved, l_bpFaced.

    Returns
    -------
    pd.DataFrame with columns:
        match_idx, player_id,
        deciding_set_win_pct, tb_win_pct, bp_save_pct, challenge_success_rate
    """
    required_cols = {
        "winner_id", "loser_id", "tourney_date", "score", "best_of",
        "w_bpSaved", "w_bpFaced", "l_bpSaved", "l_bpFaced",
    }
    missing = required_cols - set(matches_df.columns)
    if missing:
        logger.warning("compute_player_features: missing columns %s", missing)
        # Add missing columns as NaN
        for col in missing:
            matches_df = matches_df.copy()
            matches_df[col] = np.nan

    df = matches_df.copy()
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df["match_idx"] = df.index

    # Pre-compute per-match flags
    df["_has_deciding"] = df.apply(
        lambda r: _is_deciding_set_match(r["score"], r.get("best_of", 3)), axis=1
    )
    df["_has_tb"] = df["score"].apply(_has_tiebreak)

    # Coerce bp columns to numeric
    for col in ["w_bpSaved", "w_bpFaced", "l_bpSaved", "l_bpFaced"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # -----------------------------------------------------------------------
    # Melt into long format: one row per player per match
    # -----------------------------------------------------------------------
    winner_rows = pd.DataFrame({
        "match_idx":      df["match_idx"].values,
        "player_id":      pd.to_numeric(df["winner_id"], errors="coerce"),
        "match_date":     df["tourney_date"].values,
        "won":            True,
        "had_deciding":   df["_has_deciding"].values,
        "had_tb":         df["_has_tb"].values,
        "bp_saved":       df["w_bpSaved"].values,
        "bp_faced":       df["w_bpFaced"].values,
    })

    loser_rows = pd.DataFrame({
        "match_idx":      df["match_idx"].values,
        "player_id":      pd.to_numeric(df["loser_id"], errors="coerce"),
        "match_date":     df["tourney_date"].values,
        "won":            False,
        "had_deciding":   df["_has_deciding"].values,
        "had_tb":         df["_has_tb"].values,
        "bp_saved":       df["l_bpSaved"].values,
        "bp_faced":       df["l_bpFaced"].values,
    })

    long_df = pd.concat([winner_rows, loser_rows], ignore_index=True)
    long_df = long_df.dropna(subset=["player_id"])
    long_df["player_id"] = long_df["player_id"].astype(int)
    long_df = long_df.sort_values(["player_id", "match_date"]).reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Rolling feature computation per player
    # -----------------------------------------------------------------------
    # We compute rolling stats using date-based windows.
    # For each player, we shift by 1 to exclude the current match.

    results = []

    for player_id, grp in long_df.groupby("player_id", sort=False):
        grp = grp.sort_values("match_date").reset_index(drop=True)
        n = len(grp)

        match_dates = grp["match_date"].values
        match_idxs  = grp["match_idx"].values
        won_arr     = grp["won"].values.astype(float)
        deciding    = grp["had_deciding"].values.astype(float)
        tb          = grp["had_tb"].values.astype(float)
        bp_saved    = grp["bp_saved"].values
        bp_faced    = grp["bp_faced"].values

        # Safe bp_save_pct per row (fraction for this match)
        bp_save_row = np.where(
            bp_faced > 0, bp_saved / bp_faced, np.nan
        )

        dec_win_pct_arr = np.full(n, np.nan)
        tb_win_pct_arr  = np.full(n, np.nan)
        bp_save_pct_arr = np.full(n, np.nan)

        for i in range(n):
            cur_date = match_dates[i]

            # Window for deciding set: 1095 days before current match
            ds_cutoff = cur_date - pd.Timedelta(days=DECIDING_SET_WINDOW_DAYS)
            # Window for tiebreak / bp: 730 days before current match
            tb_cutoff = cur_date - pd.Timedelta(days=TB_WINDOW_DAYS)
            bp_cutoff = cur_date - pd.Timedelta(days=BP_WINDOW_DAYS)

            # Only use PAST matches (indices < i)
            if i < MIN_PERIODS:
                # Fewer than MIN_PERIODS prior matches — leave as NaN
                continue

            # Past match indices
            past_dates   = match_dates[:i]
            past_won     = won_arr[:i]
            past_dec     = deciding[:i]
            past_tb      = tb[:i]
            past_bp_save = bp_save_row[:i]

            # Feature 1: deciding_set_win_pct
            mask_ds = (past_dates >= ds_cutoff) & (past_dec == 1.0)
            n_ds = mask_ds.sum()
            if n_ds >= MIN_PERIODS:
                dec_win_pct_arr[i] = past_won[mask_ds].mean()

            # Feature 2: tb_win_pct
            mask_tb = (past_dates >= tb_cutoff) & (past_tb == 1.0)
            n_tb = mask_tb.sum()
            if n_tb >= MIN_PERIODS:
                tb_win_pct_arr[i] = past_won[mask_tb].mean()

            # Feature 3: bp_save_pct — mean of individual match bp save rates
            mask_bp = past_dates >= bp_cutoff
            bp_vals_window = past_bp_save[mask_bp]
            valid_bp = bp_vals_window[~np.isnan(bp_vals_window)]
            if len(valid_bp) >= MIN_PERIODS:
                bp_save_pct_arr[i] = valid_bp.mean()

        player_result = pd.DataFrame({
            "match_idx":               match_idxs,
            "player_id":               player_id,
            "deciding_set_win_pct":    dec_win_pct_arr,
            "tb_win_pct":              tb_win_pct_arr,
            "bp_save_pct":             bp_save_pct_arr,
        })
        results.append(player_result)

    if not results:
        return pd.DataFrame(columns=[
            "match_idx", "player_id",
            "deciding_set_win_pct", "tb_win_pct", "bp_save_pct",
            "challenge_success_rate",
        ])

    out = pd.concat(results, ignore_index=True)
    # Feature 4: challenge_success_rate — always 0.5 (no data available)
    out["challenge_success_rate"] = 0.5
    return out
