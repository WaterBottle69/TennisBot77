"""
main.py — Orchestrator for the tennis mentality score backtest pipeline.

Run:  python main.py

Steps:
  1. Ingest  — download & cache ATP match CSVs 2011–2024 + players + SLAM PBP
  2. Elo     — build Elo history from all matches
  3. Features — compute rolling per-player features (no lookahead)
  4. Score   — build mentality scores per match
  5. Elo diff — attach pre-match elo_diff to each row
  6. Sanity  — career-average mentality scores for selected players
  7. Backtest — walk-forward, ablation, permutation (500), stratified
  8. Report  — print to terminal + save results.json
"""

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure the package root is on the path so imports work both from repo root
# and from inside the tennis_backtest/ directory
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

import ingest
import elo as elo_module
import features as feat_module
import score as score_module
import backtest as bt_module
import report as report_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")

ATP_YEARS   = list(range(2011, 2025))
SLAM_YEARS  = list(range(2014, 2024))
SLAM_NAMES  = ["ausopen", "frenchopen", "wimbledon", "usopen"]

SANITY_NAMES = {
    "Djokovic":  "Djokovic",
    "Nadal":     "Nadal",
    "Federer":   "Federer",
    "Murray":    "Murray",
    "Kyrgios":   "Kyrgios",
    "Medvedev":  "Medvedev",
}


def _attach_elo_diff(matches_df: pd.DataFrame, elo_sys: elo_module.EloSystem) -> pd.DataFrame:
    """Add elo_diff column = winner pre-match Elo minus loser pre-match Elo."""
    df = matches_df.copy()
    winner_elos = []
    loser_elos  = []

    for row in df.itertuples(index=False):
        try:
            w_id = int(row.winner_id)
            l_id = int(row.loser_id)
            date = row.tourney_date
        except Exception:
            winner_elos.append(np.nan)
            loser_elos.append(np.nan)
            continue

        winner_elos.append(elo_sys.get_elo_before_match(w_id, date))
        loser_elos.append(elo_sys.get_elo_before_match(l_id, date))

    df["winner_elo"] = winner_elos
    df["loser_elo"]  = loser_elos
    df["elo_diff"]   = df["winner_elo"] - df["loser_elo"]
    return df


def _sanity_check(matches_scored: pd.DataFrame) -> dict[str, float]:
    """Compute career-average mentality score for a set of notable players."""
    results: dict[str, float] = {}

    for short_name, display in SANITY_NAMES.items():
        # Match on winner_name or loser_name containing the last name
        mask_w = matches_scored["winner_name"].str.contains(
            short_name, case=False, na=False
        )
        mask_l = matches_scored["loser_name"].str.contains(
            short_name, case=False, na=False
        )

        winner_scores = matches_scored.loc[mask_w, "winner_mentality"].dropna()
        loser_scores  = matches_scored.loc[mask_l, "loser_mentality"].dropna()
        all_scores    = pd.concat([winner_scores, loser_scores])

        if all_scores.empty:
            results[display] = float("nan")
        else:
            results[display] = float(all_scores.mean())

    return results


def main() -> None:
    logger.info("=== Tennis Mentality Score Backtest Pipeline ===")

    # ------------------------------------------------------------------
    # 1. Ingest
    # ------------------------------------------------------------------
    logger.info("Step 1/8 — Ingesting ATP matches 2011–2024 ...")
    matches_df = ingest.load_atp_matches(ATP_YEARS)
    if matches_df.empty:
        logger.error("No ATP match data loaded. Check network connectivity.")
        sys.exit(1)
    logger.info("  Loaded %d matches", len(matches_df))

    logger.info("Step 1b — Ingesting SLAM PBP (skipping 404s silently) ...")
    _slam_df = ingest.load_slam_pbp(SLAM_YEARS, SLAM_NAMES)
    logger.info(
        "  Loaded %d SLAM PBP rows%s",
        len(_slam_df),
        " (challenge columns unavailable — Feature 4 excluded)" if not _slam_df.empty else "",
    )

    # ------------------------------------------------------------------
    # 2. Elo system
    # ------------------------------------------------------------------
    logger.info("Step 2/8 — Building Elo ratings ...")
    elo_sys = elo_module.EloSystem(k=32, initial=1500.0)
    elo_sys.build_from_matches(matches_df)
    logger.info("  Elo built for %d players", len(elo_sys._ratings))

    # ------------------------------------------------------------------
    # 3. Rolling features
    # ------------------------------------------------------------------
    logger.info("Step 3/8 — Computing rolling player features (no lookahead) ...")
    logger.info("  This may take a few minutes for 14 years of data ...")
    player_features_df = feat_module.compute_player_features(matches_df)
    logger.info("  Computed %d player-match feature rows", len(player_features_df))

    # ------------------------------------------------------------------
    # 4. Mentality scores
    # ------------------------------------------------------------------
    logger.info("Step 4/8 — Building mentality scores per match ...")
    matches_scored = score_module.build_mentality_scores(matches_df, player_features_df)
    logger.info(
        "  Mentality scores computed (non-null winner: %d, loser: %d)",
        matches_scored["winner_mentality"].notna().sum(),
        matches_scored["loser_mentality"].notna().sum(),
    )

    # ------------------------------------------------------------------
    # 5. Attach Elo diff
    # ------------------------------------------------------------------
    logger.info("Step 5/8 — Attaching pre-match Elo diff ...")
    matches_scored = _attach_elo_diff(matches_scored, elo_sys)
    logger.info(
        "  elo_diff non-null rows: %d", matches_scored["elo_diff"].notna().sum()
    )

    # ------------------------------------------------------------------
    # 6. Sanity check
    # ------------------------------------------------------------------
    logger.info("Step 6/8 — Sanity check on notable players ...")
    sanity = _sanity_check(matches_scored)
    for player, val in sanity.items():
        logger.info("  %-22s %.1f", player, val if not np.isnan(val) else -1)

    # ------------------------------------------------------------------
    # 7. Backtest
    # ------------------------------------------------------------------
    logger.info("Step 7/8 — Walk-forward backtest ...")
    wf_df = bt_module.run_walk_forward(matches_scored)
    logger.info("  Walk-forward complete: %d test years", len(wf_df))

    logger.info("  Running ablation ...")
    ablation_df = bt_module.run_ablation(matches_scored, player_features_df)

    logger.info("  Running permutation test (500 shuffles) — this takes ~1–2 min ...")
    perm_dict = bt_module.run_permutation_test(matches_scored, n_shuffles=500)

    logger.info("  Running stratified analysis ...")
    strat_df = bt_module.run_stratified(matches_scored)

    # ------------------------------------------------------------------
    # 8. Report
    # ------------------------------------------------------------------
    logger.info("Step 8/8 — Printing report ...")
    skipped = ingest.get_skipped() if hasattr(ingest, "get_skipped") else []

    report_module.print_report(
        wf_df       = wf_df,
        ablation_df = ablation_df,
        perm_dict   = perm_dict,
        strat_df    = strat_df,
        sanity_dict = sanity,
        skipped_list= skipped,
    )

    results_path = HERE / "results.json"
    report_module.save_results_json(wf_df, ablation_df, perm_dict, strat_df, results_path)
    logger.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
