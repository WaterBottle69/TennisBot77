"""
report.py — Print and save the backtest report for the tennis mentality score pipeline.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BAR_CHAR   = "█"
BAR_SCALE  = 40   # max bar length in characters
DIVIDER    = "=" * 70
THIN_DIV   = "-" * 70


def _bar(value: float, min_val: float = 0.0, max_val: float = 1.0, width: int = 16) -> str:
    """Return a simple ASCII bar proportional to value within [min_val, max_val]."""
    if max_val <= min_val or np.isnan(value):
        return ""
    ratio = max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))
    n = int(round(ratio * width))
    return BAR_CHAR * n


def _improvement_bar(improvement: float, width: int = 8) -> str:
    """Bar for improvement values (can be small positive/negative numbers)."""
    if np.isnan(improvement):
        return ""
    # Scale: map improvement to a visible bar; treat 0.005 as max
    ratio = max(0.0, min(1.0, improvement / 0.005))
    n = int(round(ratio * width))
    return BAR_CHAR * n


def print_report(
    wf_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    perm_dict: Dict,
    strat_df: pd.DataFrame,
    sanity_dict: Dict[str, float],
    skipped_list: List[str],
) -> None:
    """Print the full backtest report to stdout."""

    print()
    print(DIVIDER)
    print("  TENNIS MENTALITY SCORE — BACKTEST REPORT")
    print(DIVIDER)

    # -----------------------------------------------------------------------
    # SANITY CHECK
    # -----------------------------------------------------------------------
    print()
    print("▌ SANITY CHECK — career average mentality scores")

    if sanity_dict:
        max_score = max(v for v in sanity_dict.values() if not np.isnan(v)) if sanity_dict else 100.0
        for player, score in sorted(sanity_dict.items(), key=lambda x: -x[1] if not np.isnan(x[1]) else -999):
            bar = _bar(score, 0, 100, width=16)
            flag = ""
            if player in ("Djokovic", "Nadal") and score < 60:
                flag = "  ⚠ BELOW EXPECTED"
            if player == "Kyrgios" and score > 50:
                flag = "  ⚠ ABOVE EXPECTED"
            print(f"  {player:<22} {score:>5.1f}  {bar}{flag}")
    else:
        print("  (no sanity data available)")

    if skipped_list:
        print()
        print("  Skipped downloads:")
        for s in skipped_list:
            print(f"    - {s}")

    # -----------------------------------------------------------------------
    # WALK-FORWARD RESULTS
    # -----------------------------------------------------------------------
    print()
    print("▌ WALK-FORWARD RESULTS (Brier score — lower is better)")
    print(
        f"{'Year':<8}{'N Test':<10}{'Elo Only':<14}{'Elo+Mental':<14}"
        f"{'Improvement':<14}{'Acc Elo':<10}{'Acc Full':<10}"
    )
    print(THIN_DIV)

    if not wf_df.empty:
        for _, row in wf_df.iterrows():
            imp = row["improvement"]
            imp_str = f"+{imp:.5f}" if imp >= 0 else f"{imp:.5f}"
            check = " ✓" if imp > 0 else " ✗"
            print(
                f"{int(row['year']):<8}"
                f"{int(row['n_test']):<10}"
                f"{row['brier_baseline']:<14.5f}"
                f"{row['brier_full']:<14.5f}"
                f"{imp_str:<14}"
                f"{row['acc_elo'] * 100:.1f}%{'':<4}"
                f"{row['acc_full'] * 100:.1f}%{check}"
            )

        # Averages row
        avg_base = wf_df["brier_baseline"].mean()
        avg_full = wf_df["brier_full"].mean()
        avg_imp  = wf_df["improvement"].mean()
        avg_imp_str = f"+{avg_imp:.5f}" if avg_imp >= 0 else f"{avg_imp:.5f}"
        print(THIN_DIV)
        print(
            f"{'AVERAGE':<8}{'':<10}"
            f"{avg_base:<14.5f}"
            f"{avg_full:<14.5f}"
            f"{avg_imp_str:<14}"
        )
    else:
        print("  (no walk-forward results)")

    # -----------------------------------------------------------------------
    # ABLATION
    # -----------------------------------------------------------------------
    print()
    print("▌ ABLATION — feature importance (Brier cost of removing each feature)")

    if not ablation_df.empty:
        for _, row in ablation_df.iterrows():
            imp_str = f"+{row['importance']:.6f}" if row['importance'] >= 0 else f"{row['importance']:.6f}"
            bar = _improvement_bar(row["importance"])
            print(f"  {row['feature']:<28} {imp_str:<12} {bar:<10}  rank {int(row['rank'])}")
    else:
        print("  (no ablation results)")

    # -----------------------------------------------------------------------
    # PERMUTATION TEST
    # -----------------------------------------------------------------------
    print()
    print("▌ PERMUTATION TEST")

    real_b  = perm_dict.get("real_brier", np.nan)
    null_m  = perm_dict.get("null_mean",  np.nan)
    null_s  = perm_dict.get("null_std",   np.nan)
    p_val   = perm_dict.get("p_value",    np.nan)

    print(f"  Real Brier:        {real_b:.6f}" if not np.isnan(real_b) else "  Real Brier:        N/A")
    if not np.isnan(null_m) and not np.isnan(null_s):
        print(f"  Null mean ± std:   {null_m:.6f} ± {null_s:.6f}")
    else:
        print("  Null mean ± std:   N/A")
    if not np.isnan(p_val):
        sig_str = "SIGNIFICANT ✓" if p_val < 0.05 else "NOT SIGNIFICANT ✗"
        print(f"  p-value:           {p_val:.4f}")
        print(f"  Result:            {sig_str}")
    else:
        print("  p-value:           N/A")

    # -----------------------------------------------------------------------
    # STRATIFIED RESULTS
    # -----------------------------------------------------------------------
    print()
    print("▌ STRATIFIED RESULTS")

    if not strat_df.empty:
        for _, row in strat_df.iterrows():
            imp = row["improvement"]
            imp_str = f"+{imp:.6f}" if imp >= 0 else f"{imp:.6f}"
            bar = _improvement_bar(imp)
            print(f"  {row['stratum']:<22} {imp_str:<12} {bar}")
    else:
        print("  (no stratified results)")

    # -----------------------------------------------------------------------
    # INTERPRETATION
    # -----------------------------------------------------------------------
    print()
    print("▌ INTERPRETATION")
    print("  Brier improvement > 0.001  →  meaningful signal")
    print("  Brier improvement > 0.003  →  strong signal")
    print("  p < 0.05                   →  statistically significant")

    print()
    print(DIVIDER)
    print("  Data: JeffSackmann/tennis_atp (CC BY-NC-SA)")
    print("  Note: SLAM PBP challenge data unavailable — Feature 4 excluded")
    print(DIVIDER)
    print()


def save_results_json(
    wf_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    perm_dict: Dict,
    strat_df: pd.DataFrame,
    path: Path,
) -> None:
    """Serialize all backtest results to a JSON file at the given path."""
    def _df_to_records(df: pd.DataFrame):
        if df is None or df.empty:
            return []
        # Convert to Python native types
        records = []
        for rec in df.to_dict(orient="records"):
            clean = {}
            for k, v in rec.items():
                if isinstance(v, float) and np.isnan(v):
                    clean[k] = None
                elif hasattr(v, "item"):
                    clean[k] = v.item()
                else:
                    clean[k] = v
            records.append(clean)
        return records

    def _clean_dict(d: dict):
        clean = {}
        for k, v in d.items():
            if isinstance(v, float) and np.isnan(v):
                clean[k] = None
            elif hasattr(v, "item"):
                clean[k] = v.item()
            else:
                clean[k] = v
        return clean

    payload = {
        "walk_forward":    _df_to_records(wf_df),
        "ablation":        _df_to_records(ablation_df),
        "permutation":     _clean_dict(perm_dict) if perm_dict else {},
        "stratified":      _df_to_records(strat_df),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    logger.info("Results saved to %s", path)
