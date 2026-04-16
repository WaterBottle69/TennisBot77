"""
wfo.py — Deep Walk-Forward Optimization
========================================
Expanding-window + rolling-window WFO across ATP data 2015–2025.

Per fold:
  • Grid-search (min_edge × kelly_frac × vig) optimised for Sharpe ratio
  • Out-of-sample evaluation with best params
  • Full per-bet records stored for post-hoc analysis

Outputs
-------
  wfo_bets.csv          – every OOS bet (date, model_prob, edge, pnl, …)
  wfo_optimal_params.json – best params per fold + global consensus
  wfo_report.pdf        – 15+ seaborn heatmaps & diagnostic charts
  wfo_interactive.html  – Plotly P&L + Sharpe dashboard
"""

import os
import sys
import json
import logging
import warnings
import itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Try importing from the sister file ──────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from backtester_kelly import load_atp_matches, build_features, ELO_INIT, ELO_K

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ATP_DIR       = os.path.expanduser("~/Downloads/tennis_atp-master")
MODEL_PATH    = os.path.join(_DIR, "best_xgb_model.json")
FEATURES_PATH = os.path.join(_DIR, "model_features.json")

PDF_OUT       = os.path.join(_DIR, "wfo_report.pdf")
CSV_OUT       = os.path.join(_DIR, "wfo_bets.csv")
JSON_OUT      = os.path.join(_DIR, "wfo_optimal_params.json")
HTML_OUT      = os.path.join(_DIR, "wfo_interactive.html")

# ─────────────────────────────────────────────────────────────────────────────
# Parameter grid (keep small — 5×5×4 = 100 combos per fold)
# ─────────────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "min_edge":   [0.02, 0.04, 0.06, 0.08, 0.10],
    "kelly_frac": [0.05, 0.10, 0.20, 0.35, 0.50],
    "vig":        [0.05, 0.07, 0.09, 0.11],
}

BANKROLL_INIT = 500.0
MAX_BET_FRAC  = 0.20   # cap on fraction of bankroll per bet


# ─────────────────────────────────────────────────────────────────────────────
# 1. Simulation engine — returns per-bet DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def simulate_betting_full(
    dates:        np.ndarray,
    model_prob:   np.ndarray,
    market_prob:  np.ndarray,
    outcomes:     np.ndarray,
    surfaces:     np.ndarray,
    p1_ranks:     np.ndarray,
    bankroll_init: float = BANKROLL_INIT,
    kelly_frac:   float  = 0.25,
    min_edge:     float  = 0.04,
    vig:          float  = 0.08,
) -> tuple[pd.DataFrame, dict]:
    """
    Full simulation: returns (per_bet_df, summary_dict).

    Binary Kalshi-style payout:
      Win  → profit = bet * (1 - mkp_adj) / mkp_adj
      Lose → profit = -bet
    """
    bankroll = bankroll_init
    records  = []

    for date, mp, mkp_raw, outcome, surf, rank in zip(
            dates, model_prob, market_prob, outcomes, surfaces, p1_ranks):

        edge_raw = mp - mkp_raw
        if edge_raw < min_edge or mkp_raw <= 0.05 or mkp_raw >= 0.95:
            continue

        mkp_adj  = min(mkp_raw + vig / 2.0, 0.95)
        edge_adj = mp - mkp_adj
        if edge_adj <= 0:
            continue

        # Confidence-tiered Kelly
        if mp >= 0.70:
            tier = 0.40
        elif mp >= 0.60:
            tier = 0.25
        elif mp >= 0.55:
            tier = 0.12
        else:
            tier = 0.05

        f_star   = edge_adj / (1.0 - mkp_adj)
        frac     = min(f_star * tier * (kelly_frac / 0.25), MAX_BET_FRAC)
        bet      = bankroll * frac

        if outcome == 1:
            profit = bet * (1.0 - mkp_adj) / mkp_adj
        else:
            profit = -bet

        bankroll = max(bankroll + profit, 0.01)

        records.append({
            "date":        date,
            "model_prob":  round(float(mp),       4),
            "market_prob": round(float(mkp_raw),  4),
            "edge":        round(float(edge_raw), 4),
            "edge_adj":    round(float(edge_adj), 4),
            "mkp_adj":     round(float(mkp_adj),  4),
            "kelly_frac_used": round(float(frac), 4),
            "bet_size":    round(float(bet),      4),
            "outcome":     int(outcome),
            "profit":      round(float(profit),   4),
            "bankroll":    round(float(bankroll), 4),
            "surface":     str(surf),
            "p1_rank":     float(rank),
            "min_edge":    min_edge,
            "kelly_frac":  kelly_frac,
            "vig":         vig,
        })

    bets_df   = pd.DataFrame(records)
    summary   = _compute_metrics(bets_df, bankroll_init)
    return bets_df, summary


# ─────────────────────────────────────────────────────────────────────────────
# 2. Risk metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(bets_df: pd.DataFrame, bankroll_init: float) -> dict:
    if bets_df.empty:
        return {k: 0.0 for k in [
            "n_bets","win_rate","roi","sharpe","sortino","calmar",
            "max_drawdown","max_dd_duration_days","total_profit",
            "avg_edge","avg_bet","bankroll_final","kelly_growth"]}

    n     = len(bets_df)
    wins  = (bets_df["outcome"] == 1).sum()
    wr    = wins / n

    profits     = bets_df["profit"].values
    stakes      = bets_df["bet_size"].values
    total_stake = stakes.sum()
    total_profit= profits.sum()
    roi         = total_profit / total_stake * 100 if total_stake > 0 else 0.0

    # Bankroll curve for drawdown
    bankroll_curve = bets_df["bankroll"].values
    peak     = np.maximum.accumulate(bankroll_curve)
    dd       = (bankroll_curve - peak) / peak
    max_dd   = float(dd.min())           # most negative

    # Max drawdown duration (in bets, converted to approx days ~2 matches/day)
    in_dd = False
    dd_start = 0
    max_dd_dur = 0
    for i, v in enumerate(dd):
        if v < 0:
            if not in_dd:
                in_dd    = True
                dd_start = i
            max_dd_dur = max(max_dd_dur, i - dd_start)
        else:
            in_dd = False
    max_dd_dur_days = max_dd_dur / 2.0   # rough conversion

    # Sharpe (per-bet returns; annualised not needed for comparison)
    rets = profits / np.maximum(stakes, 1e-9)
    sharpe  = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0

    # Sortino (downside std only)
    down = rets[rets < 0]
    sortino = float(rets.mean() / down.std()) if len(down) > 1 and down.std() > 0 else 0.0

    # Calmar = ROI% / |max_dd%|
    calmar = roi / (abs(max_dd) * 100 + 1e-9)

    # Kelly geometric growth proxy: geometric mean of (1 + r)
    kelly_growth = float(np.exp(np.log1p(rets).mean())) - 1.0

    return {
        "n_bets":             n,
        "win_rate":           round(wr * 100, 2),
        "roi":                round(roi, 3),
        "sharpe":             round(sharpe, 4),
        "sortino":            round(sortino, 4),
        "calmar":             round(calmar, 4),
        "max_drawdown":       round(max_dd * 100, 3),
        "max_dd_duration_days": round(max_dd_dur_days, 1),
        "total_profit":       round(total_profit, 4),
        "avg_edge":           round(float(bets_df["edge"].mean()), 4),
        "avg_bet":            round(float(stakes.mean()), 4),
        "bankroll_final":     round(float(bankroll_curve[-1]), 4),
        "kelly_growth":       round(kelly_growth * 100, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. Walk-forward fold generators
# ─────────────────────────────────────────────────────────────────────────────

def expanding_folds(years, train_start=2015, test_window=1):
    """
    Expanding train window; test = next `test_window` years.
    Yields (train_years, test_years, label).
    """
    years = sorted(set(years))
    y0    = max(train_start, years[0])
    avail = [y for y in years if y >= y0]
    fold_years = sorted(set(avail))
    n = len(fold_years)
    # Need at least 4 training years before first test
    for i in range(4, n - test_window + 1):
        train_ys = fold_years[:i]
        test_ys  = fold_years[i: i + test_window]
        label    = f"EXP_{min(train_ys)}-{max(train_ys)}→{min(test_ys)}"
        yield train_ys, test_ys, label


def rolling_folds(years, train_window=4, test_window=1):
    """
    Fixed-size rolling train window.
    Yields (train_years, test_years, label).
    """
    years     = sorted(set(years))
    n         = len(years)
    for i in range(train_window, n - test_window + 1):
        train_ys = years[i - train_window: i]
        test_ys  = years[i: i + test_window]
        label    = f"ROLL_{min(train_ys)}-{max(train_ys)}→{min(test_ys)}"
        yield train_ys, test_ys, label


# ─────────────────────────────────────────────────────────────────────────────
# 4. Grid search per fold
# ─────────────────────────────────────────────────────────────────────────────

def grid_search_fold(
    train_data: dict,
    param_grid: dict = PARAM_GRID,
    objective:  str  = "sharpe",
) -> dict:
    """
    Exhaustive grid search over param_grid using in-sample data.
    Returns best params dict + IS score.
    """
    keys   = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    best_score  = -np.inf
    best_params = {}

    for combo in combos:
        params = dict(zip(keys, combo))
        _, summary = simulate_betting_full(**train_data, **params)
        score = summary.get(objective, 0.0)
        # Penalise if too few bets (unreliable estimate)
        if summary["n_bets"] < 20:
            score = -99.0
        if score > best_score:
            best_score  = score
            best_params = params

    best_params["is_score"] = round(best_score, 4)
    return best_params


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main WFO runner
# ─────────────────────────────────────────────────────────────────────────────

class WalkForwardOptimizer:
    def __init__(self, feat_df: pd.DataFrame, model_probs: np.ndarray):
        self.df       = feat_df.reset_index(drop=True)
        self.probs    = model_probs
        self.years    = feat_df["year"].values

        self.fold_results   = []   # list of dicts per fold
        self.all_oos_bets   = []   # list of DataFrames

    def _data_dict(self, mask: np.ndarray) -> dict:
        sub  = self.df[mask]
        prob = self.probs[mask]
        idx  = sub.index.values
        return dict(
            dates       = sub["tourney_date"].values,
            model_prob  = prob,
            market_prob = sub["elo_prob_p1"].values,
            outcomes    = sub["target"].values,
            surfaces    = sub["surface_norm"].values,
            p1_ranks    = sub["P1_Rank"].values,
            bankroll_init = BANKROLL_INIT,
        )

    def run(self, scheme: str = "expanding", **kwargs) -> pd.DataFrame:
        """
        scheme: "expanding" | "rolling" | "both"
        Returns combined OOS bets DataFrame.
        """
        folds_iter = []
        if scheme in ("expanding", "both"):
            folds_iter += list(expanding_folds(self.years, **kwargs))
        if scheme in ("rolling", "both"):
            folds_iter += list(rolling_folds(self.years,
                                             train_window=kwargs.get("train_window", 4),
                                             test_window=kwargs.get("test_window", 1)))

        log.info(f"Running {len(folds_iter)} WFO folds ({scheme})…")

        for train_ys, test_ys, label in folds_iter:
            train_mask = np.isin(self.years, train_ys)
            test_mask  = np.isin(self.years, test_ys)

            if train_mask.sum() < 100 or test_mask.sum() < 10:
                continue

            train_data = self._data_dict(train_mask)
            test_data  = self._data_dict(test_mask)

            # In-sample optimisation
            log.info(f"  [{label}] IS grid-search ({train_mask.sum()} rows)…")
            best_p = grid_search_fold(train_data)

            # Out-of-sample evaluation
            eval_params = {k: best_p[k] for k in ["min_edge", "kelly_frac", "vig"]}
            oos_bets, oos_summary = simulate_betting_full(**test_data, **eval_params)
            oos_bets["fold"]   = label
            oos_bets["scheme"] = "expanding" if label.startswith("EXP") else "rolling"

            # Also record IS performance for overfitting check
            _, is_summary = simulate_betting_full(**train_data, **eval_params)

            self.fold_results.append({
                "fold":          label,
                "scheme":        oos_bets["scheme"].iloc[0] if not oos_bets.empty else "?",
                "train_years":   str(train_ys),
                "test_years":    str(test_ys),
                **{f"best_{k}": v for k, v in best_p.items()},
                **{f"oos_{k}": v for k, v in oos_summary.items()},
                **{f"is_{k}":  v for k, v in is_summary.items()},
            })

            if not oos_bets.empty:
                self.all_oos_bets.append(oos_bets)

            log.info(
                f"  [{label}] best={eval_params} "
                f"OOS sharpe={oos_summary['sharpe']:.3f} "
                f"roi={oos_summary['roi']:.2f}% n={oos_summary['n_bets']}"
            )

        combined = pd.concat(self.all_oos_bets, ignore_index=True) \
                   if self.all_oos_bets else pd.DataFrame()
        return combined

    def fold_summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Heatmap & chart helpers
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("RdYlGn", as_cmap=True)

def _heatmap_param_surface(
    feat_df:    pd.DataFrame,
    model_probs: np.ndarray,
    mask:       np.ndarray,
    metric:     str = "sharpe",
    title:      str = "",
    ax:         plt.Axes = None,
):
    """Heatmap of metric over min_edge × kelly_frac (vig fixed at modal value)."""
    min_edges  = PARAM_GRID["min_edge"]
    kelly_fracs= PARAM_GRID["kelly_frac"]
    best_vig   = 0.07

    matrix = np.full((len(min_edges), len(kelly_fracs)), np.nan)
    sub    = feat_df[mask]
    probs  = model_probs[mask]

    for i, me in enumerate(min_edges):
        for j, kf in enumerate(kelly_fracs):
            _, s = simulate_betting_full(
                dates       = sub["tourney_date"].values,
                model_prob  = probs,
                market_prob = sub["elo_prob_p1"].values,
                outcomes    = sub["target"].values,
                surfaces    = sub["surface_norm"].values,
                p1_ranks    = sub["P1_Rank"].values,
                bankroll_init = BANKROLL_INIT,
                min_edge    = me,
                kelly_frac  = kf,
                vig         = best_vig,
            )
            v = s.get(metric, 0.0)
            if s["n_bets"] >= 15:
                matrix[i, j] = v

    df_heat = pd.DataFrame(matrix,
                           index=min_edges,
                           columns=kelly_fracs)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))

    sns.heatmap(df_heat, ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Kelly Fraction")
    ax.set_ylabel("Min Edge")
    ax.set_title(title or f"{metric.capitalize()} Surface (vig={best_vig})")
    return ax


def _heatmap_vig_edge(
    feat_df:    pd.DataFrame,
    model_probs: np.ndarray,
    mask:       np.ndarray,
    metric:     str = "sharpe",
    ax:         plt.Axes = None,
):
    """Heatmap of metric over vig × min_edge (kelly fixed at 0.25)."""
    vigs      = PARAM_GRID["vig"]
    min_edges = PARAM_GRID["min_edge"]

    matrix = np.full((len(vigs), len(min_edges)), np.nan)
    sub    = feat_df[mask]
    probs  = model_probs[mask]

    for i, v in enumerate(vigs):
        for j, me in enumerate(min_edges):
            _, s = simulate_betting_full(
                dates       = sub["tourney_date"].values,
                model_prob  = probs,
                market_prob = sub["elo_prob_p1"].values,
                outcomes    = sub["target"].values,
                surfaces    = sub["surface_norm"].values,
                p1_ranks    = sub["P1_Rank"].values,
                bankroll_init = BANKROLL_INIT,
                min_edge    = me,
                kelly_frac  = 0.25,
                vig         = v,
            )
            sv = s.get(metric, 0.0)
            if s["n_bets"] >= 15:
                matrix[i, j] = sv

    df_heat = pd.DataFrame(matrix, index=vigs, columns=min_edges)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4))

    sns.heatmap(df_heat, ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                linewidths=0.4, cbar_kws={"shrink": 0.8})
    ax.set_xlabel("Min Edge")
    ax.set_ylabel("Vig")
    ax.set_title(f"{metric.capitalize()} — Vig × Min Edge (kelly=0.25)")
    return ax


def _heatmap_surface_year(feat_df: pd.DataFrame, model_probs: np.ndarray, ax=None):
    """Accuracy by surface × year pivot heatmap."""
    df = feat_df.copy()
    df["predicted"] = (model_probs >= 0.5).astype(int)
    df["correct"]   = (df["predicted"] == df["target"]).astype(int)

    pivot = df.groupby(["year", "surface_norm"])["correct"].mean().unstack("surface_norm")
    pivot = pivot[[c for c in ["Hard", "Clay", "Grass"] if c in pivot.columns]]
    pivot = pivot.dropna(how="all")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                vmin=0.45, vmax=0.65, linewidths=0.4)
    ax.set_title("Model Accuracy — Year × Surface")
    ax.set_xlabel("Surface")
    ax.set_ylabel("Year")
    return ax


def _heatmap_monthly_pnl(oos_bets: pd.DataFrame, ax=None):
    """Monthly P&L calendar heatmap."""
    if oos_bets.empty:
        return ax

    df = oos_bets.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"])
    df["year"]  = df["date"].dt.year
    df["month"] = df["date"].dt.month

    pivot = df.groupby(["year", "month"])["profit"].sum().unstack("month")
    pivot.columns = [f"M{c:02d}" for c in pivot.columns]

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 5))

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".0f",
                center=0, linewidths=0.3)
    ax.set_title("Monthly P&L ($) — OOS Bets")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    return ax


def _heatmap_calibration(model_probs: np.ndarray, targets: np.ndarray, ax=None):
    """Model confidence bin × actual win rate heatmap (calibration check)."""
    bins   = np.arange(0.45, 1.01, 0.05)
    labels = [f"{b:.2f}" for b in bins[:-1]]
    cat    = pd.cut(model_probs, bins=bins, labels=labels)

    df = pd.DataFrame({"bin": cat, "outcome": targets})
    df = df.dropna(subset=["bin"])
    win_rate = df.groupby("bin", observed=True)["outcome"].mean().reset_index()
    count    = df.groupby("bin", observed=True)["outcome"].count().reset_index()
    win_rate = win_rate.merge(count, on="bin", suffixes=("_rate","_count"))

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 3))

    pivot = win_rate.set_index("bin")[["outcome_rate"]].T
    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                vmin=0.3, vmax=0.9, linewidths=0.4)
    ax.set_title("Calibration: Model Confidence → Actual Win Rate")
    ax.set_xlabel("Model Confidence Bin")
    ax.set_yticks([])
    return ax


def _heatmap_edge_surface_roi(oos_bets: pd.DataFrame, ax=None):
    """ROI by edge bin × surface."""
    if oos_bets.empty:
        return ax

    df = oos_bets.copy()
    df["edge_bin"] = pd.cut(df["edge"],
                             bins=[0.0, 0.04, 0.06, 0.08, 0.10, 0.99],
                             labels=["0-4%","4-6%","6-8%","8-10%",">10%"])
    df["roi_bet"] = df["profit"] / df["bet_size"].clip(lower=1e-6) * 100

    pivot = df.groupby(["edge_bin", "surface"], observed=True)["roi_bet"].mean().unstack("surface")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".1f",
                center=0, linewidths=0.4)
    ax.set_title("ROI (%) — Edge Bin × Surface")
    ax.set_xlabel("Surface")
    ax.set_ylabel("Edge Bin")
    return ax


def _heatmap_rank_tier_accuracy(feat_df: pd.DataFrame, model_probs: np.ndarray, ax=None):
    """Accuracy by rank tier (P1) × surface."""
    df = feat_df.copy()
    df["pred"]    = (model_probs >= 0.5).astype(int)
    df["correct"] = (df["pred"] == df["target"]).astype(int)
    df["rank_tier"] = pd.cut(df["P1_Rank"],
                              bins=[0, 10, 50, 100, 200, 500, 9999],
                              labels=["Top-10","11-50","51-100","101-200","201-500","500+"])
    pivot = df.groupby(["rank_tier", "surface_norm"], observed=True)["correct"] \
               .mean().unstack("surface_norm")
    pivot = pivot[[c for c in ["Hard","Clay","Grass"] if c in pivot.columns]]

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                vmin=0.4, vmax=0.75, linewidths=0.4)
    ax.set_title("Accuracy — Rank Tier × Surface")
    return ax


def _heatmap_feature_correlation(feat_df: pd.DataFrame, ax=None):
    """Spearman correlation matrix of numeric features."""
    num_cols = [c for c in [
        "P1_Rank","P2_Rank","P1_Age","P2_Age","P1_Height_cm","P2_Height_cm",
        "Elo_Diff","Rank_Diff","Rank_Points_Diff","Age_Diff","elo_prob_p1",
        "Height_Diff","Age_Diff_Sq","Surface_Height_Grass",
    ] if c in feat_df.columns]

    corr_mat = feat_df[num_cols].dropna().corr(method="spearman")

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 9))

    mask = np.triu(np.ones_like(corr_mat, dtype=bool))
    sns.heatmap(corr_mat, ax=ax, mask=mask, cmap="coolwarm",
                annot=True, fmt=".2f", linewidths=0.2,
                vmin=-1, vmax=1, square=True, cbar_kws={"shrink": 0.6})
    ax.set_title("Feature Correlation (Spearman)")
    return ax


def _heatmap_fold_stability(fold_df: pd.DataFrame, ax=None):
    """IS vs OOS Sharpe per fold."""
    if fold_df.empty:
        return ax

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))

    fold_labels = fold_df["fold"].values
    is_sharpe   = fold_df["is_sharpe"].values if "is_sharpe"  in fold_df else np.zeros(len(fold_df))
    oos_sharpe  = fold_df["oos_sharpe"].values if "oos_sharpe" in fold_df else np.zeros(len(fold_df))

    x   = np.arange(len(fold_labels))
    w   = 0.35
    ax.bar(x - w/2, is_sharpe,  w, label="IS Sharpe",  color="#2C7BB6", alpha=0.8)
    ax.bar(x + w/2, oos_sharpe, w, label="OOS Sharpe", color="#D7191C", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Walk-Forward Fold Stability — IS vs OOS Sharpe")
    ax.legend()
    return ax


def _heatmap_param_consensus(fold_df: pd.DataFrame, ax=None):
    """How often each param value is selected across folds."""
    if fold_df.empty:
        return ax

    rows = []
    for param in ["best_min_edge", "best_kelly_frac", "best_vig"]:
        if param not in fold_df.columns:
            continue
        vc = fold_df[param].value_counts(normalize=True)
        short = param.replace("best_", "")
        for val, freq in vc.items():
            rows.append({"param": short, "value": str(val), "freq": freq})

    if not rows:
        return ax

    df_p = pd.DataFrame(rows)
    pivot = df_p.pivot(index="value", columns="param", values="freq").fillna(0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    ax.set_title("Parameter Selection Frequency Across Folds")
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Value")
    return ax


def _plot_oos_equity(oos_bets: pd.DataFrame, ax=None):
    """Equity curve of all OOS bets chronologically."""
    if oos_bets.empty:
        return ax

    df = oos_bets.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")

    bankroll = BANKROLL_INIT + df["profit"].cumsum()

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df["date"].values, bankroll.values, linewidth=1.2, color="#2C7BB6")
    ax.fill_between(df["date"].values, BANKROLL_INIT, bankroll.values,
                    alpha=0.15, color="#2C7BB6")
    ax.axhline(BANKROLL_INIT, color="grey", linestyle="--", linewidth=0.8)
    ax.set_title("OOS Cumulative Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Bankroll ($)")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
    return ax


# ─────────────────────────────────────────────────────────────────────────────
# 7. Plotly interactive dashboard
# ─────────────────────────────────────────────────────────────────────────────

def build_plotly_dashboard(oos_bets: pd.DataFrame, fold_df: pd.DataFrame) -> None:
    if not _HAS_PLOTLY or oos_bets.empty:
        log.warning("Plotly not available or no OOS bets — skipping HTML export.")
        return

    df = oos_bets.copy()
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date")
    df["cumulative_pnl"]  = df["profit"].cumsum()
    df["bankroll_plotly"] = BANKROLL_INIT + df["cumulative_pnl"]

    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "OOS Equity Curve", "Daily P&L",
            "Edge Distribution", "OOS Sharpe by Fold",
            "Win Rate by Surface", "Model Prob Distribution"
        ],
        vertical_spacing=0.12,
    )

    # 1 — Equity
    fig.add_trace(go.Scatter(x=df["date"], y=df["bankroll_plotly"],
                             mode="lines", name="Bankroll",
                             line=dict(color="#2C7BB6")), row=1, col=1)

    # 2 — Daily P&L
    daily = df.groupby("date")["profit"].sum().reset_index()
    colors_bar = ["#1A9641" if v >= 0 else "#D7191C" for v in daily["profit"]]
    fig.add_trace(go.Bar(x=daily["date"], y=daily["profit"],
                         name="Daily P&L", marker_color=colors_bar), row=1, col=2)

    # 3 — Edge histogram
    fig.add_trace(go.Histogram(x=df["edge"], nbinsx=30,
                               name="Edge", marker_color="#7B3294"), row=2, col=1)

    # 4 — OOS Sharpe by fold
    if not fold_df.empty and "oos_sharpe" in fold_df:
        fold_colors = ["#1A9641" if v >= 0 else "#D7191C"
                       for v in fold_df["oos_sharpe"]]
        fig.add_trace(go.Bar(x=fold_df["fold"], y=fold_df["oos_sharpe"],
                             name="OOS Sharpe", marker_color=fold_colors), row=2, col=2)

    # 5 — Win rate by surface
    wr_surf = df.groupby("surface")["outcome"].mean().reset_index()
    fig.add_trace(go.Bar(x=wr_surf["surface"], y=wr_surf["outcome"],
                         name="Win Rate", marker_color="#FDAE61"), row=3, col=1)

    # 6 — Model prob histogram
    fig.add_trace(go.Histogram(x=df["model_prob"], nbinsx=25,
                               name="Model Prob", marker_color="#2C7BB6"), row=3, col=2)

    fig.update_layout(
        title_text="Walk-Forward OOS Dashboard",
        height=1000,
        showlegend=False,
        template="plotly_white",
    )

    fig.write_html(HTML_OUT)
    log.info(f"Interactive dashboard saved → {HTML_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. PDF report
# ─────────────────────────────────────────────────────────────────────────────

def generate_wfo_pdf(
    feat_df:    pd.DataFrame,
    model_probs: np.ndarray,
    oos_bets:   pd.DataFrame,
    fold_df:    pd.DataFrame,
    oos_mask:   np.ndarray,
    pdf_path:   str = PDF_OUT,
):
    log.info(f"Generating PDF report → {pdf_path}")

    with PdfPages(pdf_path) as pdf:

        # ── Page 1: Title / KPI summary ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 8))
        ax.axis("off")
        if not oos_bets.empty:
            m = _compute_metrics(oos_bets, BANKROLL_INIT)
            lines = [
                f"Walk-Forward Optimization Report",
                f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "",
                f"Total OOS Bets     : {m['n_bets']:,}",
                f"Win Rate           : {m['win_rate']:.2f}%",
                f"ROI                : {m['roi']:.2f}%",
                f"Sharpe Ratio       : {m['sharpe']:.4f}",
                f"Sortino Ratio      : {m['sortino']:.4f}",
                f"Calmar Ratio       : {m['calmar']:.4f}",
                f"Max Drawdown       : {m['max_drawdown']:.2f}%",
                f"Max DD Duration    : {m['max_dd_duration_days']:.1f} days",
                f"Avg Edge           : {m['avg_edge']*100:.2f}%",
                f"Avg Bet            : ${m['avg_bet']:.2f}",
                f"Total Profit       : ${m['total_profit']:.2f}",
                f"Bankroll Final     : ${m['bankroll_final']:.2f}",
                f"Kelly Growth       : {m['kelly_growth']:.2f}%",
            ]
            for i, line in enumerate(lines):
                weight = "bold" if i in (0, 1) else "normal"
                size   = 16 if i == 0 else (12 if i == 1 else 13)
                ax.text(0.1, 0.95 - i * 0.055, line,
                        transform=ax.transAxes, fontsize=size, weight=weight)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 2: OOS Equity Curve ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        _plot_oos_equity(oos_bets, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 3: Fold Stability ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 4))
        _heatmap_fold_stability(fold_df, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 4: ROI + Sharpe surfaces ───────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("OOS Parameter Surface", fontsize=14)
        _heatmap_param_surface(feat_df, model_probs, oos_mask,
                               metric="roi", title="ROI Surface", ax=axes[0])
        _heatmap_param_surface(feat_df, model_probs, oos_mask,
                               metric="sharpe", title="Sharpe Surface", ax=axes[1])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 5: Max Drawdown + N_bets surfaces ───────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("Drawdown & Activity Surface", fontsize=14)
        _heatmap_param_surface(feat_df, model_probs, oos_mask,
                               metric="max_drawdown", title="Max Drawdown Surface", ax=axes[0])
        _heatmap_param_surface(feat_df, model_probs, oos_mask,
                               metric="n_bets", title="N-Bets Surface", ax=axes[1])
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 6: Vig × Edge Sharpe ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        _heatmap_vig_edge(feat_df, model_probs, oos_mask, metric="sharpe", ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 7: Surface × Year Accuracy ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 8))
        _heatmap_surface_year(feat_df, model_probs, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 8: Monthly P&L Calendar ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(13, 5))
        _heatmap_monthly_pnl(oos_bets, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 9: Calibration ──────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 3))
        _heatmap_calibration(model_probs[oos_mask], feat_df[oos_mask]["target"].values, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 10: Edge × Surface ROI ──────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 5))
        _heatmap_edge_surface_roi(oos_bets, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 11: Rank Tier × Accuracy ───────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 6))
        _heatmap_rank_tier_accuracy(feat_df, model_probs, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 12: Feature Correlation ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(11, 9))
        _heatmap_feature_correlation(feat_df, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 13: Parameter Consensus ────────────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 5))
        _heatmap_param_consensus(fold_df, ax=ax)
        pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 14: Distribution diagnostics ───────────────────────────────
        if not oos_bets.empty:
            fig, axes = plt.subplots(2, 2, figsize=(12, 9))
            fig.suptitle("OOS Bet Diagnostics", fontsize=14)

            axes[0,0].hist(oos_bets["model_prob"], bins=30, color="#2C7BB6", edgecolor="white")
            axes[0,0].set_title("Model Probability Distribution"); axes[0,0].set_xlabel("Prob")

            axes[0,1].hist(oos_bets["edge"], bins=30, color="#1A9641", edgecolor="white")
            axes[0,1].set_title("Edge Distribution"); axes[0,1].set_xlabel("Edge")

            axes[1,0].hist(oos_bets["bet_size"], bins=30, color="#FDAE61", edgecolor="white")
            axes[1,0].set_title("Bet Size Distribution"); axes[1,0].set_xlabel("$ Bet")

            axes[1,1].scatter(oos_bets["edge"], oos_bets["profit"],
                              alpha=0.3, s=8, color="#7B3294")
            axes[1,1].axhline(0, color="black", linewidth=0.8)
            axes[1,1].set_title("Edge vs Profit"); axes[1,1].set_xlabel("Edge"); axes[1,1].set_ylabel("$")

            plt.tight_layout()
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        # ── Page 15: Fold KPI table ──────────────────────────────────────────
        if not fold_df.empty:
            show_cols = ["fold", "oos_n_bets", "oos_win_rate", "oos_roi",
                         "oos_sharpe", "oos_max_drawdown",
                         "best_min_edge", "best_kelly_frac", "best_vig"]
            show_cols = [c for c in show_cols if c in fold_df.columns]

            fig, ax = plt.subplots(figsize=(14, max(3, len(fold_df) * 0.5 + 2)))
            ax.axis("off")
            tbl_data = fold_df[show_cols].round(3).astype(str).values.tolist()
            table = ax.table(cellText=tbl_data,
                             colLabels=[c.replace("_"," ").title() for c in show_cols],
                             loc="center", cellLoc="center")
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            ax.set_title("Walk-Forward Fold Summary", fontsize=13, pad=12)
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

    log.info(f"PDF saved → {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # ── Load & featurise data ───────────────────────────────────────────────
    log.info("Loading ATP match data…")
    raw = load_atp_matches(ATP_DIR)
    log.info("Building features (chronological ELO)…")
    feat_df = build_features(raw)
    feat_df = feat_df[feat_df["year"] >= 2012].reset_index(drop=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model_probs = None
    if _HAS_XGB and os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        with open(FEATURES_PATH) as f:
            feature_names = json.load(f)
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)

        available = [c for c in feature_names if c in feat_df.columns]
        X = feat_df[available].fillna(0).values
        model_probs = model.predict_proba(X)[:, 1]
        log.info(f"XGBoost model loaded, {len(available)} features used.")
    else:
        log.warning("XGBoost model not found — using ELO prob as model proxy.")
        model_probs = feat_df["elo_prob_p1"].values

    model_probs = np.clip(model_probs, 0.01, 0.99)

    # ── Define OOS period ───────────────────────────────────────────────────
    oos_mask  = feat_df["year"].values >= 2020
    train_mask = ~oos_mask

    log.info(f"Data: {(~oos_mask).sum():,} IS rows | {oos_mask.sum():,} OOS rows")

    # ── Run WFO ────────────────────────────────────────────────────────────
    wfo = WalkForwardOptimizer(feat_df, model_probs)
    oos_bets = wfo.run(scheme="both")
    fold_df  = wfo.fold_summary()

    # ── Save CSV ────────────────────────────────────────────────────────────
    if not oos_bets.empty:
        oos_bets.to_csv(CSV_OUT, index=False)
        log.info(f"Per-bet CSV saved → {CSV_OUT}  ({len(oos_bets):,} rows)")

    # ── Save optimal params JSON ─────────────────────────────────────────────
    if not fold_df.empty:
        param_cols = ["best_min_edge", "best_kelly_frac", "best_vig"]
        consensus  = {}
        for col in param_cols:
            if col in fold_df.columns:
                consensus[col.replace("best_", "")] = float(
                    fold_df[col].mode().iloc[0])

        json_out = {
            "generated":      datetime.now().isoformat(),
            "consensus_params": consensus,
            "fold_params":    fold_df[
                ["fold"] + [c for c in param_cols if c in fold_df.columns]
            ].to_dict(orient="records"),
        }
        with open(JSON_OUT, "w") as f:
            json.dump(json_out, f, indent=2)
        log.info(f"Optimal params JSON saved → {JSON_OUT}")
        log.info(f"Consensus params: {consensus}")

    # ── Generate PDF ─────────────────────────────────────────────────────────
    generate_wfo_pdf(feat_df, model_probs, oos_bets, fold_df, oos_mask)

    # ── Generate Plotly HTML ──────────────────────────────────────────────────
    build_plotly_dashboard(oos_bets, fold_df)

    log.info("Walk-forward optimization complete.")
    if not fold_df.empty:
        top = fold_df.nlargest(3, "oos_sharpe")[["fold","oos_sharpe","oos_roi","best_min_edge","best_kelly_frac","best_vig"]]
        log.info(f"\nTop 3 folds by OOS Sharpe:\n{top.to_string(index=False)}")


if __name__ == "__main__":
    main()
