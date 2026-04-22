"""
wfo_clean.py  —  Walk-Forward Optimisation Backtest
=====================================================
Self-contained: reads clean_tennis_data.csv, trains XGBoost per fold,
grid-searches betting parameters, and writes a 20-page PDF.

Market probability proxy: rank-points ratio
  market_prob_p1 = P1_Rank_Points / (P1_Rank_Points + P2_Rank_Points)
  (higher ATP points → higher implied win probability)
"""

import os, sys, json, logging, warnings, itertools
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, roc_auc_score, brier_score_loss, log_loss,
    roc_curve, confusion_matrix,
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

_DIR     = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(_DIR, "clean_tennis_data.csv")
PDF_OUT  = os.path.join(_DIR, "wfo_clean_report.pdf")
CSV_OUT  = os.path.join(_DIR, "wfo_clean_bets.csv")
JSON_OUT = os.path.join(_DIR, "wfo_clean_params.json")

# ─── Betting simulation parameters ───────────────────────────────────────────
BANKROLL_INIT = 500.0
MAX_BET_FRAC  = 0.20

PARAM_GRID = {
    "min_edge":   [0.02, 0.04, 0.06, 0.08, 0.10],
    "kelly_frac": [0.05, 0.10, 0.20, 0.35, 0.50],
    "vig":        [0.05, 0.07, 0.09, 0.11],
}

BASE_FEATURES = [
    "Surface_Hard", "Surface_Clay", "Surface_Grass",
    "Best_Of_Sets",
    "P1_Is_Right_Handed", "P1_Height_cm", "P1_Age",
    "P1_Rank", "P1_Rank_Points",
    "P2_Is_Right_Handed", "P2_Height_cm", "P2_Age",
    "P2_Rank", "P2_Rank_Points",
]
DERIVED_FEATURES = [
    "Height_Diff", "Age_Diff", "Rank_Diff",
    "Rank_Points_Diff", "Surface_Height_Grass", "Age_Diff_Sq",
    "market_prob",
]
ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ─── 1. Data loading & feature engineering ───────────────────────────────────

def load_and_engineer(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, low_memory=False)
    df["year"]  = df["tourney_date"] // 10000
    df["month"] = (df["tourney_date"] % 10000) // 100
    df["target"] = df["Target_P1_Wins"].astype(int)

    # Derived differential features
    df["Height_Diff"]       = df["P1_Height_cm"]    - df["P2_Height_cm"]
    df["Age_Diff"]          = df["P1_Age"]           - df["P2_Age"]
    df["Rank_Diff"]         = df["P1_Rank"]          - df["P2_Rank"]
    df["Rank_Points_Diff"]  = df["P1_Rank_Points"]   - df["P2_Rank_Points"]
    df["Surface_Height_Grass"] = df["Height_Diff"] * df["Surface_Grass"]
    df["Age_Diff_Sq"]       = np.sign(df["Age_Diff"]) * (df["Age_Diff"] / 10.0) ** 2

    # Market probability proxy: rank-points ratio (higher points = favourite)
    total_pts = df["P1_Rank_Points"] + df["P2_Rank_Points"]
    df["market_prob"] = np.where(
        total_pts > 0,
        df["P1_Rank_Points"] / total_pts,
        0.5,
    ).clip(0.05, 0.95)

    # Surface label for groupby
    df["surface_label"] = np.select(
        [df["Surface_Hard"] == 1, df["Surface_Clay"] == 1, df["Surface_Grass"] == 1],
        ["Hard", "Clay", "Grass"],
        default="Hard",
    )

    # Fill any residual NaNs with column median
    for col in ALL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    return df.reset_index(drop=True)


# ─── 2. Betting simulation ────────────────────────────────────────────────────

def simulate_betting(
    dates, model_prob, market_prob, outcomes, surfaces, p1_ranks,
    bankroll_init=BANKROLL_INIT,
    kelly_frac=0.25, min_edge=0.04, vig=0.08,
):
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

        tier = 0.40 if mp >= 0.70 else (0.25 if mp >= 0.60 else (0.12 if mp >= 0.55 else 0.05))
        f_star = edge_adj / (1.0 - mkp_adj)
        frac   = min(f_star * tier * (kelly_frac / 0.25), MAX_BET_FRAC)
        bet    = bankroll * frac

        profit   = bet * (1.0 - mkp_adj) / mkp_adj if outcome == 1 else -bet
        bankroll = max(bankroll + profit, 0.01)

        records.append({
            "date": date, "model_prob": round(float(mp), 4),
            "market_prob": round(float(mkp_raw), 4),
            "edge": round(float(edge_raw), 4),
            "edge_adj": round(float(edge_adj), 4),
            "kelly_frac_used": round(float(frac), 4),
            "bet_size": round(float(bet), 4),
            "outcome": int(outcome), "profit": round(float(profit), 4),
            "bankroll": round(float(bankroll), 4),
            "surface": str(surf), "p1_rank": float(rank),
            "min_edge": min_edge, "kelly_frac": kelly_frac, "vig": vig,
        })

    return pd.DataFrame(records)


def compute_metrics(bets_df: pd.DataFrame, bankroll_init=BANKROLL_INIT) -> dict:
    zeros = {k: 0.0 for k in ["n_bets","win_rate","roi","sharpe","sortino",
                                "calmar","max_drawdown","max_dd_days",
                                "total_profit","avg_edge","avg_bet",
                                "bankroll_final","kelly_growth"]}
    if bets_df.empty:
        return zeros

    n      = len(bets_df)
    wr     = (bets_df["outcome"] == 1).sum() / n
    stakes = bets_df["bet_size"].values
    profs  = bets_df["profit"].values
    roi    = profs.sum() / stakes.sum() * 100 if stakes.sum() > 0 else 0.0

    curve  = bets_df["bankroll"].values
    peak   = np.maximum.accumulate(curve)
    dd     = (curve - peak) / peak
    max_dd = float(dd.min())

    in_dd, dd_start, max_dd_len = False, 0, 0
    for i, v in enumerate(dd):
        if v < 0:
            if not in_dd: in_dd, dd_start = True, i
            max_dd_len = max(max_dd_len, i - dd_start)
        else:
            in_dd = False

    rets    = profs / np.maximum(stakes, 1e-9)
    sharpe  = float(rets.mean() / rets.std()) if rets.std() > 0 else 0.0
    down    = rets[rets < 0]
    sortino = float(rets.mean() / down.std()) if len(down) > 1 and down.std() > 0 else 0.0
    calmar  = roi / (abs(max_dd) * 100 + 1e-9)
    kg      = float(np.exp(np.log1p(rets).mean())) - 1.0

    return {
        "n_bets": n, "win_rate": round(wr * 100, 2),
        "roi": round(roi, 3), "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4), "calmar": round(calmar, 4),
        "max_drawdown": round(max_dd * 100, 3),
        "max_dd_days": round(max_dd_len / 2.0, 1),
        "total_profit": round(float(profs.sum()), 4),
        "avg_edge": round(float(bets_df["edge"].mean()), 4),
        "avg_bet": round(float(stakes.mean()), 4),
        "bankroll_final": round(float(curve[-1]), 4),
        "kelly_growth": round(kg * 100, 4),
    }


# ─── 3. XGBoost training helper ──────────────────────────────────────────────

def train_xgb(X_train, y_train, X_test):
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", verbosity=0,
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_test, np.zeros(len(X_test)))],
              verbose=False)
    return model.predict_proba(X_test)[:, 1]


# ─── 4. Grid search per fold ─────────────────────────────────────────────────

def grid_search(train_data: dict, objective="sharpe") -> dict:
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    best_score, best_params = -np.inf, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        bets   = simulate_betting(**train_data, **params)
        m      = compute_metrics(bets)
        score  = m.get(objective, 0.0) if m["n_bets"] >= 20 else -99.0
        if score > best_score:
            best_score, best_params = score, params
    best_params["is_score"] = round(best_score, 4)
    return best_params


# ─── 5. Walk-Forward folds ────────────────────────────────────────────────────

def expanding_folds(years, min_train=6, test_w=1):
    ys = sorted(set(years))
    for i in range(min_train, len(ys) - test_w + 1):
        train_ys = ys[:i]
        test_ys  = ys[i: i + test_w]
        yield train_ys, test_ys, f"EXP_{ys[0]}-{ys[i-1]}→{test_ys[0]}"


def rolling_folds(years, train_w=6, test_w=1):
    ys = sorted(set(years))
    for i in range(train_w, len(ys) - test_w + 1):
        train_ys = ys[i - train_w: i]
        test_ys  = ys[i: i + test_w]
        yield train_ys, test_ys, f"ROLL_{train_ys[0]}-{train_ys[-1]}→{test_ys[0]}"


# ─── 6. Main WFO runner ───────────────────────────────────────────────────────

class WFO:
    def __init__(self, df: pd.DataFrame):
        self.df    = df
        self.years = df["year"].values
        self.fold_results  = []
        self.all_oos_bets  = []
        self.fold_probs    = {}   # fold_label → (y_true, y_prob, mask)

    def _data_dict(self, mask):
        sub = self.df[mask]
        return dict(
            dates       = sub["tourney_date"].values,
            market_prob = sub["market_prob"].values,
            outcomes    = sub["target"].values,
            surfaces    = sub["surface_label"].values,
            p1_ranks    = sub["P1_Rank"].values,
            bankroll_init = BANKROLL_INIT,
        )

    def run(self, scheme="both"):
        folds = []
        if scheme in ("expanding", "both"):
            folds += list(expanding_folds(self.years))
        if scheme in ("rolling", "both"):
            folds += list(rolling_folds(self.years))

        log.info(f"Running {len(folds)} WFO folds ({scheme})…")

        for train_ys, test_ys, label in folds:
            tr_mask = np.isin(self.years, train_ys)
            te_mask = np.isin(self.years, test_ys)
            if tr_mask.sum() < 200 or te_mask.sum() < 30:
                continue

            X_tr = self.df.loc[tr_mask, ALL_FEATURES].fillna(0).values
            y_tr = self.df.loc[tr_mask, "target"].values
            X_te = self.df.loc[te_mask, ALL_FEATURES].fillna(0).values
            y_te = self.df.loc[te_mask, "target"].values

            log.info(f"  [{label}] training XGB on {tr_mask.sum():,} rows…")
            oos_probs = train_xgb(X_tr, y_tr, X_te)
            oos_probs = np.clip(oos_probs, 0.01, 0.99)

            # IS probs (for calibration / feature charts)
            is_probs  = train_xgb(X_tr, y_tr, X_tr)
            is_probs  = np.clip(is_probs, 0.01, 0.99)

            # Grid-search on IS data
            is_data = self._data_dict(tr_mask)
            is_data["model_prob"] = is_probs
            best_p = grid_search(is_data)

            eval_p = {k: best_p[k] for k in ["min_edge", "kelly_frac", "vig"]}

            # OOS simulation
            oos_data = self._data_dict(te_mask)
            oos_data["model_prob"] = oos_probs
            oos_bets = simulate_betting(**oos_data, **eval_p)
            oos_bets["fold"]   = label
            oos_bets["scheme"] = "expanding" if label.startswith("EXP") else "rolling"

            oos_m = compute_metrics(oos_bets)
            is_bets_eval = simulate_betting(**{**is_data, **eval_p})
            is_m  = compute_metrics(is_bets_eval)

            # Model accuracy
            oos_acc = accuracy_score(y_te, (oos_probs >= 0.5).astype(int))
            oos_auc = roc_auc_score(y_te, oos_probs) if len(np.unique(y_te)) > 1 else 0.5

            self.fold_results.append({
                "fold": label,
                "scheme": "expanding" if label.startswith("EXP") else "rolling",
                "train_years": str(train_ys),
                "test_years":  str(test_ys),
                "n_train": int(tr_mask.sum()),
                "n_test":  int(te_mask.sum()),
                "oos_accuracy": round(oos_acc * 100, 2),
                "oos_auc": round(oos_auc, 4),
                **{f"best_{k}": v for k, v in best_p.items()},
                **{f"oos_{k}": v for k, v in oos_m.items()},
                **{f"is_{k}":  v for k, v in is_m.items()},
            })
            self.fold_probs[label] = (y_te, oos_probs, te_mask)

            if not oos_bets.empty:
                self.all_oos_bets.append(oos_bets)

            log.info(
                f"  [{label}] acc={oos_acc*100:.1f}% auc={oos_auc:.3f} "
                f"sharpe={oos_m['sharpe']:.3f} roi={oos_m['roi']:.2f}% "
                f"n_bets={oos_m['n_bets']}"
            )

        combined = pd.concat(self.all_oos_bets, ignore_index=True) \
                   if self.all_oos_bets else pd.DataFrame()
        return combined

    def fold_df(self):
        return pd.DataFrame(self.fold_results)


# ─── 7. PDF Generation ────────────────────────────────────────────────────────

def _savefig(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf(df, oos_bets, fold_df, wfo_obj, pdf_path=PDF_OUT):
    log.info(f"Generating PDF → {pdf_path}")

    # Aggregate OOS truth/prob for global metrics
    all_y_true, all_y_prob = [], []
    for label, (yt, yp, _) in wfo_obj.fold_probs.items():
        all_y_true.append(yt)
        all_y_prob.append(yp)

    if all_y_true:
        agg_y_true = np.concatenate(all_y_true)
        agg_y_prob = np.concatenate(all_y_prob)
        agg_y_pred = (agg_y_prob >= 0.5).astype(int)
        agg_acc    = accuracy_score(agg_y_true, agg_y_pred)
        agg_auc    = roc_auc_score(agg_y_true, agg_y_prob)
        agg_brier  = brier_score_loss(agg_y_true, agg_y_prob)
        agg_ll     = log_loss(agg_y_true, agg_y_prob)
    else:
        agg_y_true = agg_y_prob = agg_y_pred = np.array([])
        agg_acc = agg_auc = agg_brier = agg_ll = 0.0

    global_m = compute_metrics(oos_bets) if not oos_bets.empty else {}

    # Colour palette
    C = ["#2C7BB6", "#D7191C", "#1A9641", "#FDAE61", "#7B3294", "#4DAC26"]
    sns.set_style("whitegrid")

    with PdfPages(pdf_path) as pdf:

        # ── PAGE 1 · Cover / Executive Summary ───────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#0D1117")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off(); ax.set_facecolor("#0D1117")

        ax.text(0.5, 0.96, "Walk-Forward Optimisation Backtest Report",
                color="white", fontsize=22, fontweight="bold",
                ha="center", transform=ax.transAxes)
        ax.text(0.5, 0.91,
                f"Tennis ATP · clean_tennis_data.csv · 2000–2024  |  Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                color="#888888", fontsize=11, ha="center", transform=ax.transAxes)

        kpis = [
            ("Dataset Matches",     f"{len(df):,}",              "2000–2024"),
            ("WFO Folds",           f"{len(fold_df)}",           "expanding + rolling"),
            ("OOS Bets Placed",     f"{global_m.get('n_bets',0):,}", "after edge/vig filter"),
            ("Aggregate Accuracy",  f"{agg_acc*100:.2f}%",       "all OOS folds"),
            ("Aggregate AUC",       f"{agg_auc:.4f}",            "ROC area"),
            ("Brier Score",         f"{agg_brier:.4f}",          "lower = better calibration"),
            ("OOS Win Rate",        f"{global_m.get('win_rate',0):.2f}%", "triggered bets"),
            ("OOS ROI",             f"{global_m.get('roi',0):+.2f}%",    "profit ÷ staked"),
            ("Sharpe Ratio",        f"{global_m.get('sharpe',0):.4f}",   "per-bet return ratio"),
            ("Sortino Ratio",       f"{global_m.get('sortino',0):.4f}",  "downside-adjusted"),
            ("Calmar Ratio",        f"{global_m.get('calmar',0):.4f}",   "ROI / max drawdown"),
            ("Max Drawdown",        f"{global_m.get('max_drawdown',0):.2f}%",  "bankroll peak→trough"),
            ("Max DD Duration",     f"{global_m.get('max_dd_days',0):.0f} days", "approx"),
            ("Total Profit",        f"${global_m.get('total_profit',0):+,.2f}",  f"start ${BANKROLL_INIT:.0f}"),
            ("Final Bankroll",      f"${global_m.get('bankroll_final',0):,.2f}", "OOS combined"),
            ("Avg Edge",            f"{global_m.get('avg_edge',0)*100:.2f}%",   "model vs market"),
        ]
        tbl = ax.table(cellText=[[k, v, n_] for k, v, n_ in kpis],
                       colLabels=["Metric", "Value", "Note"],
                       cellLoc="center", loc="center",
                       bbox=[0.04, 0.10, 0.92, 0.74])
        tbl.auto_set_font_size(False); tbl.set_fontsize(10)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#333333")
            if r == 0:
                cell.set_facecolor("#1F6FEB"); cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#161B22"); cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#0D1117"); cell.set_text_props(color="white")
        _savefig(pdf, fig)

        # ── PAGE 2 · OOS Equity Curve ─────────────────────────────────────
        if not oos_bets.empty:
            fig, ax = plt.subplots(figsize=(13, 5))
            df_eq = oos_bets.copy()
            df_eq["date"] = pd.to_datetime(df_eq["date"].astype(str), format="%Y%m%d", errors="coerce")
            df_eq = df_eq.dropna(subset=["date"]).sort_values("date")
            bankroll_curve = BANKROLL_INIT + df_eq["profit"].cumsum()

            ax.plot(df_eq["date"], bankroll_curve, lw=1.5, color=C[0], label="Bankroll")
            ax.fill_between(df_eq["date"], BANKROLL_INIT, bankroll_curve,
                            where=bankroll_curve >= BANKROLL_INIT,
                            alpha=0.18, color=C[2], label="Above start")
            ax.fill_between(df_eq["date"], BANKROLL_INIT, bankroll_curve,
                            where=bankroll_curve < BANKROLL_INIT,
                            alpha=0.18, color=C[1], label="Below start")
            ax.axhline(BANKROLL_INIT, color="grey", linestyle="--", lw=0.9)
            ax.set_title("OOS Cumulative Equity Curve (All Folds Chronological)", fontsize=13)
            ax.set_xlabel("Date"); ax.set_ylabel("Bankroll ($)")
            ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            ax.legend()
            _savefig(pdf, fig)

        # ── PAGE 3 · Drawdown Curve ───────────────────────────────────────
        if not oos_bets.empty:
            fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
            fig.suptitle("Drawdown Analysis", fontsize=13)

            axes[0].plot(df_eq["date"], bankroll_curve, lw=1.4, color=C[0])
            axes[0].axhline(BANKROLL_INIT, color="grey", linestyle="--", lw=0.8)
            axes[0].set_ylabel("Bankroll ($)")
            axes[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("${x:,.0f}"))
            axes[0].set_title("Equity")

            peak_curve = bankroll_curve.cummax()
            dd_pct = (bankroll_curve - peak_curve) / peak_curve * 100
            axes[1].fill_between(df_eq["date"], dd_pct, 0,
                                  alpha=0.5, color=C[1])
            axes[1].plot(df_eq["date"], dd_pct, lw=0.8, color=C[1])
            axes[1].axhline(0, color="black", lw=0.7)
            axes[1].set_ylabel("Drawdown (%)")
            axes[1].set_xlabel("Date")
            axes[1].set_title("Rolling Drawdown from Peak")
            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 4 · Fold Stability — IS vs OOS Sharpe ───────────────────
        if not fold_df.empty:
            fig, axes = plt.subplots(2, 1, figsize=(14, 9))
            fig.suptitle("Walk-Forward Fold Stability", fontsize=14)

            for scheme, color_is, color_oos, ax in [
                ("expanding", "#2C7BB6", "#D7191C", axes[0]),
                ("rolling",   "#1A9641", "#FDAE61", axes[1]),
            ]:
                sub = fold_df[fold_df["scheme"] == scheme].reset_index(drop=True)
                if sub.empty:
                    ax.text(0.5, 0.5, f"No {scheme} folds", ha="center", transform=ax.transAxes)
                    continue
                x = np.arange(len(sub))
                w = 0.35
                ax.bar(x - w/2, sub.get("is_sharpe", [0]*len(sub)),  w,
                       label="IS Sharpe",  color=color_is,  alpha=0.8)
                ax.bar(x + w/2, sub.get("oos_sharpe", [0]*len(sub)), w,
                       label="OOS Sharpe", color=color_oos, alpha=0.8)
                ax.axhline(0, color="black", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["fold"], rotation=35, ha="right", fontsize=7)
                ax.set_title(f"{scheme.capitalize()} Window Folds")
                ax.set_ylabel("Sharpe Ratio")
                ax.legend()

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 5 · OOS Accuracy & AUC by Fold ──────────────────────────
        if not fold_df.empty and "oos_accuracy" in fold_df.columns:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("OOS Model Performance by Fold", fontsize=13)

            for scheme, ax, color in [
                ("expanding", axes[0], C[0]),
                ("rolling",   axes[1], C[2]),
            ]:
                sub = fold_df[fold_df["scheme"] == scheme].reset_index(drop=True)
                if sub.empty:
                    ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
                    continue
                x   = np.arange(len(sub))
                ax.bar(x, sub["oos_accuracy"], color=color, alpha=0.8, label="Accuracy %")
                ax.plot(x, sub.get("oos_auc", [0]*len(sub)) * 100,
                        "o--", color=C[1], lw=1.5, label="AUC × 100")
                ax.axhline(50, color="grey", linestyle=":", lw=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(sub["fold"], rotation=35, ha="right", fontsize=7)
                ax.set_title(f"{scheme.capitalize()} — Accuracy & AUC")
                ax.set_ylabel("%")
                ax.set_ylim(45, 75)
                ax.legend()

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 6 · OOS ROI & N-Bets by Fold ────────────────────────────
        if not fold_df.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("OOS Betting Performance by Fold", fontsize=13)

            ax = axes[0]
            colors_roi = [C[2] if v >= 0 else C[1]
                          for v in fold_df.get("oos_roi", [0]*len(fold_df))]
            ax.bar(range(len(fold_df)), fold_df.get("oos_roi", [0]*len(fold_df)),
                   color=colors_roi, alpha=0.85)
            ax.axhline(0, color="black", lw=0.8)
            ax.set_xticks(range(len(fold_df)))
            ax.set_xticklabels(fold_df["fold"], rotation=35, ha="right", fontsize=6)
            ax.set_title("OOS ROI (%) per Fold")
            ax.set_ylabel("ROI %")

            ax = axes[1]
            ax.bar(range(len(fold_df)), fold_df.get("oos_n_bets", [0]*len(fold_df)),
                   color=C[0], alpha=0.85)
            ax.set_xticks(range(len(fold_df)))
            ax.set_xticklabels(fold_df["fold"], rotation=35, ha="right", fontsize=6)
            ax.set_title("OOS Number of Bets per Fold")
            ax.set_ylabel("Bets")

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 7 · Parameter Heatmap — Sharpe Surface ───────────────────
        if not oos_bets.empty:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle("Parameter Sensitivity — OOS Data", fontsize=13)

            for metric, ax, title in [
                ("sharpe", axes[0], "Sharpe Surface"),
                ("roi",    axes[1], "ROI Surface"),
            ]:
                mes  = PARAM_GRID["min_edge"]
                kfs  = PARAM_GRID["kelly_frac"]
                mat  = np.full((len(mes), len(kfs)), np.nan)
                sub_d = dict(
                    dates=oos_bets["date"].values,
                    model_prob=oos_bets["model_prob"].values,
                    market_prob=oos_bets["market_prob"].values,
                    outcomes=oos_bets["outcome"].values,
                    surfaces=oos_bets["surface"].values,
                    p1_ranks=oos_bets["p1_rank"].values,
                    bankroll_init=BANKROLL_INIT,
                )
                for i, me in enumerate(mes):
                    for j, kf in enumerate(kfs):
                        b = simulate_betting(**sub_d, min_edge=me, kelly_frac=kf, vig=0.07)
                        m = compute_metrics(b)
                        if m["n_bets"] >= 15:
                            mat[i, j] = m.get(metric, 0.0)

                sns.heatmap(pd.DataFrame(mat, index=mes, columns=kfs),
                            ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                            linewidths=0.4, cbar_kws={"shrink": 0.8})
                ax.set_xlabel("Kelly Fraction")
                ax.set_ylabel("Min Edge")
                ax.set_title(title)

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 8 · Vig × Edge Sharpe Heatmap ───────────────────────────
        if not oos_bets.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            vigs = PARAM_GRID["vig"]
            mes  = PARAM_GRID["min_edge"]
            mat  = np.full((len(vigs), len(mes)), np.nan)
            sub_d = dict(
                dates=oos_bets["date"].values,
                model_prob=oos_bets["model_prob"].values,
                market_prob=oos_bets["market_prob"].values,
                outcomes=oos_bets["outcome"].values,
                surfaces=oos_bets["surface"].values,
                p1_ranks=oos_bets["p1_rank"].values,
                bankroll_init=BANKROLL_INIT,
            )
            for i, v in enumerate(vigs):
                for j, me in enumerate(mes):
                    b = simulate_betting(**sub_d, min_edge=me, kelly_frac=0.25, vig=v)
                    m = compute_metrics(b)
                    if m["n_bets"] >= 15:
                        mat[i, j] = m.get("sharpe", 0.0)

            sns.heatmap(pd.DataFrame(mat, index=vigs, columns=mes),
                        ax=ax, cmap="RdYlGn", annot=True, fmt=".3f",
                        linewidths=0.4)
            ax.set_title("Sharpe — Vig × Min Edge (Kelly=0.25)")
            ax.set_xlabel("Min Edge"); ax.set_ylabel("Vig")
            _savefig(pdf, fig)

        # ── PAGE 9 · Monthly P&L Calendar Heatmap ────────────────────────
        if not oos_bets.empty:
            fig, ax = plt.subplots(figsize=(14, 5))
            df_mo = oos_bets.copy()
            df_mo["date2"] = pd.to_datetime(df_mo["date"].astype(str),
                                             format="%Y%m%d", errors="coerce")
            df_mo = df_mo.dropna(subset=["date2"])
            df_mo["yr"]  = df_mo["date2"].dt.year
            df_mo["mo"]  = df_mo["date2"].dt.month
            pivot = df_mo.groupby(["yr","mo"])["profit"].sum().unstack("mo")
            pivot.columns = [f"M{c:02d}" for c in pivot.columns]
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".0f",
                        center=0, linewidths=0.3, cbar_kws={"shrink":0.8})
            ax.set_title("Monthly P&L ($) — OOS Bets")
            ax.set_xlabel("Month"); ax.set_ylabel("Year")
            _savefig(pdf, fig)

        # ── PAGE 10 · Model Calibration ───────────────────────────────────
        if len(agg_y_true) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Model Calibration & Discrimination (All OOS Folds)", fontsize=13)

            ax = axes[0]
            prob_true, prob_pred = calibration_curve(agg_y_true, agg_y_prob, n_bins=12)
            ax.plot(prob_pred, prob_true, "o-", color=C[0], lw=2, label="Model")
            ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect")
            ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.15, color=C[1])
            ax.set_title("Reliability Diagram (Calibration Curve)")
            ax.set_xlabel("Mean Predicted Probability")
            ax.set_ylabel("Fraction of Positives")
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.text(0.05, 0.92, f"Brier={agg_brier:.4f}", transform=ax.transAxes, fontsize=10)
            ax.legend()

            ax = axes[1]
            fpr, tpr, _ = roc_curve(agg_y_true, agg_y_prob)
            ax.plot(fpr, tpr, color=C[0], lw=2, label=f"AUC = {agg_auc:.4f}")
            ax.plot([0,1],[0,1], "--", color="grey")
            ax.fill_between(fpr, tpr, alpha=0.12, color=C[0])
            ax.set_title("ROC Curve")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right")

            _savefig(pdf, fig)

        # ── PAGE 11 · Confidence Bin vs Actual Win Rate (Calibration Heatmap)
        if len(agg_y_true) > 0:
            fig, ax = plt.subplots(figsize=(11, 3))
            bins   = np.arange(0.45, 1.01, 0.05)
            labels = [f"{b:.2f}-{b+.05:.2f}" for b in bins[:-1]]
            cats   = pd.cut(agg_y_prob, bins=bins, labels=labels)
            cal_df = pd.DataFrame({"bin": cats, "outcome": agg_y_true}).dropna(subset=["bin"])
            wr_df  = cal_df.groupby("bin", observed=True)["outcome"].agg(["mean","count"]).reset_index()
            pivot  = wr_df.set_index("bin")[["mean"]].T
            annot  = [[f"{v:.2f}\n(n={n_})" for v, n_ in
                       zip(wr_df["mean"].values, wr_df["count"].values)]]
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=annot, fmt="",
                        vmin=0.3, vmax=0.9, linewidths=0.4, cbar_kws={"shrink":0.8})
            ax.set_title("Confidence Bin → Actual Win Rate (Calibration Check)")
            ax.set_xlabel("Model Confidence Bin"); ax.set_yticks([])
            _savefig(pdf, fig)

        # ── PAGE 12 · Edge × Surface ROI Heatmap ─────────────────────────
        if not oos_bets.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            df_es = oos_bets.copy()
            df_es["edge_bin"] = pd.cut(df_es["edge"],
                                        bins=[0,.04,.06,.08,.10,.99],
                                        labels=["<4%","4-6%","6-8%","8-10%",">10%"])
            df_es["roi_bet"] = df_es["profit"] / df_es["bet_size"].clip(lower=1e-6) * 100
            pivot = df_es.groupby(["edge_bin","surface"], observed=True)["roi_bet"] \
                         .mean().unstack("surface").fillna(0)
            sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".1f",
                        center=0, linewidths=0.4)
            ax.set_title("ROI (%) — Edge Bin × Surface")
            ax.set_xlabel("Surface"); ax.set_ylabel("Edge Bin")
            _savefig(pdf, fig)

        # ── PAGE 13 · Accuracy by Surface × Year (Heatmap) ───────────────
        if len(agg_y_true) > 0:
            # Rebuild per-row predictions for all OOS folds
            pred_rows = []
            for label, (yt, yp, te_mask) in wfo_obj.fold_probs.items():
                sub = df[te_mask].copy()
                sub["y_true"] = yt
                sub["y_prob"] = yp
                sub["y_pred"] = (yp >= 0.5).astype(int)
                sub["correct"] = (sub["y_pred"] == sub["y_true"]).astype(int)
                pred_rows.append(sub)

            if pred_rows:
                pred_df = pd.concat(pred_rows, ignore_index=True)
                pivot = pred_df.groupby(["year","surface_label"])["correct"] \
                               .mean().unstack("surface_label")
                pivot = pivot[[c for c in ["Hard","Clay","Grass"] if c in pivot.columns]]
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(pivot, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                            vmin=0.45, vmax=0.70, linewidths=0.4)
                ax.set_title("Model Accuracy — Year × Surface (OOS Only)")
                ax.set_xlabel("Surface"); ax.set_ylabel("Year")
                _savefig(pdf, fig)

        # ── PAGE 14 · Rank Tier × Surface Accuracy ────────────────────────
        if pred_rows:
            fig, ax = plt.subplots(figsize=(9, 6))
            pred_df["rank_tier"] = pd.cut(pred_df["P1_Rank"],
                                           bins=[0,10,50,100,200,500,9999],
                                           labels=["Top-10","11-50","51-100","101-200","201-500","500+"])
            pivot2 = pred_df.groupby(["rank_tier","surface_label"], observed=True)["correct"] \
                            .mean().unstack("surface_label")
            pivot2 = pivot2[[c for c in ["Hard","Clay","Grass"] if c in pivot2.columns]]
            sns.heatmap(pivot2, ax=ax, cmap="RdYlGn", annot=True, fmt=".2f",
                        vmin=0.40, vmax=0.75, linewidths=0.4)
            ax.set_title("Accuracy — P1 Rank Tier × Surface (OOS)")
            _savefig(pdf, fig)

        # ── PAGE 15 · Feature Correlation Heatmap ────────────────────────
        corr_cols = [c for c in ALL_FEATURES + ["target"] if c in df.columns]
        fig, ax = plt.subplots(figsize=(13, 11))
        corr_mat = df[corr_cols].corr(method="spearman")
        mask = np.triu(np.ones_like(corr_mat, dtype=bool))
        sns.heatmap(corr_mat, ax=ax, mask=mask, cmap="coolwarm",
                    annot=True, fmt=".2f", linewidths=0.2,
                    vmin=-1, vmax=1, square=True, cbar_kws={"shrink":0.6},
                    annot_kws={"size": 7})
        ax.set_title("Feature Correlation Matrix (Spearman) — Full Dataset")
        plt.tight_layout()
        _savefig(pdf, fig)

        # ── PAGE 16 · Parameter Consensus Heatmap ────────────────────────
        if not fold_df.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            rows = []
            for param in ["best_min_edge","best_kelly_frac","best_vig"]:
                if param not in fold_df.columns:
                    continue
                for val, freq in fold_df[param].value_counts(normalize=True).items():
                    rows.append({"param": param.replace("best_",""), "value": str(val), "freq": freq})
            if rows:
                p_df = pd.DataFrame(rows)
                pivot_p = p_df.pivot(index="value", columns="param", values="freq").fillna(0)
                sns.heatmap(pivot_p, ax=ax, cmap="Blues", annot=True, fmt=".2f",
                            linewidths=0.3)
                ax.set_title("Parameter Selection Frequency Across All Folds")
                ax.set_xlabel("Parameter"); ax.set_ylabel("Value")
            else:
                ax.text(0.5, 0.5, "No param data", ha="center", transform=ax.transAxes)
            _savefig(pdf, fig)

        # ── PAGE 17 · Bet Diagnostics (4-panel) ───────────────────────────
        if not oos_bets.empty:
            fig, axes = plt.subplots(2, 2, figsize=(13, 9))
            fig.suptitle("OOS Bet Diagnostics", fontsize=14)

            axes[0,0].hist(oos_bets["model_prob"], bins=35, color=C[0], edgecolor="white")
            axes[0,0].set_title("Model Probability Distribution")
            axes[0,0].set_xlabel("P(P1 wins)")

            axes[0,1].hist(oos_bets["edge"], bins=35, color=C[2], edgecolor="white")
            axes[0,1].axvline(0.04, color=C[1], linestyle="--", label="Min edge")
            axes[0,1].set_title("Edge Distribution"); axes[0,1].set_xlabel("Edge")
            axes[0,1].legend()

            axes[1,0].hist(oos_bets["bet_size"], bins=35, color=C[3], edgecolor="white")
            axes[1,0].set_title("Bet Size Distribution"); axes[1,0].set_xlabel("$ Bet")

            sc = axes[1,1].scatter(oos_bets["edge"], oos_bets["profit"],
                                   alpha=0.25, s=7, c=oos_bets["outcome"],
                                   cmap="RdYlGn", vmin=0, vmax=1)
            axes[1,1].axhline(0, color="black", lw=0.8)
            axes[1,1].axvline(0.04, color=C[1], linestyle="--", lw=0.8)
            axes[1,1].set_title("Edge vs Profit (green=win)")
            axes[1,1].set_xlabel("Edge"); axes[1,1].set_ylabel("$")
            plt.colorbar(sc, ax=axes[1,1], label="Outcome")

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 18 · Win/Loss by surface + Kelly distribution ────────────
        if not oos_bets.empty:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Surface & Sizing Analysis", fontsize=13)

            ax = axes[0]
            surf_stats = oos_bets.groupby("surface").agg(
                win_rate=("outcome", "mean"),
                n_bets=("outcome", "count"),
                roi=("profit", lambda x: x.sum() / oos_bets.loc[x.index,"bet_size"].sum() * 100),
            ).reset_index()
            x = np.arange(len(surf_stats))
            bars = ax.bar(x, surf_stats["win_rate"] * 100, color=[C[0],C[2],C[3]][:len(x)], alpha=0.85)
            for bar, row in zip(bars, surf_stats.itertuples()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f"{bar.get_height():.1f}%\n(n={row.n_bets})\nROI:{row.roi:.1f}%",
                        ha="center", fontsize=9)
            ax.set_xticks(x); ax.set_xticklabels(surf_stats["surface"])
            ax.set_title("Win Rate, Count & ROI by Surface")
            ax.set_ylabel("Win Rate (%)")
            ax.set_ylim(0, 80)

            ax = axes[1]
            ax.hist(oos_bets["kelly_frac_used"], bins=30, color=C[4], edgecolor="white")
            ax.set_title("Kelly Fraction Used per Bet")
            ax.set_xlabel("Fraction of Bankroll")
            ax.set_ylabel("Count")

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 19 · Annual accuracy bar + probability distributions ─────
        if pred_rows:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Accuracy Analysis (OOS Folds)", fontsize=13)

            ax = axes[0]
            ann = pred_df.groupby("year")["correct"].agg(["mean","count"])
            bars = ax.bar(ann.index, ann["mean"] * 100, color=C[0], alpha=0.85)
            ax.axhline(pred_df["correct"].mean() * 100, color=C[1], linestyle="--", lw=1.2, label="Overall avg")
            for yr, row in ann.iterrows():
                ax.text(yr, row["mean"]*100 + 0.5, f"{row['mean']*100:.0f}%\n({row['count']})",
                        ha="center", fontsize=7)
            ax.set_ylim(40, 80)
            ax.set_xlabel("Year"); ax.set_ylabel("Accuracy (%)")
            ax.set_title("Annual Accuracy")
            ax.legend()

            ax = axes[1]
            ax.hist(pred_df.loc[pred_df["y_true"]==1,"y_prob"], bins=30,
                    alpha=0.65, color=C[2], label="P1 Won")
            ax.hist(pred_df.loc[pred_df["y_true"]==0,"y_prob"], bins=30,
                    alpha=0.65, color=C[1], label="P1 Lost")
            ax.axvline(0.5, color="black", linestyle="--", lw=0.8)
            ax.set_title("Probability Distribution by Outcome")
            ax.set_xlabel("P(P1 wins)"); ax.set_ylabel("Count")
            ax.legend()

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 20 · Confusion Matrix + Rank analysis ────────────────────
        if pred_rows and len(pred_df) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(13, 5))
            fig.suptitle("Classification Diagnostics", fontsize=13)

            ax = axes[0]
            cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0,1]); ax.set_yticks([0,1])
            ax.set_xticklabels(["P1 Loses","P1 Wins"])
            ax.set_yticklabels(["P1 Loses","P1 Wins"])
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                            color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
            ax.set_title("Confusion Matrix (All OOS)"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            plt.colorbar(im, ax=ax)

            ax = axes[1]
            rank_diff = pred_df["P1_Rank"] - pred_df["P2_Rank"]
            bins = [-np.inf,-100,-20,0,20,100,np.inf]
            labels = ["P1 much better","P1 better","P1 slight",
                      "P2 slight","P2 better","P2 much better"]
            pred_df["rank_bucket"] = pd.cut(rank_diff, bins=bins, labels=labels)
            bucket_acc = pred_df.groupby("rank_bucket", observed=True)["correct"].mean() * 100
            bucket_n   = pred_df.groupby("rank_bucket", observed=True)["correct"].count()
            bucket_acc.plot(kind="bar", ax=ax, color=C[0], alpha=0.85, rot=30)
            for i, (a, n_) in enumerate(zip(bucket_acc, bucket_n)):
                ax.text(i, a + 0.5, f"{a:.0f}%\nn={n_}", ha="center", fontsize=8)
            ax.set_ylim(40, 90)
            ax.set_ylabel("Accuracy (%)")
            ax.set_title("Accuracy by Rank Differential")

            plt.tight_layout()
            _savefig(pdf, fig)

        # ── PAGE 21 · Fold KPI Table ──────────────────────────────────────
        if not fold_df.empty:
            show_cols = ["fold","n_train","n_test","oos_accuracy","oos_auc",
                         "oos_n_bets","oos_win_rate","oos_roi","oos_sharpe",
                         "oos_max_drawdown","best_min_edge","best_kelly_frac","best_vig"]
            show_cols = [c for c in show_cols if c in fold_df.columns]
            pretty    = [c.replace("_"," ").replace("oos ","OOS ").replace("best ","").title()
                         for c in show_cols]

            # split into chunks of 20 rows per page
            chunk_size = 20
            for chunk_start in range(0, len(fold_df), chunk_size):
                chunk = fold_df[show_cols].iloc[chunk_start:chunk_start+chunk_size]
                fig, ax = plt.subplots(figsize=(16, max(3, len(chunk)*0.55 + 2)))
                ax.axis("off")
                tbl_data = chunk.round(3).astype(str).values.tolist()
                table = ax.table(cellText=tbl_data,
                                  colLabels=pretty,
                                  loc="center", cellLoc="center")
                table.auto_set_font_size(False)
                table.set_fontsize(7)
                table.scale(1.2, 1.6)
                page_lbl = f" (rows {chunk_start+1}–{chunk_start+len(chunk)})" if len(fold_df) > chunk_size else ""
                ax.set_title(f"Walk-Forward Fold Summary{page_lbl}", fontsize=12, pad=12)
                _savefig(pdf, fig)

        # ── PAGE 22 · Methodology ─────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        ax = fig.add_axes([0.05, 0.02, 0.90, 0.95])
        ax.set_axis_off()
        text = (
            "Methodology & Design Notes\n"
            "══════════════════════════════════════════════════════════════════\n\n"
            "DATA\n"
            f"  · Source: clean_tennis_data.csv  ({len(df):,} matches, 2000–2024)\n"
            "  · 16 raw columns: surface dummies, best-of, handedness, height, age,\n"
            "    ATP rank, ATP rank-points, and P1 win label (Target_P1_Wins)\n"
            "  · Zero missing values; P1/P2 already randomly assigned in the CSV\n\n"
            "DERIVED FEATURES (added at runtime)\n"
            "  Height_Diff, Age_Diff, Rank_Diff, Rank_Points_Diff,\n"
            "  Surface_Height_Grass, Age_Diff_Sq, market_prob\n\n"
            "MARKET PROBABILITY PROXY\n"
            "  market_prob_p1 = P1_Rank_Points / (P1_Rank_Points + P2_Rank_Points)\n"
            "  Higher ATP points → implied higher win probability\n"
            "  Clipped to [0.05, 0.95] to avoid extreme odds\n\n"
            "WALK-FORWARD SCHEME\n"
            "  Expanding window : train from 2000 up to year T, test year T+1\n"
            "  Rolling window   : train on 6 years, test next year\n"
            "  Both schemes run; OOS periods never overlap training data\n\n"
            "MODEL (per fold)\n"
            "  XGBoost classifier: n_estimators=300, max_depth=4, lr=0.05,\n"
            "  subsample=0.8, colsample_bytree=0.8, min_child_weight=10\n"
            "  Trained fresh on each fold's training window\n\n"
            "PARAMETER GRID SEARCH (in-sample Sharpe optimisation)\n"
            "  min_edge   : [0.02, 0.04, 0.06, 0.08, 0.10]\n"
            "  kelly_frac : [0.05, 0.10, 0.20, 0.35, 0.50]\n"
            "  vig        : [0.05, 0.07, 0.09, 0.11]\n"
            "  100 combinations per fold; objective = Sharpe ratio\n\n"
            "BETTING SIMULATION\n"
            "  · Bet triggered when model_prob − market_prob > min_edge (post-vig)\n"
            "  · Confidence-tiered Kelly fraction:\n"
            "    model_prob ≥ 0.70 → tier 0.40 | ≥ 0.60 → 0.25\n"
            "    ≥ 0.55 → 0.12 | < 0.55 → 0.05\n"
            "  · Position cap: 20% of bankroll per bet\n"
            "  · Starting bankroll: $500\n"
            "  · Binary Kalshi-style payout:\n"
            "    Win  → profit = bet × (1 − mkp_adj) / mkp_adj\n"
            "    Lose → loss   = bet\n\n"
            "ANTI-LEAKAGE\n"
            "  · Model trained strictly on past data; future years never seen\n"
            "  · market_prob uses only features available at match time\n"
            "  · No post-match statistics used as features\n"
        )
        ax.text(0.0, 1.0, text, va="top", ha="left",
                fontsize=9.5, fontfamily="monospace",
                transform=ax.transAxes, color="#111111")
        _savefig(pdf, fig)

    log.info(f"PDF saved → {pdf_path}")


# ─── 8. Entry point ───────────────────────────────────────────────────────────

def main():
    log.info("Loading and engineering features from clean_tennis_data.csv…")
    df = load_and_engineer(CSV_PATH)
    log.info(f"Dataset: {len(df):,} rows, years {df['year'].min()}–{df['year'].max()}")

    wfo = WFO(df)
    log.info("Starting Walk-Forward Optimisation (both expanding + rolling)…")
    oos_bets = wfo.run(scheme="both")
    fold_df  = wfo.fold_df()

    if not oos_bets.empty:
        oos_bets.to_csv(CSV_OUT, index=False)
        log.info(f"Per-bet CSV: {CSV_OUT}  ({len(oos_bets):,} rows)")

    if not fold_df.empty:
        param_cols = ["best_min_edge","best_kelly_frac","best_vig"]
        consensus  = {}
        for col in param_cols:
            if col in fold_df.columns:
                consensus[col.replace("best_","")] = float(fold_df[col].mode().iloc[0])
        json_payload = {
            "generated":       datetime.now().isoformat(),
            "consensus_params": consensus,
            "fold_params":     fold_df[["fold"] + [c for c in param_cols if c in fold_df.columns]]
                               .to_dict(orient="records"),
        }
        with open(JSON_OUT, "w") as f:
            json.dump(json_payload, f, indent=2)
        log.info(f"Consensus params: {consensus}")
        log.info(f"Params JSON: {JSON_OUT}")

    generate_pdf(df, oos_bets, fold_df, wfo, PDF_OUT)

    log.info("Done.")
    if not fold_df.empty and "oos_sharpe" in fold_df.columns:
        top = fold_df.nlargest(5, "oos_sharpe")[
            ["fold","oos_sharpe","oos_roi","oos_n_bets","best_min_edge","best_kelly_frac","best_vig"]
        ]
        print("\n" + "="*70)
        print("TOP 5 FOLDS BY OOS SHARPE")
        print("="*70)
        print(top.to_string(index=False))
        print("="*70 + "\n")

    if not oos_bets.empty:
        m = compute_metrics(oos_bets)
        print(f"OVERALL OOS  |  bets={m['n_bets']:,}  win={m['win_rate']:.1f}%  "
              f"roi={m['roi']:+.2f}%  sharpe={m['sharpe']:.4f}  "
              f"maxDD={m['max_drawdown']:.1f}%  bankroll=${m['bankroll_final']:,.2f}")


if __name__ == "__main__":
    main()
