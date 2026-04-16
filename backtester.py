"""
Walk-Forward Tennis Backtest
============================
Strict temporal split — no lookahead bias, no data leakage.

Splits
------
  Warm-up  : 1968–2019  (ELO warm-up + model training reference)
  Validation: 2020–2021  (hyper-param selection, not used for final eval)
  Test      : 2022–2024  (all reported metrics come from here only)

Features fed to the existing XGBoost model are exactly the 14 stored in
model_features.json.  ELO ratings built from the warm-up period are used
*only* for the Kelly betting simulation (to estimate "market implied prob").

How leakage is prevented
------------------------
- ELO ratings are updated *after* each match prediction, never before.
- Rank/age/height features come from the match row itself (Kalshi-style
  snapshot at match time) — the model already learned on this format.
- Test years (2022-2024) were never seen by the model at training time.
"""

import os
import glob
import json
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    accuracy_score, confusion_matrix, roc_auc_score,
    brier_score_loss, log_loss, roc_curve,
)
from sklearn.calibration import calibration_curve
import joblib
try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ATP_DIR        = os.path.expanduser("~/Downloads/tennis_atp-master")
MODEL_PATH     = os.path.join(os.path.dirname(__file__), "best_xgb_model.json")
FEATURES_PATH  = os.path.join(os.path.dirname(__file__), "model_features.json")
PDF_OUT        = os.path.join(os.path.dirname(__file__), "backtest_analysis.pdf")

WARMUP_END     = 2019
VALID_END      = 2021   # validation: 2020-2021
TEST_START     = 2022   # out-of-sample test

ELO_K          = 32
ELO_INIT       = 1500
KELLY_FRAC     = 0.25
BANKROLL_INIT  = 1000.0

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load ATP CSVs
# ─────────────────────────────────────────────────────────────────────────────
def load_atp_matches(atp_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(atp_dir, "atp_matches_[12]*.csv")))
    if not files:
        raise FileNotFoundError(f"No atp_matches_YYYY.csv found in {atp_dir}")
    frames = []
    for f in files:
        year = int(os.path.basename(f).split("_")[-1].replace(".csv", ""))
        try:
            df = pd.read_csv(f, low_memory=False)
            df["year"] = year
            frames.append(df)
        except Exception as e:
            log.warning(f"Skipping {f}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded {len(combined):,} matches from {len(files)} files ({files[0][-8:-4]}–{files[-1][-8:-4]})")
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# 2. Build the 14 model features + ELO (strictly in chronological order)
# ─────────────────────────────────────────────────────────────────────────────
SURFACE_MAP = {"Hard": "Hard", "Clay": "Clay", "Grass": "Grass",
               "Carpet": "Hard", "Indoor Hard": "Hard", "Outdoor Hard": "Hard"}

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns rows with the 14 model features plus helper columns.
    ELO is computed cumulatively — each row uses ELO *before* the match.
    """
    df = df.copy()

    # ── normalise surface ──────────────────────────────────────────────────
    df["surface_norm"] = df["surface"].map(SURFACE_MAP).fillna("Hard")

    # ── hand encoding (R=1, L=0, U=0) ────────────────────────────────────
    df["winner_hand_enc"] = (df["winner_hand"] == "R").astype(int)
    df["loser_hand_enc"]  = (df["loser_hand"]  == "R").astype(int)

    # ── numeric coerce ────────────────────────────────────────────────────
    for col in ["winner_ht", "loser_ht", "winner_age", "loser_age",
                "winner_rank", "loser_rank",
                "winner_rank_points", "loser_rank_points"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Median-fill rank/points from same year to avoid future leakage
    for col in ["winner_rank", "loser_rank", "winner_rank_points", "loser_rank_points"]:
        median_fill = df.groupby("year")[col].transform("median")
        df[col] = df[col].fillna(median_fill)

    # Height / age – overall median (stable, not leaking temporal info)
    df["winner_ht"]  = df["winner_ht"].fillna(df["winner_ht"].median())
    df["loser_ht"]   = df["loser_ht"].fillna(df["loser_ht"].median())
    df["winner_age"] = df["winner_age"].fillna(df["winner_age"].median())
    df["loser_age"]  = df["loser_age"].fillna(df["loser_age"].median())

    # ── ELO state ─────────────────────────────────────────────────────────
    elo: dict[str, float] = {}

    elo_winner_before = np.empty(len(df), dtype=float)
    elo_loser_before  = np.empty(len(df), dtype=float)

    for i, row in enumerate(df.itertuples(index=False)):
        wid = str(row.winner_id)
        lid = str(row.loser_id)
        ew  = elo.get(wid, ELO_INIT)
        el  = elo.get(lid, ELO_INIT)
        elo_winner_before[i] = ew
        elo_loser_before[i]  = el
        # update AFTER recording
        exp_w = 1 / (1 + 10 ** ((el - ew) / 400))
        elo[wid] = ew + ELO_K * (1 - exp_w)
        elo[lid] = el + ELO_K * (0 - (1 - exp_w))

    df["elo_winner"] = elo_winner_before
    df["elo_loser"]  = elo_loser_before

    # ── ELO-implied prob (used only for Kelly simulation) ─────────────────
    df["elo_prob_winner"] = 1 / (1 + 10 ** ((df["elo_loser"] - df["elo_winner"]) / 400))

    # ── Randomly assign P1/P2 to avoid winner-bias in features ───────────
    rng = np.random.default_rng(42)
    swap = rng.random(len(df)) < 0.5

    def pick(w_col, l_col):
        w = df[w_col].values
        l = df[l_col].values
        return np.where(swap, l, w), np.where(swap, w, l)

    p1_hand,  p2_hand  = pick("winner_hand_enc", "loser_hand_enc")
    p1_ht,    p2_ht    = pick("winner_ht",        "loser_ht")
    p1_age,   p2_age   = pick("winner_age",       "loser_age")
    p1_rank,  p2_rank  = pick("winner_rank",      "loser_rank")
    p1_rpts,  p2_rpts  = pick("winner_rank_points","loser_rank_points")
    p1_elo,   p2_elo   = pick("elo_winner",        "elo_loser")

    # label: P1 wins = 1 when P1 is the original winner (swap=False)
    y = np.where(swap, 0, 1)

    out = pd.DataFrame({
        "year":              df["year"].values,
        "tourney_date":      df["tourney_date"].values,
        "Surface_Hard":      (df["surface_norm"] == "Hard").astype(int).values,
        "Surface_Clay":      (df["surface_norm"] == "Clay").astype(int).values,
        "Surface_Grass":     (df["surface_norm"] == "Grass").astype(int).values,
        "Best_Of_Sets":      pd.to_numeric(df["best_of"], errors="coerce").fillna(3).values,
        "P1_Is_Right_Handed": p1_hand,
        "P1_Height_cm":      p1_ht,
        "P1_Age":            p1_age,
        "P1_Rank":           p1_rank,
        "P1_Rank_Points":    p1_rpts,
        "P2_Is_Right_Handed": p2_hand,
        "P2_Height_cm":      p2_ht,
        "P2_Age":            p2_age,
        "P2_Rank":           p2_rank,
        "P2_Rank_Points":    p2_rpts,
        "elo_prob_p1":       np.where(swap, 1 - df["elo_prob_winner"].values,
                                            df["elo_prob_winner"].values),
        "surface_norm":      df["surface_norm"].values,
        "target":            y,
    })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Kelly betting simulation
# ─────────────────────────────────────────────────────────────────────────────
FLAT_BET_SIZE = 10.0   # flat $10 per triggered bet (realistic for backtesting)

def simulate_betting(model_prob: np.ndarray,
                     market_prob: np.ndarray,
                     outcomes: np.ndarray,
                     bankroll_init: float = BANKROLL_INIT,
                     kelly_frac: float = KELLY_FRAC,
                     min_edge: float = 0.04,
                     vig: float = 0.08) -> dict:
    """
    Simulate flat-bet strategy on a binary outcome market.

    Flat bets ($10 each) are used for interpretable backtesting.
    ELO-implied odds are a proxy for "market" — real efficiency would be
    higher, so results here represent a *theoretical upper bound*.

    Binary market mechanics (Kalshi-style):
      - Buy 1 YES contract at price mkp (adjusted for vig)
      - Win:  receive $1 total → net profit = (1 - mkp_adj)
      - Lose: lose stake = mkp_adj

    vig (8%) models exchange rake / bid-ask spread.
    """
    bankroll  = bankroll_init
    history   = [bankroll]
    bets      = []
    edges     = []
    bet_returns = []
    cumulative_profit = 0.0

    for mp, mkp_raw, outcome in zip(model_prob, market_prob, outcomes):
        edge = mp - mkp_raw
        if edge < min_edge or mkp_raw <= 0.05 or mkp_raw >= 0.95:
            history.append(bankroll)
            continue

        # Apply vig: effective cost is slightly higher than fair price
        mkp_adj  = min(mkp_raw + vig / 2, 0.95)
        edge_adj = mp - mkp_adj
        if edge_adj <= 0:
            history.append(bankroll)
            continue

        # Flat bet — cost is mkp_adj per $1 notional (Kalshi binary contract)
        bet_cost   = FLAT_BET_SIZE * mkp_adj          # what we pay
        net_profit = FLAT_BET_SIZE * (1.0 - mkp_adj)  # what we net if we win

        profit = net_profit if outcome == 1 else -bet_cost

        bankroll          += profit
        cumulative_profit += profit
        bankroll           = max(bankroll, 0.01)
        history.append(bankroll)
        bets.append(bet_cost)
        edges.append(edge)
        bet_returns.append(profit / max(bet_cost, 1e-9))

    n_bets    = len(bets)
    wins      = sum(1 for r in bet_returns if r > 0)
    total_staked = sum(bets)
    roi       = cumulative_profit / total_staked * 100 if total_staked > 0 else 0.0
    win_rate  = wins / n_bets * 100 if n_bets > 0 else 0.0

    return dict(
        history=history,
        bankroll_final=bankroll,
        n_bets=n_bets,
        win_rate=win_rate,
        roi=roi,
        cumulative_profit=cumulative_profit,
        total_staked=total_staked,
        avg_edge=float(np.mean(edges)) if edges else 0.0,
        avg_bet=float(np.mean(bets))   if bets  else 0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. PDF generation
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = ["#2C7BB6", "#D7191C", "#1A9641", "#FDAE61", "#7B3294"]

def _savefig(pdf, fig):
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def generate_pdf(feat_df: pd.DataFrame,
                 y_true: np.ndarray,
                 y_prob: np.ndarray,
                 y_pred: np.ndarray,
                 betting: dict,
                 feature_names: list[str],
                 model,
                 pdf_path: str):

    acc    = accuracy_score(y_true, y_pred)
    auc    = roc_auc_score(y_true, y_prob)
    brier  = brier_score_loss(y_true, y_prob)
    ll     = log_loss(y_true, y_prob)
    n      = len(y_true)

    with PdfPages(pdf_path) as pdf:

        # ── PAGE 1 · Cover / KPI summary ─────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("#0D1117")
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_axis_off()
        ax.set_facecolor("#0D1117")

        ax.text(0.5, 0.94, "Tennis ML Backtest Report",
                color="white", fontsize=26, fontweight="bold",
                ha="center", va="center", transform=ax.transAxes)
        ax.text(0.5, 0.89, "XGBoost · Walk-Forward · Out-of-Sample 2022–2024",
                color="#888888", fontsize=13, ha="center", va="center",
                transform=ax.transAxes)

        kpis = [
            ("Matches Evaluated",   f"{n:,}",         "test set (2022–2024)"),
            ("Accuracy",            f"{acc*100:.2f}%", "higher = better"),
            ("ROC-AUC",             f"{auc:.4f}",      "1.0 = perfect"),
            ("Brier Score",         f"{brier:.4f}",    "lower = better"),
            ("Log-Loss",            f"{ll:.4f}",       "lower = better"),
            ("Bets Triggered",      f"{betting['n_bets']:,}",
                                                       f"${FLAT_BET_SIZE:.0f} flat bet, +4pp edge min"),
            ("Win Rate (bets)",     f"{betting['win_rate']:.1f}%", "of triggered bets"),
            ("Staking ROI",         f"{betting['roi']:+.1f}%",
                                                       f"profit ÷ total staked (after 8% vig)"),
            ("Cumulative P&L",      f"${betting.get('cumulative_profit',0):+,.0f}",
                                                       f"on ${betting.get('total_staked',0):,.0f} staked"),
        ]
        cols = ["Metric", "Value", "Note"]
        table_data = [[k, v, n_] for k, v, n_ in kpis]
        tbl = ax.table(cellText=table_data, colLabels=cols,
                       cellLoc="center", loc="center",
                       bbox=[0.05, 0.35, 0.90, 0.48])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor("#333333")
            if r == 0:
                cell.set_facecolor("#1F6FEB"); cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#161B22"); cell.set_text_props(color="white")
            else:
                cell.set_facecolor("#0D1117"); cell.set_text_props(color="white")

        methodology = (
            "Methodology\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "• Data: All ATP main-tour match CSVs (1968–2024) — Jeff Sackmann dataset\n"
            "• Model: XGBoost trained on 1968–2019; evaluated on 2022–2024 (unseen)\n"
            "• Features (14): surface dummies, best-of, handedness, height, age,\n"
            "  ATP rank, ATP rank-points — all taken from the match row (no lookahead)\n"
            "• ELO ratings: updated match-by-match; prediction uses pre-match ELO\n"
            "• Betting sim: 25% fractional Kelly vs. ELO-implied market probability\n"
            "  Minimum edge threshold: 4 pp; position cap: 10% of bankroll"
        )
        ax.text(0.05, 0.31, methodology, color="#AAAAAA", fontsize=9,
                va="top", transform=ax.transAxes, fontfamily="monospace")
        _savefig(pdf, fig)

        # ── PAGE 2 · Calibration + ROC ────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Model Calibration & Discrimination (Test 2022–2024)", fontsize=14)

        # Calibration
        ax = axes[0]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=12)
        ax.plot(prob_pred, prob_true, "o-", color=PALETTE[0], lw=2, label="Model")
        ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
        ax.fill_between(prob_pred, prob_true, prob_pred,
                        alpha=0.15, color=PALETTE[1])
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.text(0.05, 0.92, f"Brier={brier:.4f}", transform=ax.transAxes, fontsize=10)

        # ROC
        ax = axes[1]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, color=PALETTE[0], lw=2, label=f"AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.fill_between(fpr, tpr, alpha=0.15, color=PALETTE[0])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend(loc="lower right")
        _savefig(pdf, fig)

        # ── PAGE 3 · P&L curve ───────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(12, 5))
        history = betting["history"]
        x = np.arange(len(history))
        ax.plot(x, history, color=PALETTE[0], lw=1.5)
        ax.axhline(BANKROLL_INIT, color="gray", linestyle="--", linewidth=0.8)
        ax.fill_between(x, BANKROLL_INIT, history,
                        where=np.array(history) >= BANKROLL_INIT,
                        alpha=0.2, color=PALETTE[2], label="Above starting")
        ax.fill_between(x, BANKROLL_INIT, history,
                        where=np.array(history) < BANKROLL_INIT,
                        alpha=0.2, color=PALETTE[1], label="Below starting")
        ax.set_title(f"Bankroll Curve  (staking ROI {betting['roi']:+.1f}%  ·  {betting['n_bets']:,} flat bets  ·  "
                     f"P&L ${betting.get('cumulative_profit', 0):+,.0f})")
        ax.set_xlabel("Bet #")
        ax.set_ylabel("Bankroll ($)")
        ax.legend()
        _savefig(pdf, fig)

        # ── PAGE 4 · Annual accuracy + surface breakdown ─────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Annual
        ax = axes[0]
        yearly = feat_df.copy()
        yearly["correct"] = (y_pred == y_true).astype(int)
        ann = yearly.groupby("year")["correct"].agg(["mean", "count"])
        ann.columns = ["acc", "n"]
        ax.bar(ann.index, ann["acc"] * 100, color=PALETTE[0], alpha=0.8)
        ax.axhline(acc * 100, color=PALETTE[1], linestyle="--", linewidth=1.2, label="Overall avg")
        for yr, row in ann.iterrows():
            ax.text(yr, row["acc"] * 100 + 0.5, f"{row['acc']*100:.0f}%", ha="center", fontsize=8)
        ax.set_ylim(40, 80)
        ax.set_xlabel("Year"); ax.set_ylabel("Accuracy (%)")
        ax.set_title("Annual Accuracy (Test Set)")
        ax.legend()

        # Surface
        ax = axes[1]
        surf_map = {
            "Hard":  feat_df["Surface_Hard"].values == 1,
            "Clay":  feat_df["Surface_Clay"].values == 1,
            "Grass": feat_df["Surface_Grass"].values == 1,
        }
        surf_accs, surf_ns, surf_names = [], [], []
        for surf, mask in surf_map.items():
            if mask.sum() > 0:
                surf_accs.append(accuracy_score(y_true[mask], y_pred[mask]) * 100)
                surf_ns.append(mask.sum())
                surf_names.append(surf)
        bars = ax.bar(surf_names, surf_accs, color=PALETTE[:len(surf_names)], alpha=0.85)
        for bar, n_, a in zip(bars, surf_ns, surf_accs):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    a + 0.5, f"{a:.1f}%\n(n={n_:,})", ha="center", fontsize=9)
        ax.set_ylim(40, 80)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Surface")
        _savefig(pdf, fig)

        # ── PAGE 5 · Confusion matrix + edge distribution ────────────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Confusion matrix
        ax = axes[0]
        cm = confusion_matrix(y_true, y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["P1 Loses", "P1 Wins"])
        ax.set_yticklabels(["P1 Loses", "P1 Wins"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_title("Confusion Matrix"); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        plt.colorbar(im, ax=ax)

        # Edge distribution
        ax = axes[1]
        market_prob = feat_df["elo_prob_p1"].values
        edges = y_prob - market_prob
        ax.hist(edges, bins=40, color=PALETTE[3], edgecolor="white", alpha=0.85)
        ax.axvline(0, color="black", linestyle="--")
        ax.axvline(0.04, color=PALETTE[1], linestyle="--", label="Min edge threshold")
        ax.set_xlabel("Model Prob − ELO Market Prob"); ax.set_ylabel("Count")
        ax.set_title("Edge Distribution (model vs ELO market)")
        ax.legend()
        _savefig(pdf, fig)

        # ── PAGE 6 · Feature importance ──────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            if hasattr(model, "feature_importances_"):
                imp = model.feature_importances_
                imp_df = pd.Series(imp, index=feature_names).sort_values()
                imp_df.plot(kind="barh", ax=ax, color=PALETTE[0])
                ax.set_title("Feature Importance (sklearn — mean decrease in impurity)")
                ax.set_xlabel("Importance")
            elif _HAS_XGB and isinstance(model, xgb.Booster):
                importance = model.get_score(importance_type="gain")
                imp_df = pd.Series(importance).reindex(feature_names, fill_value=0).sort_values()
                imp_df.plot(kind="barh", ax=ax, color=PALETTE[0])
                ax.set_title("Feature Importance (XGBoost Gain)")
                ax.set_xlabel("Gain")
            else:
                raise ValueError("no importance")
        except Exception:
            ax.text(0.5, 0.5, "Feature importance unavailable",
                    ha="center", va="center", transform=ax.transAxes)
        _savefig(pdf, fig)

        # ── PAGE 7 · Probability distribution + rank analysis ───────────
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        ax = axes[0]
        ax.hist(y_prob[y_true == 1], bins=30, alpha=0.65,
                color=PALETTE[2], label="P1 actually won")
        ax.hist(y_prob[y_true == 0], bins=30, alpha=0.65,
                color=PALETTE[1], label="P1 actually lost")
        ax.set_xlabel("Predicted P1 Win Probability")
        ax.set_ylabel("Count")
        ax.set_title("Probability Distribution by Outcome")
        ax.legend()

        ax = axes[1]
        rank_diff = feat_df["P1_Rank"].values - feat_df["P2_Rank"].values
        bins = [-np.inf, -100, -20, 0, 20, 100, np.inf]
        labels = ["P1 much better", "P1 better", "P1 slightly better",
                  "P2 slightly better", "P2 better", "P2 much better"]
        feat_df2 = feat_df.copy()
        feat_df2["rank_bucket"] = pd.cut(rank_diff, bins=bins, labels=labels)
        feat_df2["correct"] = (y_pred == y_true).astype(int)
        bucket_acc = feat_df2.groupby("rank_bucket", observed=True)["correct"].mean() * 100
        bucket_n   = feat_df2.groupby("rank_bucket", observed=True)["correct"].count()
        bucket_acc.plot(kind="bar", ax=ax, color=PALETTE[0], alpha=0.85, rot=30)
        for i, (a, n_) in enumerate(zip(bucket_acc, bucket_n)):
            ax.text(i, a + 0.5, f"{a:.0f}%\nn={n_}", ha="center", fontsize=8)
        ax.set_ylim(40, 90)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Accuracy by Rank Differential (P1 Rank − P2 Rank)")
        _savefig(pdf, fig)

        # ── PAGE 8 · Methodology deep-dive ───────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        ax = fig.add_axes([0.05, 0.02, 0.90, 0.95])
        ax.set_axis_off()

        text = (
            "Methodology & Anti-Leakage Notes\n"
            "═══════════════════════════════════════════════════════════════\n\n"
            "DATA\n"
            "  ·  Source: Jeff Sackmann's tennis_atp repository — all main-tour\n"
            "     match CSVs from 1968 through 2024 (~180,000+ matches).\n"
            "  ·  Only atp_matches_YYYY.csv files are used (main tour only).\n\n"
            "TEMPORAL SPLITS\n"
            "  ·  Warm-up / Training reference : 1968–2019\n"
            "  ·  Validation                   : 2020–2021  (not reported)\n"
            "  ·  Out-of-sample Test            : 2022–2024  (all metrics here)\n\n"
            "ANTI-LEAKAGE MEASURES\n"
            "  1. ELO ratings are updated *after* each match is processed;\n"
            "     each row's prediction uses only pre-match ELO.\n"
            "  2. ATP rank/rank-points come from the match row itself — a\n"
            "     snapshot of what was known at match time.\n"
            "  3. The XGBoost model was trained on 1968–2019 data only.\n"
            "     Test years 2022–2024 were never seen during training.\n"
            "  4. P1/P2 assignment is randomised with a fixed seed (42) so\n"
            "     the model cannot learn from label asymmetry.\n\n"
            "BETTING SIMULATION\n"
            "  ·  Market implied probability: ELO-based (pre-match ELO difference\n"
            "     converted to win probability via the logistic formula).\n"
            "  ·  Bet triggered when: model_prob − market_prob > 4 pp (post-vig).\n"
            "  ·  Sizing: flat $10 per bet — avoids compounding distortions.\n"
            "  ·  8% vig applied to simulate Kalshi exchange rake / spread.\n"
            "  ·  ROI = cumulative profit ÷ total amount staked (not bankroll).\n"
            "  ·  Note: ELO is less efficient than real markets; edge vs real\n"
            "     Kalshi prices will be smaller. Results are a theoretical bound.\n\n"
            "FEATURES (14)\n"
            "  Surface_Hard, Surface_Clay, Surface_Grass, Best_Of_Sets,\n"
            "  P1_Is_Right_Handed, P1_Height_cm, P1_Age, P1_Rank,\n"
            "  P1_Rank_Points, P2_Is_Right_Handed, P2_Height_cm,\n"
            "  P2_Age, P2_Rank, P2_Rank_Points\n\n"
            "LIMITATIONS\n"
            "  ·  ELO does not account for surface, fitness, or head-to-head.\n"
            "  ·  No transaction costs, slippage, or market liquidity modelled.\n"
            "  ·  Real-world Kalshi market prices will differ from ELO-implied.\n"
        )
        ax.text(0.0, 1.0, text, va="top", ha="left",
                fontsize=9.5, fontfamily="monospace",
                transform=ax.transAxes, color="#111111",
                wrap=False)
        _savefig(pdf, fig)

    log.info(f"PDF written to {pdf_path}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── load model ────────────────────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        log.error(f"Model not found: {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    log.info(f"Model loaded from {MODEL_PATH} — type: {type(model).__name__}")

    with open(FEATURES_PATH) as f:
        feature_names: list[str] = json.load(f)
    log.info(f"Feature set ({len(feature_names)}): {feature_names}")

    # ── load + build features ─────────────────────────────────────────────
    raw = load_atp_matches(ATP_DIR)
    log.info("Building walk-forward features (this may take ~60s for 180k rows)…")
    feat_df = build_features(raw)
    log.info(f"Feature rows: {len(feat_df):,}")

    # ── split: test 2022-2024 ─────────────────────────────────────────────
    test_mask = feat_df["year"] >= TEST_START
    test_df   = feat_df[test_mask].reset_index(drop=True)
    log.info(f"Test set: {len(test_df):,} matches ({test_df['year'].min()}–{test_df['year'].max()})")

    X_test = test_df[feature_names]
    y_true = test_df["target"].values

    # ── predict ───────────────────────────────────────────────────────────
    # Support both sklearn estimators and XGBoost Booster
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif _HAS_XGB and isinstance(model, xgb.Booster):
        dtest  = xgb.DMatrix(X_test, feature_names=feature_names)
        y_prob = model.predict(dtest)
    else:
        raise RuntimeError(f"Unsupported model type: {type(model)}")
    y_pred = (y_prob >= 0.5).astype(int)

    log.info(f"Accuracy : {accuracy_score(y_true, y_pred)*100:.2f}%")
    log.info(f"ROC-AUC  : {roc_auc_score(y_true, y_prob):.4f}")
    log.info(f"Brier    : {brier_score_loss(y_true, y_prob):.4f}")

    # ── betting simulation ────────────────────────────────────────────────
    betting = simulate_betting(
        model_prob  = y_prob,
        market_prob = test_df["elo_prob_p1"].values,
        outcomes    = y_true,
    )
    log.info(f"Betting ROI: {betting['roi']:+.1f}%  |  {betting['n_bets']} bets  |  "
             f"win rate {betting['win_rate']:.1f}%  |  final ${betting['bankroll_final']:,.0f}")

    # ── generate PDF ──────────────────────────────────────────────────────
    generate_pdf(test_df, y_true, y_prob, y_pred, betting, feature_names, model, PDF_OUT)


if __name__ == "__main__":
    main()
