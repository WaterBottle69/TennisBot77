"""
alpha_discovery.py — Systematic alpha signal discovery for TennisBot.

Tests 10 new correlation factors not currently in config.py:
  1.  First-serve % differential            (rolling 20-match window)
  2.  Second-serve won % differential        (rolling 20-match window)
  3.  Ace rate differential                  (rolling 20-match window)
  4.  Break-point save rate differential     (rolling 20-match window)
  5.  Double-fault rate differential         (rolling 20-match window)
  6.  Days-rest differential                 (schedule gaps)
  7.  Prior-match fatigue (minutes played)   (last match)
  8.  Ranking trajectory                     (90-day rank delta)
  9.  Surface-switch penalty                 (cross-surface transition)
 10.  Tournament-round depth effect          (late-round favourite edge)

Methodology per signal:
  - Expanding-window WFO (train on years < Y, test on year Y)
  - Test years: 2010–2024 (ensures enough training data)
  - Betting rule: bet in direction of signal when |logit_adj| > threshold
  - Kelly fraction 0.05 with vig 0.07, min_edge 0.12
  - Permutation test (1 000 shuffles) for OOS p-value
  - Reports ROI, Sharpe, MaxDD, n_bets, p_value per signal
"""

import math
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings("ignore")

# ── Constants ──────────────────────────────────────────────────────────────────
ROLLING_N      = 20       # matches for rolling serve stats
MIN_SERVE_ROWS = 5        # min prior matches needed to use serve signal
TEST_START     = 2010
VIG            = 0.07
MIN_EDGE       = 0.12
KELLY_FRAC     = 0.05
BANKROLL_INIT  = 1_000.0
MAX_BET        = 250.0
N_PERMUTE      = 1_000
ELO_K          = 32.0
ELO_INIT       = 1_500.0

# ── Data loading ───────────────────────────────────────────────────────────────

def load_atp() -> pd.DataFrame:
    import glob
    files = sorted(glob.glob("massive_tennis_dataset/atp_tour/atp_matches_2*.csv"))
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f, low_memory=False))
        except Exception:
            pass
    df = pd.concat(dfs, ignore_index=True)
    df["tourney_date"] = pd.to_datetime(
        df["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    df = df[df["tourney_level"].isin(["G", "M", "A", "F"])].copy()
    df = df.dropna(subset=["tourney_date", "winner_rank", "loser_rank"]).copy()
    df["winner_rank"] = pd.to_numeric(df["winner_rank"], errors="coerce")
    df["loser_rank"]  = pd.to_numeric(df["loser_rank"],  errors="coerce")
    df = df.dropna(subset=["winner_rank", "loser_rank"])
    df = df.sort_values("tourney_date").reset_index(drop=True)
    df["year"] = df["tourney_date"].dt.year
    df["surface"] = df["surface"].fillna("Hard")
    df["round_num"] = df["round"].map({
        "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
    }).fillna(3)
    return df


# ── Elo ────────────────────────────────────────────────────────────────────────

def compute_elo(df: pd.DataFrame):
    elos = defaultdict(lambda: ELO_INIT)
    elo_w, elo_l = [], []
    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]
        ew, el = elos[w], elos[l]
        elo_w.append(ew); elo_l.append(el)
        exp_w = 1.0 / (1.0 + 10.0 ** ((el - ew) / 400.0))
        elos[w] += ELO_K * (1 - exp_w)
        elos[l] += ELO_K * (0 - (1 - exp_w))
    df = df.copy()
    df["elo_w"] = elo_w
    df["elo_l"] = elo_l
    df["elo_prob_w"] = 1.0 / (1.0 + 10.0 ** ((np.array(elo_l) - np.array(elo_w)) / 400.0))
    return df


# ── Rolling player serve stats ─────────────────────────────────────────────────

def build_rolling_serve(df: pd.DataFrame) -> dict:
    """
    Returns dict: player_id → deque of recent (1stPct, 2ndWonPct, acePct, bpSavePct, dfPct).
    Processes df in chronological order, yielding per-row stats BEFORE the match is added.
    """
    from collections import deque
    store: dict = defaultdict(lambda: deque(maxlen=ROLLING_N))

    stats_w, stats_l = [], []
    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]

        def _mean(dq, idx):
            vals = [x[idx] for x in dq if x[idx] is not None]
            return np.mean(vals) if len(vals) >= MIN_SERVE_ROWS else np.nan

        def _get(pid):
            dq = store[pid]
            return {
                "first_pct":   _mean(dq, 0),
                "second_won":  _mean(dq, 1),
                "ace_pct":     _mean(dq, 2),
                "bp_save":     _mean(dq, 3),
                "df_pct":      _mean(dq, 4),
            }

        stats_w.append(_get(w))
        stats_l.append(_get(l))

        def _entry(row_data, prefix):
            svpt = pd.to_numeric(row_data.get(f"{prefix}_svpt"), errors="coerce")
            st1  = pd.to_numeric(row_data.get(f"{prefix}_1stIn"),  errors="coerce")
            s1w  = pd.to_numeric(row_data.get(f"{prefix}_1stWon"), errors="coerce")
            s2w  = pd.to_numeric(row_data.get(f"{prefix}_2ndWon"), errors="coerce")
            ace  = pd.to_numeric(row_data.get(f"{prefix}_ace"),    errors="coerce")
            df_  = pd.to_numeric(row_data.get(f"{prefix}_df"),     errors="coerce")
            bpf  = pd.to_numeric(row_data.get(f"{prefix}_bpFaced"),errors="coerce")
            bps  = pd.to_numeric(row_data.get(f"{prefix}_bpSaved"),errors="coerce")
            if svpt and svpt > 0:
                first_pct  = st1 / svpt  if (st1 is not None and not math.isnan(st1)) else None
                s2_pts     = svpt - (st1 if (st1 and not math.isnan(st1)) else 0)
                second_won = (s2w / s2_pts) if (s2_pts > 0 and s2w is not None and not math.isnan(s2w)) else None
                ace_pct    = ace / svpt   if (ace is not None and not math.isnan(ace))  else None
                df_pct     = df_  / svpt  if (df_ is not None and not math.isnan(df_))  else None
            else:
                first_pct = second_won = ace_pct = df_pct = None
            bp_save = (bps / bpf) if (bpf and bpf > 0 and bps is not None and not math.isnan(bps)) else None
            return (first_pct, second_won, ace_pct, bp_save, df_pct)

        store[w].append(_entry(row, "w"))
        store[l].append(_entry(row, "l"))

    return stats_w, stats_l


# ── Days-rest computation ──────────────────────────────────────────────────────

def build_rest_days(df: pd.DataFrame):
    last_match: dict = {}
    rest_w, rest_l = [], []
    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]
        d = row["tourney_date"]
        rest_w.append((d - last_match[w]).days if w in last_match else np.nan)
        rest_l.append((d - last_match[l]).days if l in last_match else np.nan)
        last_match[w] = d
        last_match[l] = d
    return rest_w, rest_l


# ── Prior match minutes ────────────────────────────────────────────────────────

def build_prior_minutes(df: pd.DataFrame):
    last_mins: dict = {}
    mins_w, mins_l = [], []
    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]
        mins_w.append(last_mins.get(w, np.nan))
        mins_l.append(last_mins.get(l, np.nan))
        m = pd.to_numeric(row.get("minutes"), errors="coerce")
        if not np.isnan(m):
            last_mins[w] = m
            last_mins[l] = m
    return mins_w, mins_l


# ── Ranking trajectory (90-day delta) ─────────────────────────────────────────

def build_rank_trajectory(df: pd.DataFrame):
    rank_hist: dict = defaultdict(list)
    traj_w, traj_l = [], []

    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]
        d = row["tourney_date"]
        wr, lr = row["winner_rank"], row["loser_rank"]

        def _traj(pid, cur_rank):
            hist = rank_hist[pid]
            cutoff = d - pd.Timedelta(days=90)
            old = [r for dt, r in hist if dt >= cutoff and dt < d]
            if old:
                return cur_rank - np.mean(old)  # positive = getting worse
            return np.nan

        traj_w.append(_traj(w, wr))
        traj_l.append(_traj(l, lr))
        rank_hist[w].append((d, wr))
        rank_hist[l].append((d, lr))

    return traj_w, traj_l


# ── Surface switch ─────────────────────────────────────────────────────────────

def build_surface_switch(df: pd.DataFrame):
    last_surf: dict = {}
    switch_w, switch_l = [], []
    for _, row in df.iterrows():
        w, l = row["winner_id"], row["loser_id"]
        s = row["surface"]
        switch_w.append(0 if last_surf.get(w) == s else (1 if w in last_surf else np.nan))
        switch_l.append(0 if last_surf.get(l) == s else (1 if l in last_surf else np.nan))
        last_surf[w] = s
        last_surf[l] = s
    return switch_w, switch_l


# ── Kelly betting simulation ───────────────────────────────────────────────────

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def simulate_bets(model_probs, signal_adj, targets, coef, threshold=0.0):
    bankroll = BANKROLL_INIT
    pnl = []
    for mp, adj, tgt in zip(model_probs, signal_adj, targets):
        if np.isnan(adj):
            continue
        adjusted = sigmoid(logit(mp) + coef * adj)
        market_p = mp  # use raw Elo as market proxy
        market_p_vig = market_p / (market_p + (1 - market_p) * (1 + VIG))
        edge = adjusted - market_p_vig
        if abs(edge) < MIN_EDGE:
            continue
        if edge > 0:
            bet_on_winner = True
        else:
            bet_on_winner = False
            edge = -edge
            adjusted = 1 - adjusted
            market_p_vig = 1 - market_p_vig
        k = (edge / (1 - market_p_vig)) * KELLY_FRAC
        k = min(k, MAX_BET / bankroll if bankroll > 0 else 0)
        k = max(k, 0)
        stake = bankroll * k
        if stake < 1.0:
            continue
        win = (tgt == 1) if bet_on_winner else (tgt == 0)
        payoff = stake * (1 / adjusted - 1) if win else -stake
        bankroll = max(bankroll + payoff, 0)
        pnl.append(payoff)
        if bankroll <= 0:
            break

    if not pnl:
        return {"roi": 0.0, "sharpe": 0.0, "max_dd": 0.0, "n_bets": 0}
    pnl = np.array(pnl)
    cum  = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum + BANKROLL_INIT)
    dd   = (peak - (cum + BANKROLL_INIT)) / peak
    roi  = cum[-1] / BANKROLL_INIT * 100
    sharpe = pnl.mean() / (pnl.std() + 1e-9) * np.sqrt(252)
    return {"roi": roi, "sharpe": sharpe, "max_dd": float(dd.max() * 100), "n_bets": len(pnl)}


def fit_coef(adj_train, prob_train, target_train):
    """Simple logistic regression coefficient for signal via grid search."""
    valid = ~np.isnan(adj_train)
    if valid.sum() < 100:
        return 0.0
    a = adj_train[valid]
    p = prob_train[valid]
    t = target_train[valid]
    best_coef, best_ll = 0.0, -np.inf
    for c in np.linspace(-3, 3, 61):
        pred = sigmoid(logit(p) + c * a)
        ll = np.mean(t * np.log(pred + 1e-9) + (1 - t) * np.log(1 - pred + 1e-9))
        if ll > best_ll:
            best_ll, best_coef = ll, c
    return best_coef


# ── Walk-forward evaluator ─────────────────────────────────────────────────────

def walk_forward_signal(df, signal_col, test_start=TEST_START):
    years = sorted(df["year"].unique())
    test_years = [y for y in years if y >= test_start]
    results = []

    for fold_year in test_years:
        train = df[df["year"] < fold_year]
        test  = df[df["year"] == fold_year]
        if len(train) < 500 or len(test) < 50:
            continue

        # drop rows with NaN signal in training
        train_valid = train.dropna(subset=[signal_col])
        if len(train_valid) < 200:
            results.append({"year": fold_year, "roi": 0, "sharpe": 0, "n_bets": 0, "max_dd": 0})
            continue

        coef = fit_coef(
            train_valid[signal_col].values,
            train_valid["elo_prob_w"].values,
            train_valid["target"].values,
        )

        test_valid = test.dropna(subset=[signal_col])
        if len(test_valid) < 20:
            results.append({"year": fold_year, "roi": 0, "sharpe": 0, "n_bets": 0, "max_dd": 0})
            continue

        res = simulate_bets(
            test_valid["elo_prob_w"].values,
            test_valid[signal_col].values,
            test_valid["target"].values,
            coef,
        )
        res["year"] = fold_year
        results.append(res)

    return pd.DataFrame(results)


def permutation_pvalue(df, signal_col, coef, n_perm=N_PERMUTE):
    test = df[df["year"] >= TEST_START].dropna(subset=[signal_col])
    if len(test) < 50:
        return 1.0
    real_res = simulate_bets(
        test["elo_prob_w"].values,
        test[signal_col].values,
        test["target"].values,
        coef,
    )
    real_roi = real_res["roi"]
    rng = np.random.default_rng(42)
    null_rois = []
    sig = test[signal_col].values.copy()
    for _ in range(n_perm):
        rng.shuffle(sig)
        null = simulate_bets(test["elo_prob_w"].values, sig, test["target"].values, coef)
        null_rois.append(null["roi"])
    p = np.mean(np.array(null_rois) >= real_roi)
    return p


# ── Main ────────────────────────────────────────────────────────────────────────

def main():
    print("Loading ATP tour data (2000-2024)…")
    df = load_atp()
    print(f"  {len(df):,} matches | {df.year.min()}-{df.year.max()}")

    print("Computing Elo…")
    df = compute_elo(df)

    # Random P1/P2 assignment (eliminate winner-label bias)
    rng = np.random.default_rng(42)
    swap = rng.random(len(df)) < 0.5
    df["target"] = np.where(swap, 0, 1)
    df["elo_prob_w"] = np.where(swap, 1 - df["elo_prob_w"], df["elo_prob_w"])
    # Re-orient: elo_prob_w is now P(p1 wins), target=1 means p1 won

    print("Computing rolling serve stats…")
    # Need to process in original winner/loser order for serve stats
    df_orig = df.copy()
    # Serve stats are always from winner perspective, so we don't randomise before computing
    df_tmp = df_orig.sort_values("tourney_date").reset_index(drop=True)
    sw, sl = build_rolling_serve(df_tmp)

    print("Computing schedule signals…")
    rw, rl = build_rest_days(df_tmp)
    mw, ml = build_prior_minutes(df_tmp)
    tw, tl = build_rank_trajectory(df_tmp)
    ssw, ssl = build_surface_switch(df_tmp)

    # Now build signal columns (oriented to P1/P2 with the same swap mask)
    sw_arr = np.array([x["first_pct"]  for x in sw], dtype=float)
    sl_arr = np.array([x["first_pct"]  for x in sl], dtype=float)
    s2w    = np.array([x["second_won"] for x in sw], dtype=float)
    s2l    = np.array([x["second_won"] for x in sl], dtype=float)
    aw     = np.array([x["ace_pct"]    for x in sw], dtype=float)
    al     = np.array([x["ace_pct"]    for x in sl], dtype=float)
    bpw    = np.array([x["bp_save"]    for x in sw], dtype=float)
    bpl    = np.array([x["bp_save"]    for x in sl], dtype=float)
    dfw    = np.array([x["df_pct"]     for x in sw], dtype=float)
    dfl    = np.array([x["df_pct"]     for x in sl], dtype=float)
    rw_a   = np.array(rw, dtype=float)
    rl_a   = np.array(rl, dtype=float)
    mw_a   = np.array(mw, dtype=float)
    ml_a   = np.array(ml, dtype=float)
    tw_a   = np.array(tw, dtype=float)
    tl_a   = np.array(tl, dtype=float)
    ssw_a  = np.array(ssw, dtype=float)
    ssl_a  = np.array(ssl, dtype=float)

    # Apply swap: P1 = winner if not swapped, loser if swapped
    def _orient(w_arr, l_arr):
        return np.where(swap, l_arr, w_arr), np.where(swap, w_arr, l_arr)

    p1_1st, p2_1st = _orient(sw_arr, sl_arr)
    p1_2nd, p2_2nd = _orient(s2w,    s2l)
    p1_ace, p2_ace = _orient(aw,     al)
    p1_bp,  p2_bp  = _orient(bpw,    bpl)
    p1_df,  p2_df  = _orient(dfw,    dfl)
    p1_rest,p2_rest= _orient(rw_a,   rl_a)
    p1_min, p2_min = _orient(mw_a,   ml_a)
    p1_traj,p2_traj= _orient(tw_a,   tl_a)
    p1_sw,  p2_sw  = _orient(ssw_a,  ssl_a)

    # Signal definitions (positive = favours p1)
    # We use differences so the signal is symmetric
    df["sig_first_serve"]   = p1_1st - p2_1st          # higher 1st serve % → better
    df["sig_second_won"]    = p1_2nd - p2_2nd           # higher 2nd won % → better
    df["sig_ace_rate"]      = p1_ace - p2_ace           # higher ace rate → better
    df["sig_bp_save"]       = p1_bp  - p2_bp            # higher BP save % → better
    df["sig_df_rate"]       = p2_df  - p1_df            # p1 lower DF is better → invert
    df["sig_rest_diff"]     = p1_rest- p2_rest          # more rest → better
    df["sig_fatigue_mins"]  = p2_min - p1_min           # p1 played fewer mins → better
    df["sig_rank_traj"]     = p2_traj- p1_traj          # p1 improving (lower is better) → negate
    df["sig_surface_switch"]= p2_sw  - p1_sw            # p1 NOT switching, p2 IS → advantage p1

    # Round depth: higher round = later in tournament = favourites more reliable
    df["sig_round_depth"]   = (df["round_num"] * np.log(df["winner_rank"] / df["loser_rank"].clip(1))).where(
        ~swap, -(df["round_num"] * np.log(df["winner_rank"] / df["loser_rank"].clip(1)))
    )
    # correct orientation: in late rounds, the better-ranked P1 should win more
    p1_rank = np.where(swap, df["loser_rank"].values,  df["winner_rank"].values)
    p2_rank = np.where(swap, df["winner_rank"].values, df["loser_rank"].values)
    df["sig_round_depth"] = df["round_num"] * np.log(p2_rank / p1_rank.clip(1))

    signals = {
        "1. First-serve % diff":     "sig_first_serve",
        "2. 2nd-serve won % diff":   "sig_second_won",
        "3. Ace rate diff":          "sig_ace_rate",
        "4. BP save rate diff":      "sig_bp_save",
        "5. Double-fault rate diff": "sig_df_rate",
        "6. Rest days diff":         "sig_rest_diff",
        "7. Fatigue (prior mins)":   "sig_fatigue_mins",
        "8. Rank trajectory (90d)":  "sig_rank_traj",
        "9. Surface switch penalty": "sig_surface_switch",
        "10. Round-depth × rank":    "sig_round_depth",
    }

    summary = []
    print("\n" + "="*78)
    print(f"  {'Signal':<30}  {'ROI%':>7}  {'Sharpe':>7}  {'MaxDD%':>7}  {'Bets':>6}  {'pVal':>6}")
    print("="*78)

    for name, col in signals.items():
        wf = walk_forward_signal(df, col)
        if wf.empty or wf["n_bets"].sum() == 0:
            print(f"  {name:<30}  {'NO BETS':>7}")
            continue

        agg_roi    = wf["roi"].mean()
        agg_sharpe = wf["sharpe"].mean()
        agg_maxdd  = wf["max_dd"].max()
        total_bets = int(wf["n_bets"].sum())

        # Fit global coef on all training data (pre-2024) for permutation test
        train_all = df[df["year"] < 2024].dropna(subset=[col])
        coef_global = fit_coef(
            train_all[col].values,
            train_all["elo_prob_w"].values,
            train_all["target"].values,
        )
        pval = permutation_pvalue(df, col, coef_global)

        print(f"  {name:<30}  {agg_roi:>+7.2f}%  {agg_sharpe:>7.3f}  {agg_maxdd:>7.1f}%  {total_bets:>6,}  {pval:>6.3f}")
        summary.append({
            "signal": name,
            "col": col,
            "roi_pct": round(agg_roi, 3),
            "sharpe": round(agg_sharpe, 3),
            "max_dd_pct": round(agg_maxdd, 1),
            "n_bets": total_bets,
            "p_value": round(pval, 3),
            "coef": round(coef_global, 4),
        })

        # Per-year detail
        for _, r in wf.iterrows():
            mark = "✓" if r["roi"] > 0 else "✗"
            print(f"    {mark} {int(r['year'])}: ROI={r['roi']:+.1f}%  Sharpe={r['sharpe']:.2f}  Bets={int(r['n_bets'])}")

        print()

    print("="*78)

    if summary:
        sumdf = pd.DataFrame(summary).sort_values("roi_pct", ascending=False)
        print("\n  RANKED BY OOS ROI:")
        for _, r in sumdf.iterrows():
            star = "  *** SIGNIFICANT ***" if r["p_value"] < 0.05 else ""
            print(f"    {r['signal']:<32}  ROI={r['roi_pct']:+.2f}%  p={r['p_value']:.3f}{star}")

        # Save profitable + significant signals
        profitable = sumdf[(sumdf["roi_pct"] > 0) & (sumdf["p_value"] < 0.10)]
        if not profitable.empty:
            print("\n  PROFITABLE SIGNALS (ROI>0, p<0.10):")
            for _, r in profitable.iterrows():
                print(f"    {r['signal']:<32}  coef={r['coef']:+.4f}  ROI={r['roi_pct']:+.2f}%  p={r['p_value']:.3f}")
        else:
            print("\n  No signals met both ROI>0 and p<0.10 criteria.")

        sumdf.to_csv("alpha_discovery_results.csv", index=False)
        print("\n  Full results saved → alpha_discovery_results.csv")

    return sumdf if summary else pd.DataFrame()


if __name__ == "__main__":
    results = main()
