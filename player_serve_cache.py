"""
player_serve_cache.py — Pre-match serve quality lookup built from historical ATP data.

Computes rolling 20-match averages per player:
  - second_serve_won_pct  : points won on 2nd serve  (WFO coef=3.0, p=0.000)
  - bp_save_rate          : break points saved rate   (WFO coef=2.9, p=0.000)

Used by main.py to apply the serve-quality logit edge before the match starts.
Falls back to tour medians if a player has fewer than 5 appearances.
"""

import glob
import logging
import os
from collections import defaultdict, deque

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

_ATP_GLOB = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "massive_tennis_dataset", "atp_tour", "atp_matches_*.csv",
)

ROLLING_N = 20
MIN_MATCHES = 5
TOUR_DEFAULT_2ND = 0.535
TOUR_DEFAULT_BP_SAVE = 0.617


def _load_df() -> pd.DataFrame:
    files = sorted(glob.glob(_ATP_GLOB))
    if not files:
        return pd.DataFrame()
    needed = ["winner_name", "loser_name", "w_2ndWon", "w_svpt", "w_1stIn",
              "w_bpSaved", "w_bpFaced", "l_2ndWon", "l_svpt", "l_1stIn",
              "l_bpSaved", "l_bpFaced", "tourney_date"]
    chunks = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=lambda c: c in needed, low_memory=False)
            chunks.append(df)
        except Exception:
            continue
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def _build_cache(df: pd.DataFrame) -> dict:
    """Build per-player rolling serve stats. Returns {name: {second_serve_won_pct, bp_save_rate}}."""
    if df.empty:
        return {}

    for col in ["w_2ndWon", "w_svpt", "w_1stIn", "w_bpSaved", "w_bpFaced",
                "l_2ndWon", "l_svpt", "l_1stIn", "l_bpSaved", "l_bpFaced"]:
        if col not in df.columns:
            df[col] = np.nan

    df = df.sort_values("tourney_date", na_position="first").reset_index(drop=True)

    # Rolling buffers: player → deque of (second_serve_won_pts, second_serve_pts, bp_saved, bp_faced)
    buf: dict = defaultdict(lambda: deque(maxlen=ROLLING_N))

    def _push(name, s2won, s2pts, bps, bpf):
        if pd.notna(s2won) and pd.notna(s2pts) and s2pts > 0:
            buf[name].append((s2won, s2pts, bps if pd.notna(bps) else 0, bpf if pd.notna(bpf) else 0))

    for _, row in df.iterrows():
        w = row.get("winner_name", "")
        l = row.get("loser_name", "")
        w2won = row.get("w_2ndWon", np.nan)
        wsvpt = row.get("w_svpt", np.nan)
        w1st  = row.get("w_1stIn", np.nan)
        wbps  = row.get("w_bpSaved", np.nan)
        wbpf  = row.get("w_bpFaced", np.nan)
        l2won = row.get("l_2ndWon", np.nan)
        lsvpt = row.get("l_svpt", np.nan)
        l1st  = row.get("l_1stIn", np.nan)
        lbps  = row.get("l_bpSaved", np.nan)
        lbpf  = row.get("l_bpFaced", np.nan)

        w2pts = wsvpt - w1st if pd.notna(wsvpt) and pd.notna(w1st) else np.nan
        l2pts = lsvpt - l1st if pd.notna(lsvpt) and pd.notna(l1st) else np.nan

        _push(w, w2won, w2pts, wbps, wbpf)
        _push(l, l2won, l2pts, lbps, lbpf)

    cache = {}
    for name, dq in buf.items():
        if len(dq) < MIN_MATCHES:
            continue
        arr = list(dq)
        s2won_total = sum(a[0] for a in arr)
        s2pts_total = sum(a[1] for a in arr)
        bps_total   = sum(a[2] for a in arr)
        bpf_total   = sum(a[3] for a in arr)
        cache[name] = {
            "second_serve_won_pct": s2won_total / s2pts_total if s2pts_total > 0 else TOUR_DEFAULT_2ND,
            "bp_save_rate":         bps_total   / bpf_total   if bpf_total   > 0 else TOUR_DEFAULT_BP_SAVE,
        }

    log.info("[ServeCache] Built serve-quality cache for %d players", len(cache))
    return cache


_CACHE: dict = {}
_LOADED = False


def load_cache():
    global _CACHE, _LOADED
    if _LOADED:
        return
    _LOADED = True
    try:
        df = _load_df()
        _CACHE = _build_cache(df)
    except Exception as exc:
        log.warning("[ServeCache] Build failed — using tour defaults: %s", exc)
        _CACHE = {}


def _fuzzy_lookup(name: str):
    if name in _CACHE:
        return _CACHE[name]
    name_lower = name.lower()
    for k, v in _CACHE.items():
        if k.lower() == name_lower:
            return v
    last = name.split()[-1].lower() if name else ""
    for k, v in _CACHE.items():
        if last and last in k.lower():
            return v
    return None


def get_serve_stats(player_name: str) -> dict:
    """
    Return pre-match rolling serve stats for a player.
    Falls back to tour medians if unknown.
    """
    if not _LOADED:
        load_cache()
    result = _fuzzy_lookup(player_name)
    if result:
        return result
    return {
        "second_serve_won_pct": TOUR_DEFAULT_2ND,
        "bp_save_rate":         TOUR_DEFAULT_BP_SAVE,
    }
