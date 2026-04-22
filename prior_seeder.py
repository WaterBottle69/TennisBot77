"""
prior_seeder.py — Historical-data seeding + local persistence for BayesianServeProbUpdater.

Computes a (prior_mean, concentration) pair from scraped stats AND a local
rolling-average cache so that each session starts smarter than the last.

Why not persist the raw Bayesian posterior?
-------------------------------------------
The posterior after one match absorbs match-specific noise (opponent quality,
fatigue, weather, a hot/cold streak).  Carrying it into the next match would
bias the prior with irrelevant noise.  What IS worth persisting is a smoothed
rolling average of observed serve-win rates across multiple matches — that
signal is stable and genuinely predictive.

Cache strategy
--------------
• Per player-slug, per surface: exponentially-weighted moving average (EWMA)
  of observed serve-win rates, plus match count.
• EWMA lambda = 0.15: each new match has 15% weight; about 6–7 matches to
  fully converge.  Prevents a single bad day from dominating the estimate.
• Entries older than TTL_DAYS (90) are ignored — player form drifts over time.
• File: prior_cache.json in the project root (a few KB at most).
• Thread-safe writes via atomic rename.

Concentration schedule
-----------------------
    Source                              concentration  (~games to converge)
    ─────────────────────────────────── ────────────── ─────────────────────
    Cache ≥ 10 matches                  130            ~2 games
    Cache 6–9 matches                   115            ~3 games
    Cache 3–5 matches                   100            ~4 games
    Cache 1–2 matches                   85             ~6 games
    Scraped stats (fs_pct + pts_won)    120            ~3 games
    Scraped stats (partial)             80             ~6 games
    Cold start (nothing)                50             ~15 games (old default)

    When both cache AND scraped stats are available, cache wins (it is
    match-observed data vs. career averages).

ATP surface baselines (Sackmann tennis_atp, 2010–2024, ~70k matches)
----------------------------------------------------------------------
    Surface     p_serve_avg   multiplier_vs_hard
    ─────────── ──────────── ─────────────────────
    Hard (out)   0.635         1.000  (reference)
    Hard (in)    0.650         1.024
    Clay         0.575         0.906
    Grass        0.668         1.052
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from typing import Optional, Tuple

log = logging.getLogger(__name__)

# ── constants ─────────────────────────────────────────────────────────────────

_SURFACE_BASELINES: dict[str, dict] = {
    "hard": {
        "p_serve":           0.635,
        "pts_won_1st_serve": 0.720,
        "pts_won_2nd_serve": 0.500,
        "first_serve_in":    0.620,
    },
    "clay": {
        "p_serve":           0.575,
        "pts_won_1st_serve": 0.690,
        "pts_won_2nd_serve": 0.470,
        "first_serve_in":    0.640,
    },
    "grass": {
        "p_serve":           0.668,
        "pts_won_1st_serve": 0.760,
        "pts_won_2nd_serve": 0.520,
        "first_serve_in":    0.600,
    },
    "indoor": {
        "p_serve":           0.650,
        "pts_won_1st_serve": 0.730,
        "pts_won_2nd_serve": 0.510,
        "first_serve_in":    0.625,
    },
}

_SURFACE_MULTIPLIER: dict[str, float] = {
    "hard":   1.000,
    "clay":   0.906,
    "grass":  1.052,
    "indoor": 1.024,
}

_HARD_BASELINE_P = _SURFACE_BASELINES["hard"]["p_serve"]

# Cache EWMA smoothing factor: 0.15 means ~6–7 matches to fully absorb new data
_EWMA_LAMBDA = 0.15

# Ignore cache entries older than this many days (player form drifts)
_TTL_DAYS = 90

# Minimum observed points in a match to trust the sample
_MIN_POINTS_TO_RECORD = 20


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalise_surface(surface: str) -> str:
    s = surface.lower().strip()
    if "clay" in s:
        return "clay"
    if "grass" in s or "wimbledon" in s:
        return "grass"
    if "indoor" in s or "carpet" in s:
        return "indoor"
    return "hard"


def _safe_float(v, default: float) -> float:
    if v is None:
        return default
    try:
        f = float(v)
        if f > 1.0:
            f /= 100.0
        if 0.0 < f < 1.0:
            return f
    except (TypeError, ValueError):
        pass
    return default


def _conc_from_n(n: int) -> float:
    """Map number of observed matches → concentration parameter."""
    if n >= 10:  return 130.0
    if n >= 6:   return 115.0
    if n >= 3:   return 100.0
    if n >= 1:   return 85.0
    return 50.0


# ── PriorCache ────────────────────────────────────────────────────────────────

class PriorCache:
    """
    Local JSON cache for per-player, per-surface serve-win rate estimates.

    Structure on disk (prior_cache.json):
    {
        "carlos-alcaraz": {
            "hard": {"ewma": 0.682, "n": 8, "updated": "2026-04-22T14:30:00Z"},
            "clay": {"ewma": 0.621, "n": 5, "updated": "2026-04-10T09:15:00Z"}
        },
        ...
    }
    """

    def __init__(self, path: str):
        self._path = path
        self._data: dict = {}
        self._load()

    # ── persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                self._data = json.load(f)
            log.debug("[PRIOR-CACHE] Loaded %d player entries from %s",
                      len(self._data), self._path)
        except Exception as exc:
            log.warning("[PRIOR-CACHE] Could not load %s: %s — starting fresh", self._path, exc)
            self._data = {}

    def _save(self) -> None:
        """Atomic write via temp file + rename to prevent corruption on crash."""
        dir_name = os.path.dirname(self._path) or "."
        try:
            fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
            os.replace(tmp, self._path)
        except Exception as exc:
            log.warning("[PRIOR-CACHE] Save failed: %s", exc)
            try:
                os.unlink(tmp)
            except Exception:
                pass

    # ── read ──────────────────────────────────────────────────────────────────

    def get(self, slug: str, surface: str) -> Optional[Tuple[float, int]]:
        """
        Return (ewma_serve_rate, n_matches) for this player+surface combo,
        or None if no valid entry exists.

        Entries older than _TTL_DAYS are treated as non-existent.
        """
        surf = _normalise_surface(surface)
        entry = self._data.get(slug, {}).get(surf)
        if not entry:
            return None

        # TTL check
        try:
            updated = datetime.fromisoformat(entry["updated"].replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - updated).days
            if age_days > _TTL_DAYS:
                log.debug("[PRIOR-CACHE] Stale entry for %s/%s (%d days old) — ignoring",
                          slug, surf, age_days)
                return None
        except Exception:
            return None

        ewma = float(entry.get("ewma", 0))
        n    = int(entry.get("n", 0))
        if not (0.3 < ewma < 0.9):
            return None
        return ewma, n

    # ── write ─────────────────────────────────────────────────────────────────

    def record_match(
        self,
        slug: str,
        surface: str,
        points_won: int,
        points_total: int,
    ) -> None:
        """
        Update the rolling EWMA estimate after a completed match.

        Args:
            slug:          player slug (e.g. 'carlos-alcaraz')
            surface:       court surface string
            points_won:    serve points won in this match
            points_total:  total serve points played in this match
        """
        if not slug or points_total < _MIN_POINTS_TO_RECORD:
            log.debug("[PRIOR-CACHE] Skipping record for %s — too few points (%d)",
                      slug, points_total)
            return

        observed = points_won / points_total
        surf = _normalise_surface(surface)

        if slug not in self._data:
            self._data[slug] = {}

        existing = self._data[slug].get(surf)
        if existing and self.get(slug, surf) is not None:
            old_ewma = float(existing["ewma"])
            old_n    = int(existing["n"])
            new_ewma = _EWMA_LAMBDA * observed + (1.0 - _EWMA_LAMBDA) * old_ewma
            new_n    = old_n + 1
        else:
            # First observation for this player+surface — use it directly
            new_ewma = observed
            new_n    = 1

        # Clamp to realistic range
        new_ewma = max(0.40, min(0.85, new_ewma))

        self._data[slug][surf] = {
            "ewma":    round(new_ewma, 5),
            "n":       new_n,
            "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

        log.info(
            "[PRIOR-CACHE] %s / %s  observed=%.4f  ewma %.4f → %.4f  n=%d",
            slug, surf, observed,
            existing["ewma"] if existing else observed,
            new_ewma, new_n,
        )
        self._save()

    # ── housekeeping ──────────────────────────────────────────────────────────

    def prune_stale(self) -> int:
        """Remove all entries older than _TTL_DAYS. Returns number pruned."""
        pruned = 0
        for slug in list(self._data):
            for surf in list(self._data[slug]):
                if self.get(slug, surf) is None:
                    del self._data[slug][surf]
                    pruned += 1
            if not self._data[slug]:
                del self._data[slug]
        if pruned:
            self._save()
            log.info("[PRIOR-CACHE] Pruned %d stale entries", pruned)
        return pruned

    def summary(self) -> str:
        lines = []
        for slug, surfaces in sorted(self._data.items()):
            parts = []
            for surf, entry in surfaces.items():
                result = self.get(slug, surf)
                if result:
                    ewma, n = result
                    parts.append(f"{surf}={ewma:.3f}(n={n})")
            if parts:
                lines.append(f"  {slug}: {', '.join(parts)}")
        return "\n".join(lines) if lines else "  (empty)"


# ── module-level singleton ────────────────────────────────────────────────────

_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prior_cache.json")
prior_cache = PriorCache(_CACHE_PATH)


# ── public API ────────────────────────────────────────────────────────────────

def compute_serve_prior(
    player_stats: dict,
    surface: str = "hard",
    cache: Optional[PriorCache] = None,
) -> Tuple[float, float]:
    """
    Compute (prior_mean, concentration) for BayesianServeProbUpdater.

    Priority:
      1. Local cache (match-observed rolling average) — highest quality
      2. Scraped stats (tennisstats career averages)
      3. ATP surface baseline (cold start fallback)

    Args:
        player_stats: dict from TennisStatsScraper.get_player_data()
        surface:      court surface string
        cache:        PriorCache instance (defaults to module-level singleton)

    Returns:
        (prior_mean, concentration)
    """
    if cache is None:
        cache = prior_cache

    surf = _normalise_surface(surface)
    base = _SURFACE_BASELINES[surf]
    slug = player_stats.get("slug", "")

    # ── 1. Cache lookup ───────────────────────────────────────────────────────
    if slug:
        cached = cache.get(slug, surf)
        if cached is not None:
            ewma, n = cached
            conc = _conc_from_n(n)
            log.debug(
                "[PRIOR] %s  surface=%s  source=cache  ewma=%.4f  n=%d  conc=%.0f",
                slug, surf, ewma, n, conc,
            )
            return ewma, conc

    # ── 2. Scraped stats ──────────────────────────────────────────────────────
    fs_in = _safe_float(player_stats.get("first_serve_pct"),        base["first_serve_in"])
    p1st  = _safe_float(player_stats.get("pts_won_1st_serve") or
                        player_stats.get("first_serve_won_pct"),    base["pts_won_1st_serve"])
    p2nd  = _safe_float(player_stats.get("pts_won_2nd_serve") or
                        player_stats.get("second_serve_won_pct"),   base["pts_won_2nd_serve"])

    p_serve = fs_in * p1st + (1.0 - fs_in) * p2nd

    if p_serve > 0:
        p_serve_adj = (p_serve / _HARD_BASELINE_P) * base["p_serve"]
    else:
        p_serve_adj = base["p_serve"] * _SURFACE_MULTIPLIER[surf]

    p_serve_adj = max(0.45, min(0.80, p_serve_adj))

    has_fs_in = player_stats.get("first_serve_pct") is not None
    has_p1st  = (player_stats.get("pts_won_1st_serve") is not None or
                 player_stats.get("first_serve_won_pct") is not None)

    if has_fs_in and has_p1st:
        concentration = 120.0
    elif has_fs_in or has_p1st:
        concentration = 80.0
    else:
        concentration = 50.0

    log.debug(
        "[PRIOR] %s  surface=%s  source=scraped  fs_in=%.3f  p1st=%.3f  "
        "→ p_serve=%.4f  conc=%.0f",
        slug or "?", surf, fs_in, p1st, p_serve_adj, concentration,
    )
    return p_serve_adj, concentration


def compute_return_prior(
    player_stats: dict,
    surface: str = "hard",
    cache: Optional[PriorCache] = None,
) -> Tuple[float, float]:
    """Return (prior_mean, concentration) for the returning player."""
    p_serve, conc = compute_serve_prior(player_stats, surface, cache=cache)
    p_return = max(0.20, min(0.55, 1.0 - p_serve))
    return p_return, conc


def reseed_for_surface(
    bayes_updater,
    player_stats: dict,
    surface: str,
    is_server: bool = True,
    cache: Optional[PriorCache] = None,
) -> None:
    """
    Re-initialise a BayesianServeProbUpdater in place when the actual court
    surface becomes known.  Blends the new surface prior with any live
    observations already absorbed.
    """
    if is_server:
        new_mean, new_conc = compute_serve_prior(player_stats, surface, cache=cache)
    else:
        new_mean, new_conc = compute_return_prior(player_stats, surface, cache=cache)

    live_obs = (bayes_updater.alpha + bayes_updater.beta) - (
        bayes_updater.alpha_pre + bayes_updater.beta_pre
    )
    blend_weight = max(0.0, 1.0 - live_obs / 40.0)

    blended_mean = blend_weight * new_mean + (1.0 - blend_weight) * bayes_updater.get_posterior_mean()
    blended_conc = blend_weight * new_conc + (1.0 - blend_weight) * (bayes_updater.alpha + bayes_updater.beta)
    blended_conc = max(20.0, blended_conc)

    bayes_updater.alpha_pre = blended_mean * blended_conc
    bayes_updater.beta_pre  = (1.0 - blended_mean) * blended_conc
    bayes_updater.alpha     = bayes_updater.alpha_pre + (bayes_updater.alpha - bayes_updater.alpha_pre * blend_weight)
    bayes_updater.beta      = bayes_updater.beta_pre  + (bayes_updater.beta  - bayes_updater.beta_pre  * blend_weight)

    log.info(
        "[PRIOR-RESEED] surface=%s  new_mean=%.4f  new_conc=%.0f  "
        "live_obs=%.0f  blend=%.2f  posterior=%.4f",
        surface, new_mean, new_conc, live_obs, blend_weight,
        bayes_updater.get_posterior_mean(),
    )
