"""
adaptive_controller.py — Rolling Performance Self-Calibrator
=============================================================
Tracks the bot's own bet outcomes and dynamically adjusts trading parameters.

The key insight: if the bot is consistently winning more than the model predicts,
it can afford to relax edge thresholds (capture more volume). If it's losing more
than expected, it tightens up immediately (capital preservation).

No changes to config.py on disk — all adjustments are in-memory and reset
when the bankroll recovers past its high-water mark.

Usage:
    controller = AdaptiveController(base_min_edge=0.04)
    # After each bet settles:
    controller.record_result(won=True, expected_p=0.65)
    # Feed into bet_manager:
    min_edge_adj = controller.current_min_edge
    kelly_mult   = controller.kelly_multiplier
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------
WINDOW_SIZE         = 25     # number of recent bets to evaluate
OUTPERFORM_RATIO    = 1.15   # actual/expected win rate threshold to relax
UNDERPERFORM_RATIO  = 0.85   # threshold to tighten
SEVERE_RATIO        = 0.60   # threshold to trigger protection mode
PROTECTION_DURATION = 3600   # seconds (60 min) to stay in protection mode

RELAX_FACTOR        = 0.90   # multiply min_edge by this when outperforming (lower = more bets)
TIGHTEN_FACTOR      = 1.15   # multiply min_edge by this when underperforming
KELLY_MULT_NORMAL   = 1.0
KELLY_MULT_HOT      = 1.10   # slightly larger when outperforming
KELLY_MULT_COLD     = 0.80   # smaller when underperforming
KELLY_MULT_PROTECT  = 0.0    # halt new bets


@dataclass
class BetRecord:
    won:        bool
    expected_p: float    # what our model said the win probability was
    edge:       float
    stake:      float
    timestamp:  float


class AdaptiveController:
    """
    Maintains a rolling window of recent bets and adjusts edge + Kelly
    parameters autonomously.
    """

    def __init__(self, base_min_edge: float = 0.04, base_kelly_tiers: list = None):
        self.base_min_edge   = base_min_edge
        self._min_edge_adj   = base_min_edge    # current live value
        self._kelly_mult     = KELLY_MULT_NORMAL

        self._window: deque[BetRecord] = deque(maxlen=WINDOW_SIZE)
        self._protection_until: float = 0.0
        self._high_water_bankroll: float = 0.0
        self._current_bankroll: float = 0.0

        # Stats for logging
        self._total_bets  = 0
        self._total_wins  = 0
        self._last_update = time.time()

    # ----------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------

    @property
    def current_min_edge(self) -> float:
        return self._min_edge_adj

    @property
    def kelly_multiplier(self) -> float:
        # If in protection mode, block new bets
        if time.time() < self._protection_until:
            remaining = self._protection_until - time.time()
            log.warning(f"[ADAPTIVE] Protection mode active — {remaining/60:.1f} min remaining. Blocking new bets.")
            return KELLY_MULT_PROTECT
        return self._kelly_mult

    @property
    def is_in_protection(self) -> bool:
        return time.time() < self._protection_until

    def update_bankroll(self, bankroll: float):
        """Call each time the balance is fetched to track high-water mark."""
        self._current_bankroll = bankroll
        if bankroll > self._high_water_bankroll:
            self._high_water_bankroll = bankroll
            # If we've recovered to a new high, exit protection mode
            if self.is_in_protection:
                log.info(f"[ADAPTIVE] New bankroll high ${bankroll:.2f} — exiting protection mode early.")
                self._protection_until = 0.0

    def record_result(self, won: bool, expected_p: float, edge: float = 0.0, stake: float = 0.0):
        """
        Call after each bet settles. Triggers parameter recalibration.
        """
        self._total_bets += 1
        if won:
            self._total_wins += 1

        record = BetRecord(
            won=won,
            expected_p=expected_p,
            edge=edge,
            stake=stake,
            timestamp=time.time(),
        )
        self._window.append(record)
        self._recalibrate()

    def status_dict(self) -> dict:
        """Return current state for dashboard/logging."""
        ratio, _, _ = self._compute_ratio()
        return {
            "total_bets":       self._total_bets,
            "total_wins":       self._total_wins,
            "window_size":      len(self._window),
            "calibration_ratio": round(ratio, 3) if ratio else None,
            "current_min_edge": round(self._min_edge_adj, 4),
            "kelly_multiplier": round(self._kelly_mult, 3),
            "in_protection":    self.is_in_protection,
        }

    # ----------------------------------------------------------------
    # Internal
    # ----------------------------------------------------------------

    def _compute_ratio(self):
        """
        Compute actual win rate vs expected win rate over the rolling window.
        Returns (ratio, actual_win_rate, expected_win_rate).
        """
        if len(self._window) < max(5, WINDOW_SIZE // 3):
            return None, None, None   # not enough data yet

        actual_wins    = sum(1 for r in self._window if r.won)
        actual_rate    = actual_wins / len(self._window)
        expected_rate  = sum(r.expected_p for r in self._window) / len(self._window)

        if expected_rate <= 0:
            return None, None, None

        ratio = actual_rate / expected_rate
        return ratio, actual_rate, expected_rate

    def _recalibrate(self):
        ratio, actual, expected = self._compute_ratio()
        if ratio is None:
            return

        log.info(
            f"[ADAPTIVE] Calibration: actual={actual:.1%} "
            f"expected={expected:.1%} ratio={ratio:.2f} "
            f"(n={len(self._window)})"
        )

        # ── Severe protection trigger ───────────────────────────────
        if ratio < SEVERE_RATIO:
            self._protection_until = time.time() + PROTECTION_DURATION
            self._min_edge_adj = self.base_min_edge * 2.0
            self._kelly_mult   = KELLY_MULT_PROTECT
            log.warning(
                f"[ADAPTIVE] ⚠️  PROTECTION MODE ACTIVATED "
                f"(ratio={ratio:.2f}). Halting new bets for 60 min. "
                f"MIN_EDGE → {self._min_edge_adj:.4f}"
            )
            return

        # ── Underperforming ─────────────────────────────────────────
        if ratio < UNDERPERFORM_RATIO:
            new_edge = min(self._min_edge_adj * TIGHTEN_FACTOR, self.base_min_edge * 2.5)
            self._min_edge_adj = new_edge
            self._kelly_mult   = KELLY_MULT_COLD
            log.info(
                f"[ADAPTIVE] 🔴 Underperforming (ratio={ratio:.2f}). "
                f"Tightening MIN_EDGE → {new_edge:.4f}, Kelly × {KELLY_MULT_COLD}"
            )

        # ── Outperforming ───────────────────────────────────────────
        elif ratio > OUTPERFORM_RATIO:
            new_edge = max(self._min_edge_adj * RELAX_FACTOR, self.base_min_edge * 0.6)
            self._min_edge_adj = new_edge
            self._kelly_mult   = KELLY_MULT_HOT
            log.info(
                f"[ADAPTIVE] 🟢 Outperforming (ratio={ratio:.2f}). "
                f"Relaxing MIN_EDGE → {new_edge:.4f}, Kelly × {KELLY_MULT_HOT}"
            )

        # ── On track ────────────────────────────────────────────────
        else:
            # Gently drift back toward base if we've strayed
            self._min_edge_adj = self._min_edge_adj * 0.95 + self.base_min_edge * 0.05
            self._kelly_mult   = KELLY_MULT_NORMAL
            log.debug(
                f"[ADAPTIVE] ✅ On track (ratio={ratio:.2f}). "
                f"MIN_EDGE holding at {self._min_edge_adj:.4f}"
            )
