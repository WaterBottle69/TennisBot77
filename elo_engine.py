"""
elo_engine.py — Core Elo rating engine with sports-specific multipliers.

Tracks two teams through a live game, applying:
  - Per-event Elo deltas (weighted by multipliers)
  - Substitution/lineup changes from OpenCV
  - Halftime side-switch adjustment
  - Blowout accelerator
  - H2H historical bias seed
"""

import math
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional
from config import Config

log = logging.getLogger(__name__)


@dataclass
class GameState:
    clock_minutes: float     = 0.0
    period: int              = 1        # 1 = first half, 2 = second half, OT = 3+
    score_a: int             = 0
    score_b: int             = 0
    halftime_occurred: bool  = False
    blowout_applied: bool    = False
    momentum_streak_a: int   = 0        # consecutive scoring events for A
    momentum_streak_b: int   = 0


class EloEngine:
    def __init__(self, config: Config):
        self.cfg    = config
        self.ratings: Dict[str, float] = {
            "team_a": config.BASE_ELO,
            "team_b": config.BASE_ELO,
        }
        self.game_state = GameState()
        self._event_log = []

    # ──────────────────────────────────────────────────────────────────────────
    # Seeding
    # ──────────────────────────────────────────────────────────────────────────

    def seed_from_history(self, h2h: dict):
        """
        Adjust starting Elo based on historical head-to-head record.
        h2h = {
            "team_a_win_rate": 0.72,   # A wins 72% of games vs B
            "avg_point_diff": 8.4,     # A wins by 8.4 pts on avg
            "seasons_sampled": 3
        }
        """
        win_rate  = h2h.get("team_a_win_rate", 0.5)
        pt_diff   = h2h.get("avg_point_diff", 0.0)
        threshold = self.cfg.H2H_DOMINANCE_THRESHOLD

        if win_rate >= threshold:
            bias = self.cfg.ELO_MULTIPLIERS["h2h_dominance"] * abs(pt_diff)
            self.ratings["team_a"] += bias
            self.ratings["team_b"] -= bias
            log.info(f"H2H bias applied: +{bias:.1f} to A (win rate {win_rate:.0%})")

        elif win_rate <= (1 - threshold):
            bias = self.cfg.ELO_MULTIPLIERS["h2h_underdog"] * abs(pt_diff)
            self.ratings["team_b"] += abs(bias)
            self.ratings["team_a"] -= abs(bias)
            log.info(f"H2H bias applied: +{abs(bias):.1f} to B (A win rate only {win_rate:.0%})")

    # ──────────────────────────────────────────────────────────────────────────
    # Live event processing
    # ──────────────────────────────────────────────────────────────────────────

    def apply_event(self, event: dict):
        """
        Process a live event dict:
        {
            "type": "3pt_made",
            "team": "team_a",         # which team the event belongs to
            "player_id": "p123",
            "clock": "Q2 04:32",
            "clock_minutes": 28.4,    # absolute game minutes elapsed
            "score_a": 54,
            "score_b": 48
        }
        """
        etype        = event.get("type", "")
        team         = event.get("team", "team_a")
        opponent     = "team_b" if team == "team_a" else "team_a"
        clock_min    = event.get("clock_minutes", self.game_state.clock_minutes)
        period       = event.get("period", self.game_state.period)

        # Update game state
        self.game_state.clock_minutes = clock_min
        self.game_state.score_a       = event.get("score_a", self.game_state.score_a)
        self.game_state.score_b       = event.get("score_b", self.game_state.score_b)
        self.game_state.period        = period

        # Halftime adjustment (one-time)
        if period == 2 and not self.game_state.halftime_occurred:
            self._apply_halftime()

        # Blowout check
        self._check_blowout()

        # Momentum tracking
        self._update_momentum(team, opponent, etype)

        # Core Elo delta
        multiplier = self.cfg.ELO_MULTIPLIERS.get(etype, 0.0)
        if multiplier == 0.0:
            return  # untracked event type

        delta = self.cfg.K_FACTOR * multiplier
        self.ratings[team]     += delta
        self.ratings[opponent] -= delta * 0.5  # partial drag on opponent

        self._event_log.append({
            "clock": event.get("clock"),
            "event": etype,
            "team": team,
            "delta": delta,
            "elo_a": self.ratings["team_a"],
            "elo_b": self.ratings["team_b"],
        })

        log.debug(f"{etype} by {team}: delta={delta:+.1f} | "
                  f"Elo A={self.ratings['team_a']:.1f} B={self.ratings['team_b']:.1f}")

    def apply_cv_observations(self, observations: list):
        """
        Process OpenCV observations:
        [
            {"type": "substitution", "player_tier": "star", "action": "out", "team": "team_a"},
            {"type": "position_change", "player_id": "p55", "from": "PG", "to": "SF", "team": "team_b"},
        ]
        """
        for obs in observations:
            otype = obs.get("type")
            team  = obs.get("team", "team_a")

            if otype == "substitution":
                tier   = obs.get("player_tier", "bench")   # "star" | "starter" | "bench"
                action = obs.get("action", "out")           # "in" | "out"
                key    = f"{tier}_player_{action}"
                mult   = self.cfg.ELO_MULTIPLIERS.get(key, 0.0)
                if mult:
                    delta = self.cfg.K_FACTOR * mult
                    opponent = "team_b" if team == "team_a" else "team_a"
                    self.ratings[team]     += delta
                    self.ratings[opponent] -= delta * 0.3
                    log.info(f"CV Sub: {key} for {team} → delta={delta:+.1f}")

            elif otype == "injury_substitution":
                delta = self.cfg.K_FACTOR * self.cfg.ELO_MULTIPLIERS["injury_substitution"]
                self.ratings[team] += delta
                log.warning(f"CV Injury sub detected for {team}: delta={delta:+.1f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Game-state triggers
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_halftime(self):
        """
        At halftime teams switch defensive/offensive sides.
        Teams that had momentum in H1 sometimes lose it in H2.
        Apply a small mean-reversion toward equal Elo.
        """
        self.game_state.halftime_occurred = True
        mult = self.cfg.ELO_MULTIPLIERS["halftime_switch"]
        avg  = (self.ratings["team_a"] + self.ratings["team_b"]) / 2
        self.ratings["team_a"] += (avg - self.ratings["team_a"]) * (mult - 1.0) * 0.3
        self.ratings["team_b"] += (avg - self.ratings["team_b"]) * (mult - 1.0) * 0.3
        log.info(f"Halftime reversion applied. Elo A={self.ratings['team_a']:.1f} "
                 f"B={self.ratings['team_b']:.1f}")

    def _check_blowout(self):
        """
        If one team is winning by BLOWOUT_POINT_DIFF points after
        BLOWOUT_TIME_THRESHOLD minutes, apply a one-time Elo boost.
        """
        if self.game_state.blowout_applied:
            return
        diff = self.game_state.score_a - self.game_state.score_b
        if (abs(diff) >= self.cfg.BLOWOUT_POINT_DIFF and
                self.game_state.clock_minutes >= self.cfg.BLOWOUT_TIME_THRESHOLD):
            shift = self.cfg.BLOWOUT_ELO_SHIFT
            if diff > 0:
                self.ratings["team_a"] += shift
                self.ratings["team_b"] -= shift
            else:
                self.ratings["team_b"] += shift
                self.ratings["team_a"] -= shift
            self.game_state.blowout_applied = True
            log.info(f"Blowout detected ({diff:+d} pts). "
                     f"Elo shift applied: {shift:.0f} pts.")

    def _update_momentum(self, team: str, opponent: str, etype: str):
        """Track consecutive scoring runs and apply momentum multiplier."""
        scoring_events = {"3pt_made", "2pt_made", "free_throw_made"}
        if etype not in scoring_events:
            return

        if team == "team_a":
            self.game_state.momentum_streak_a += 1
            self.game_state.momentum_streak_b = 0
            streak = self.game_state.momentum_streak_a
        else:
            self.game_state.momentum_streak_b += 1
            self.game_state.momentum_streak_a = 0
            streak = self.game_state.momentum_streak_b

        if streak >= 3:
            bonus = self.cfg.K_FACTOR * self.cfg.ELO_MULTIPLIERS["momentum_shift"]
            self.ratings[team]     += bonus
            self.ratings[opponent] -= bonus * 0.5
            log.info(f"Momentum streak x{streak} for {team}: bonus={bonus:+.1f}")

    # ──────────────────────────────────────────────────────────────────────────
    # Win probability output
    # ──────────────────────────────────────────────────────────────────────────

    def win_probability(self) -> Dict[str, float]:
        """
        Standard Elo win expectancy formula.
        Returns dict with team_a / team_b probabilities summing to 1.0.
        """
        diff = self.ratings["team_a"] - self.ratings["team_b"]
        p_a  = 1.0 / (1.0 + math.pow(10, -diff / 400.0))
        p_b  = 1.0 - p_a
        return {"team_a": round(p_a, 4), "team_b": round(p_b, 4)}

    def summary(self) -> dict:
        return {
            "ratings":     self.ratings.copy(),
            "win_prob":    self.win_probability(),
            "game_state":  self.game_state.__dict__,
            "events_seen": len(self._event_log),
        }
