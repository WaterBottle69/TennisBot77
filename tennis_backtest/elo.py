"""
elo.py — Elo rating system for the tennis mentality score backtest pipeline.

Provides EloSystem class for computing player ratings from match history,
including history tracking and pre-match Elo lookups via binary search.
"""

import bisect
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class EloSystem:
    """Simple Elo rating system for tennis matches."""

    def __init__(self, k: float = 32, initial: float = 1500.0) -> None:
        self.k = k
        self.initial = initial
        self._ratings: Dict[int, float] = {}
        # history: player_id -> [(date, elo), ...] sorted ascending by date
        self.history: Dict[int, List[Tuple[pd.Timestamp, float]]] = {}

    def get_elo(self, player_id: int) -> float:
        """Return current Elo for player_id, defaulting to initial if unseen."""
        return self._ratings.get(player_id, self.initial)

    def update(
        self, winner_id: int, loser_id: int
    ) -> Tuple[float, float]:
        """Update Elo ratings after a match.

        Returns (new_winner_elo, new_loser_elo).
        """
        r_w = self.get_elo(winner_id)
        r_l = self.get_elo(loser_id)

        expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
        expected_l = 1.0 - expected_w

        new_r_w = r_w + self.k * (1.0 - expected_w)
        new_r_l = r_l + self.k * (0.0 - expected_l)

        self._ratings[winner_id] = new_r_w
        self._ratings[loser_id] = new_r_l

        return new_r_w, new_r_l

    def build_from_matches(self, df: pd.DataFrame) -> Dict[int, float]:
        """Process matches chronologically, building ratings and history.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: winner_id, loser_id, tourney_date (Timestamp).

        Returns
        -------
        dict mapping player_id -> final Elo rating
        """
        self._ratings = {}
        self.history = {}

        required = {"winner_id", "loser_id", "tourney_date"}
        missing = required - set(df.columns)
        if missing:
            logger.warning("build_from_matches: missing columns %s", missing)
            return {}

        sorted_df = df.sort_values("tourney_date")

        for row in sorted_df.itertuples(index=False):
            try:
                w_id = int(row.winner_id)
                l_id = int(row.loser_id)
                match_date = row.tourney_date
            except Exception:
                continue

            new_w, new_l = self.update(w_id, l_id)

            if w_id not in self.history:
                self.history[w_id] = []
            if l_id not in self.history:
                self.history[l_id] = []

            self.history[w_id].append((match_date, new_w))
            self.history[l_id].append((match_date, new_l))

        return dict(self._ratings)

    def get_elo_before_match(
        self,
        player_id: int,
        match_date: pd.Timestamp,
        history: Optional[Dict[int, List[Tuple[pd.Timestamp, float]]]] = None,
    ) -> float:
        """Return the most recent Elo strictly before match_date.

        Uses binary search on the history list (sorted ascending by date).
        Returns initial Elo if no prior entry exists.
        """
        if history is None:
            history = self.history

        player_hist = history.get(player_id)
        if not player_hist:
            return self.initial

        # Extract dates for binary search
        dates = [entry[0] for entry in player_hist]

        # Find insertion point for match_date — we want strictly before
        idx = bisect.bisect_left(dates, match_date)
        if idx == 0:
            return self.initial

        return player_hist[idx - 1][1]
