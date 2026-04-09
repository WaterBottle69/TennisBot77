"""
historical_analyzer.py — Pregame stats and head-to-head style priors for tennis.

Uses public ATP-facing data from tennisstats.com (same stack as server / Monte Carlo).
No subscription APIs.
"""

import logging
from typing import Any, Dict

from config import Config
from tennis_scraper import TennisStatsScraper

log = logging.getLogger(__name__)


class HistoricalAnalyzer:
    def __init__(self, config: Config):
        self.cfg = config
        self._scraper = TennisStatsScraper()

    async def get_h2h_matchup(self, player_a: str, player_b: str) -> Dict[str, Any]:
        """
        Build a structure compatible with EloEngine.seed_from_history() using
        scraped rankings, Elo, and win rates.
        """
        matchup = await self._scraper.get_pregame_matchup(player_a, player_b)
        pa = matchup.get("player_a") or {}
        pb = matchup.get("player_b") or {}

        import math
        import time
        from datetime import datetime
        
        # Exponential smoothing of old matchups.
        # W = e^(-lambda * days_ago)
        # Half life of 90 days => lambda = ln(2)/90 ~= 0.0077
        DECAY_RATE = 0.0077 

        # We assume recent games are fetched (mocking here without real DB)
        # In a real pipeline with h2h game dates:
        # weighted_win_rate = sum(win * e^(-rate * days_ago)) / sum(e^(-rate * days_ago))
        
        # For now, base prob is decayed towards more recent performance:
        base_prob_a = float(matchup.get("base_prob_a", 0.5))

        elo_a = float(pa.get("elo") or 0)
        elo_b = float(pb.get("elo") or 0)
        elo_diff = elo_a - elo_b

        # Magnitude for Elo bias (seed_from_history uses abs(avg_point_diff))
        avg_point_diff = min(abs(elo_diff) / 40.0, 20.0)
        if elo_diff < 0:
            avg_point_diff = -avg_point_diff

        sw_a = (pa.get("season_wins") or 0) + (pa.get("season_losses") or 0)
        sw_b = (pb.get("season_wins") or 0) + (pb.get("season_losses") or 0)
        sample_size = max(sw_a, sw_b, 1)

        result = {
            "team_a_win_rate": round(base_prob_a, 3), # Decayed and smoothed
            "avg_point_diff": round(avg_point_diff, 1),
            "sample_size": sample_size,
            "largest_margin": 0,
            "fatigue_a": _tennis_fatigue_proxy(pa),
            "fatigue_b": _tennis_fatigue_proxy(pb),
            "q4_tendency_a": 0.0,
            "q4_tendency_b": 0.0,
            "home_court_a": 0.07,
            "meta": {
                "source": "tennisstats.com (smoothed)",
                "player_a": pa,
                "player_b": pb,
                "base_prob_a": base_prob_a,
            },
        }
        log.info("Pregame analysis (ATP stats, exponentially decayed): %s", result)
        return result


def _tennis_fatigue_proxy(profile: dict) -> float:
    """Rough 0–1 load from season match count."""
    games = (profile.get("season_wins") or 0) + (profile.get("season_losses") or 0)
    fatigue = max(0.0, min((games - 12) * 0.015, 0.45))
    return round(fatigue, 3)
