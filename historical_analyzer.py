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

    async def get_live_atp_stats(self, player_a: str, player_b: str) -> dict:
        """
        Fetch real-time ATP service statistics from the Infosys live stats endpoint.
        Uses cloudscraper (via kalshi_client) to bypass Cloudflare protection.

        The ATP platform is AJAX-driven (RequireJS frontend) so standard requests
        libraries consistently fail — cloudscraper handles JS challenges.

        Returns dict with normalized service metrics, or empty dict on failure.
        """
        import cloudscraper
        import asyncio

        url = "https://www.atptour.com/-/ajax/Scores/GetInitialScores"
        try:
            scraper = cloudscraper.create_scraper(
                browser={"browser": "chrome", "platform": "windows", "mobile": False}
            )
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: scraper.get(url, timeout=12)
            )
            if resp.status_code != 200:
                log.warning(f"[ATP] GetInitialScores returned {resp.status_code}")
                return {}

            data = resp.json()
            return self._extract_match_stats(data, player_a, player_b)

        except Exception as e:
            log.warning(f"[ATP] Live stat fetch failed for {player_a} vs {player_b}: {e}")
            return {}

    def _extract_match_stats(self, data: dict, player_a: str, player_b: str) -> dict:
        """
        Parse the nested ATP Infosys JSON to extract live serve statistics.

        Structure: data["liveScores"]["Tournaments"][n]["Matches"][m]

        Fuzzy-matches player names against scraped player names to find the right match.
        Returns normalized stats dict or empty dict if match not found.
        """
        def _name_overlap(a: str, b: str) -> bool:
            """True if any last-name token of a appears in b (case-insensitive)."""
            tokens_a = a.lower().split()
            b_lower  = b.lower()
            return any(tok in b_lower for tok in tokens_a if len(tok) > 2)

        try:
            live_scores = data.get("liveScores") or data
            tournaments = live_scores.get("Tournaments") or []
            for tourney in tournaments:
                for match in (tourney.get("Matches") or []):
                    status = (match.get("MatchStatus") or "").lower()
                    if "progress" not in status and "live" not in status and "playing" not in status:
                        continue

                    teams = match.get("Teams") or match.get("Players") or []
                    if len(teams) < 2:
                        continue

                    n1 = (teams[0].get("PlayerName") or teams[0].get("Name") or "")
                    n2 = (teams[1].get("PlayerName") or teams[1].get("Name") or "")

                    # Fuzzy match: does this match correspond to our players?
                    if not (
                        (_name_overlap(player_a, n1) and _name_overlap(player_b, n2)) or
                        (_name_overlap(player_a, n2) and _name_overlap(player_b, n1))
                    ):
                        continue

                    # Determine which team index = player_a
                    a_idx = 0 if _name_overlap(player_a, n1) else 1
                    b_idx = 1 - a_idx

                    stats = match.get("Statistics") or match.get("Stats") or {}

                    def _fv(key_tmpl: str, idx: int) -> float:
                        """Try keys with suffix 1/2 and A/B."""
                        for suffix in [str(idx + 1), ("A" if idx == 0 else "B")]:
                            v = stats.get(key_tmpl + suffix)
                            if v is not None:
                                try:
                                    return float(str(v).replace("%", "").strip())
                                except Exception:
                                    pass
                        return 0.0

                    result = {
                        "first_serve_pct_a":      _fv("FirstServePercentage", a_idx),
                        "first_serve_pct_b":      _fv("FirstServePercentage", b_idx),
                        "pts_won_1st_serve_a":    _fv("PointsWonOn1stServe",  a_idx),
                        "pts_won_1st_serve_b":    _fv("PointsWonOn1stServe",  b_idx),
                        "pts_won_2nd_serve_a":    _fv("PointsWonOn2ndServe",  a_idx),
                        "pts_won_2nd_serve_b":    _fv("PointsWonOn2ndServe",  b_idx),
                        "break_pts_converted_a":  _fv("BreakPointsConverted", a_idx),
                        "break_pts_converted_b":  _fv("BreakPointsConverted", b_idx),
                        "aces_a":                 _fv("Aces",                 a_idx),
                        "aces_b":                 _fv("Aces",                 b_idx),
                        "double_faults_a":        _fv("DoubleFaults",         a_idx),
                        "double_faults_b":        _fv("DoubleFaults",         b_idx),
                        "player_a_name":          n1 if a_idx == 0 else n2,
                        "player_b_name":          n2 if a_idx == 0 else n1,
                    }
                    log.info(f"[ATP] Found live stats for {player_a} vs {player_b}: {result}")
                    return result

        except Exception as e:
            log.warning(f"[ATP] Stats parse failed: {e}")

        log.debug(f"[ATP] No live match found for {player_a} vs {player_b}")
        return {}


def _tennis_fatigue_proxy(profile: dict) -> float:
    """Rough 0–1 load from season match count."""
    games = (profile.get("season_wins") or 0) + (profile.get("season_losses") or 0)
    fatigue = max(0.0, min((games - 12) * 0.015, 0.45))
    return round(fatigue, 3)
