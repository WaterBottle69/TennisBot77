"""
atp_stats_centre.py — ATP Stats Centre live match scraper.

Scrapes https://www.atptour.com/en/scores/stats-centre/live/{year}/{tournament_id}/{match_id}
to pull per-point serve/break stats for the Markov engine.

Match ID discovery flow:
  1. Hit the ATP AJAX live scores endpoint (GetInitialScores)
  2. Fuzzy-match player names to find the live match
  3. Extract TournamentId + MatchId from the JSON
  4. Construct the stats-centre URL and scrape the HTML stats table
"""

import asyncio
import logging
import re
import json
import time
from datetime import datetime
from typing import Optional, Dict

log = logging.getLogger(__name__)

try:
    import cloudscraper
    _CS_AVAIL = True
except ImportError:
    _CS_AVAIL = False
    log.warning("[ATP-STATS] cloudscraper not installed — ATP stats-centre scraping disabled.")

try:
    from bs4 import BeautifulSoup
    _BS4_AVAIL = True
except ImportError:
    _BS4_AVAIL = False

ATP_AJAX_URL  = "https://www.atptour.com/-/ajax/Scores/GetInitialScores"
ATP_STATS_BASE = "https://www.atptour.com/en/scores/stats-centre/live"


def _name_overlap(a: str, b: str) -> bool:
    tokens_a = a.lower().split()
    b_lower = b.lower()
    return any(tok in b_lower for tok in tokens_a if len(tok) > 2)


def _safe_pct(v: str) -> Optional[float]:
    """Parse '63%', '63', or '0.63' → 0.63."""
    v = v.replace("%", "").strip()
    try:
        f = float(v)
        return f / 100.0 if f > 1.0 else f
    except Exception:
        return None


class ATPStatsCentreScraper:
    """
    Discovers and scrapes live ATP Stats Centre pages.
    Instance is long-lived; caches the match URL and stats within each session.
    """

    STATS_TTL   = 30.0   # seconds before re-fetching stats page
    MATCHID_TTL = 120.0  # seconds before re-discovering match URL

    def __init__(self):
        self._url_cache: Dict[str, dict]   = {}   # pair_key → {url, ts}
        self._stats_cache: Dict[str, dict] = {}   # url → {stats, ts}

    # ── Public API ─────────────────────────────────────────────────────────────

    async def fetch_stats(self, player_a: str, player_b: str,
                          stats_url: str = None) -> dict:
        """Return live serve/break stats for the named match. Empty dict on failure."""
        if stats_url is None:
            stats_url = await self._find_match_url(player_a, player_b)
        if not stats_url:
            log.debug("[ATP-STATS] No stats URL found for %s vs %s", player_a, player_b)
            return {}

        cached = self._stats_cache.get(stats_url, {})
        if cached and time.time() - cached.get("ts", 0) < self.STATS_TTL:
            return cached["stats"]

        stats = await self._scrape_stats_page(stats_url, player_a, player_b)
        self._stats_cache[stats_url] = {"stats": stats, "ts": time.time()}
        if stats:
            log.info("[ATP-STATS] %s vs %s → %s", player_a, player_b, stats)
        return stats

    # ── Internal ───────────────────────────────────────────────────────────────

    async def _find_match_url(self, player_a: str, player_b: str) -> Optional[str]:
        pair_key = f"{player_a}|{player_b}"
        cached = self._url_cache.get(pair_key, {})
        if cached and time.time() - cached.get("ts", 0) < self.MATCHID_TTL:
            return cached.get("url")

        if not _CS_AVAIL:
            return None

        loop   = asyncio.get_event_loop()
        sc     = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        year   = datetime.now().year

        try:
            resp = await loop.run_in_executor(
                None, lambda: sc.get(ATP_AJAX_URL, timeout=12)
            )
            if resp.status_code != 200:
                return None

            data        = resp.json()
            live_scores = data.get("liveScores") or data
            tournaments = live_scores.get("Tournaments") or []

            for tourney in tournaments:
                t_id = (
                    tourney.get("TournamentId") or
                    tourney.get("Id") or
                    tourney.get("TId") or ""
                )
                for match in (tourney.get("Matches") or []):
                    status = (match.get("MatchStatus") or "").lower()
                    if not any(k in status for k in ("progress", "live", "playing")):
                        continue

                    teams = match.get("Teams") or match.get("Players") or []
                    if len(teams) < 2:
                        continue

                    n1 = (teams[0].get("PlayerName") or teams[0].get("Name") or "")
                    n2 = (teams[1].get("PlayerName") or teams[1].get("Name") or "")

                    matched = (
                        (_name_overlap(player_a, n1) and _name_overlap(player_b, n2)) or
                        (_name_overlap(player_a, n2) and _name_overlap(player_b, n1))
                    )
                    if not matched:
                        continue

                    m_id = (
                        match.get("MatchId") or
                        match.get("Id") or
                        match.get("MId") or ""
                    )
                    if t_id and m_id:
                        url = f"{ATP_STATS_BASE}/{year}/{t_id}/{m_id}"
                        self._url_cache[pair_key] = {"url": url, "ts": time.time()}
                        log.info("[ATP-STATS] Discovered stats URL: %s", url)
                        return url

        except Exception as exc:
            log.warning("[ATP-STATS] URL discovery failed: %s", exc)

        return None

    async def _scrape_stats_page(self, url: str, player_a: str, player_b: str) -> dict:
        if not _CS_AVAIL or not _BS4_AVAIL:
            return {}

        loop = asyncio.get_event_loop()
        sc   = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )

        try:
            resp = await loop.run_in_executor(
                None, lambda: sc.get(url, timeout=14)
            )
            if resp.status_code != 200:
                log.warning("[ATP-STATS] %s returned %s", url, resp.status_code)
                return {}
            return self._parse_html(resp.text, player_a, player_b)
        except Exception as exc:
            log.warning("[ATP-STATS] scrape failed (%s): %s", url, exc)
            return {}

    def _parse_html(self, html: str, player_a: str, player_b: str) -> dict:
        """Parse the ATP Stats Centre HTML. Tries embedded JSON first, then table."""
        result = {}

        # 1. Look for embedded JSON blobs (Next.js, AJAX injection, etc.)
        for js_candidate in re.findall(r'\{[^<]{200,}\}', html):
            try:
                data = json.loads(js_candidate)
                r = self._extract_from_json(data)
                if len(r) >= 3:
                    return r
            except Exception:
                pass

        # 2. BeautifulSoup table parse
        if not _BS4_AVAIL:
            return result

        soup = BeautifulSoup(html, "html.parser")

        # Stats table rows — ATP uses varying class names across redesigns
        selectors = [
            "tr.stat-row", ".stat-row", ".stats-row",
            "tr.match-stat", ".match-stat",
            ".score-stats tr", ".stats-table tr",
        ]
        rows = []
        for sel in selectors:
            rows = soup.select(sel)
            if rows:
                break

        for row in rows:
            cells = row.find_all(["td", "div"], recursive=False) or row.find_all(["td", "div"])
            if len(cells) < 3:
                continue
            # ATP table: [p1_val, stat_label, p2_val]  OR  [stat_label, p1_val, p2_val]
            mid = len(cells) // 2
            label = cells[mid].get_text(strip=True).lower()
            val_a = cells[0].get_text(strip=True)
            val_b = cells[-1].get_text(strip=True)
            self._apply_stat(result, label, val_a, val_b)

        return result

    def _extract_from_json(self, data: dict) -> dict:
        result = {}
        stats_list = (
            data.get("MatchStats") or data.get("matchStats") or
            data.get("stats") or data.get("Statistics") or []
        )
        if isinstance(stats_list, list):
            for s in stats_list:
                name = (s.get("name") or s.get("label") or s.get("Name") or "").lower()
                v1   = str(s.get("p1") or s.get("player1") or s.get("value1") or s.get("V1") or "")
                v2   = str(s.get("p2") or s.get("player2") or s.get("value2") or s.get("V2") or "")
                self._apply_stat(result, name, v1, v2)
        return result

    def _apply_stat(self, result: dict, label: str, val_a: str, val_b: str):
        """Map a stat label + two values into the normalized result dict."""
        l = label.lower()
        if "1st serve" in l and "%" in l:
            result["first_serve_pct_a"] = _safe_pct(val_a) or 0.0
            result["first_serve_pct_b"] = _safe_pct(val_b) or 0.0
        elif "1st serve" in l and ("won" in l or "point" in l):
            result["pts_won_1st_serve_a"] = _safe_pct(val_a) or 0.0
            result["pts_won_1st_serve_b"] = _safe_pct(val_b) or 0.0
        elif "2nd serve" in l and ("won" in l or "point" in l):
            result["pts_won_2nd_serve_a"] = _safe_pct(val_a) or 0.0
            result["pts_won_2nd_serve_b"] = _safe_pct(val_b) or 0.0
        elif "break point" in l and "convert" in l:
            result["break_pts_converted_a"] = _safe_pct(val_a) or 0.0
            result["break_pts_converted_b"] = _safe_pct(val_b) or 0.0
        elif "ace" in l:
            try:
                result["aces_a"] = float(val_a.replace("%", ""))
                result["aces_b"] = float(val_b.replace("%", ""))
            except Exception:
                pass
        elif "double fault" in l:
            try:
                result["double_faults_a"] = float(val_a.replace("%", ""))
                result["double_faults_b"] = float(val_b.replace("%", ""))
            except Exception:
                pass
        elif "serve speed" in l or "avg speed" in l:
            try:
                result["serve_speed_avg_a"] = float(re.sub(r"[^\d.]", "", val_a))
                result["serve_speed_avg_b"] = float(re.sub(r"[^\d.]", "", val_b))
            except Exception:
                pass


# Module-level singleton
atp_stats_scraper = ATPStatsCentreScraper()
