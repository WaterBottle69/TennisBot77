"""
sportradar_scraper.py — SportRadar Tennis v3 WebSocket + REST client.

WebSocket push feed delivers point-by-point updates at ~200-500ms latency,
replacing the 10-second polling loop for live score data.

Sign-up: https://developer.sportradar.com  (free sandbox trial, no credit card)
Product: Tennis v3 (Trial)  →  copy your API key into kalshi_keys.json as
         "sportradar_api_key": "YOUR_KEY"

WebSocket URL:
  wss://api.sportradar.com/tennis/trial/v3/en/stream/events/subscribe?api_key=KEY

REST endpoints used:
  Live schedule : GET /schedules/live/schedule.json?api_key=KEY
  Daily schedule: GET /schedules/{date}/schedule.json?api_key=KEY
  Match summary : GET /sport_events/{match_id}/summary.json?api_key=KEY
"""

import asyncio
import json
import logging
import re
import ssl
import time
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_REST_BASE  = "https://api.sportradar.com/tennis/trial/v3/en"
_WS_URL     = "wss://api.sportradar.com/tennis/trial/v3/en/stream/events/subscribe"

# Reconnection back-off: 1 → 2 → 4 → 8 → … → max 60 seconds
_BACKOFF_INITIAL = 1.0
_BACKOFF_MAX     = 60.0

# How long (seconds) to cache a REST live-schedule response
_LIVE_SCHED_TTL  = 15.0


# ── helpers ──────────────────────────────────────────────────────────────────

def _name_overlap(a: str, b: str) -> bool:
    """True when any token of `a` appears in `b` (case-insensitive, len>2)."""
    tokens = a.lower().split()
    b_lo   = b.lower()
    return any(t in b_lo for t in tokens if len(t) > 2)


def _names_match(pa: str, pb: str, n1: str, n2: str) -> bool:
    """Return True if (pa, pb) matches (n1, n2) in either order."""
    fwd = _name_overlap(pa, n1) and _name_overlap(pb, n2)
    rev = _name_overlap(pa, n2) and _name_overlap(pb, n1)
    return fwd or rev


def _reversed(pa: str, pb: str, n1: str, n2: str) -> bool:
    """True when the Kalshi (pa, pb) order is the reverse of SportRadar (n1, n2)."""
    return _name_overlap(pa, n2) and _name_overlap(pb, n1)


def _map_score_str(s: str) -> int:
    """'0'→0, '15'→1, '30'→2, '40'→3, 'A'/'AD'/'ADV'→4."""
    s = str(s).strip().upper()
    if s in ("0", ""):             return 0
    if s == "15":                  return 1
    if s == "30":                  return 2
    if s in ("40", "3"):           return 3
    if s in ("A", "AD", "ADV"):    return 4
    try:
        return min(4, int(s))
    except ValueError:
        return 0


def _parse_point_field(point_str: str, server_is_home: bool) -> Tuple[int, int]:
    """
    Parse SportRadar 'point' field "server_score-receiver_score"
    into (home_pts, away_pts) integers.

    Example: "30-40", serving=away → away=30, home=40 → home_pts=40, away_pts=30
    """
    parts = re.split(r"[-:]", str(point_str or "0-0"), maxsplit=1)
    srv_pts = _map_score_str(parts[0]) if parts else 0
    rcv_pts = _map_score_str(parts[1]) if len(parts) > 1 else 0
    if server_is_home:
        return srv_pts, rcv_pts
    else:
        return rcv_pts, srv_pts


def _extract_stats(statistics: dict, home_key: str = "home",
                   away_key: str = "away") -> dict:
    """
    Convert SportRadar statistics block → normalised serve-stats dict.
    Keys match what historical_analyzer / NonIIDLiveMatchState expect.
    """
    h = statistics.get(home_key) or statistics.get("home_team") or {}
    a = statistics.get(away_key) or statistics.get("away_team") or {}

    def _pct(v) -> float:
        if v is None:
            return 0.0
        try:
            f = float(v)
            return f / 100.0 if f > 1.0 else f
        except Exception:
            return 0.0

    def _int(v) -> float:
        try:
            return float(v or 0)
        except Exception:
            return 0.0

    return {
        "first_serve_pct_a":      _pct(h.get("first_serve_in_pct") or h.get("first_serve_percentage")),
        "first_serve_pct_b":      _pct(a.get("first_serve_in_pct") or a.get("first_serve_percentage")),
        "pts_won_1st_serve_a":    _pct(h.get("first_serve_won_pct") or h.get("first_serve_points_won_pct")),
        "pts_won_1st_serve_b":    _pct(a.get("first_serve_won_pct") or a.get("first_serve_points_won_pct")),
        "pts_won_2nd_serve_a":    _pct(h.get("second_serve_won_pct") or h.get("second_serve_points_won_pct")),
        "pts_won_2nd_serve_b":    _pct(a.get("second_serve_won_pct") or a.get("second_serve_points_won_pct")),
        "break_pts_converted_a":  _pct(h.get("break_points_won") or 0) if not h.get("break_points_attempted") else
                                  ((_int(h.get("break_points_won")) / max(1, _int(h.get("break_points_attempted")))) if h.get("break_points_attempted") else 0.0),
        "break_pts_converted_b":  _pct(a.get("break_points_won") or 0) if not a.get("break_points_attempted") else
                                  ((_int(a.get("break_points_won")) / max(1, _int(a.get("break_points_attempted")))) if a.get("break_points_attempted") else 0.0),
        "aces_a":         _int(h.get("aces")),
        "aces_b":         _int(a.get("aces")),
        "double_faults_a": _int(h.get("double_faults")),
        "double_faults_b": _int(a.get("double_faults")),
        "serve_speed_avg_a": _int(h.get("avg_first_serve_speed") or h.get("first_serve_speed_avg")),
        "serve_speed_avg_b": _int(a.get("avg_first_serve_speed") or a.get("first_serve_speed_avg")),
    }


def _parse_sport_event_status(ses: dict, competitors: list,
                               player_a: str, player_b: str) -> Optional[dict]:
    """
    Convert a SportRadar sport_event_status dict + competitors list into the
    standard score dict yielded by poll_live_score_real.

    Returns None if the match is not currently live.
    """
    status = ses.get("status", "")
    if status not in ("live", "inprogress"):
        return None

    # Identify home/away sides and whether Kalshi (pa, pb) is reversed
    n1 = (competitors[0].get("name") or "") if competitors else ""
    n2 = (competitors[1].get("name") or "") if len(competitors) > 1 else ""

    swapped = _reversed(player_a, player_b, n1, n2)

    # Sets won
    home_sets = int(ses.get("home_score", 0) or 0)
    away_sets = int(ses.get("away_score", 0) or 0)

    # Period (set) scores
    period_scores = ses.get("period_scores") or []
    # Current set in progress is the last period that has no winner yet,
    # or the last period listed
    cur_ps = period_scores[-1] if period_scores else {}
    home_games = int(cur_ps.get("home_score", 0) or 0)
    away_games = int(cur_ps.get("away_score", 0) or 0)

    # Serving player
    serving = (ses.get("serving") or ses.get("current_ct_team") or "home").lower()
    home_serving = serving == "home"

    # Game points
    point_str = ses.get("point") or ses.get("game") or "0-0"
    home_pts, away_pts = _parse_point_field(point_str, server_is_home=home_serving)

    # Flip everything if Kalshi has the players in reverse order
    if swapped:
        sets   = (away_sets, home_sets)
        games  = (away_games, home_games)
        points = (away_pts, home_pts)
        p1_serving = not home_serving
    else:
        sets   = (home_sets, away_sets)
        games  = (home_games, away_games)
        points = (home_pts, away_pts)
        p1_serving = home_serving

    return {
        "sets":        sets,
        "games":       games,
        "points":      points,
        "p1_serving":  p1_serving,
        "is_live":     True,
        "status":      "LIVE",
        "source":      "sportradar",
        "data_age_secs": 0.0,
    }


# ── REST helpers ──────────────────────────────────────────────────────────────

class SportRadarRESTClient:
    """
    Thin async REST wrapper for SportRadar Tennis v3.
    Used for:
      - Discovering the sport_event id of a live match (needed for summary)
      - Fetching pre-game competitor profiles / head-to-head data
    """

    def __init__(self, api_key: str):
        self._key     = api_key
        self._session: Optional[aiohttp.ClientSession] = None
        self._live_cache: Tuple[float, list] = (0.0, [])

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _get_json(self, path: str) -> Optional[dict]:
        url = f"{_REST_BASE}/{path}?api_key={self._key}"
        try:
            session = self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=12)) as r:
                if r.status == 200:
                    return await r.json(content_type=None)
                if r.status == 403:
                    log.warning("[SPORTRADAR] 403 Forbidden — check API key and quota")
                elif r.status == 429:
                    log.warning("[SPORTRADAR] 429 Rate-limited — back off before retrying")
                else:
                    log.warning("[SPORTRADAR] HTTP %s for %s", r.status, path)
        except Exception as exc:
            log.warning("[SPORTRADAR] REST GET failed (%s): %s", path, exc)
            self._session = None
        return None

    async def get_live_matches(self) -> list:
        """
        Return a list of live sport_event dicts from the live schedule endpoint.
        Cached for _LIVE_SCHED_TTL seconds.
        """
        now = time.time()
        if now - self._live_cache[0] < _LIVE_SCHED_TTL:
            return self._live_cache[1]

        data = await self._get_json("schedules/live/schedule.json")
        if not data:
            return self._live_cache[1]

        matches = []
        for item in (data.get("sport_events") or []):
            ses   = item.get("sport_event_status") or {}
            se    = item.get("sport_event") or item
            comps = se.get("competitors") or []
            matches.append({
                "id":          se.get("id", ""),
                "status":      ses.get("status", ""),
                "competitors": comps,
                "sport_event_status": ses,
            })

        self._live_cache = (now, matches)
        return matches

    async def find_live_match_id(self, player_a: str, player_b: str) -> Optional[str]:
        """Return the sport_event id for a live match between player_a and player_b."""
        matches = await self.get_live_matches()
        for m in matches:
            comps = m.get("competitors", [])
            n1 = comps[0].get("name", "") if comps else ""
            n2 = comps[1].get("name", "") if len(comps) > 1 else ""
            if _names_match(player_a, player_b, n1, n2):
                return m["id"]
        return None

    async def get_match_summary(self, match_id: str) -> Optional[dict]:
        """Full match summary including statistics."""
        return await self._get_json(f"sport_events/{match_id}/summary.json")

    async def get_daily_schedule(self, date_str: str) -> list:
        """
        Return sport_event list for a given date (YYYY-MM-DD).
        Used by poll_live_score_real to find upcoming (NS) matches.
        """
        data = await self._get_json(f"schedules/{date_str}/schedule.json")
        if not data:
            return []
        return data.get("sport_events") or []

    async def get_pregame_profile(self, player_a: str, player_b: str) -> dict:
        """
        Fetch head-to-head and competitor profiles from the match summary
        of a scheduled (or recently live) match.

        Returns a dict with keys matching what TennisStatsScraper returns
        so it can be merged directly with pregame_matchup output.
        """
        match_id = await self.find_live_match_id(player_a, player_b)
        if not match_id:
            return {}

        summary = await self.get_match_summary(match_id)
        if not summary:
            return {}

        stats = summary.get("statistics") or {}
        serve_stats = _extract_stats(stats)

        se    = summary.get("sport_event") or {}
        comps = se.get("competitors") or []
        swapped = bool(comps) and _reversed(player_a, player_b,
                                             comps[0].get("name", ""),
                                             comps[1].get("name", "") if len(comps) > 1 else "")
        if swapped:
            # suffix _a and _b refer to Kalshi player_a / player_b, not home/away
            serve_stats = {
                k.replace("_a", "_TMP").replace("_b", "_a").replace("_TMP", "_b"): v
                for k, v in serve_stats.items()
            }

        return serve_stats


# ── WebSocket stream ─────────────────────────────────────────────────────────

class SportRadarLiveStream:
    """
    Connects to the SportRadar Tennis v3 WebSocket push feed and filters
    events for a specific match (player_a vs player_b).

    Usage:
        async for update in stream.listen():
            # update is the same dict shape as poll_live_score_real yields
            process(update)

    The generator runs indefinitely, reconnecting on drops with exponential
    back-off. Call stream.stop() to terminate gracefully.
    """

    def __init__(self, player_a: str, player_b: str, api_key: str):
        self.player_a   = player_a
        self.player_b   = player_b
        self._key       = api_key
        self._stopped   = False
        self._ws_url    = f"{_WS_URL}?api_key={api_key}"
        self._last_stats: dict = {}
        self._match_id: Optional[str] = None
        self._competitors: list = []

    def stop(self):
        self._stopped = True

    def last_serve_stats(self) -> dict:
        """Return the most recently received serve statistics (may be empty)."""
        return self._last_stats.copy()

    async def listen(self) -> AsyncGenerator[dict, None]:
        """
        Async generator that yields score update dicts.
        Reconnects automatically on WebSocket errors.
        """
        backoff = _BACKOFF_INITIAL
        while not self._stopped:
            try:
                connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
                async with aiohttp.ClientSession(connector=connector) as session:
                    log.info("[SPORTRADAR-WS] Connecting to WebSocket feed …")
                    async with session.ws_connect(
                        self._ws_url,
                        heartbeat=30.0,
                        timeout=aiohttp.ClientWSTimeout(ws_close=10.0),
                    ) as ws:
                        log.info("[SPORTRADAR-WS] Connected. Waiting for events for %s vs %s",
                                 self.player_a, self.player_b)
                        backoff = _BACKOFF_INITIAL  # reset on successful connect

                        async for msg in ws:
                            if self._stopped:
                                return

                            if msg.type == aiohttp.WSMsgType.TEXT:
                                update = self._handle_message(msg.data)
                                if update is not None:
                                    yield update

                            elif msg.type == aiohttp.WSMsgType.ERROR:
                                log.warning("[SPORTRADAR-WS] WSMsgType.ERROR — reconnecting")
                                break

                            elif msg.type == aiohttp.WSMsgType.CLOSED:
                                log.info("[SPORTRADAR-WS] Connection closed — reconnecting")
                                break

            except aiohttp.WSServerHandshakeError as exc:
                if exc.status == 401:
                    log.error("[SPORTRADAR-WS] 401 Unauthorized — check your API key. Stopping WS.")
                    return
                if exc.status == 403:
                    log.error("[SPORTRADAR-WS] 403 Forbidden — trial quota may be exhausted. Stopping WS.")
                    return
                log.warning("[SPORTRADAR-WS] Handshake error %s — retrying in %.0fs", exc.status, backoff)
            except asyncio.CancelledError:
                return
            except Exception as exc:
                log.warning("[SPORTRADAR-WS] Connection error: %s — retrying in %.0fs", exc, backoff)

            if not self._stopped:
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _BACKOFF_MAX)

    def _handle_message(self, raw: str) -> Optional[dict]:
        """
        Parse a raw WebSocket message and return a score dict if it belongs
        to the target match and carries a score update, else None.
        """
        try:
            envelope = json.loads(raw)
        except json.JSONDecodeError:
            return None

        # SportRadar wraps everything in a "payload" key
        payload  = envelope.get("payload") or envelope
        meta     = payload.get("metadata") or {}
        event_type = (meta.get("event_type") or meta.get("type") or "").lower()

        se_block = (payload.get("event") or payload.get("sport_event") or {})
        se       = se_block.get("sport_event") or se_block
        ses      = (se_block.get("sport_event_status")
                    or payload.get("sport_event_status")
                    or {})

        competitors = se.get("competitors") or self._competitors
        if competitors:
            self._competitors = competitors

        n1 = competitors[0].get("name", "") if competitors else ""
        n2 = competitors[1].get("name", "") if len(competitors) > 1 else ""

        # Filter: only process events for our target match
        if n1 or n2:
            if not _names_match(self.player_a, self.player_b, n1, n2):
                return None

        # Store match id for REST lookups
        mid = se.get("id", "")
        if mid:
            self._match_id = mid

        # Collect serve statistics when present
        stats = (se_block.get("statistics") or payload.get("statistics") or {})
        if stats:
            parsed_stats = _extract_stats(stats)
            if any(v for v in parsed_stats.values()):
                swapped = _reversed(self.player_a, self.player_b, n1, n2)
                if swapped:
                    parsed_stats = {
                        k.replace("_a", "_TMP").replace("_b", "_a").replace("_TMP", "_b"): v
                        for k, v in parsed_stats.items()
                    }
                self._last_stats = parsed_stats

        # Only yield on score-bearing events
        if event_type not in ("score_update", "match_started", "period_score",
                               "server_change", "tiebreak_start", ""):
            if event_type == "match_ended":
                log.info("[SPORTRADAR-WS] Match ended: %s vs %s", self.player_a, self.player_b)
                self._stopped = True
            return None

        if not ses:
            return None

        status = ses.get("status", "")
        if status == "closed":
            log.info("[SPORTRADAR-WS] Match closed: %s vs %s", self.player_a, self.player_b)
            self._stopped = True
            return None

        update = _parse_sport_event_status(ses, competitors, self.player_a, self.player_b)
        if update is None:
            return None

        # Attach tournament name if available
        tournament = (se.get("sport_event_context", {})
                       .get("competition", {})
                       .get("name", "")
                       or se.get("tournament", {}).get("name", ""))
        update["tournament"] = tournament

        log.debug("[SPORTRADAR-WS] %s vs %s — sets=%s games=%s pts=%s srv=%s",
                  self.player_a, self.player_b,
                  update["sets"], update["games"],
                  update["points"], update["p1_serving"])
        return update


# ── module-level singletons ───────────────────────────────────────────────────
# These are instantiated with a placeholder key and re-keyed in
# live_score_scraper.py once config is loaded.

_rest_client: Optional[SportRadarRESTClient] = None


def get_rest_client(api_key: str) -> SportRadarRESTClient:
    global _rest_client
    if _rest_client is None or _rest_client._key != api_key:
        if _rest_client is not None:
            asyncio.get_event_loop().create_task(_rest_client.close())
        _rest_client = SportRadarRESTClient(api_key)
    return _rest_client
