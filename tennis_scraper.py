"""
api-tennis.com Scraper
======================
Replaces the old tennisstats.com scraper. Uses the api-tennis.com REST API
for player rankings, profiles, and serve statistics.

API key is read from kalshi_keys.json ("api_tennis_key") or the
API_TENNIS_KEY environment variable.

Endpoints used:
  get_fixtures  → builds name→player_key cache from recent matches
  get_players   → per-season stats: rank, surface W/L, titles, DOB
  get_livescore → (bonus) live scores with serve statistics per point
  get_H2H       → head-to-head history

Height and handedness come from Jeff Sackmann's free GitHub CSV
(atp_players.csv / wta_players.csv) — these are permanent player
attributes that never change and are not provided by the API.
"""

import aiohttp
import asyncio
import csv
import io
import logging
import re
import ssl
import time as _time
from datetime import date, datetime, timedelta
from difflib import get_close_matches

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_API_BASE = "https://api.api-tennis.com/tennis/"

# Sackmann GitHub CSVs for height + handedness (static player attributes)
_SACKMANN_ATP = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_players.csv"
_SACKMANN_WTA = "https://raw.githubusercontent.com/JeffSackmann/tennis_wta/master/wta_players.csv"

_NAME_BLACKLIST = {
    "argentina", "jordan", "brazil", "france", "germany", "england", "spain",
    "italy", "usa", "mexico", "canada", "australia", "china", "japan",
    "team", "club", "all-stars", "field", "other", "any", "combined",
}


def _name_to_slug(name: str) -> str:
    """'Carlos Alcaraz' → 'carlos-alcaraz'"""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


def _age_from_dob(dob_str: str) -> int:
    """Parse 'DD.MM.YYYY' or 'YYYYMMDD' and return age in years."""
    try:
        if "." in dob_str:
            d = datetime.strptime(dob_str, "%d.%m.%Y")
        else:
            d = datetime.strptime(dob_str, "%Y%m%d")
        today = date.today()
        return today.year - d.year - ((today.month, today.day) < (d.month, d.day))
    except Exception:
        return 25


class ApiTennisScraper:
    """
    Drop-in replacement for TennisStatsScraper using api-tennis.com.

    Public interface is identical so historical_analyzer.py / main.py
    require no changes.

    Data returned per player (same keys as the old scraper):
      slug           – URL-safe name (e.g. 'carlos-alcaraz')
      player_key     – api-tennis.com numeric ID
      ranking        – int current singles rank
      elo            – 0 (API does not provide ELO; disables pts_rank edge gracefully)
      win_rate       – float career wins/(wins+losses)
      career_wins    – int
      career_losses  – int
      season_wins    – int current year
      season_losses  – int current year
      surface_win_rate – float win rate on current match surface
      recent_win_rate  – float win rate over last 2 seasons
      age            – int
      height_cm      – int (from Sackmann CSV)
      hand           – 'R' or 'L' (from Sackmann CSV)
      profile_url    – str
    """

    _KEY_CACHE_TTL  = 3600 * 6   # rebuild name→key cache every 6 hours
    _PROFILE_TTL    = 3600       # cache player profiles for 1 hour
    _SACKMANN_TTL   = 86400 * 7  # refresh Sackmann CSV weekly
    _RANKINGS_TTL   = 3600       # rankings cache TTL

    def __init__(self, api_key: str = "", **_kwargs):
        import os
        self._api_key = api_key or os.getenv("API_TENNIS_KEY", "")
        # name slug → player_key int
        self._name_to_key: dict = {}
        self._name_to_key_ts: float = 0.0
        # player_key → profile dict
        self._profile_cache: dict = {}
        self._profile_cache_ts: dict = {}
        # slug → rankings entry (for fetch_rankings compat)
        self._rankings: dict = {}
        self._rankings_ts: float = 0.0
        # Sackmann data: lower_fullname → {height_cm, hand}
        self._sackmann: dict = {}
        self._sackmann_ts: float = 0.0
        self._session: aiohttp.ClientSession = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ------------------------------------------------------------------
    # Core HTTP helper
    # ------------------------------------------------------------------

    async def _get_json(self, params: dict) -> dict:
        """GET request to api-tennis.com, returns parsed JSON dict."""
        params = dict(params)
        params["APIkey"] = self._api_key
        try:
            session = self._get_session()
            async with session.get(
                _API_BASE, params=params, timeout=aiohttp.ClientTimeout(total=20)
            ) as resp:
                if resp.status != 200:
                    log.warning("[API-TENNIS] HTTP %s for %s", resp.status, params.get("method"))
                    return {}
                return await resp.json(content_type=None)
        except Exception as exc:
            log.warning("[API-TENNIS] Request failed (%s): %s", params.get("method"), exc)
            self._session = None
            return {}

    # ------------------------------------------------------------------
    # Name → player_key cache
    # ------------------------------------------------------------------

    async def _ensure_key_cache(self):
        """Build/refresh the name→player_key lookup from recent fixtures."""
        if self._name_to_key and _time.time() - self._name_to_key_ts < self._KEY_CACHE_TTL:
            return

        today = date.today()
        start = (today - timedelta(days=14)).isoformat()
        stop  = today.isoformat()

        log.info("[API-TENNIS] Refreshing player key cache (last 14 days of fixtures)...")
        data = await self._get_json({
            "method": "get_fixtures",
            "date_start": start,
            "date_stop":  stop,
        })

        added = 0
        for event in data.get("result", []):
            # Skip doubles
            etype = event.get("event_type_type", "")
            if "Doubles" in etype or "doubles" in etype:
                continue
            p1 = event.get("event_first_player", "")
            k1 = event.get("first_player_key")
            p2 = event.get("event_second_player", "")
            k2 = event.get("second_player_key")
            if p1 and k1:
                self._name_to_key[_name_to_slug(p1)] = int(k1)
                added += 1
            if p2 and k2:
                self._name_to_key[_name_to_slug(p2)] = int(k2)
                added += 1

        self._name_to_key_ts = _time.time()
        log.info("[API-TENNIS] Key cache built: %d unique player slugs", len(self._name_to_key))

    def _resolve_key(self, name: str) -> int:
        """Fuzzy-match player name to a player_key. Returns 0 if not found."""
        slug = _name_to_slug(name)

        if slug in self._name_to_key:
            return self._name_to_key[slug]

        # Last-name-only (no hyphen) → find highest-priority entry ending with it
        if "-" not in slug and len(slug) >= 3:
            for k in self._name_to_key:
                if k.endswith(f"-{slug}"):
                    return self._name_to_key[k]

        candidates = get_close_matches(slug, self._name_to_key.keys(), n=1, cutoff=0.75)
        if candidates:
            resolved = candidates[0]
            if any(p in _NAME_BLACKLIST for p in resolved.split("-")):
                return 0
            return self._name_to_key[resolved]

        return 0

    # ------------------------------------------------------------------
    # Sackmann CSV (height + handedness)
    # ------------------------------------------------------------------

    async def _ensure_sackmann(self):
        """Download Sackmann player CSVs for height and handedness."""
        if self._sackmann and _time.time() - self._sackmann_ts < self._SACKMANN_TTL:
            return

        log.info("[API-TENNIS] Refreshing Sackmann player CSV (height + hand)...")
        for url in [_SACKMANN_ATP, _SACKMANN_WTA]:
            try:
                session = self._get_session()
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                    if resp.status != 200:
                        log.warning("[SACKMANN] HTTP %s for %s", resp.status, url)
                        continue
                    text = await resp.text()
                reader = csv.DictReader(io.StringIO(text))
                for row in reader:
                    fname = row.get("name_first", "").strip().lower()
                    lname = row.get("name_last", "").strip().lower()
                    if not fname or not lname:
                        continue
                    key = f"{fname}-{lname}"
                    ht_raw = row.get("height", "").strip()
                    hand_raw = row.get("hand", "").strip().upper()
                    dob_raw = row.get("dob", "").strip()
                    self._sackmann[key] = {
                        "height_cm": int(ht_raw) if ht_raw.isdigit() else None,
                        "hand": hand_raw if hand_raw in ("R", "L") else None,
                        "dob": dob_raw,
                    }
            except Exception as exc:
                log.warning("[SACKMANN] Failed to load %s: %s", url, exc)

        self._sackmann_ts = _time.time()
        log.info("[API-TENNIS] Sackmann loaded: %d players", len(self._sackmann))

    def _sackmann_lookup(self, full_name: str) -> dict:
        """Return {height_cm, hand} for a player by full name, or {} if unknown."""
        slug = _name_to_slug(full_name)  # "jannik-sinner"
        if slug in self._sackmann:
            return self._sackmann[slug]
        # Fuzzy fallback
        candidates = get_close_matches(slug, self._sackmann.keys(), n=1, cutoff=0.80)
        return self._sackmann.get(candidates[0], {}) if candidates else {}

    # ------------------------------------------------------------------
    # Player profile from API
    # ------------------------------------------------------------------

    async def _fetch_api_profile(self, player_key: int) -> dict:
        """Fetch and cache a player profile from get_players."""
        if player_key in self._profile_cache:
            if _time.time() - self._profile_cache_ts.get(player_key, 0) < self._PROFILE_TTL:
                return self._profile_cache[player_key]

        data = await self._get_json({"method": "get_players", "player_key": player_key})
        result = data.get("result", [])
        if not result or not isinstance(result, list):
            return {}

        raw = result[0]
        stats = [s for s in raw.get("stats", []) if s.get("type") == "singles"]
        cur_year = str(date.today().year)

        # Current rank: most recent singles season with a valid rank
        ranking = 999
        for s in sorted(stats, key=lambda x: x.get("season", ""), reverse=True):
            r = s.get("rank", "")
            if r and str(r).isdigit() and int(r) > 0:
                ranking = int(r)
                break

        # Career totals (singles)
        career_wins   = sum(int(s.get("matches_won", 0) or 0) for s in stats)
        career_losses = sum(int(s.get("matches_lost", 0) or 0) for s in stats)
        win_rate = career_wins / (career_wins + career_losses) if (career_wins + career_losses) > 0 else 0.5

        # Season W/L
        season_wins = season_losses = 0
        for s in stats:
            if s.get("season") == cur_year:
                season_wins   = int(s.get("matches_won",  0) or 0)
                season_losses = int(s.get("matches_lost", 0) or 0)
                break

        # Recent win rate (last 2 full seasons of singles)
        recent_w = recent_l = 0
        recent_seasons = sorted(
            [s for s in stats if s.get("season", "") and s["season"].isdigit()],
            key=lambda x: int(x["season"]), reverse=True
        )[:2]
        for s in recent_seasons:
            recent_w += int(s.get("matches_won",  0) or 0)
            recent_l += int(s.get("matches_lost", 0) or 0)
        recent_win_rate = recent_w / (recent_w + recent_l) if (recent_w + recent_l) > 0 else win_rate

        # Surface win rates (all-time singles)
        def _surf_rate(won_key, lost_key):
            w = sum(int(s.get(won_key, 0) or 0) for s in stats)
            l = sum(int(s.get(lost_key, 0) or 0) for s in stats)
            return round(w / (w + l), 4) if (w + l) > 0 else 0.5

        hard_wr  = _surf_rate("hard_won",  "hard_lost")
        clay_wr  = _surf_rate("clay_won",  "clay_lost")
        grass_wr = _surf_rate("grass_won", "grass_lost")

        # Age from DOB
        dob_str = raw.get("player_bday", "")
        age = _age_from_dob(dob_str) if dob_str else 25

        profile = {
            "player_key":      player_key,
            "player_full_name": raw.get("player_full_name", raw.get("player_name", "")),
            "ranking":         ranking,
            "elo":             0,      # API doesn't provide ELO; disables pts_rank edge
            "win_rate":        round(win_rate, 4),
            "career_wins":     career_wins,
            "career_losses":   career_losses,
            "season_wins":     season_wins,
            "season_losses":   season_losses,
            "recent_win_rate": round(recent_win_rate, 4),
            "hard_win_rate":   hard_wr,
            "clay_win_rate":   clay_wr,
            "grass_win_rate":  grass_wr,
            "age":             age,
            "profile_url":     f"https://api-tennis.com/player/{player_key}",
        }

        self._profile_cache[player_key] = profile
        self._profile_cache_ts[player_key] = _time.time()
        return profile

    # ------------------------------------------------------------------
    # Public API (same interface as TennisStatsScraper)
    # ------------------------------------------------------------------

    async def fetch_rankings(self) -> dict:
        """
        Return a slug→data dict of known players (built from key cache).
        Used by main.py for the ranking-threshold gate.
        """
        if self._rankings and _time.time() - self._rankings_ts < self._RANKINGS_TTL:
            return self._rankings

        await self._ensure_key_cache()

        # Build a lightweight rankings dict from the key cache
        rankings = {}
        for slug, key in self._name_to_key.items():
            rankings[slug] = {
                "slug": slug,
                "player_key": key,
                "ranking": 999,  # filled in when profile is fetched
                "elo": 0,
                "win_rate": 0.5,
                "career_wins": 0,
                "career_losses": 0,
                "profile_url": f"https://api-tennis.com/player/{key}",
                "season_wins": None,
                "season_losses": None,
            }

        self._rankings = rankings
        self._rankings_ts = _time.time()
        return rankings

    async def fetch_player_profile(self, slug: str) -> dict:
        """Compatibility shim: resolve slug → key → profile."""
        await self._ensure_key_cache()
        key = self._name_to_key.get(slug) or self._resolve_key(slug)
        if not key:
            return {}
        return await self._fetch_api_profile(key)

    async def get_player_data(self, player_name: str) -> dict:
        """
        Main entry point. Resolves player name → API profile → merged data dict.
        Falls back to a low-quality generic profile if the player is not found.
        """
        await self._ensure_key_cache()
        await self._ensure_sackmann()

        player_key = self._resolve_key(player_name)

        if not player_key:
            log.warning("[API-TENNIS] '%s' not found in key cache — using generic profile", player_name)
            return self._generic_profile(player_name)

        profile = await self._fetch_api_profile(player_key)
        if not profile:
            log.warning("[API-TENNIS] Empty profile for key=%s ('%s')", player_key, player_name)
            return self._generic_profile(player_name)

        # Merge Sackmann height + hand
        full_name = profile.get("player_full_name", player_name)
        sack = self._sackmann_lookup(full_name) or self._sackmann_lookup(player_name)
        height_cm = sack.get("height_cm") or 185
        hand      = sack.get("hand") or "R"

        slug = _name_to_slug(player_name)
        result = {
            "slug":             slug,
            "player_key":       player_key,
            "ranking":          profile["ranking"],
            "elo":              profile["elo"],
            "win_rate":         profile["win_rate"],
            "career_wins":      profile["career_wins"],
            "career_losses":    profile["career_losses"],
            "season_wins":      profile["season_wins"],
            "season_losses":    profile["season_losses"],
            "recent_win_rate":  profile["recent_win_rate"],
            "surface_win_rate": profile["hard_win_rate"],  # overridden per-match in main.py
            "hard_win_rate":    profile["hard_win_rate"],
            "clay_win_rate":    profile["clay_win_rate"],
            "grass_win_rate":   profile["grass_win_rate"],
            "age":              profile["age"],
            "height_cm":        height_cm,
            "hand":             hand,
            "profile_url":      profile["profile_url"],
        }

        log.info(
            "[API-TENNIS] Loaded '%s' → rank=%s  win_rate=%.3f  age=%d  hand=%s  ht=%dcm",
            player_name, result["ranking"], result["win_rate"],
            result["age"], result["hand"], result["height_cm"],
        )
        return result

    def _generic_profile(self, player_name: str) -> dict:
        return {
            "slug":             _name_to_slug(player_name),
            "data_quality":     "low",
            "ranking":          100,
            "elo":              0,
            "win_rate":         0.50,
            "career_wins":      0,
            "career_losses":    0,
            "season_wins":      0,
            "season_losses":    0,
            "recent_win_rate":  0.50,
            "surface_win_rate": 0.50,
            "hard_win_rate":    0.50,
            "clay_win_rate":    0.50,
            "grass_win_rate":   0.50,
            "age":              25,
            "height_cm":        185,
            "hand":             "R",
            "profile_url":      "",
        }

    async def fetch_recent_matches(self, slug: str, n: int = 10) -> list:
        """
        Return the last N singles matches for a player as
        [[opp_rank, surface, win_loss, days_since], ...] (chronological order).
        Used by the LSTM sequential model.
        """
        await self._ensure_key_cache()
        player_key = self._name_to_key.get(slug) or self._resolve_key(slug)
        if not player_key:
            return [[0.0, 0.0, 0.0, 0.0]] * n

        today = date.today()
        start = (today - timedelta(days=180)).isoformat()
        stop  = today.isoformat()

        data = await self._get_json({
            "method":     "get_fixtures",
            "date_start": start,
            "date_stop":  stop,
        })

        now = datetime.now()
        sequence = []

        for event in data.get("result", []):
            etype = event.get("event_type_type", "")
            if "Singles" not in etype and "singles" not in etype:
                continue
            if event.get("event_status") not in ("Finished", "FT"):
                continue

            p1_key = event.get("first_player_key")
            p2_key = event.get("second_player_key")
            if player_key not in (p1_key, p2_key):
                continue

            try:
                m_date = datetime.strptime(event["event_date"], "%Y-%m-%d")
                days_since = (now - m_date).days

                # Surface from tournament name heuristic
                tour_name = event.get("tournament_name", "").lower()
                if any(w in tour_name for w in ("clay", "tierra", "roland")):
                    surf = 2.0
                elif any(w in tour_name for w in ("grass", "wimbledon", "halle")):
                    surf = 3.0
                else:
                    surf = 1.0  # default hard

                winner = event.get("event_winner", "")
                if player_key == p1_key:
                    win_loss = 1.0 if winner == "First Player" else 0.0
                    opp_key  = p2_key
                else:
                    win_loss = 1.0 if winner == "Second Player" else 0.0
                    opp_key  = p1_key

                # Opponent rank from cache (best effort)
                opp_profile = self._profile_cache.get(opp_key, {})
                opp_rank = float(opp_profile.get("ranking", 100))

                sequence.append([opp_rank, surf, win_loss, float(days_since)])
                if len(sequence) >= n:
                    break

            except Exception as exc:
                log.debug("[API-TENNIS] fetch_recent_matches row error: %s", exc)

        # Pad with zeros if fewer than n
        while len(sequence) < n:
            sequence.insert(0, [0.0, 0.0, 0.0, 0.0])

        return sequence[-n:]  # chronological (oldest → newest)

    async def get_pregame_matchup(self, player_a_name: str, player_b_name: str) -> dict:
        """
        Compatibility shim used by historical_analyzer.py.
        Returns a matchup dict with base_prob_a and player metadata.
        """
        pa, pb = await asyncio.gather(
            self.get_player_data(player_a_name),
            self.get_player_data(player_b_name),
        )

        elo_a = pa.get("ranking", 100)
        elo_b = pb.get("ranking", 100)

        # Simple Elo-style win probability from ranking (lower rank = stronger)
        # Using rank inverse as a proxy since we don't have true Elo
        pts_a = 1.0 / max(elo_a, 1)
        pts_b = 1.0 / max(elo_b, 1)
        base_prob_a = pts_a / (pts_a + pts_b)

        return {
            "player_a":    pa,
            "player_b":    pb,
            "base_prob_a": round(base_prob_a, 4),
            "meta": {
                "player_a": pa,
                "player_b": pb,
            },
        }

    def _resolve_slug(self, name: str, rankings: dict = None):  # noqa: ARG002
        """Compatibility shim used by historical_analyzer.py."""
        return _name_to_slug(name)


# Module alias so existing imports (from tennis_scraper import TennisStatsScraper) still work
TennisStatsScraper = ApiTennisScraper
