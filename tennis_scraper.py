"""
TennisStats.com Scraper
=======================
Pulls live ATP rankings and player profiles from https://tennisstats.com.

URLs used (no auth, fully public):
  Rankings : https://tennisstats.com/rankings/atp
  Profile  : https://tennisstats.com/players/{player-slug}
  H2H      : https://tennisstats.com/h2h/{player-a}-vs-{player-b}

Why we switched from atptour.com:
  atptour.com now blocks aiohttp with an SSL cert verification error and
  hides win-rate data behind JavaScript rendering, making it effectively
  unusable for server-side scraping without a headless browser.
"""

import aiohttp
import asyncio
import ssl
import re
import logging
from bs4 import BeautifulSoup
from difflib import get_close_matches
from functools import lru_cache

log = logging.getLogger(__name__)

# Reusable SSL context that skips certificate verification.
# tennisstats.com has a valid cert, but this guards against any local CA issues.
_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

# Blacklist of terms that should NEVER be resolved as tennis players
# (e.g. countries, teams, generic sport terms)
_NAME_BLACKLIST = {
    "argentina", "jordan", "brazil", "france", "germany", "england", "spain", 
    "italy", "usa", "mexico", "canada", "australia", "china", "japan",
    "team", "club", "all-stars", "field", "other", "any", "combined",
}


def _name_to_slug(name: str) -> str:
    """Convert 'Carlos Alcaraz' → 'carlos-alcaraz'."""
    return re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")


class TennisStatsScraper:
    """
    Scraper backed by tennisstats.com.

    Data returned per player:
      slug         – URL-safe identifier (e.g. 'carlos-alcaraz')
      ranking      – int ATP singles rank
      elo          – int ELO points (tennisstats proprietary score)
      win_rate     – float career wins / (wins + losses)
      career_wins  – int
      career_losses– int
      season_wins  – int  (current year)
      season_losses– int
      aces_per_match – float (current year, 3-set matches)
      double_faults  – float (current year, 3-set matches)
      break_pts_won  – float (current year, 3-set matches)
      prize_money  – str  (formatted, e.g. '$53,902,993')
      profile_url  – str  full URL
    """

    BASE = "https://tennisstats.com"

    def __init__(self):
        self._rankings: dict | None = None   # slug → data dict, populated lazily
        self._profile_cache: dict = {}       # slug → profile dict

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _get(self, url: str) -> str | None:
        """Fetch a URL and return its HTML, or None on failure."""
        try:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            async with aiohttp.ClientSession(
                headers=_HEADERS, connector=connector
            ) as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as r:
                    if r.status == 200:
                        return await r.text()
                    log.warning("HTTP %s for %s", r.status, url)
        except Exception as exc:
            log.error("Fetch failed for %s: %s", url, exc)
        return None

    # ------------------------------------------------------------------
    # Rankings
    # ------------------------------------------------------------------

    async def fetch_rankings(self) -> dict:
        """
        Scrape the full ATP singles rankings from tennisstats.com.

        Returns a dict keyed by *player slug* (e.g. 'carlos-alcaraz') where
        each value is a basic data dict with ranking, elo, career W-L etc.
        Cached in memory for the lifetime of this scraper instance.
        """
        if self._rankings is not None:
            return self._rankings

        rankings: dict = {}
        
        for tour in ["atp", "wta"]:
            url = f"{self.BASE}/rankings/{tour}"
            html = await self._get(url)
            if not html:
                log.error(f"Could not fetch {tour.upper()} rankings from tennisstats.com")
                continue

            soup = BeautifulSoup(html, "html.parser")

            # Each player row is an <a> tag whose href starts with /players/
            for a in soup.find_all("a", href=re.compile(r"^/players/[^/]+$")):
                href = a["href"]
                slug = href.split("/players/")[-1].strip("/")
                if not slug:
                    continue

                raw = a.get_text(" ", strip=True)

                # Pattern in the raw text for a rankings row:
                # "1Carlos Alcaraz8913,590 22 19 4.33 1.83 $53,902,993"
                # Fields: rank, name, age, elo, ?, wins, losses, aces, df, prize
                # We use regex to extract rank + career record robustly.
                rank_match = re.match(r"^(\d+)", raw)
                rank = int(rank_match.group(1)) if rank_match else 999

                # ELO is the first standalone number >= 1000 after the player name
                elo_match = re.search(r"\b(\d{1,2},\d{3})\b", raw)
                elo = int(elo_match.group(1).replace(",", "")) if elo_match else 0

                # Career W-L: two consecutive numbers separated by whitespace
                # (wins then losses) — appears after the elo score
                wl_match = re.search(r"\b(\d{1,3})\s+(\d{1,3})\b", raw)
                wins   = int(wl_match.group(1)) if wl_match else 0
                losses = int(wl_match.group(2)) if wl_match else 0
                win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0.5

                prize_match = re.search(r"\$[\d,]+", raw)
                prize = prize_match.group(0) if prize_match else "N/A"

                rankings[slug] = {
                    "slug":         slug,
                    "ranking":      rank,
                    "elo":          elo,
                    "win_rate":     round(win_rate, 4),
                    "career_wins":  wins,
                    "career_losses": losses,
                    "prize_money":  prize,
                    "profile_url":  f"{self.BASE}{href}",
                    # Will be enriched when we load the full profile
                    "season_wins":   None,
                    "season_losses": None,
                    "aces_per_match": None,
                    "double_faults":  None,
                    "break_pts_won":  None,
                }

        log.info("Loaded %d players from tennisstats.com ATP/WTA rankings", len(rankings))
        self._rankings = rankings
        return rankings

    # ------------------------------------------------------------------
    # Player profile
    # ------------------------------------------------------------------

    async def fetch_player_profile(self, slug: str) -> dict:
        """
        Fetch the detailed profile page for one player and return a merged
        data dict (ranking data + profile enrichments).
        """
        if slug in self._profile_cache:
            return self._profile_cache[slug]

        url = f"{self.BASE}/players/{slug}"
        html = await self._get(url)
        if not html:
            return {}

        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text(" ", strip=True)

        result: dict = {"slug": slug, "profile_url": url}

        # ATP rank
        rank_m = re.search(r"ATP\s+Rank\s*(\d+)", text, re.I)
        if rank_m:
            result["ranking"] = int(rank_m.group(1))

        # ELO points
        elo_m = re.search(r"Points\s+([\d,]+)", text, re.I)
        if elo_m:
            result["elo"] = int(elo_m.group(1).replace(",", ""))

        # Career W-L  e.g. "367 - 92"
        career_m = re.search(r"(\d+)\s*-\s*(\d+)\s*\$", text)
        if career_m:
            cw, cl = int(career_m.group(1)), int(career_m.group(2))
            result["career_wins"]   = cw
            result["career_losses"] = cl
            result["win_rate"]      = round(cw / (cw + cl), 4) if (cw + cl) > 0 else 0.5

        # Season W-L from the overview sentence
        # "In 2026, Carlos Alcaraz is … 17 wins to 2 losses"
        season_m = re.search(r"(\d+)\s+wins?\s+to\s+(\d+)\s+loss", text, re.I)
        if season_m:
            result["season_wins"]   = int(season_m.group(1))
            result["season_losses"] = int(season_m.group(2))

        # Aces per match (2026, 3-set)
        aces_m = re.search(r"average\s+of\s+([\d.]+)\s+aces\s+per\s+match", text, re.I)
        if aces_m:
            result["aces_per_match"] = float(aces_m.group(1))

        # Double faults per match
        df_m = re.search(
            r"averaged\s+([\d.]+)\s+double\s+faults\s+per\s+match\s+in\s+best\s+of\s+3",
            text, re.I
        )
        if df_m:
            result["double_faults"] = float(df_m.group(1))

        # Average break points won per match
        bp_m = re.search(
            r"average\s+of\s+([\d.]+)\s+break\s+points\s+won\s+per\s+match",
            text, re.I
        )
        if bp_m:
            result["break_pts_won"] = float(bp_m.group(1))

        # First-serve % from overview sentence
        fs_m = re.search(r"([\d.]+)%\s+on\s+first\s+serves", text, re.I)
        if fs_m:
            result["first_serve_pct"] = float(fs_m.group(1))

        # Break-point conversion %
        bpc_m = re.search(r"converts\s+([\d.]+)%\s+of\s+their\s+break\s+points", text, re.I)
        if bpc_m:
            result["bp_conversion_pct"] = float(bpc_m.group(1))

        # Prize money
        pm_m = re.search(r"\$([\d,]+)", text)
        if pm_m:
            result["prize_money"] = "$" + pm_m.group(1)

        # Height
        ht_m = re.search(r"([\d\.]+)m", text)
        if ht_m:
            try:
                # Convert 1.83m -> 183cm
                result["height_cm"] = int(float(ht_m.group(1)) * 100)
            except ValueError:
                pass
                
        # Age
        # Often appears right before the W-L record or Career Winnings, 
        # or we just look for Age \d+
        age_m = re.search(r"Hand\s+(\d+)", text)
        if age_m:
            result["age"] = int(age_m.group(1))
            
        # Handedness
        if "Right-handed" in text or "Right-Handed" in text:
            result["hand"] = "R"
        elif "Left-handed" in text or "Left-Handed" in text:
            result["hand"] = "L"

        self._profile_cache[slug] = result
        return result

    async def fetch_recent_matches(self, slug: str, n: int = 10) -> list:
        """
        Scrape the 'Latest Results' for a player to feed the LSTM sequential model.
        Returns a list of match features: [[opp_rank, surface, win_loss, days_since], ...]
        """
        url = f"{self.BASE}/players/{slug}"
        html = await self._get(url)
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        match_rows = soup.select(".h2h-history-row")[:n]
        
        # Ensure we have the latest rankings to resolve opponent ranks
        rankings = await self.fetch_rankings()
        
        from datetime import datetime
        now = datetime.now()
        
        sequence = []
        surface_map = {"hard": 1, "clay": 2, "grass": 3}
        
        for row in match_rows:
            try:
                # 1. Date (index 1)
                date_str = row.select_one("div:nth-child(1) p").get_text(strip=True)
                # Normalise scraped dates like 'Jan 292026' → 'Jan 29 2026'
                date_str = re.sub(r'([A-Za-z])(\d{1,2})(\d{4})', r'\1 \2 \3', date_str)
                m_date = datetime.strptime(date_str.strip(), "%b %d %Y")
                days_since = (now - m_date).days
                
                # 2. Surface (index 2)
                surf_text = row.select_one("div:nth-child(2) p span").get_text(strip=True).lower()
                surf_val = surface_map.get(surf_text, 0)
                
                # 3. Result (index 3)
                res_text = row.select_one("div:nth-child(3) div").get_text(strip=True).upper()
                win_loss = 1.0 if res_text == "W" else 0.0
                
                # 4. Opponent Rank (index 6 - name lookup)
                opp_name = row.select_one("div:nth-child(6) p").get_text(strip=True)
                opp_slug = _name_to_slug(opp_name)
                opp_rank = rankings.get(opp_slug, {}).get("ranking", 100) # Default to 100 if unknown
                
                sequence.append([float(opp_rank), float(surf_val), win_loss, float(days_since)])
            except Exception as e:
                log.warning(f"Failed to parse history row: {e}")
                continue
                
        # Pad with zeros if less than n
        while len(sequence) < n:
            sequence.append([0.0, 0.0, 0.0, 0.0])
            
        return list(reversed(sequence)) # Return chronological order (oldest to newest)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_player_data(self, player_name: str) -> dict:
        """
        Resolve a player by name (fuzzy-matched against slugs) and return
        a fully enriched data dict.

        Falls back to a generic profile if the player is not found.
        """
        rankings = await self.fetch_rankings()
        slug = self._resolve_slug(player_name, rankings)

        if slug:
            base = rankings.get(slug, {}).copy()
            # Enrich with detailed profile page
            profile = await self.fetch_player_profile(slug)
            base.update({k: v for k, v in profile.items() if v is not None})
            log.info(
                "Loaded %s from tennisstats.com  rank=%s  win_rate=%.3f",
                player_name, base.get("ranking", "?"), base.get("win_rate", 0.5),
            )
            return base

        log.warning("Player '%s' not found on tennisstats.com – using generic profile", player_name)
        return {
            "slug":          _name_to_slug(player_name),
            "ranking":       100,
            "elo":           500,
            "win_rate":      0.50,
            "career_wins":   0,
            "career_losses": 0,
            "prize_money":   "N/A",
            "profile_url":   "",
        }

    def _resolve_slug(self, name: str, rankings: dict) -> str | None:
        """
        Find the best-matching slug for a given name string.

        Priority:
          1. Exact slug match      ('carlos-alcaraz' → 'carlos-alcaraz')
          2. Slug built from name  ('Carlos Alcaraz' → 'carlos-alcaraz')
          3. Fuzzy match on slugs  (handles minor typos / middle names)
        """
        slug_from_name = _name_to_slug(name)

        if slug_from_name in rankings:
            return slug_from_name

        # If Kalshi provides ONLY a last name (e.g. "Cerundolo"), there is NO hyphen.
        # Find the highest ranked player (first in dict) whose slug ends with this last name.
        if "-" not in slug_from_name and len(slug_from_name) >= 3:
            for k in rankings:
                if k.endswith(f"-{slug_from_name}"):
                    return k

        # Fuzzy fallback using the ENTIRE name string against known rankings
        candidates = get_close_matches(slug_from_name, rankings.keys(), n=1, cutoff=0.8)
        
        if candidates:
            resolved = candidates[0]
            if any(p in _NAME_BLACKLIST for p in resolved.split("-")):
                log.warning("Fuzzy match '%s' for '%s' is on the blacklist and will be rejected.", resolved, name)
                return slug_from_name
            return resolved

        # If not found in the initial rankings cache, DO NOT fallback to random players 
        # sharing a last name. We must return the exact full name slug so the scraper 
        # attempts to hit their exact profile page (which may exist if they rank >100).
        return slug_from_name

    async def get_pregame_matchup(self, player_a_name: str, player_b_name: str) -> dict:
        """
        Main entry-point used by server.py.

        Fetches both players in parallel and returns a matchup dict with
        a computed base win probability.
        """
        pa, pb = await asyncio.gather(
            self.get_player_data(player_a_name),
            self.get_player_data(player_b_name),
        )

        rank_diff = pb.get("ranking", 50) - pa.get("ranking", 50)
        elo_diff  = pa.get("elo", 0) - pb.get("elo", 0)

        # Blended probability:
        #   40 % from ranking gap  (each rank worth ~1 %)
        #   40 % from ELO gap      (each 100 ELO points ≈ 4 %)
        #   20 % from win-rate gap
        base_prob_a = 0.5
        base_prob_a += (rank_diff * 0.01)
        base_prob_a += (elo_diff / 100) * 0.04
        base_prob_a += (pa.get("win_rate", 0.5) - pb.get("win_rate", 0.5)) * 0.20

        base_prob_a = max(0.05, min(0.95, base_prob_a))

        return {
            "player_a":    pa,
            "player_b":    pb,
            "base_prob_a": round(base_prob_a, 4),
            "base_prob_b": round(1.0 - base_prob_a, 4),
        }


# ---------------------------------------------------------------------------
# Backwards-compatibility alias
# ATCScraper is the name imported by server.py and test files.
# ---------------------------------------------------------------------------
ATCScraper = TennisStatsScraper
