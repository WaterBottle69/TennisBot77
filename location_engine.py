"""
location_engine.py — Tournament venue scraper + environmental data pipeline.

Currently active in model:
  - Sun azimuth/elevation → serve glare penalty on p_serve

Collected but NOT yet used in model (reserved for future physics engine):
  - Altitude above sea level
  - Humidity, temperature, wind speed/direction, atmospheric pressure
  - Court surface, court orientation override
"""

import asyncio
import logging
import math
import time
import datetime
import re
import ssl
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict

log = logging.getLogger(__name__)

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode = ssl.CERT_NONE

# ── Cache TTLs ─────────────────────────────────────────────────────────────────
_VENUE_CACHE_TTL   = 86400.0   # 24 h — venue coords don't change during a tournament
_WEATHER_CACHE_TTL = 900.0     # 15 min — weather is slow-moving

# ── Sun penalty tuning ─────────────────────────────────────────────────────────
# Max reduction applied to p_serve when server directly faces the sun.
_MAX_SUN_PENALTY     = 0.06    # 6% reduction in serve-win probability
_GLARE_ANGLE_DEG     = 50.0    # Angular window (server gaze vs sun azimuth) within which glare is possible
_GLARE_ELEV_LOW_DEG  = 10.0    # Sun below this elevation → no glare (below horizon / obscured)
_GLARE_ELEV_PEAK_DEG = 35.0    # Sun at this elevation → maximum glare (worst toss angle)
_GLARE_ELEV_HIGH_DEG = 75.0    # Sun above this elevation → minimal glare (directly overhead)

# ── Known tournament → venue lookup ───────────────────────────────────────────
# Tuples: (city, country, lat, lon, altitude_m, court_surface, court_orientation_deg)
# court_orientation_deg: direction from baseline-A to baseline-B (0 = N–S, 90 = E–W).
# Standard outdoor courts are 0° (N–S) to avoid morning/afternoon sun in players' eyes.
# Indoor courts → use 0.0 by default; glare physically irrelevant but penalty will be 0 (sun elev negative indoors).
_KNOWN_VENUES: Dict[str, tuple] = {
    # Grand Slams
    "wimbledon":          ("London",        "UK",          51.4344,   -0.2134,    13,  "grass",   0.0),
    "roland garros":      ("Paris",         "France",      48.8460,    2.2494,    43,  "clay",    0.0),
    "us open":            ("New York",      "USA",         40.6971,  -73.8517,     4,  "hard",    0.0),
    "australian open":    ("Melbourne",     "Australia",  -37.8202,  144.9792,    26,  "hard",    0.0),
    # Masters 1000 / WTA 1000
    "madrid open":        ("Madrid",        "Spain",       40.4168,   -3.7038,   667,  "clay",    0.0),
    "rome":               ("Rome",          "Italy",       41.9028,   12.4964,    20,  "clay",    0.0),
    "monte carlo":        ("Monte Carlo",   "Monaco",      43.7384,    7.4246,    55,  "clay",    0.0),
    "miami open":         ("Miami",         "USA",         25.6853,  -80.2414,     2,  "hard",    0.0),
    "indian wells":       ("Indian Wells",  "USA",         33.7175, -116.3427,   460,  "hard",    0.0),
    "paris masters":      ("Paris",         "France",      48.8566,    2.3522,    43,  "hard",  -15.0),  # Bercy indoor-ish
    "shanghai":           ("Shanghai",      "China",       31.2304,  121.4737,     4,  "hard",    0.0),
    "toronto":            ("Toronto",       "Canada",      43.6532,  -79.3832,    76,  "hard",    0.0),
    "montreal":           ("Montreal",      "Canada",      45.5017,  -73.5673,    27,  "hard",    0.0),
    "cincinnati":         ("Cincinnati",    "USA",         39.1031,  -84.5120,   261,  "hard",    0.0),
    # ATP 500 / WTA 500
    "barcelona":          ("Barcelona",     "Spain",       41.3851,    2.1734,    12,  "clay",    0.0),
    "hamburg":            ("Hamburg",       "Germany",     53.5753,    9.9952,    14,  "clay",    0.0),
    "düsseldorf":         ("Düsseldorf",    "Germany",     51.2217,    6.7762,    38,  "clay",    0.0),
    "geneva":             ("Geneva",        "Switzerland", 46.2044,    6.1432,   375,  "clay",    0.0),
    "lyon":               ("Lyon",          "France",      45.7640,    4.8357,   173,  "clay",    0.0),
    "houston":            ("Houston",       "USA",         29.7604,  -95.3698,    15,  "clay",    0.0),
    "eastbourne":         ("Eastbourne",    "UK",          50.7684,    0.2904,     8,  "grass",   0.0),
    "halle":              ("Halle",         "Germany",     51.4820,    8.0431,   116,  "grass",   0.0),
    "queens":             ("London",        "UK",          51.4907,   -0.2065,    10,  "grass",   0.0),
    "queen's club":       ("London",        "UK",          51.4907,   -0.2065,    10,  "grass",   0.0),
    "metz":               ("Metz",          "France",      49.1193,    6.1757,   180,  "hard",    0.0),
    "vienna":             ("Vienna",        "Austria",     48.2082,   16.3738,   151,  "hard",    0.0),
    "stockholm":          ("Stockholm",     "Sweden",      59.3293,   18.0686,    28,  "hard",    0.0),
    "antwerp":            ("Antwerp",       "Belgium",     51.2194,    4.4025,    12,  "hard",    0.0),
    "tokyo":              ("Tokyo",         "Japan",       35.6762,  139.6503,     6,  "hard",    0.0),
    "beijing":            ("Beijing",       "China",       39.9042,  116.4074,    44,  "hard",    0.0),
    # Challengers (common)
    "münchen":            ("München",       "Germany",     48.1351,   11.5820,   519,  "clay",    0.0),
    "munich":             ("München",       "Germany",     48.1351,   11.5820,   519,  "clay",    0.0),
    "bordeaux":           ("Bordeaux",      "France",      44.8378,   -0.5792,     6,  "clay",    0.0),
    "tunis":              ("Tunis",         "Tunisia",     36.8190,   10.1658,    66,  "clay",    0.0),
    "tallahassee":        ("Tallahassee",   "USA",         30.4383,  -84.2807,    55,  "clay",    0.0),
    "nottingham":         ("Nottingham",    "UK",          52.9548,   -1.1581,    47,  "grass",   0.0),
    "surbiton":           ("London",        "UK",          51.3940,   -0.3044,    15,  "grass",   0.0),
    "rosmalen":           ("Rosmalen",      "Netherlands", 51.7158,    5.3537,    13,  "grass",   0.0),
}


# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class VenueInfo:
    """Venue location and court metadata for a tournament."""
    tournament:            str
    city:                  str
    country:               str
    latitude:              float
    longitude:             float

    # ── Not yet used in model — reserved for future physics engine ─────────────
    altitude_m:            float  = 0.0       # metres above sea level
    court_surface:         str    = "unknown"  # hard / clay / grass / carpet
    court_orientation_deg: float  = 0.0        # 0 = N–S baseline axis; 90 = E–W


@dataclass
class WeatherData:
    """
    Real-time weather at the venue.

    All fields are fetched and stored; NONE are currently wired into the model.
    Reserved for Phase-2 physics engine (ball drag, bounce coefficient, etc.).
    """
    temperature_c:    float = 20.0
    humidity_pct:     float = 50.0
    wind_speed_kmh:   float = 0.0
    wind_direction_deg: float = 0.0   # degrees clockwise from N
    pressure_hpa:     float = 1013.0
    conditions:       str   = "unknown"
    fetched_at:       float = field(default_factory=time.time)

    # ── Future modifiers (not computed yet) ────────────────────────────────────
    _ball_drag_factor:     float = 1.0   # altitude-adjusted air resistance
    _bounce_coefficient:   float = 1.0   # humidity effect on ball bounce
    _wind_serve_modifier:  float = 0.0   # head/tail wind on serve toss accuracy


@dataclass
class SunData:
    """
    Sun position at the venue and derived serve-impact metrics.

    p1_sun_penalty / p2_sun_penalty: reduction applied to that player's
    serve-win-point probability (p_serve) when they are the active server.
    These are per-tick values — recomputed each poll using p1_serving flag.
    """
    azimuth_deg:    float              # 0 = N, 90 = E, 180 = S, 270 = W
    elevation_deg:  float              # degrees above horizon (negative = below)
    timestamp:      float              # unix time when computed

    glare_active:          bool  = False
    p1_sun_penalty:        float = 0.0  # penalty when P1 is serving
    p2_sun_penalty:        float = 0.0  # penalty when P2 is serving
    p1_facing_azimuth_deg: float = 0.0  # direction P1 faces when serving
    p2_facing_azimuth_deg: float = 0.0  # direction P2 faces when serving
    description:           str   = ""


# ── Pure-Python NOAA solar position algorithm ──────────────────────────────────

def _julian_day(dt: datetime.datetime) -> float:
    """Convert UTC datetime to Julian Day Number."""
    a = (14 - dt.month) // 12
    y = dt.year + 4800 - a
    m = dt.month + 12 * a - 3
    jdn = (dt.day + (153 * m + 2) // 5 + 365 * y
           + y // 4 - y // 100 + y // 400 - 32045)
    return jdn + (dt.hour - 12) / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0


def compute_solar_position(lat_deg: float, lon_deg: float,
                           dt: datetime.datetime) -> Tuple[float, float]:
    """
    Compute sun azimuth and elevation using the NOAA simplified algorithm.
    Accurate to ~0.01° for dates within ±50 years of J2000.

    Args:
        lat_deg: latitude  (+N / -S)
        lon_deg: longitude (+E / -W)
        dt:      UTC datetime

    Returns:
        (azimuth_deg, elevation_deg)
        azimuth: 0 = N, 90 = E, 180 = S, 270 = W
        elevation: degrees above horizon (negative = sun below horizon)
    """
    jd = _julian_day(dt)
    t = (jd - 2451545.0) / 36525.0          # Julian centuries from J2000.0

    # Geometric mean longitude (deg)
    L0 = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360
    # Mean anomaly (deg)
    M = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    M_r = math.radians(M)

    # Equation of center
    C = ((1.914602 - t * (0.004817 + 0.000014 * t)) * math.sin(M_r)
         + (0.019993 - 0.000101 * t) * math.sin(2 * M_r)
         + 0.000289 * math.sin(3 * M_r))

    # Sun's true longitude → apparent longitude
    sun_lon = L0 + C
    omega   = 125.04 - 1934.136 * t
    app_lon = sun_lon - 0.00569 - 0.00478 * math.sin(math.radians(omega))

    # Mean obliquity → corrected obliquity
    mean_obl = (23.0
                + (26.0 + (21.448 - t * (46.8150 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0)
    obl_corr = mean_obl + 0.00256 * math.cos(math.radians(omega))

    # Solar declination
    decl = math.degrees(math.asin(
        math.sin(math.radians(obl_corr)) * math.sin(math.radians(app_lon))
    ))

    # Equation of time (minutes) — correct NOAA form with orbital eccentricity
    e     = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    y_val = math.tan(math.radians(obl_corr / 2)) ** 2
    L0_r  = math.radians(L0)
    eot_min = 4.0 * math.degrees(
        y_val * math.sin(2 * L0_r)
        - 2.0 * e * math.sin(M_r)
        + 4.0 * e * y_val * math.sin(M_r) * math.cos(2 * L0_r)
        - 0.5 * y_val ** 2 * math.sin(4 * L0_r)
        - 1.25 * e ** 2 * math.sin(2 * M_r)
    )

    # True solar time → hour angle
    utc_min = dt.hour * 60.0 + dt.minute + dt.second / 60.0
    ha      = (utc_min - (720.0 - 4.0 * lon_deg - eot_min)) * 0.25

    lat_r  = math.radians(lat_deg)
    decl_r = math.radians(decl)
    ha_r   = math.radians(ha)

    # Solar zenith angle
    cos_z = (math.sin(lat_r) * math.sin(decl_r)
             + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r))
    cos_z    = max(-1.0, min(1.0, cos_z))
    zenith   = math.degrees(math.acos(cos_z))
    elevation = 90.0 - zenith

    # Azimuth (0 = N, 90 = E, 180 = S, 270 = W)
    sin_z = math.sin(math.radians(zenith))
    if sin_z < 1e-10:
        return 0.0, elevation

    cos_az = ((math.sin(lat_r) * cos_z - math.sin(decl_r))
              / (math.cos(lat_r) * sin_z))
    cos_az  = max(-1.0, min(1.0, cos_az))
    az_raw  = math.degrees(math.acos(cos_az))
    azimuth = (360.0 - az_raw) if ha > 0 else az_raw   # afternoon → west side

    return azimuth, elevation


# ── Court-end tracking ─────────────────────────────────────────────────────────

def server_facing_azimuth(
    p1_serving: bool,
    games: Tuple[int, int],
    sets: Tuple[int, int],
    court_orientation_deg: float = 0.0,
    p1_starts_baseline_a: bool = True,
) -> Tuple[float, float]:
    """
    Determine the compass direction each player faces when serving.

    Players switch ends after every odd-total-game count within each set,
    and at the start of a new set if the final-set game count was odd.

    Args:
        p1_serving:           True if player 1 is the current server
        games:                (games_a, games_b) in the CURRENT set
        sets:                 (sets_a, sets_b) won so far (for multi-set correction)
        court_orientation_deg: baseline-A → baseline-B direction (0 = N–S)
        p1_starts_baseline_a:  P1 starts at baseline-A for the first game of match

    Returns:
        (p1_facing_azimuth_deg, p2_facing_azimuth_deg)
        where azimuth is the direction the player faces when serving
        (toward the opposite end = toward their opponent).
    """
    # Baseline-A faces (court_orientation_deg + 180) % 360
    # Baseline-B faces court_orientation_deg
    dir_from_A = (court_orientation_deg + 180.0) % 360.0
    dir_from_B =  court_orientation_deg

    # Total games completed in the current set
    total_games_set = games[0] + games[1]

    # Switches happen after games 1, 3, 5, 7 ... (odd totals)
    switches_in_set = (total_games_set + 1) // 2   # floor((total+1)/2)

    # Carry over parity from completed sets:
    # If the previous set ended on an odd total, one extra switch already happened.
    sets_parity = (sets[0] + sets[1]) % 2          # odd sets completed = extra switch

    total_switches = switches_in_set + sets_parity

    # Who is at baseline A?
    p1_at_A = p1_starts_baseline_a if (total_switches % 2 == 0) else not p1_starts_baseline_a

    p1_facing = dir_from_A if p1_at_A else dir_from_B
    p2_facing = dir_from_B if p1_at_A else dir_from_A

    return p1_facing, p2_facing


def compute_sun_penalty(server_facing_deg: float,
                        sun_azimuth_deg: float,
                        sun_elevation_deg: float) -> float:
    """
    Return the p_serve reduction [0, _MAX_SUN_PENALTY] for a player serving
    toward the given compass direction when the sun is at (azimuth, elevation).

    Max penalty (_MAX_SUN_PENALTY) occurs when:
      - angular offset between gaze and sun ≤ 0°
      - sun elevation ≈ _GLARE_ELEV_PEAK_DEG (worst toss angle)
    Zero penalty when:
      - sun is below _GLARE_ELEV_LOW_DEG  (not glaring)
      - sun is above _GLARE_ELEV_HIGH_DEG  (straight up, not in face)
      - angular offset > _GLARE_ANGLE_DEG  (sun not in server's visual field)
    """
    elev = sun_elevation_deg
    if elev < _GLARE_ELEV_LOW_DEG or elev > _GLARE_ELEV_HIGH_DEG:
        return 0.0

    # Angular difference between server's gaze and sun azimuth
    diff = abs(server_facing_deg - sun_azimuth_deg) % 360.0
    if diff > 180.0:
        diff = 360.0 - diff

    if diff > _GLARE_ANGLE_DEG:
        return 0.0

    # Directional factor: 1.0 at 0° offset → 0.0 at _GLARE_ANGLE_DEG
    dir_factor = (_GLARE_ANGLE_DEG - diff) / _GLARE_ANGLE_DEG

    # Elevation factor: peaks at _GLARE_ELEV_PEAK_DEG, 0 at the limits
    if elev <= _GLARE_ELEV_PEAK_DEG:
        elev_factor = (elev - _GLARE_ELEV_LOW_DEG) / (_GLARE_ELEV_PEAK_DEG - _GLARE_ELEV_LOW_DEG)
    else:
        elev_factor = (_GLARE_ELEV_HIGH_DEG - elev) / (_GLARE_ELEV_HIGH_DEG - _GLARE_ELEV_PEAK_DEG)
    elev_factor = max(0.0, elev_factor)

    return _MAX_SUN_PENALTY * dir_factor * elev_factor


# ── LocationEngine ─────────────────────────────────────────────────────────────

class LocationEngine:
    """
    Async engine that scrapes venue location data and computes environmental
    modifiers for the betting model.

    Usage per match:
        engine = LocationEngine()
        venue  = await engine.get_venue_info("ATP Challenger München")
        # Each tick:
        sun = engine.get_sun_data(venue, score_update)
        penalty = sun.p1_sun_penalty if p1_serving else sun.p2_sun_penalty
        p_serve_adjusted = max(0.01, p_serve - penalty)
    """

    def __init__(self):
        self._venue_cache: Dict[str, Tuple[float, VenueInfo]]   = {}   # key → (ts, info)
        self._weather_cache: Dict[str, Tuple[float, WeatherData]] = {}  # key → (ts, data)
        self._session = None

    def _get_session(self):
        import aiohttp
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=_SSL_CTX)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    # ── Venue info ─────────────────────────────────────────────────────────────

    async def get_venue_info(self, tournament_name: str) -> Optional[VenueInfo]:
        """
        Return venue coordinates for a tournament.  Priority order:
          1. Internal _KNOWN_VENUES dict  (instant)
          2. Nominatim geocoding via OSM   (async HTTP, free, no key)
        Results are cached for _VENUE_CACHE_TTL seconds.
        """
        key = tournament_name.lower().strip()
        now = time.time()

        # Cache hit
        if key in self._venue_cache:
            ts, info = self._venue_cache[key]
            if now - ts < _VENUE_CACHE_TTL:
                return info

        info = self._lookup_known_venue(key) or await self._geocode_venue(tournament_name)
        if info:
            self._venue_cache[key] = (now, info)
            log.info("[LOCATION] Venue resolved: %s → %s, %s (%.4f, %.4f) alt=%dm",
                     tournament_name, info.city, info.country,
                     info.latitude, info.longitude, info.altitude_m)
        return info

    def _lookup_known_venue(self, key: str) -> Optional[VenueInfo]:
        """Match tournament name against _KNOWN_VENUES using keyword scanning."""
        for kw, (city, country, lat, lon, alt, surface, orient) in _KNOWN_VENUES.items():
            if kw in key:
                return VenueInfo(
                    tournament=key, city=city, country=country,
                    latitude=lat, longitude=lon, altitude_m=float(alt),
                    court_surface=surface, court_orientation_deg=float(orient),
                )
        return None

    async def _geocode_venue(self, tournament_name: str) -> Optional[VenueInfo]:
        """
        Extract a city from the tournament name and geocode via Nominatim (OSM).
        No API key required. Rate-limited to 1 req/s by OSM policy.
        """
        city = self._parse_city_from_name(tournament_name)
        if not city:
            log.debug("[LOCATION] Could not parse city from: %s", tournament_name)
            return None

        try:
            import aiohttp
            url = (
                "https://nominatim.openstreetmap.org/search"
                f"?q={city}&format=json&limit=1&addressdetails=1"
            )
            headers = {"User-Agent": "TennisBot/1.0 (github.com/tennis-bot)"}
            session = self._get_session()
            async with session.get(url, headers=headers,
                                   timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status != 200:
                    log.warning("[LOCATION] Nominatim HTTP %d for %s", r.status, city)
                    return None
                results = await r.json(content_type=None)

            if not results:
                log.debug("[LOCATION] No geocode result for city: %s", city)
                return None

            place = results[0]
            lat   = float(place["lat"])
            lon   = float(place["lon"])
            addr  = place.get("address", {})
            country = addr.get("country", "Unknown")

            # Fetch altitude from Open-Elevation (free, no key)
            alt = await self._fetch_altitude(lat, lon)

            return VenueInfo(
                tournament=tournament_name, city=city, country=country,
                latitude=lat, longitude=lon, altitude_m=alt,
            )

        except Exception as exc:
            log.warning("[LOCATION] Geocode failed for '%s': %s", tournament_name, exc)
            return None

    # Generic names that cannot be geocoded and should be skipped
    _UNRESOLVABLE = frozenset({
        "unknown tournament", "atp/wta", "atp", "wta", "itf",
        "tennis", "tour", "challenger", "grand slam", "unknown",
        "tournament", "match", "final", "semifinal",
    })

    @staticmethod
    def _parse_city_from_name(name: str) -> str:
        """
        Extract a city keyword from a tournament name like:
          "ATP Challenger München", "WTA 125 Lyon", "Madrid Open",
          "Internazionali BNL d'Italia Rome", "BNPPH Indoors Antwerp"

        Returns "" for generic/unresolvable names so geocoding is skipped.
        """
        low = name.lower().strip()
        if low in LocationEngine._UNRESOLVABLE or not low:
            return ""

        # Strip common prefix words (case-insensitive)
        prefixes = r"\b(ATP|WTA|ITF|Challenger|Masters|Open|International|Internazionali|" \
                   r"Grand Slam|Indoor|Indoors|Outdoor|Series|Tour|BNL|BNP|1000|500|250|125|" \
                   r"d'|di|de|du)\b"
        cleaned = re.sub(prefixes, " ", name, flags=re.IGNORECASE)
        cleaned = re.sub(r"[''']s?", "", cleaned)
        cleaned = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", cleaned)
        tokens  = [t.strip() for t in cleaned.split() if len(t.strip()) > 2]
        candidate = tokens[-1] if tokens else ""

        # Reject if the extracted token is itself a generic word
        if candidate.lower() in LocationEngine._UNRESOLVABLE:
            return ""
        return candidate

    # ── Altitude ───────────────────────────────────────────────────────────────

    async def _fetch_altitude(self, lat: float, lon: float) -> float:
        """
        Fetch elevation (metres) from Open-Elevation public API.
        Not yet used in model — reserved for ball-speed physics.
        """
        try:
            import aiohttp
            url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat:.4f},{lon:.4f}"
            session = self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=8)) as r:
                if r.status != 200:
                    return 0.0
                data = await r.json(content_type=None)
                results = data.get("results", [])
                if results:
                    return float(results[0].get("elevation", 0.0))
        except Exception as exc:
            log.debug("[LOCATION] Altitude fetch failed: %s", exc)
        return 0.0

    # ── Weather (active: feeds age×temperature edge adjustment) ───────────────

    async def get_weather(self, lat: float, lon: float,
                          city: str = "") -> WeatherData:
        """
        Fetch real-time weather via wttr.in (free, no API key).
        temperature_c is consumed by the age×temperature win-prob adjustment
        in main.py. Other fields (humidity, wind) are cached for future use.
        """
        key = f"{lat:.2f},{lon:.2f}"
        now = time.time()

        if key in self._weather_cache:
            ts, data = self._weather_cache[key]
            if now - ts < _WEATHER_CACHE_TTL:
                return data

        try:
            import aiohttp
            query = city if city else f"{lat:.4f},{lon:.4f}"
            url   = f"https://wttr.in/{query}?format=j1"
            session = self._get_session()
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                if r.status != 200:
                    return WeatherData()
                raw = await r.json(content_type=None)

            cc     = raw.get("current_condition", [{}])[0]
            temp_c = float(cc.get("temp_C", 20))
            humid  = float(cc.get("humidity", 50))
            wind_k = float(cc.get("windspeedKmph", 0))
            wind_d = float(cc.get("winddirDegree", 0))
            press  = float(cc.get("pressure", 1013))
            desc   = (cc.get("weatherDesc") or [{"value": "unknown"}])[0].get("value", "unknown")

            data = WeatherData(
                temperature_c=temp_c,
                humidity_pct=humid,
                wind_speed_kmh=wind_k,
                wind_direction_deg=wind_d,
                pressure_hpa=press,
                conditions=desc,
            )
            self._weather_cache[key] = (now, data)
            log.debug("[WEATHER] %s: %.0f°C, humid=%.0f%%, wind=%.0fkm/h @ %.0f°, %s",
                      query, temp_c, humid, wind_k, wind_d, desc)
            return data

        except Exception as exc:
            log.debug("[LOCATION] Weather fetch failed: %s", exc)
            return WeatherData()

    # ── Sun data (active in model) ─────────────────────────────────────────────

    def get_sun_data(
        self,
        venue: VenueInfo,
        score_update: dict,
        dt: Optional[datetime.datetime] = None,
    ) -> SunData:
        """
        Compute current sun position and serve-glare penalties for both players.

        Called every live tick (cheap — pure arithmetic, no I/O).
        Penalties are applied by main.py to p_serve before Markov calculation.

        Args:
            venue:        VenueInfo from get_venue_info()
            score_update: live score dict from poll_live_score_real()
                          must contain 'games', 'sets', 'p1_serving'
            dt:           UTC datetime (default: now)
        """
        if dt is None:
            dt = datetime.datetime.utcnow()

        azimuth, elevation = compute_solar_position(
            venue.latitude, venue.longitude, dt
        )

        # Determine which direction each player faces when serving
        games = score_update.get("games", (0, 0))
        sets  = score_update.get("sets",  (0, 0))

        p1_face, p2_face = server_facing_azimuth(
            p1_serving=bool(score_update.get("p1_serving", True)),
            games=games,
            sets=sets,
            court_orientation_deg=venue.court_orientation_deg,
        )

        p1_penalty = compute_sun_penalty(p1_face, azimuth, elevation)
        p2_penalty = compute_sun_penalty(p2_face, azimuth, elevation)
        glare_active = (p1_penalty > 0.001 or p2_penalty > 0.001)

        # Human-readable description
        if elevation < 0:
            desc = "Sun below horizon — no glare"
        elif not glare_active:
            desc = (f"Sun: az={azimuth:.0f}° elev={elevation:.1f}° — "
                    f"outside glare window for both players")
        else:
            desc = (
                f"Sun: az={azimuth:.0f}° elev={elevation:.1f}° | "
                f"P1 faces {p1_face:.0f}° (penalty={p1_penalty:.1%}) | "
                f"P2 faces {p2_face:.0f}° (penalty={p2_penalty:.1%})"
            )

        return SunData(
            azimuth_deg=azimuth,
            elevation_deg=elevation,
            timestamp=time.time(),
            glare_active=glare_active,
            p1_sun_penalty=p1_penalty,
            p2_sun_penalty=p2_penalty,
            p1_facing_azimuth_deg=p1_face,
            p2_facing_azimuth_deg=p2_face,
            description=desc,
        )
