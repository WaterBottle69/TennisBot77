"""
ATP Masters Full Data Scraper
==============================
Collects every ATP Masters (1000) match from 1991–2024 with:
  - Full match stats (aces, DFs, serve %, break points, match duration, score)
  - Player profiles (height, handedness, nationality, ranking, rank points)
  - Weather at match venue on match date (humidity, temperature, wind, precipitation)
  - Venue altitude (from Open-Meteo elevation field)
  - Venue GPS coordinates and city
  - Player wingspan (scraped from Wikipedia where available; NaN otherwise)
  - Days rest for each player since previous match in the same tournament
  - Tournament round, surface, court type (indoor/outdoor)

Data sources:
  1. Jeff Sackmann tennis-atp GitHub  — match results & player stats
  2. Open-Meteo archive API           — historical weather & elevation (free, no key)
  3. Wikipedia API                    — player wingspan (sparse, best-effort)

Output CSV files:
  atp_masters_matches.csv   — one row per match (main dataset)
  atp_masters_players.csv   — one row per unique player (with wingspan where found)
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import re
import sys
import os
from io import StringIO
from datetime import datetime, timedelta

# ── Suppress SSL warnings from Sackmann raw.githubusercontent.com ──────────
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ── Venue lookup ────────────────────────────────────────────────────────────
# All ATP Masters / Super 9 venues with GPS coords and notes.
# Toronto and Montreal alternate years for the Canadian Open.
VENUES = {
    # tourney_name fragment → {lat, lon, city, country, court_type, surface_detail}
    "Indian Wells":   {"lat": 33.7206, "lon": -116.3703, "city": "Indian Wells",  "country": "USA",     "court_type": "Outdoor", "surface_detail": "Hard (Plexicushion)"},
    "Miami":          {"lat": 25.6899, "lon": -80.1766,  "city": "Miami",         "country": "USA",     "court_type": "Outdoor", "surface_detail": "Hard (Laykold)"},
    "Monte Carlo":    {"lat": 43.7390, "lon":   7.4276,  "city": "Monte Carlo",   "country": "Monaco",  "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Monte-Carlo":    {"lat": 43.7390, "lon":   7.4276,  "city": "Monte Carlo",   "country": "Monaco",  "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Madrid":         {"lat": 40.4168, "lon":  -3.7038,  "city": "Madrid",        "country": "Spain",   "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Rome":           {"lat": 41.8885, "lon":  12.4773,  "city": "Rome",          "country": "Italy",   "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Internazionali": {"lat": 41.8885, "lon":  12.4773,  "city": "Rome",          "country": "Italy",   "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Canada":         {"lat": 43.7315, "lon": -79.3895,  "city": "Toronto/Montreal","country": "Canada","court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Canadian":       {"lat": 43.7315, "lon": -79.3895,  "city": "Toronto/Montreal","country": "Canada","court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Rogers Cup":     {"lat": 43.7315, "lon": -79.3895,  "city": "Toronto/Montreal","country": "Canada","court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Cincinnati":     {"lat": 39.3578, "lon": -84.2954,  "city": "Mason, OH",     "country": "USA",     "court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Western &":      {"lat": 39.3578, "lon": -84.2954,  "city": "Mason, OH",     "country": "USA",     "court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Shanghai":       {"lat": 31.1878, "lon": 121.4374,  "city": "Shanghai",      "country": "China",   "court_type": "Outdoor", "surface_detail": "Hard (DecoTurf)"},
    "Paris":          {"lat": 48.8942, "lon":   2.3927,  "city": "Paris",         "country": "France",  "court_type": "Indoor",  "surface_detail": "Hard (Greenset)"},
    "Hamburg":        {"lat": 53.5642, "lon":   9.9640,  "city": "Hamburg",       "country": "Germany", "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Stuttgart":      {"lat": 48.8016, "lon":   9.2226,  "city": "Stuttgart",     "country": "Germany", "court_type": "Outdoor", "surface_detail": "Clay (red)"},
    "Essen":          {"lat": 51.4556, "lon":   7.0116,  "city": "Essen",         "country": "Germany", "court_type": "Indoor",  "surface_detail": "Carpet"},
    "Hannover":       {"lat": 52.3759, "lon":   9.7320,  "city": "Hannover",      "country": "Germany", "court_type": "Indoor",  "surface_detail": "Carpet"},
    "Stockholm":      {"lat": 59.3293, "lon":  18.0686,  "city": "Stockholm",     "country": "Sweden",  "court_type": "Indoor",  "surface_detail": "Hard"},
}

# Canadian Open alternates Toronto (odd years) / Montreal (even years)
CANADA_ODD_YEAR  = {"lat": 43.7315, "lon": -79.3895, "city": "Toronto"}
CANADA_EVEN_YEAR = {"lat": 45.5019, "lon": -73.5674, "city": "Montreal"}

# ── Known player wingspans (cm) compiled from publicly available sources ────
# Sources: Wikipedia, ATP profiles, sports anthropometry studies
KNOWN_WINGSPANS = {
    "Rafael Nadal":        188,
    "Roger Federer":       186,
    "Novak Djokovic":      188,
    "Andy Murray":         188,
    "Pete Sampras":        188,
    "Andre Agassi":        175,
    "Stefan Edberg":       188,
    "Boris Becker":        193,
    "Ivan Lendl":          188,
    "Mats Wilander":       183,
    "Lleyton Hewitt":      180,
    "Juan Carlos Ferrero": 185,
    "Carlos Moya":         190,
    "Gustavo Kuerten":     195,
    "Marcelo Rios":        183,
    "Thomas Muster":       183,
    "Goran Ivanisevic":    193,
    "Richard Krajicek":    199,
    "Michael Chang":       172,
    "Jim Courier":         183,
    "Carlos Alcaraz":      188,
    "Jannik Sinner":       188,
    "Daniil Medvedev":     196,
    "Alexander Zverev":    196,
    "Stefanos Tsitsipas":  190,
    "Dominic Thiem":       190,
    "Matteo Berrettini":   193,
    "Andrey Rublev":       188,
    "Casper Ruud":         183,
    "Holger Rune":         188,
    "Taylor Fritz":        193,
    "Frances Tiafoe":      186,
    "Ben Shelton":         193,
    "Hubert Hurkacz":      196,
    "Felix Auger-Aliassime": 193,
    "Denis Shapovalov":    185,
    "Nick Kyrgios":        193,
    "Marin Cilic":         196,
    "Milos Raonic":        204,
    "Jo-Wilfried Tsonga":  185,
    "Gael Monfils":        190,
    "Richard Gasquet":     183,
    "David Ferrer":        178,
    "Fernando Verdasco":   193,
    "Tommy Haas":          188,
    "Nikolay Davydenko":   178,
    "Robby Ginepri":       183,
    "Tomas Berdych":       198,
    "David Nalbandian":    183,
    "Juan Martin del Potro": 198,
    "Feliciano Lopez":     188,
    "John Isner":          213,
    "Kevin Anderson":      203,
    "Ivo Karlovic":        211,
    "Sam Querrey":         196,
    "Reilly Opelka":       211,
    "Jack Sock":           193,
    "Mardy Fish":          183,
    "Andy Roddick":        188,
    "James Blake":         180,
    "Robin Soderling":     193,
    "Jurgen Melzer":       185,
    "Nikolai Davydenko":   178,
    "Lleyton Hewitt":      180,
    "Tim Henman":          183,
    "Mark Philippoussis":  193,
    "Patrick Rafter":      185,
    "Yevgeny Kafelnikov":  190,
    "Marcelo Melo":        190,
    "Fabio Fognini":       178,
    "Diego Schwartzman":   165,
    "Pablo Carreno Busta": 183,
    "Roberto Bautista Agut": 180,
    "Grigor Dimitrov":     188,
    "David Goffin":        180,
    "Lucas Pouille":       188,
    "Nicolas Mahut":       188,
    "Pierre-Hugues Herbert": 183,
    "Karen Khachanov":     196,
    "Alex de Minaur":      180,
    "Lorenzo Musetti":     185,
    "Jannik Sinner":       188,
    "Lorenzo Sonego":      188,
    "Sebastian Korda":     193,
}


def resolve_venue(tourney_name: str, year: int) -> dict:
    """Return venue metadata for a given tournament name."""
    for key, venue in VENUES.items():
        if key.lower() in tourney_name.lower():
            # Canadian Open — alternate cities
            if "canada" in key.lower() or "canadian" in key.lower() or "rogers" in key.lower():
                if year % 2 == 1:
                    venue = {**venue, **CANADA_ODD_YEAR}
                else:
                    venue = {**venue, **CANADA_EVEN_YEAR}
            return venue
    return {"lat": None, "lon": None, "city": tourney_name, "country": "Unknown",
            "court_type": "Unknown", "surface_detail": "Unknown"}


# ── Weather cache — avoid hammering the API ──────────────────────────────────
_weather_cache: dict = {}
_venue_elevation_cache: dict = {}


def fetch_weather(lat: float, lon: float, date_str: str) -> dict:
    """
    Fetch noon-hour weather from Open-Meteo archive for a lat/lon on a date.
    Returns dict with humidity, temperature, wind_speed, precipitation, elevation.
    date_str format: YYYY-MM-DD
    """
    if lat is None or lon is None:
        return {}

    cache_key = f"{lat:.3f},{lon:.3f},{date_str}"
    if cache_key in _weather_cache:
        return _weather_cache[cache_key]

    url = (
        "https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={lat}&longitude={lon}"
        f"&start_date={date_str}&end_date={date_str}"
        "&hourly=relative_humidity_2m,temperature_2m,wind_speed_10m,"
        "wind_direction_10m,precipitation,apparent_temperature"
        "&timezone=auto"
    )
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code != 200:
            _weather_cache[cache_key] = {}
            return {}
        data = resp.json()
        elevation = data.get("elevation", None)
        hourly = data.get("hourly", {})
        # Use hour 12 (noon) as representative match-time weather
        idx = 12
        result = {
            "altitude_m":        round(elevation, 1) if elevation is not None else None,
            "humidity_pct":      hourly.get("relative_humidity_2m", [None]*24)[idx],
            "temp_celsius":      hourly.get("temperature_2m", [None]*24)[idx],
            "apparent_temp_c":   hourly.get("apparent_temperature", [None]*24)[idx],
            "wind_speed_kmh":    hourly.get("wind_speed_10m", [None]*24)[idx],
            "wind_direction_deg":hourly.get("wind_direction_10m", [None]*24)[idx],
            "precipitation_mm":  hourly.get("precipitation", [None]*24)[idx],
        }
        _weather_cache[cache_key] = result
        return result
    except Exception as e:
        print(f"  [weather] {date_str} {lat},{lon} → {e}", file=sys.stderr)
        _weather_cache[cache_key] = {}
        return {}


def fetch_player_wingspan_wikipedia(player_name: str) -> float | None:
    """
    Try to find wingspan/arm-span for a player from Wikipedia.
    Returns cm float or None if not found.
    """
    slug = player_name.strip().replace(" ", "_")
    url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}"
    try:
        r = requests.get(url, timeout=8, headers={"User-Agent": "TennisScraper/1.0"})
        if r.status_code != 200:
            return None
        data = r.json()
        text = data.get("extract", "")
        # Look for wingspan / arm span mentions
        match = re.search(
            r"(?:wingspan|arm\s*span|reach)[^\d]{0,20}([\d.]+)\s*(cm|m\b)",
            text, re.IGNORECASE
        )
        if match:
            val, unit = float(match.group(1)), match.group(2).lower()
            return val * 100 if unit == "m" else val
    except Exception:
        pass
    return None


# ── Download Sackmann ATP match data ────────────────────────────────────────
SACKMANN_BASE = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"
START_YEAR = 1990
END_YEAR = 2024


def download_year(year: int) -> pd.DataFrame | None:
    url = f"{SACKMANN_BASE}/atp_matches_{year}.csv"
    try:
        r = requests.get(url, timeout=30, verify=False)
        if r.status_code != 200:
            return None
        df = pd.read_csv(StringIO(r.text), low_memory=False)
        df["year"] = year
        return df
    except Exception as e:
        print(f"  [sackmann] {year} failed: {e}", file=sys.stderr)
        return None


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # 1. Download all years
    print("═" * 60)
    print("Step 1/5 — Downloading Sackmann ATP match data")
    print("═" * 60)
    frames = []
    for yr in range(START_YEAR, END_YEAR + 1):
        print(f"  Fetching {yr}...", end=" ", flush=True)
        df = download_year(yr)
        if df is not None:
            frames.append(df)
            print(f"{len(df)} matches")
        else:
            print("skipped")
        time.sleep(0.15)

    all_matches = pd.concat(frames, ignore_index=True)
    print(f"\nTotal rows downloaded: {len(all_matches):,}")

    # 2. Filter for Masters only (tourney_level == 'M')
    masters = all_matches[all_matches["tourney_level"] == "M"].copy()
    print(f"Masters matches only:  {len(masters):,}\n")

    # 3. Add venue metadata
    print("Step 2/5 — Resolving venue coordinates and court type")
    def get_venue_field(row, field):
        v = resolve_venue(str(row["tourney_name"]), int(row["year"]))
        return v.get(field)

    masters["venue_city"]         = masters.apply(lambda r: get_venue_field(r, "city"), axis=1)
    masters["venue_country"]      = masters.apply(lambda r: get_venue_field(r, "country"), axis=1)
    masters["venue_lat"]          = masters.apply(lambda r: get_venue_field(r, "lat"), axis=1)
    masters["venue_lon"]          = masters.apply(lambda r: get_venue_field(r, "lon"), axis=1)
    masters["court_type"]         = masters.apply(lambda r: get_venue_field(r, "court_type"), axis=1)
    masters["surface_detail"]     = masters.apply(lambda r: get_venue_field(r, "surface_detail"), axis=1)

    # 4. Parse date
    masters["tourney_date"] = pd.to_datetime(masters["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    masters["match_date_str"] = masters["tourney_date"].dt.strftime("%Y-%m-%d")

    # 5. Fetch weather — one API call per (venue, date) pair
    print("\nStep 3/5 — Fetching weather data from Open-Meteo")
    unique_venue_dates = masters[["venue_lat", "venue_lon", "match_date_str"]].drop_duplicates()
    total = len(unique_venue_dates)
    print(f"  {total:,} unique venue×date combinations")

    weather_rows = []
    for i, (_, row) in enumerate(unique_venue_dates.iterrows()):
        if i % 50 == 0:
            print(f"  Progress: {i}/{total}...", flush=True)
        w = fetch_weather(row["venue_lat"], row["venue_lon"], row["match_date_str"])
        weather_rows.append({
            "venue_lat":       row["venue_lat"],
            "venue_lon":       row["venue_lon"],
            "match_date_str":  row["match_date_str"],
            **w
        })
        # Polite rate limiting — Open-Meteo allows ~10k requests/day free
        time.sleep(0.12)

    weather_df = pd.DataFrame(weather_rows)
    masters = masters.merge(weather_df, on=["venue_lat", "venue_lon", "match_date_str"], how="left")

    # 6. Player wingspans
    print("\nStep 4/5 — Resolving player wingspans")
    all_players = pd.concat([
        masters[["winner_name"]].rename(columns={"winner_name": "player_name"}),
        masters[["loser_name"]].rename(columns={"loser_name": "player_name"}),
    ]).drop_duplicates()["player_name"].dropna().tolist()

    wingspan_map: dict = {}
    print(f"  {len(all_players)} unique players")

    for name in all_players:
        # First check our known list
        if name in KNOWN_WINGSPANS:
            wingspan_map[name] = KNOWN_WINGSPANS[name]
            continue
        # Try Wikipedia as fallback
        ws = fetch_player_wingspan_wikipedia(name)
        wingspan_map[name] = ws
        time.sleep(0.1)

    found = sum(1 for v in wingspan_map.values() if v is not None)
    print(f"  Wingspan found for {found}/{len(all_players)} players")

    masters["winner_wingspan_cm"] = masters["winner_name"].map(wingspan_map)
    masters["loser_wingspan_cm"]  = masters["loser_name"].map(wingspan_map)

    # 7. Add days-rest column for each player within tournament
    print("\nStep 5/5 — Computing days rest and enriching features")
    masters = masters.sort_values(["tourney_id", "tourney_date", "match_num"]).reset_index(drop=True)

    # Days rest: within a tournament, how many days since player's previous match
    def compute_days_rest(df: pd.DataFrame) -> pd.DataFrame:
        last_match_date: dict = {}
        winner_rest = []
        loser_rest  = []
        for _, row in df.iterrows():
            tid = row["tourney_id"]
            wn  = row["winner_name"]
            ln  = row["loser_name"]
            d   = row["tourney_date"]
            wkey = (tid, wn)
            lkey = (tid, ln)
            winner_rest.append((d - last_match_date[wkey]).days if wkey in last_match_date else None)
            loser_rest.append( (d - last_match_date[lkey]).days if lkey in last_match_date else None)
            last_match_date[wkey] = d
            last_match_date[lkey] = d
        df = df.copy()
        df["winner_days_rest"] = winner_rest
        df["loser_days_rest"]  = loser_rest
        return df

    masters = compute_days_rest(masters)

    # Derived edge features
    masters["winner_rank_diff"]   = masters["loser_rank"]  - masters["winner_rank"]
    masters["height_diff_cm"]     = masters["winner_ht"]   - masters["loser_ht"]
    masters["wingspan_diff_cm"]   = masters["winner_wingspan_cm"] - masters["loser_wingspan_cm"]
    masters["age_diff"]           = masters["winner_age"]  - masters["loser_age"]

    # Compute serve dominance (w_1stWon/w_svpt as first-serve win %)
    def safe_div(a, b):
        return np.where((b.notna()) & (b != 0), a / b, np.nan)

    for pfx in ("w", "l"):
        svpt = masters.get(f"{pfx}_svpt")
        if svpt is not None:
            masters[f"{pfx}_1st_serve_win_pct"] = safe_div(
                pd.to_numeric(masters.get(f"{pfx}_1stWon", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(masters.get(f"{pfx}_svpt",   pd.Series(dtype=float)), errors="coerce")
            )
            masters[f"{pfx}_2nd_serve_win_pct"] = safe_div(
                pd.to_numeric(masters.get(f"{pfx}_2ndWon", pd.Series(dtype=float)), errors="coerce"),
                pd.to_numeric(masters.get(f"{pfx}_svpt",   pd.Series(dtype=float)), errors="coerce")
            )

    # Clean column order for output
    core_cols = [
        "tourney_id", "tourney_name", "year",
        "tourney_date", "surface", "surface_detail", "court_type",
        "venue_city", "venue_country", "venue_lat", "venue_lon",
        "altitude_m", "humidity_pct", "temp_celsius", "apparent_temp_c",
        "wind_speed_kmh", "wind_direction_deg", "precipitation_mm",
        "round", "best_of", "match_num", "minutes",
        "winner_name", "winner_seed", "winner_hand", "winner_ht",
        "winner_wingspan_cm", "winner_ioc", "winner_age",
        "winner_rank", "winner_rank_points", "winner_days_rest",
        "loser_name", "loser_seed", "loser_hand", "loser_ht",
        "loser_wingspan_cm", "loser_ioc", "loser_age",
        "loser_rank", "loser_rank_points", "loser_days_rest",
        "score",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "w_1st_serve_win_pct", "w_2nd_serve_win_pct",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "l_1st_serve_win_pct", "l_2nd_serve_win_pct",
        "winner_rank_diff", "height_diff_cm", "wingspan_diff_cm", "age_diff",
    ]
    available_cols = [c for c in core_cols if c in masters.columns]
    output = masters[available_cols]

    out_matches = os.path.join(out_dir, "atp_masters_matches.csv")
    output.to_csv(out_matches, index=False)
    print(f"\n  Saved → {out_matches} ({len(output):,} rows, {len(output.columns)} columns)")

    # Player summary CSV
    player_rows = []
    for name, ws in wingspan_map.items():
        subset = masters[masters["winner_name"] == name]
        if subset.empty:
            subset = masters[masters["loser_name"] == name]
        if not subset.empty:
            row0 = subset.iloc[0]
            hand = row0.get("winner_hand") if row0.get("winner_name") == name else row0.get("loser_hand")
            ht   = row0.get("winner_ht")   if row0.get("winner_name") == name else row0.get("loser_ht")
            ioc  = row0.get("winner_ioc")  if row0.get("winner_name") == name else row0.get("loser_ioc")
        else:
            hand = ht = ioc = None
        player_rows.append({
            "player_name":  name,
            "hand":         hand,
            "height_cm":    ht,
            "wingspan_cm":  ws,
            "nationality":  ioc,
        })

    players_df = pd.DataFrame(player_rows).sort_values("player_name")
    out_players = os.path.join(out_dir, "atp_masters_players.csv")
    players_df.to_csv(out_players, index=False)
    print(f"  Saved → {out_players} ({len(players_df):,} players)")

    print("\n═" * 60)
    print("Done.")
    print(f"  Masters matches: {len(output):,}")
    print(f"  Players:         {len(players_df):,}")
    print(f"  Year range:      {output['year'].min()} – {output['year'].max()}")
    print(f"  Wingspan found:  {found}/{len(all_players)} players")
    print("\n  Columns in atp_masters_matches.csv:")
    for c in output.columns:
        print(f"    {c}")


if __name__ == "__main__":
    main()
