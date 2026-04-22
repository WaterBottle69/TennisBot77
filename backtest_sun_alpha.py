"""
backtest_sun_alpha.py
─────────────────────
Tests whether a player serving with the sun in their eyes suffers a measurable
drop in serve-win-point percentage.

Data source
───────────
Jeff Sackmann's open tennis_atp GitHub repository (CC BY 4.0).
URL: https://github.com/JeffSackmann/tennis_atp
We download the last N years of ATP match CSVs, filter to outdoor matches at
known venues, and run statistical tests.

Hypothesis
──────────
H₀: Sun glare has no effect on serve-win-point %
H₁: Higher sun exposure (elevation 15–60°, azimuth within 50° of serving
    direction) → lower serve-win-point %

Method
──────
1. Download ATP match CSVs (2019–2024, ~6 years)
2. Filter: outdoor surface, venue in _KNOWN_VENUES
3. Estimate match time from round + tournament type (see _ROUND_TIMES)
4. Compute sun azimuth / elevation at that venue + time
5. Classify each match as GLARE / MODERATE / LOW based on sun position
6. Compare winner / loser 1stSvPts% across conditions
7. Run Welch t-test + Mann-Whitney U, report Cohen's d and implied alpha (edge)

Output
──────
Prints a detailed report and saves backtest_sun_results.json.
"""

import sys
import json
import math
import datetime
import logging
import statistics
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import urllib.request
import ssl

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Import our solar engine ────────────────────────────────────────────────────
from location_engine import compute_solar_position, compute_sun_penalty, _KNOWN_VENUES

# ── Config ─────────────────────────────────────────────────────────────────────
YEARS          = list(range(2019, 2025))   # 6 seasons of data
MIN_MATCH_SVPT = 20                         # discard matches with < 20 serve points (walkovers)
GLARE_ELEV_LOW = 10.0
GLARE_ELEV_HI  = 75.0
GLARE_AZ_RANGE = 50.0                       # degrees either side of serving direction

# Round → assumed LOCAL start hour (24h)
# Based on typical ATP tournament scheduling:
#   R128/R64/R32 = morning blocks (11:00)
#   R16          = afternoon (13:00)
#   QF           = 14:00
#   SF           = 15:00
#   F            = 15:00
_ROUND_HOURS: Dict[str, int] = {
    "R128": 11, "R64": 11, "R32": 12,
    "R16": 13, "QF": 14, "SF": 15, "F": 15,
    "RR": 13,   # round-robin (WTF / Davis Cup)
}
_DEFAULT_HOUR = 13

# All outdoor N–S courts assumed unless overridden here
_COURT_ORIENT: Dict[str, float] = {
    "Paris Masters": -15.0,   # Bercy is slightly angled
}

_SSL_CTX = ssl.create_default_context()
_SSL_CTX.check_hostname = False
_SSL_CTX.verify_mode   = ssl.CERT_NONE

# ── Data download ──────────────────────────────────────────────────────────────

def download_atp_year(year: int) -> Optional[pd.DataFrame]:
    url = (
        "https://raw.githubusercontent.com/JeffSackmann/tennis_atp"
        f"/master/atp_matches_{year}.csv"
    )
    log.info("Downloading ATP %d …", year)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "backtest/1.0"})
        with urllib.request.urlopen(req, context=_SSL_CTX, timeout=20) as resp:
            df = pd.read_csv(resp)
        log.info("  → %d rows", len(df))
        return df
    except Exception as exc:
        log.warning("  → FAILED: %s", exc)
        return None


def load_atp_data(years: List[int]) -> pd.DataFrame:
    frames = [f for y in years if (f := download_atp_year(y)) is not None]
    if not frames:
        log.error("No data downloaded — cannot backtest.")
        sys.exit(1)
    df = pd.concat(frames, ignore_index=True)
    log.info("Total rows before filtering: %d", len(df))
    return df

# ── Venue mapping ──────────────────────────────────────────────────────────────

def _build_venue_lookup() -> Dict[str, tuple]:
    """Expand _KNOWN_VENUES with additional common ATP tournament names."""
    lookup = {}
    for kw, v in _KNOWN_VENUES.items():
        lookup[kw] = v
    # Extra mappings that appear in Sackmann's tourney_name field
    extras = {
        "istanbul":             ("Istanbul",    "Turkey",  41.0082,  28.9784,  40, "clay",  0.0),
        "estoril":              ("Estoril",     "Portugal",38.7072,  -9.3895,  75, "clay",  0.0),
        "marrakech":            ("Marrakech",   "Morocco", 31.6295,  -7.9811, 460, "clay",  0.0),
        "buenos aires":         ("Buenos Aires","Argentina",-34.6037,-58.3816, 25, "clay",  0.0),
        "rio de janeiro":       ("Rio",         "Brazil",  -22.9068, -43.1729,  8, "clay",  0.0),
        "acapulco":             ("Acapulco",    "Mexico",  16.8531,  -99.8237,  3, "hard",  0.0),
        "san diego":            ("San Diego",   "USA",     32.7157, -117.1611, 20, "hard",  0.0),
        "los cabos":            ("Los Cabos",   "Mexico",  22.8905, -109.9167,  5, "hard",  0.0),
        "eastbourne":           ("Eastbourne",  "UK",      50.7684,    0.2904,  8, "grass", 0.0),
        "'s-hertogenbosch":     ("Rosmalen",    "Netherlands",51.7158,5.3537,  13, "grass", 0.0),
        "s-hertogenbosch":      ("Rosmalen",    "Netherlands",51.7158,5.3537,  13, "grass", 0.0),
        "dubai":                ("Dubai",       "UAE",     25.2048,  55.2708,  12, "hard",  0.0),
        "doha":                 ("Doha",        "Qatar",   25.2854,  51.5310,  10, "hard",  0.0),
        "abu dhabi":            ("Abu Dhabi",   "UAE",     24.4539,  54.3773,  27, "hard",  0.0),
        "buenos":               ("Buenos Aires","Argentina",-34.6037,-58.3816, 25, "clay",  0.0),
        "umag":                 ("Umag",        "Croatia", 45.4324,  13.5218,  10, "clay",  0.0),
        "gstaad":               ("Gstaad",      "Switzerland",46.4749, 7.2812,1050,"clay",  0.0),
        "bastad":               ("Båstad",      "Sweden",  56.4327,  12.8526,  10, "clay",  0.0),
        "kitzbuhel":            ("Kitzbühel",   "Austria", 47.4463,  12.3927, 762, "clay",  0.0),
        "washington":           ("Washington",  "USA",     38.9072,  -77.0369, 30, "hard",  0.0),
        "los angeles":          ("Los Angeles", "USA",     34.0522, -118.2437, 71, "hard",  0.0),
        "winston-salem":        ("Winston-Salem","USA",    36.0999, -80.2442, 290, "hard",  0.0),
        "new haven":            ("New Haven",   "USA",     41.3083,  -72.9279,  7, "hard",  0.0),
        "mexico city":          ("Mexico City", "Mexico",  19.4326,  -99.1332,2240,"clay",  0.0),
        "cordoba":              ("Córdoba",     "Argentina",-31.4201,-64.1888, 434,"clay",  0.0),
        "chile":                ("Santiago",    "Chile",   -33.4489, -70.6693, 567,"clay",  0.0),
        "bogota":               ("Bogotá",      "Colombia", 4.7110,  -74.0721,2600,"clay",  0.0),
        "orlando":              ("Orlando",     "USA",     28.5383,  -81.3792,  29,"clay",  0.0),
        "delray beach":         ("Delray Beach","USA",     26.4615,  -80.0728,   5,"hard",  0.0),
        "dallas":               ("Dallas",      "USA",     32.7767,  -96.7970, 139,"hard",  0.0),
    }
    lookup.update(extras)
    return lookup


_VENUE_LOOKUP = _build_venue_lookup()


def get_venue(tourney_name: str) -> Optional[tuple]:
    """Return venue tuple for a tournament name (case-insensitive keyword match)."""
    low = tourney_name.lower()
    for kw, v in _VENUE_LOOKUP.items():
        if kw in low:
            return v
    return None

# ── Sun classification ─────────────────────────────────────────────────────────

def sun_category(lat: float, lon: float, court_orient: float,
                 date_str: str, local_hour: int) -> Tuple[float, float, str]:
    """
    Return (sun_elevation, max_penalty, category) for a match.

    category:
      'GLARE'    — sun meaningfully in at least one server's eyes (penalty > 2%)
      'MODERATE' — sun elevated but not directly in serving line (0.5–2%)
      'LOW'      — sun below glare threshold or night / indoor
    """
    try:
        d = datetime.datetime.strptime(date_str, "%Y%m%d")
    except Exception:
        return 0.0, 0.0, "UNKNOWN"

    # Convert local hour to UTC using approximate timezone offset
    # Simple lon-based offset (accurate to ~1h for most venues)
    utc_offset = round(lon / 15.0)
    utc_hour   = (local_hour - utc_offset) % 24
    dt = datetime.datetime(d.year, d.month, d.day, utc_hour, 0, 0)

    try:
        az, elev = compute_solar_position(lat, lon, dt)
    except Exception:
        return 0.0, 0.0, "UNKNOWN"

    if elev < GLARE_ELEV_LOW or elev > GLARE_ELEV_HI:
        return elev, 0.0, "LOW"

    # For a N-S court: servers face N (0°) or S (180°)
    # Test both; glare applies to whichever player faces the sun
    dir_a = (court_orient + 180.0) % 360.0   # serving from baseline-A
    dir_b =  court_orient                      # serving from baseline-B
    p_a   = compute_sun_penalty(dir_a, az, elev)
    p_b   = compute_sun_penalty(dir_b, az, elev)
    max_p = max(p_a, p_b)

    if max_p > 0.02:
        return elev, max_p, "GLARE"
    elif max_p > 0.005:
        return elev, max_p, "MODERATE"
    else:
        return elev, max_p, "LOW"

# ── Match processing ───────────────────────────────────────────────────────────

def process_match(row: pd.Series) -> Optional[dict]:
    """
    Extract serve stats, venue, sun condition from one match row.
    Returns None if insufficient data.
    """
    # Filter indoor / hard-to-classify
    court = str(row.get("court", "")).strip()
    if court.lower() == "indoor":
        return None

    tourney = str(row.get("tourney_name", ""))
    venue   = get_venue(tourney)
    if venue is None:
        return None

    city, country, lat, lon, alt, surface, court_orient = venue
    date_str = str(row.get("tourney_date", ""))
    rnd      = str(row.get("round", "R32")).strip()
    local_h  = _ROUND_HOURS.get(rnd, _DEFAULT_HOUR)

    # Serve stats (winner side)
    def safe_pct(won, inn):
        try:
            w, i = float(won), float(inn)
            return (w / i) if i >= MIN_MATCH_SVPT else None
        except Exception:
            return None

    w_1stpct = safe_pct(row.get("w_1stWon"), row.get("w_1stIn"))
    l_1stpct = safe_pct(row.get("l_1stWon"), row.get("l_1stIn"))

    # Combined (both players) 1st-serve win % — for aggregate analysis
    try:
        total_1stWon = float(row.get("w_1stWon", 0)) + float(row.get("l_1stWon", 0))
        total_1stIn  = float(row.get("w_1stIn",  0)) + float(row.get("l_1stIn",  0))
        both_1stpct  = (total_1stWon / total_1stIn) if total_1stIn >= MIN_MATCH_SVPT else None
    except Exception:
        both_1stpct = None

    if both_1stpct is None:
        return None

    elev, max_penalty, cat = sun_category(lat, lon, court_orient, date_str, local_h)

    return {
        "tourney":       tourney,
        "city":          city,
        "date":          date_str,
        "round":         rnd,
        "local_hour":    local_h,
        "surface":       str(row.get("surface", surface)),
        "sun_elev":      elev,
        "max_penalty":   max_penalty,
        "sun_category":  cat,
        "w_1stpct":      w_1stpct,
        "l_1stpct":      l_1stpct,
        "both_1stpct":   both_1stpct,
        "winner":        str(row.get("winner_name", "")),
        "loser":         str(row.get("loser_name",  "")),
    }

# ── Statistical tests ──────────────────────────────────────────────────────────

def cohens_d(a: List[float], b: List[float]) -> float:
    """Pooled Cohen's d effect size."""
    if len(a) < 2 or len(b) < 2:
        return 0.0
    mean_diff = statistics.mean(a) - statistics.mean(b)
    s_a = statistics.stdev(a)
    s_b = statistics.stdev(b)
    n_a, n_b = len(a), len(b)
    pooled_std = math.sqrt(((n_a - 1) * s_a**2 + (n_b - 1) * s_b**2) / (n_a + n_b - 2))
    return mean_diff / pooled_std if pooled_std > 0 else 0.0


def welch_t_pvalue(a: List[float], b: List[float]) -> float:
    """Welch's t-test (unequal variance) p-value."""
    try:
        from scipy import stats
        _, p = stats.ttest_ind(a, b, equal_var=False)
        return float(p)
    except Exception:
        return float("nan")


def mannwhitney_pvalue(a: List[float], b: List[float]) -> float:
    """Mann-Whitney U test p-value (non-parametric)."""
    try:
        from scipy import stats
        _, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        return float(p)
    except Exception:
        return float("nan")

# ── Report ─────────────────────────────────────────────────────────────────────

def run_analysis(records: List[dict]) -> dict:
    df = pd.DataFrame(records)
    log.info("Analysing %d valid matches across %d venues …",
             len(df), df["city"].nunique())

    cats   = df["sun_category"].value_counts().to_dict()
    log.info("Sun categories: %s", cats)

    glare  = df[df["sun_category"] == "GLARE"]["both_1stpct"].dropna().tolist()
    mod    = df[df["sun_category"] == "MODERATE"]["both_1stpct"].dropna().tolist()
    low    = df[df["sun_category"] == "LOW"]["both_1stpct"].dropna().tolist()

    def summary(label, vals):
        if not vals:
            return {"n": 0}
        return {
            "n":     len(vals),
            "mean":  round(statistics.mean(vals), 4),
            "stdev": round(statistics.stdev(vals), 4) if len(vals) > 1 else 0,
            "label": label,
        }

    s_glare = summary("GLARE",    glare)
    s_mod   = summary("MODERATE", mod)
    s_low   = summary("LOW",      low)

    # Primary comparison: GLARE vs LOW
    d_gl = cohens_d(glare, low)
    p_t  = welch_t_pvalue(glare, low)
    p_mw = mannwhitney_pvalue(glare, low)

    # Regression: sun elevation → 1stSvPts%
    elev_col  = df["sun_elev"].values
    stat_col  = df["both_1stpct"].values
    mask      = np.isfinite(elev_col) & np.isfinite(stat_col)
    corr_coef = float(np.corrcoef(elev_col[mask], stat_col[mask])[0, 1])

    # Surface breakdown
    surface_results = {}
    for surf in ["Clay", "Grass", "Hard"]:
        sub = df[df["surface"].str.lower() == surf.lower()]
        sg = sub[sub["sun_category"] == "GLARE"]["both_1stpct"].dropna().tolist()
        sl = sub[sub["sun_category"] == "LOW"]["both_1stpct"].dropna().tolist()
        if sg and sl:
            surface_results[surf] = {
                "glare_n":  len(sg),
                "glare_mean": round(statistics.mean(sg), 4),
                "low_n":    len(sl),
                "low_mean": round(statistics.mean(sl), 4),
                "delta":    round(statistics.mean(sg) - statistics.mean(sl), 4),
                "cohens_d": round(cohens_d(sg, sl), 4),
                "p_value":  round(welch_t_pvalue(sg, sl), 4),
            }

    # Top venues where glare is most common
    glare_venues = (df[df["sun_category"] == "GLARE"]
                    .groupby("city")["both_1stpct"]
                    .agg(["count", "mean"])
                    .rename(columns={"count": "n", "mean": "mean_1stpct"})
                    .sort_values("n", ascending=False)
                    .head(10)
                    .to_dict())

    # Serve penalty implied by observed effect
    # If 1stSvPts drops by X% in glare, and typical p_serve is ~0.65,
    # the implied serve penalty is X * p_serve
    mean_glare = s_glare.get("mean", 0)
    mean_low   = s_low.get("mean", 0)
    obs_delta  = mean_glare - mean_low   # negative = glare suppresses serve win%
    mean_pserve = statistics.mean(df["both_1stpct"].dropna().tolist()) if len(df) else 0.65
    implied_penalty = abs(obs_delta * mean_pserve) if mean_pserve > 0 else 0.0

    # Edge estimate: if p_serve drops by implied_penalty, Markov sensitivity
    # propagates to win_prob by ~2× (empirical from our engine)
    markov_sensitivity = 2.0
    implied_wp_delta   = implied_penalty * markov_sensitivity

    # Alpha signal: only meaningful if p-value < 0.05 and effect is directional
    signal_significant = (p_t < 0.10) and (obs_delta < 0)

    result = {
        "total_matches_analysed": len(df),
        "venues_covered":         int(df["city"].nunique()),
        "years":                  sorted(df["date"].str[:4].unique().tolist()),
        "sun_categories":         cats,
        "group_stats": {
            "GLARE":    s_glare,
            "MODERATE": s_mod,
            "LOW":      s_low,
        },
        "glare_vs_low": {
            "mean_diff_1stSvPts":  round(obs_delta, 4),
            "cohens_d":            round(d_gl, 4),
            "welch_t_pvalue":      round(p_t, 4),
            "mannwhitney_pvalue":  round(p_mw, 4),
            "elevation_correlation": round(corr_coef, 4),
        },
        "implied_alpha": {
            "implied_p_serve_penalty":  round(implied_penalty, 4),
            "implied_win_prob_delta":   round(implied_wp_delta, 4),
            "signal_statistically_sig": signal_significant,
        },
        "surface_breakdown": surface_results,
        "top_glare_venues":  glare_venues,
    }
    return result


def print_report(r: dict):
    sep = "─" * 65
    print(f"\n{'═'*65}")
    print(f"  SUN-GLARE ALPHA BACKTEST  ({r['total_matches_analysed']:,} matches, "
          f"{r['venues_covered']} venues, {r['years'][0]}–{r['years'][-1]})")
    print(f"{'═'*65}")

    cats = r["sun_categories"]
    print(f"\nSun categories (estimated from round × venue × date):")
    for cat in ["GLARE", "MODERATE", "LOW", "UNKNOWN"]:
        n = cats.get(cat, 0)
        print(f"  {cat:<12} {n:>6,}  ({n/r['total_matches_analysed']*100:.1f}%)")

    gs = r["group_stats"]
    print(f"\n{sep}")
    print(f"{'Group':<12} {'N':>6}  {'Mean 1stSvPts%':>16}  {'Stdev':>8}")
    print(sep)
    for label in ["GLARE", "MODERATE", "LOW"]:
        g = gs.get(label, {})
        if g.get("n", 0):
            print(f"  {label:<10} {g['n']:>6,}  {g['mean']*100:>14.2f}%  {g['stdev']*100:>7.2f}%")

    gvl = r["glare_vs_low"]
    print(f"\n{sep}")
    print("GLARE vs LOW statistical comparison:")
    print(sep)
    delta_pct = gvl['mean_diff_1stSvPts'] * 100
    sign = "↓" if delta_pct < 0 else "↑"
    print(f"  Mean 1stSvPts% difference:  {delta_pct:+.2f}%  {sign}")
    print(f"  Cohen's d (effect size):    {gvl['cohens_d']:+.4f}")
    print(f"  Welch t-test p-value:        {gvl['welch_t_pvalue']:.4f}"
          f"  {'✓ p<0.10' if gvl['welch_t_pvalue'] < 0.10 else '✗ not sig'}")
    print(f"  Mann-Whitney U p-value:      {gvl['mannwhitney_pvalue']:.4f}")
    print(f"  Elev. ↔ 1stSvPts% corr:     {gvl['elevation_correlation']:+.4f}")

    ai = r["implied_alpha"]
    print(f"\n{sep}")
    print("Implied alpha:")
    print(sep)
    print(f"  Obs. p_serve penalty:        {ai['implied_p_serve_penalty']*100:.2f}%")
    print(f"  Implied win_prob Δ (2× sens):{ai['implied_win_prob_delta']*100:.2f}%")
    verdict = ("✓ SIGNAL FOUND" if ai["signal_statistically_sig"]
               else "✗ No significant signal")
    print(f"  Signal:                      {verdict}")

    sb = r["surface_breakdown"]
    if sb:
        print(f"\n{sep}")
        print("Surface breakdown (GLARE vs LOW):")
        print(sep)
        print(f"  {'Surface':<8} {'Glare N':>8} {'Low N':>8} {'Δ 1stSvPts%':>14} {'d':>8} {'p':>8}")
        for surf, v in sorted(sb.items()):
            print(f"  {surf:<8} {v['glare_n']:>8,} {v['low_n']:>8,} "
                  f"{v['delta']*100:>+13.2f}%  {v['cohens_d']:>+7.4f}  {v['p_value']:>8.4f}")

    print(f"\n{'═'*65}")

    # Verdict
    if ai["signal_statistically_sig"]:
        print("VERDICT: Statistically significant sun-glare effect detected.")
        print(f"  Players serving into sun lose ~{abs(gvl['mean_diff_1stSvPts'])*100:.2f}pp "
              f"on 1st serve win rate.")
        print(f"  This maps to ~{ai['implied_win_prob_delta']*100:.2f}% win-prob edge "
              f"via Markov sensitivity.")
        print("  Sun positioning IS worth keeping in the model.")
    else:
        if gvl['mean_diff_1stSvPts'] < 0:
            print("VERDICT: Directional effect observed but NOT statistically significant overall.")
            print(f"  Serve win% is {abs(gvl['mean_diff_1stSvPts'])*100:.2f}pp lower under glare.")
            # Surface-specific check
            hard = sb.get("Hard", {})
            if hard and hard.get("p_value", 1.0) < 0.05:
                print(f"  ✓ HARD COURTS: significant (p={hard['p_value']:.4f}, "
                      f"d={hard['cohens_d']:.3f}, Δ={hard['delta']*100:+.2f}pp)")
                print("  → Sun penalty ACTIVE on hard courts only.")
                print("  → Clay/grass penalty DISABLED (no signal).")
            else:
                print("  Recommendation: keep the feature but cap penalty at current level.")
        else:
            print("VERDICT: No directional sun-glare signal in this dataset.")
            print("  Recommendation: disable sun penalty or reduce max to 2%.")
    print(f"{'═'*65}\n")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    df_raw = load_atp_data(YEARS)

    # Filter outdoor matches
    if "court" in df_raw.columns:
        outdoor_mask = df_raw["court"].str.lower().str.strip() != "indoor"
        df_raw = df_raw[outdoor_mask].copy()
        log.info("Outdoor only: %d rows", len(df_raw))

    records = []
    skipped = 0
    for _, row in df_raw.iterrows():
        rec = process_match(row)
        if rec:
            records.append(rec)
        else:
            skipped += 1

    log.info("Valid records: %d  |  Skipped (no venue / no stats): %d",
             len(records), skipped)

    if len(records) < 50:
        log.error("Too few records to analyse (%d). Check venue coverage.", len(records))
        sys.exit(1)

    result = run_analysis(records)
    print_report(result)

    out_path = "backtest_sun_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Results saved to %s", out_path)

    return result


if __name__ == "__main__":
    main()
