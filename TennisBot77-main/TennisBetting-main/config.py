"""
config.py — Central configuration for the betting bot
Fill in API keys and tune constants before running.
"""

import os
import re
import json
import textwrap
from dataclasses import dataclass, field
from typing import Optional


def normalize_kalshi_pem(pem: str) -> str:
    """
    Fix PEM strings that fail cryptography's loader: CRLF, collapsed
    whitespace, and re-wrap base64 to 64-char lines.

    Do not merge lines that start with '+': base64 uses '+' and PEM wraps
    at 64 chars, so a line may legitimately begin with '+' (stripping it
    corrupts the key and breaks loading).
    """
    if not pem or not pem.strip():
        return pem
    pem = pem.strip().replace("\r\n", "\n").replace("\r", "\n")

    m = re.search(
        r"(-----BEGIN (?:RSA )?PRIVATE KEY-----)\s*(.*?)\s*(-----END (?:RSA )?PRIVATE KEY-----)",
        pem,
        re.DOTALL | re.IGNORECASE,
    )
    if m:
        begin, body, end = m.group(1), m.group(2), m.group(3)
        clean = re.sub(r"\s+", "", body)
        wrapped = "\n".join(textwrap.wrap(clean, 64))
        return f"{begin}\n{wrapped}\n{end}"

    if "-----BEGIN" not in pem:
        # Standardize Base64 (remove potential whitespace/newlines/junk)
        clean = re.sub(r"\s+", "", pem)
        wrapped = "\n".join(textwrap.wrap(clean, 64))
        # Default to RSA Private Key header as it's most common for Kalshi's .pem files
        return f"-----BEGIN RSA PRIVATE KEY-----\n{wrapped}\n-----END RSA PRIVATE KEY-----"
    
    return pem


local_api_key = os.getenv("KALSHI_API_KEY_ID", "")
local_pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
local_max_bet = 250.0  # default allocation
local_use_prod = os.getenv("KALSHI_USE_PROD", "false").lower() == "true"
if os.path.exists("kalshi_keys.json"):
    try:
        with open("kalshi_keys.json", "r") as f:
            kd = json.load(f)
            local_api_key = kd.get("api_key_id", local_api_key)
            local_max_bet = float(kd.get("max_bet_usdc", 250.0))
            raw_pem = kd.get("private_key_pem", local_pem)
            if raw_pem:
                local_pem = normalize_kalshi_pem(raw_pem)
            # Allow kalshi_keys.json to control prod vs demo
            if "use_prod" in kd:
                local_use_prod = bool(kd["use_prod"])
    except Exception:
        pass

@dataclass
class Config:
    # -- Kalshi -----------------------------------------------------------------
    # Set this to True to trade on the real exchange (requires production keys).
    # Can be set via KALSHI_USE_PROD env var OR "use_prod": true in kalshi_keys.json
    KALSHI_USE_PROD: bool        = local_use_prod
    
    # Production (Known working endpoint on this system)
    KALSHI_PROD_BASE: str        = "https://api.elections.kalshi.com/trade-api/v2"
    KALSHI_DEMO_BASE: str        = "https://demo-api.kalshi.co/trade-api/v2"
    
    # Discovery URL — Reverting to demo if production fails resolve
    KALSHI_PUBLIC_EVENT_BASE: str = "https://api.elections.kalshi.com/trade-api/v2"
    
    KALSHI_API_URL: str          = KALSHI_PROD_BASE if KALSHI_USE_PROD else KALSHI_DEMO_BASE
    
    KALSHI_API_KEY_ID: str       = local_api_key
    KALSHI_PRIVATE_KEY_PEM: str  = local_pem
    # Scan open events (paginated) for tennis / matchup-style markets
    KALSHI_EVENTS_MAX_PAGES: int = int(os.getenv("KALSHI_EVENTS_MAX_PAGES", "30"))

    # ── Polymarket (Deprecated for Kalshi) ─────────────────────────────────────────────────────────────
    POLY_PRIVATE_KEY: str        = os.getenv("POLY_PRIVATE_KEY", "")
    POLY_API_URL: str            = "https://clob.polymarket.com"
    POLY_GAMMA_URL: str          = "https://gamma-api.polymarket.com"
    POLY_FUNDER_ADDRESS: str     = os.getenv("POLY_FUNDER_ADDRESS", "")

    # Slippage tolerance (0.02 = 2%)
    MAX_SLIPPAGE: float          = 0.02

    # ── OpenCV / Video ─────────────────────────────────────────────────────────
    # 0 = webcam, or path to video file, or RTSP URL
    VIDEO_SOURCE: any            = 0
    FRAME_RATE: int              = 30
    # Confidence threshold for player detection
    CV_CONFIDENCE: float         = 0.60

    # ── Elo Engine ─────────────────────────────────────────────────────────────
    BASE_ELO: float              = 1500.0
    K_FACTOR: float              = 32.0           # Standard chess K

    # Event multipliers — tune per sport
    ELO_MULTIPLIERS: dict        = field(default_factory=lambda: {
        # Basketball scoring
        "3pt_made":              2.5,
        "2pt_made":              1.5,
        "free_throw_made":       0.8,
        "turnover":             -1.2,
        "steal":                 1.0,
        "block":                 0.9,
        "foul":                 -0.5,
        "offensive_rebound":     1.1,
        "defensive_rebound":     0.7,

        # Lineup / substitution
        "star_player_in":        2.0,
        "star_player_out":      -2.0,
        "bench_player_in":       0.5,
        "bench_player_out":     -0.3,
        "injury_substitution":  -3.5,

        # Game state
        "halftime_switch":       1.2,   # teams switch sides
        "momentum_shift":        1.8,   # 3+ consecutive scoring run
        "timeout_called":       -0.4,
        "technical_foul":       -1.0,
        "ejection":             -5.0,

        # H2H / historical
        "h2h_dominance":         1.5,   # A historically dominates B
        "h2h_underdog":         -1.0,
    })

    # Score difference threshold to apply "blowout" multiplier after N minutes
    BLOWOUT_POINT_DIFF: int      = 20
    BLOWOUT_TIME_THRESHOLD: int  = 20    # minutes into game
    BLOWOUT_ELO_SHIFT: float     = 50.0  # raw Elo points added to leader

    # ── Betting ────────────────────────────────────────────────────────────────
    # Minimum edge over market price before betting (0.03 = 3%)
    # At 1% edge, market noise dominates and Kalshi fees eat the profit.
    # 3% is the practical floor for consistent positive EV on this market.
    MIN_EDGE: float              = 0.08

    # Kelly fraction (0.25 = quarter Kelly — conservative)
    KELLY_FRACTION: float        = 0.25

    # Max single bet size in USDC
    MAX_BET_USDC: float          = local_max_bet

    # Min bet size in USDC
    MIN_BET_USDC: float          = 5.0 # lowered minimum bet so it evaluates more often on demo

    # Max total exposure per game in USDC
    MAX_GAME_EXPOSURE: float     = local_max_bet * 4

    # Extreme-odds guard: never bet a side whose market price is outside this band.
    # Any split beyond 60-40 is a market expressing a clear favorite. At these
    # odds the Markov model's output is unreliable (model noise >> true edge) and
    # the expected-value math flips negative after fees. Trades only happen in the
    # 40c–60c range where the market itself is uncertain.
    EXTREME_ODDS_MIN: float      = 0.40
    EXTREME_ODDS_MAX: float      = 0.60

    # Maximum allowed divergence between model probability and market price.
    # With a tight 40-60 band the model and market should never disagree by more
    # than 15 points on a valid signal. Larger divergences mean the model is
    # miscalibrated or the score state hasn't propagated correctly — skip.
    MAX_MODEL_DIVERGENCE: float  = 0.15

    # Scaling factors that convert a historical H2H win rate into Markov
    # serve/return point probabilities. The old value (0.1) was far too flat —
    # a player with a 90% H2H rate got p_serve=0.69 vs 0.65 for a 50-50 match,
    # barely any difference, causing large model-market gaps at extreme odds.
    # 0.25 spans the realistic ATP range (p_serve ~0.62–0.75, p_return ~0.28–0.48).
    MARKOV_SERVE_SCALE: float    = 0.25
    MARKOV_RETURN_SCALE: float   = 0.25



    # ── Historical Analyzer ─────────────────────────────────────────────────────
    # Seasons of H2H data to consider
    H2H_SEASONS_BACK: int        = 3

    # Win-rate threshold to trigger H2H Elo manipulation (0.65 = 65%)
    H2H_DOMINANCE_THRESHOLD: float = 0.65

    # -- Live Scores -----------------------------------------------------------
    LIVESCORE_API_URL: str      = "https://prod-cdn-mev-api.livescore.com/v1/api/app/live/tennis/-5?countryCode=US&locale=en"
    LIVESCORE_POLL_INTERVAL: float = 10.0  # seconds between polls

