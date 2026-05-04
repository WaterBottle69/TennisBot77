"""
config.py — Central configuration for the betting bot
Fill in API keys and tune constants before running.
"""

import os
import re
import json
import textwrap
from dataclasses import dataclass, field


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
        clean = re.sub(r"\s+", "", pem)
        wrapped = "\n".join(textwrap.wrap(clean, 64))
        return f"-----BEGIN RSA PRIVATE KEY-----\n{wrapped}\n-----END RSA PRIVATE KEY-----"
    
    return pem


local_api_key = os.getenv("KALSHI_API_KEY_ID", "")
local_pem = os.getenv("KALSHI_PRIVATE_KEY_PEM", "")
local_max_bet = 250.0  # default allocation
local_use_prod = os.getenv("KALSHI_USE_PROD", "false").lower() == "true"
local_sportradar_key = os.getenv("SPORTRADAR_API_KEY", "")
local_flaresolverr_url = os.getenv("FLARESOLVERR_URL", "http://localhost:8191/v1")
local_api_tennis_key = os.getenv("API_TENNIS_KEY", "")
_kd: dict = {}
if os.path.exists("kalshi_keys.json"):
    try:
        with open("kalshi_keys.json", "r") as f:
            _kd = json.load(f)
            local_api_key = _kd.get("api_key_id", local_api_key)
            local_max_bet = float(_kd.get("max_bet_usdc", 250.0))
            raw_pem = _kd.get("private_key_pem", local_pem)
            if raw_pem:
                local_pem = normalize_kalshi_pem(raw_pem)
            # Allow kalshi_keys.json to control prod vs demo
            if "use_prod" in _kd:
                local_use_prod = bool(_kd["use_prod"])
            if _kd.get("sportradar_api_key"):
                local_sportradar_key = _kd["sportradar_api_key"]
            # FlareSolverr URL — set this if running FlareSolverr on a different machine
            # e.g. "flaresolverr_url": "http://192.168.1.50:8191/v1"
            if _kd.get("flaresolverr_url"):
                local_flaresolverr_url = _kd["flaresolverr_url"]
            if _kd.get("api_tennis_key"):
                local_api_tennis_key = _kd["api_tennis_key"]
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
    KALSHI_EVENTS_MAX_PAGES: int = int(os.getenv("KALSHI_EVENTS_MAX_PAGES", "5"))

    # ── Polymarket (Deprecated for Kalshi) ─────────────────────────────────────────────────────────────
    POLY_PRIVATE_KEY: str        = os.getenv("POLY_PRIVATE_KEY", "")
    POLY_API_URL: str            = "https://clob.polymarket.com"
    POLY_GAMMA_URL: str          = "https://gamma-api.polymarket.com"
    POLY_FUNDER_ADDRESS: str     = os.getenv("POLY_FUNDER_ADDRESS", "")

    # Slippage tolerance (0.02 = 2%)
    MAX_SLIPPAGE: float          = 0.02

    # Kalshi per-contract trading fee: ceil(0.07 * P * (1-P) * 100) cents at entry.
    # We subtract 0.07 * P * (1-P) from raw edge so the bot only enters when the
    # post-fee EV is still positive.
    KALSHI_FEE_RATE: float       = 0.07

    # Model-reversal exit: we exit an open position once edge drops below
    # this much BELOW zero. Must be larger than the entry threshold or the
    # bot whipsaws in and out on noise.
    MODEL_REVERSAL_EXIT_EDGE: float = 0.015

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

    })

    # Score difference threshold to apply "blowout" multiplier after N minutes
    BLOWOUT_POINT_DIFF: int      = 20
    BLOWOUT_TIME_THRESHOLD: int  = 20    # minutes into game
    BLOWOUT_ELO_SHIFT: float     = 50.0  # raw Elo points added to leader

    # ── Betting ────────────────────────────────────────────────────────────────
    # No minimum edge — Kelly formula naturally returns 0 when there is no edge.
    MIN_EDGE: float              = 0.0

    # Ranking threshold gate: block any match where BOTH players are ranked
    # below this number. The XGBoost model was trained primarily on ATP top-200
    # data — its differential signal for rank-400+ Challenger players is
    # extrapolation, not learned pattern. Kalshi prices at this level reflect
    # real form data the model simply cannot see. Set to 0 to disable.
    MIN_PLAYER_RANK_THRESHOLD: int = 200

    # Alpha surface ROI gate disabled — no trade history yet to build surface from.
    MIN_ROI_THRESHOLD: float     = -1.0

    # Kelly fraction — global multiplier applied ON TOP of the confidence tier system.
    # The tier system in bet_manager._kelly_size already scales:
    #   >= 70% model confidence → 0.40x Kelly
    #   >= 60%                  → 0.25x Kelly
    #   >= 55%                  → 0.12x Kelly
    #    < 55%                  → 0.05x Kelly
    # This multiplier lets you dial all tiers up/down uniformly.
    # 1.0 = use tier fractions as-is. 0.5 = halve all tiers (more conservative).
    KELLY_FRACTION: float        = 1.0

    # Max single bet size in USDC
    MAX_BET_USDC: float          = local_max_bet

    # Min bet size in USDC
    MIN_BET_USDC: float          = 1.0

    # Max total exposure per game in USDC
    MAX_GAME_EXPOSURE: float     = local_max_bet * 4

    # No extreme-odds filter — bet any price the model finds edge on.
    EXTREME_ODDS_MIN: float      = 0.01
    EXTREME_ODDS_MAX: float      = 0.99

    # No divergence filter — model is allowed to strongly disagree with market.
    MAX_MODEL_DIVERGENCE: float  = 0.99

    MARKOV_SERVE_SCALE: float    = 0.25
    MARKOV_RETURN_SCALE: float   = 0.25

    # ── Age × Temperature edge ──────────────────────────────────────────────────
    # WFO-validated on 17,258 outdoor ATP Masters matches (1991–2024).
    # Permutation p=0.000, bootstrap 95% CI [+0.17, +0.37] on standardised coef.
    # Survives within-surface controls (Hard and Clay independently significant).
    #
    # Formula applied once per match (after venue + weather resolved):
    #   logit_adj = AGE_TEMP_AGE_COEF * (p1_age - p2_age)
    #             + AGE_TEMP_TEMP_COEF * temp_celsius * (p1_age - p2_age)
    #   base_p_a_adj = sigmoid(logit(base_p_a) + logit_adj)
    #
    # Effect flips direction at ~30°C:
    #   < 30°C: older player slightly disadvantaged  (clay/spring venues)
    #   > 30°C: older player slightly advantaged     (summer hard courts)
    AGE_TEMP_ENABLED:   bool  = True
    AGE_TEMP_AGE_COEF:  float = -0.07239   # raw logit coef for (p1_age - p2_age)
    AGE_TEMP_TEMP_COEF: float = +0.00240   # raw logit coef for temp_c × age_diff
    AGE_TEMP_MAX_ADJ:   float = 0.06       # cap adjustment at ±6% win-prob shift

    # ── Physical & Structural Edges ─────────────────────────────────────────────
    # WFO-validated on 17,258 outdoor ATP Masters matches (1991–2024).
    # All signals pass permutation test (p<0.05) and decade-stability checks.
    # Applied once per match in logit-space, capped at ±MAX_ADJ total shift.
    #
    # [1] Altitude × Height (Hard courts)  — p=0.0000, OOS ROI +11.52%
    #   Thinner air at altitude inflates serve speed; taller servers gain more.
    #   Effect confined to hard courts (clay ball-bounce absorbs the speed gain).
    #   Formula: ALT_HT_HARD_COEF * altitude_m * (p1_ht - p2_ht) * is_hard
    #
    # [2] Altitude × Age  — p=0.0000, OOS ROI +11.53%
    #   Aerobic capacity (VO2max) declines with age; thinner air amplifies this.
    #   Formula: ALT_AGE_COEF * altitude_m * (p1_age - p2_age)
    #
    # [3] Handedness mismatch — Hard: p=0.0000 (+10.6%),  Clay: p=0.0000 (+15.9%)
    #   Left-handers exploit deuce-court wide serve against RH backhands.
    #   Effect is stronger on clay (slower ball → more time to exploit the angle).
    #   lh_net = +1 if p1=LH & p2=RH,  -1 if p1=RH & p2=LH,  0 otherwise
    #   Formula: LH_HARD_COEF * lh_net * is_hard + LH_CLAY_COEF * lh_net * is_clay
    #
    # [4] Rank-points divergence from rank  — p=0.0000, OOS ROI +22.25%
    #   When a player's ranking points are higher than their ordinal rank implies,
    #   they are "better" than market odds (anchored to rank) reflect.
    #   Formula: PTS_RANK_COEF * (log(pts_a/pts_b) - log(rank_b/rank_a))
    PHYS_ENABLED:        bool  = True
    ALT_HT_HARD_COEF:    float = +0.000048  # altitude(m) × height_diff(cm) × is_hard
    ALT_AGE_COEF:        float = +0.000037  # altitude(m) × age_diff(yrs)
    LH_HARD_COEF:        float = -0.097593  # lh_net × is_hard  (negative = LH wins more)
    LH_CLAY_COEF:        float = +0.054036  # lh_net × is_clay
    PTS_RANK_COEF:       float = +0.980095  # log(pts_a/pts_b) - log(rank_b/rank_a)
    PHYS_MAX_ADJ:        float = 0.08       # cap total physical adjustment at ±8%

    # ── Serve Quality Edges ─────────────────────────────────────────────────────
    # WFO-validated on 3.4M ATP matches (2010–2024).
    # Both signals pass permutation test (p=0.000) and expanding-window WFO.
    #
    # [1] 2nd-serve won % differential — p=0.000, OOS ROI +2.24%, MaxDD 2.6%
    #   Players who consistently win more 2nd-serve points are undervalued by
    #   rank-anchored market odds. Effect is robust across all years tested.
    #   Formula: SERVE2_COEF * (p1_2nd_won_pct - p2_2nd_won_pct)
    #
    # [2] Break-point save rate differential — p=0.000, OOS ROI +2.86%
    #   Players who save break points at a higher rate demonstrate clutch
    #   performance under pressure, a persistent skill signal.
    #   Formula: BP_SAVE_COEF * (p1_bp_save_rate - p2_bp_save_rate)
    #
    # Both applied in logit-space, capped at ±SERVE_QUALITY_MAX_ADJ total shift.
    SERVE_QUALITY_ENABLED: bool  = True
    SERVE2_COEF:           float = 3.0    # logit coef for 2nd-serve won % diff
    BP_SAVE_COEF:          float = 2.9    # logit coef for BP save rate diff
    SERVE_QUALITY_MAX_ADJ: float = 0.05   # cap at ±5% win-prob shift

    # -- Live Scores -----------------------------------------------------------
    LIVESCORE_API_URL: str      = "https://prod-cdn-mev-api.livescore.com/v1/api/app/live/tennis/-5?countryCode=US&locale=en"
    LIVESCORE_DAILY_URL: str    = "https://prod-cdn-mev-api.livescore.com/v1/api/app/date/tennis/{date}/0?countryCode=US&locale=en"
    LIVESCORE_POLL_INTERVAL: float = 10.0  # seconds between polls

    # -- SportRadar Tennis v3 -------------------------------------------------
    # Free sandbox trial: https://developer.sportradar.com
    # Add your key to kalshi_keys.json as "sportradar_api_key": "YOUR_KEY"
    # or set the SPORTRADAR_API_KEY environment variable.
    SPORTRADAR_API_KEY: str     = local_sportradar_key
    SPORTRADAR_ENABLED: bool    = True   # set False to skip WS and fall back to polling
    SPORTRADAR_WS_URL: str      = "wss://api.sportradar.com/tennis/trial/v3/en/stream/events/subscribe"
    SPORTRADAR_REST_BASE: str   = "https://api.sportradar.com/tennis/trial/v3/en"

    # ── FlareSolverr (Cloudflare bypass) ───────────────────────────────────────
    # FlareSolverr is a self-hosted proxy that uses a real Chrome browser to solve
    # Cloudflare Managed Challenges — the kind that blocks aiohttp, curl_cffi,
    # and headless Playwright on tennisstats.com.
    #
    # Start it once with Docker (image is ~300 MB, container uses ~200 MB RAM):
    #   docker run -d --name flaresolverr -p 8191:8191 \
    #     ghcr.io/flaresolverr/flaresolverr:latest
    #
    # If this URL is unreachable the scraper falls back to Playwright automatically.
    FLARESOLVERR_URL: str       = local_flaresolverr_url
    API_TENNIS_KEY: str         = local_api_tennis_key

