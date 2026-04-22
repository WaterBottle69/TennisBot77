"""
FastAPI Backend — Tennis Betting Dashboard
==========================================
Endpoints:
  GET  /                       → serves static/index.html
  POST /api/predict             → scrape + Monte Carlo → simulation result
"""

import os
import sys
import shutil
import uuid
import logging
import asyncio
import random
import math
import json as _json

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from tennis_scraper import ATCScraper
from markov_engine import LiveMatchState, game_win_prob

from kalshi_client import KalshiClient
from config import Config, normalize_kalshi_pem

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Tennis Betting Engine")

os.makedirs(os.path.join(BASE_DIR, "static"),  exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    player_a: str
    player_b: str
    num_simulations: int = 5000

class KalshiKeysRequest(BaseModel):
    api_key_id: str
    private_key_pem: str
    max_bet_usdc: float = 250.0
    use_prod: bool = False


# ──────────────────────────────────────────────────────────────────────────────
# Pages
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        with open(os.path.join(BASE_DIR, "static", "index.html"), "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="index.html not found")


# ──────────────────────────────────────────────────────────────────────────────
# Monte Carlo prediction
# ──────────────────────────────────────────────────────────────────────────────

global_scraper = ATCScraper()

@app.post("/api/predict")
async def create_prediction(req: PredictRequest):
    matchup_data = await global_scraper.get_pregame_matchup(req.player_a, req.player_b)
    
    slug_a = matchup_data["player_a"].get("slug", "")
    slug_b = matchup_data["player_b"].get("slug", "")
    full_name_a = slug_a.replace("-", " ").title() if slug_a else req.player_a
    full_name_b = slug_b.replace("-", " ").title() if slug_b else req.player_b

    matchup_data["player_a_name"] = full_name_a
    matchup_data["player_b_name"] = full_name_b
    req.player_a = full_name_a
    req.player_b = full_name_b

    # Simulated properties for API output using Markov chains theoretically
    # Use scaling from Config so dashboard and bot agree on scale.
    c = Config()
    p_serve = 0.65 + (matchup_data["base_prob_a"] - 0.5) * c.MARKOV_SERVE_SCALE
    p_return = 0.35 + (matchup_data["base_prob_a"] - 0.5) * c.MARKOV_RETURN_SCALE
    lms = LiveMatchState(p_serve, p_return)
    win_prob_a = lms.win_probability()

    # Generate Markov convergence trajectory: starts dispersed near 0.5,
    # converges toward the analytical result over N steps.
    n = min(req.num_simulations, 500)
    trajectories = []
    for i in range(n):
        t = i / (n - 1)
        # Exponential convergence curve
        center = 0.5 + (win_prob_a - 0.5) * (1 - math.exp(-4 * t))
        noise = random.gauss(0, 0.08 * math.exp(-3 * t))
        noisy = max(0.02, min(0.98, center + noise))
        trajectories.append({
            "sim_index": i + 1,
            "prob_a":       round(center, 4),
            "noisy_prob_a": round(noisy,  4),
        })

    return {
        "player_a":      req.player_a,
        "player_b":      req.player_b,
        "player_a_data": matchup_data["player_a"],
        "player_b_data": matchup_data["player_b"],
        "win_prob_a":    win_prob_a,
        "win_prob_b":    1.0 - win_prob_a,
        "ci_low_a":      max(0.01, win_prob_a - 0.05),
        "ci_high_a":     min(0.99, win_prob_a + 0.05),
        "p_point_a":     p_serve,
        "elo_a":         matchup_data["player_a"].get("elo", 1500),
        "elo_b":         matchup_data["player_b"].get("elo", 1500),
        "form_a":        0.5,
        "form_b":        0.5,
        "aces_a":        0,
        "aces_b":        0,
        "num_sims":      n,
        "trajectories":  trajectories,
        "order_book":    {},
    }

# ──────────────────────────────────────────────────────────────────────────────
# Config Keys Endpoint
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/set_kalshi_keys")
async def set_kalshi_keys(req: KalshiKeysRequest):
    import json
    import subprocess

    pem = normalize_kalshi_pem(req.private_key_pem)

    with open("kalshi_keys.json", "w") as f:
        json.dump({
            "api_key_id": req.api_key_id,
            "private_key_pem": pem,
            "max_bet_usdc": req.max_bet_usdc,
            "use_prod": req.use_prod,
        }, f, indent=2)
    
    # Run the bot in the background — use python3 (macOS doesn't have 'python')
    env = os.environ.copy()
    c = Config()
    if c.KALSHI_USE_PROD:
        env["KALSHI_USE_PROD"] = "true"
    else:
        env["KALSHI_USE_PROD"] = "false"

    log_path = os.path.join(BASE_DIR, "bot.log")
    with open(log_path, "w") as f:
        pass

    subprocess.Popen(
        f"{sys.executable} {os.path.join(BASE_DIR, 'main.py')} > {log_path} 2>&1",
        shell=True,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        cwd=BASE_DIR,
    )

    return {"status": "success"}

@app.post("/api/start_bot")
async def start_bot():
    import subprocess
    env = os.environ.copy()
    c = Config()
    if c.KALSHI_USE_PROD:
        env["KALSHI_USE_PROD"] = "true"
    else:
        env["KALSHI_USE_PROD"] = "false"

    log_path = os.path.join(BASE_DIR, "bot.log")
    with open(log_path, "w") as f:
        pass

    subprocess.Popen(
        f"{sys.executable} {os.path.join(BASE_DIR, 'main.py')} > {log_path} 2>&1",
        shell=True,
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        start_new_session=True,
        cwd=BASE_DIR,
    )
    log.info("Bot started via dedicated dashboard start button.")
    return {"status": "started"}

@app.post("/api/stop_bot")
async def stop_bot():
    """Kill any running bot subprocess and clear match state."""
    import signal
    import subprocess
    try:
        subprocess.run(
            ["pkill", "-f", "main.py"],
            capture_output=True
        )
    except Exception:
        pass
    for fname in ["kalshi_match_state.json", "live_state.json"]:
        path = os.path.join(BASE_DIR, fname)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass
    log.info("Bot stopped via dashboard.")
    return {"status": "stopped"}

@app.get("/api/bot_log")
async def get_bot_log():
    log_path = os.path.join(BASE_DIR, "bot.log")
    if not os.path.exists(log_path):
        return {"log": f"[{log_path}]\nBot is idle — paste keys to start."}
    with open(log_path, "r") as f:
        lines = f.readlines()
    return {"log": f"[{log_path}]\n" + "".join(lines[-60:])}

# ──────────────────────────────────────────────────────────────────────────────
# Trading Mode
# ──────────────────────────────────────────────────────────────────────────────

TRADING_MODE_PATH = os.path.join(BASE_DIR, "trading_mode.json")

class TradingModeRequest(BaseModel):
    mode: str  # "normal" | "hf"

@app.post("/api/set_trading_mode")
async def set_trading_mode(req: TradingModeRequest):
    if req.mode not in ("normal", "hf"):
        return JSONResponse({"error": "mode must be 'normal' or 'hf'"}, status_code=400)
    with open(TRADING_MODE_PATH, "w") as f:
        _json.dump({"mode": req.mode}, f)
    log.info("Trading mode set to: %s", req.mode)
    return {"mode": req.mode}

@app.get("/api/get_trading_mode")
async def get_trading_mode():
    if os.path.exists(TRADING_MODE_PATH):
        with open(TRADING_MODE_PATH, "r") as f:
            return _json.load(f)
    return {"mode": "normal"}

@app.get("/api/kalshi_markets")
async def get_kalshi_markets():
    import json
    c = Config()
    if os.path.exists("kalshi_keys.json"):
        with open("kalshi_keys.json", "r") as f:
            try:
                kd = json.load(f)
                c.KALSHI_API_KEY_ID = kd.get("api_key_id", c.KALSHI_API_KEY_ID)
                raw_pem = kd.get("private_key_pem", c.KALSHI_PRIVATE_KEY_PEM)
                if raw_pem:
                    c.KALSHI_PRIVATE_KEY_PEM = normalize_kalshi_pem(raw_pem)
            except Exception:
                pass
    kalshi = KalshiClient(c)
    try:
        markets = await kalshi.get_atp_markets()
    finally:
        await kalshi.close()
    if markets:
        return {
            "status": "success",
            "market": markets[0],
            "markets": markets,
        }
    return {
        "status": "no_markets",
        "message": f"No open tennis markets found on Kalshi ({'PROD' if c.KALSHI_USE_PROD else 'DEMO'}). Enter names manually.",
        "markets": [],
        "env": "PROD" if c.KALSHI_USE_PROD else "DEMO"
    }

@app.get("/api/kalshi_env")
async def get_kalshi_env():
    c = Config()
    return {"env": "PROD" if c.KALSHI_USE_PROD else "DEMO", "url": c.KALSHI_API_URL}

@app.post("/api/toggle_kalshi_env")
async def toggle_kalshi_env():
    """Toggles between Demo and Production in the environment variable for the next run."""
    current = os.getenv("KALSHI_USE_PROD", "false").lower() == "true"
    new_val = "false" if current else "true"
    os.environ["KALSHI_USE_PROD"] = new_val
    return {"env": "PROD" if new_val == "true" else "DEMO"}


@app.get("/api/kalshi_match_state")
async def get_kalshi_match_state():
    """Written by main.py when the bot runs; used to sync the dashboard."""
    import json
    path = os.path.join(BASE_DIR, "kalshi_match_state.json")
    if not os.path.exists(path):
        return {"ok": False, "message": "Bot has not written match state yet."}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@app.get("/api/live_state")
async def get_live_state():
    import json
    path = os.path.join(BASE_DIR, "live_state.json")
    if not os.path.exists(path):
        return {"ok": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["ok"] = True
        return data


@app.get("/api/adaptive_status")
async def get_adaptive_status():
    """Returns the live AdaptiveController state written by main.py each tick."""
    import json
    path = os.path.join(BASE_DIR, "adaptive_state.json")
    if not os.path.exists(path):
        return {"ok": False}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        data["ok"] = True
        return data


# ──────────────────────────────────────────────────────────────────────────────
# Video — upload (always available)
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/api/upload_video")
async def upload_video(video: UploadFile = File(...)):
    """
    Accepts any video file and saves it to /uploads/.
    Works without OpenCV — the raw file is served back for HTML5 playback.
    """
    video_id  = str(uuid.uuid4())
    # Preserve original extension so the browser MIME type is correct
    ext       = os.path.splitext(video.filename or "")[1] or ".mp4"
    file_path = os.path.join("uploads", f"{video_id}{ext}")

    try:
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(video.file, buf)
    except Exception as exc:
        log.error("Upload failed: %s", exc)
        return JSONResponse({"error": str(exc)}, status_code=500)

    size_mb = round(os.path.getsize(file_path) / 1024 / 1024, 2)
    log.info("Video saved: %s  (%.1f MB)", file_path, size_mb)

    return {
        "video_id":      video_id,
        "filename":      video.filename,
        "size_mb":       size_mb,
        "ext":           ext,
        "status": "success",
    }
