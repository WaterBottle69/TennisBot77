# TennisBot77 — Implementation Plan (Co-Worker Handoff)

> **Audience**: Developer onboarding to this codebase.  
> **Goal**: Complete the remaining code fixes, verify correctness in DEMO mode, then deploy to production on Kalshi.  
> Each task has an **Acceptance Criteria** block you can run to confirm it is done.

---

## Quick Reference

| File | Role |
|------|------|
| `main.py` | Top-level orchestrator — starts match workers, polls live scores |
| `bet_manager.py` | Edge calculation, Kelly sizing, order entry/exit |
| `kalshi_client.py` | Kalshi REST + WebSocket API wrapper |
| `markov_engine.py` | Markov DP win-probability engine + `LiveMatchState` |
| `config.py` | All tunable constants and API endpoints |
| `live_score_scraper.py` | LiveScore + Flashscore polling |
| `server.py` | FastAPI dashboard (separate process from the bot) |
| `kalshi_keys.json` | **Not in repo** — you must create this (see Task 2) |

---

## Task 1 — Environment Setup

### Steps

1. Confirm Python 3.9 is available via the project venv:
   ```bash
   source test_venv/bin/activate
   python --version   # must be 3.9.x
   ```

2. Install all dependencies into the venv:
   ```bash
   pip install -r requirements.txt
   pip install websockets cloudscraper   # not in requirements.txt yet
   ```

3. Add the two missing packages to `requirements.txt`:
   ```
   websockets
   cloudscraper
   ```

4. Verify the ML models load correctly:
   ```bash
   python -c "from ml_engine import ml_engine; print('ML engine OK')"
   ```
   You may see sklearn version warnings about `best_xgb_model.json` — these are safe to ignore unless predictions are wrong.

### Acceptance Criteria
```bash
source test_venv/bin/activate
python -c "
from config import Config
from kalshi_client import KalshiClient
from markov_engine import LiveMatchState
from bet_manager import BetManager
print('All imports OK')
"
# Expected output: All imports OK  (no ImportError)
```

---

## Task 2 — Kalshi Credentials (DEMO first, then PROD)

The bot reads credentials from `kalshi_keys.json` in the project root. This file is **never committed to git** (listed in `.gitignore`).

### Steps

1. Log in to the Kalshi **Demo** exchange at `https://demo.kalshi.co`
2. Navigate to **Account → API Keys** and generate an RSA key pair
3. Create `kalshi_keys.json` in the project root:
   ```json
   {
     "api_key_id": "YOUR-KEY-ID-HERE",
     "private_key_pem": "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----",
     "max_bet_usdc": 50.0,
     "use_prod": false
   }
   ```
   - `max_bet_usdc`: Maximum single-trade size in USD — keep at **$50 for DEMO testing**
   - `use_prod`: **Must be `false` for all DEMO work**

4. Confirm the bot authenticates:
   ```bash
   python -c "
   from config import Config
   from kalshi_client import KalshiClient
   import asyncio

   async def test():
       c = Config()
       k = KalshiClient(c)
       bal = await k.get_balance()
       print(f'Balance: \${bal:.2f}')
       await k.close()

   asyncio.run(test())
   "
   ```

### Acceptance Criteria
```
# Expected (DEMO mode with keys):
KalshiClient initialized in AUTHENTICATED mode (DEMO).
Balance: $1000.00          # or whatever your demo balance is

# Expected (no keys / DRY-RUN):
Kalshi credentials not found. Running in DRY-RUN mode on DEMO.
Balance: $1000.00          # hardcoded DRY-RUN default
```

---

## Task 3 — Fix Scale-Out Double-Selling Bug (Code Change Required)

### Background

In `bet_manager.py` `_check_exits()`, the tiered scale-out block (lines ~725–745) sells a 25% tranche and then `break`s out of the inner loop — but execution falls through to the **Model Reversal Guard** check immediately below. If both conditions are true simultaneously (price at 2× AND model edge gone), `_sell_partial` AND `_sell_position` both fire on the same position in one tick. This over-sells and corrupts `_total_bet_usdc`.

### Fix

In [bet_manager.py](bet_manager.py), locate the tiered scale-out block and add a `sold_tranche` flag so the reversal guard is skipped when a tranche was just sold:

**Find this block (around line 725):**
```python
            # ── 5. Tiered scale-out (partial sells in LOW / MEDIUM vol) ─────
            if vol_regime != "HIGH" and entry_price > 0:
                gain_ratio = current_price / entry_price
                tranches = pos.get("tranches_sold", 0)
                tranche_targets = [
                    (2.0, "2× (100% gain)"),
                    (3.0, "3× (200% gain)"),
                    (4.0, "4× (300% gain)"),
                ]
                for target_ratio, label in tranche_targets:
                    tier = tranche_targets.index((target_ratio, label))
                    if gain_ratio >= target_ratio and tranches == tier:
                        total_count = pos.get("count", 1)
                        sell_count  = max(1, total_count // 4)  # sell 25%
                        log.info(...)
                        await self._sell_partial(pos_key, pos, current_price, sell_count)
                        pos["tranches_sold"] = tranches + 1
                        break

            # ── 6. Model reversal guard ─────────────────────────────────────
            if current_edge < -self.cfg.MODEL_REVERSAL_EXIT_EDGE:
```

**Change to:**
```python
            # ── 5. Tiered scale-out (partial sells in LOW / MEDIUM vol) ─────
            sold_tranche = False
            if vol_regime != "HIGH" and entry_price > 0:
                gain_ratio = current_price / entry_price
                tranches = pos.get("tranches_sold", 0)
                tranche_targets = [
                    (2.0, "2× (100% gain)"),
                    (3.0, "3× (200% gain)"),
                    (4.0, "4× (300% gain)"),
                ]
                for target_ratio, label in tranche_targets:
                    tier = tranche_targets.index((target_ratio, label))
                    if gain_ratio >= target_ratio and tranches == tier:
                        total_count = pos.get("count", 1)
                        sell_count  = max(1, total_count // 4)  # sell 25%
                        log.info(...)
                        await self._sell_partial(pos_key, pos, current_price, sell_count)
                        pos["tranches_sold"] = tranches + 1
                        sold_tranche = True
                        break

            if sold_tranche:
                continue   # skip reversal guard — partial sell already handled this tick

            # ── 6. Model reversal guard ─────────────────────────────────────
            if current_edge < -self.cfg.MODEL_REVERSAL_EXIT_EDGE:
```

### Acceptance Criteria
```bash
# Run the unit tests to confirm no regressions
python -m pytest test_algorithms.py -v -k "exit or sell or tranche"

# Then manually search the log after a DEMO run for double-sells:
grep -E "SCALE-OUT|EXIT MODEL REVERSAL" bot.log | awk 'NR>1{if (prev_ts==$1) print "DOUBLE SELL at " $1; prev_ts=$1}'
# Expected: no output (no double-sells on same timestamp)
```

---

## Task 4 — Run Full Test Suite

### Steps

1. Run all automated tests:
   ```bash
   source test_venv/bin/activate
   python -m pytest test_algorithms.py -v --tb=short 2>&1 | tee test_output.txt
   ```

2. Check for failures:
   ```bash
   grep -E "FAILED|ERROR|passed|failed" test_output.txt | tail -5
   ```

3. Run the live score test (requires internet):
   ```bash
   python test_live_scores.py
   ```

4. Run the WebSocket connectivity test:
   ```bash
   python test_ws.py
   ```

### Acceptance Criteria
```
# pytest output must show:
X passed, 0 failed
# (some warnings are acceptable — no failures)

# test_live_scores.py must output:
LiveScore API: OK
Daily schedule: OK
```

---

## Task 5 — DEMO Mode End-to-End Smoke Test

Run the bot in DEMO mode for 10 minutes and verify all subsystems.

### Steps

1. Start the bot:
   ```bash
   source test_venv/bin/activate
   python main.py
   ```

2. In a second terminal, tail the log:
   ```bash
   tail -f bot.log
   ```

3. Check for these **expected** log lines within 2 minutes:
   - `KalshiClient initialized in AUTHENTICATED mode (DEMO).` — keys loaded
   - `TENNIS DETECTED:` — at least one market found
   - `--- Processing Match Live (Markov DP)` — match worker started
   - `[WS] Connected to orderbook stream` — WebSocket live (requires auth)
   - `YES mapping verified: PLAYER_X IS YES` — YES/NO mapping resolved
   - `Updating p_serve/p_return based on live hold rate` — Markov updating live

4. Check these lines are **absent** (were bugs, now fixed):
   ```bash
   grep "server rejected WebSocket connection: HTTP 401" bot.log | wc -l
   # Expected: 0   (WS no longer retries without credentials)

   grep "\[STALE\]" bot.log | wc -l
   # Expected: near 0 or very rare   (scheduled matches show UPCOMING)
   ```

5. Verify order payloads are not sending price 1 or 99:
   ```bash
   grep "PLACING BUY" bot.log | grep -v "@[2-9][0-9]c"
   # Expected: empty (all orders at realistic cent prices)
   ```

6. Stop after 10 minutes with `Ctrl+C` and review trade summary:
   ```bash
   grep "TRADE SUMMARY" bot.log -A 10 | tail -15
   ```

### Acceptance Criteria
- No `[ERROR]` lines from `kalshi_client` or `discovery` modules
- At least one `[FILLED]` or `DRY_RUN` buy entry logged
- `_total_bet_usdc` in trade summary is non-negative
- No `Traceback` lines in the log

---

## Task 6 — Dashboard Verification

The FastAPI dashboard runs as a **separate process** from the trading bot.

### Steps

1. Start the bot first (Task 5), then in a new terminal:
   ```bash
   source test_venv/bin/activate
   uvicorn server:app --host 0.0.0.0 --port 8000 --reload
   ```

2. Open `http://localhost:8000` in a browser.

3. Verify these dashboard features:
   - **Win probability bars** update in real-time (pulls from `live_state.json`)
   - **Player names** match the active Kalshi market (pulls from `kalshi_match_state.json`)
   - **Alpha surface plot** renders at `http://localhost:8000/static/alpha_surface.html` (only populated after 5+ completed trades)
   - **Predict endpoint** returns a result:
     ```bash
     curl -s -X POST http://localhost:8000/api/predict \
       -H "Content-Type: application/json" \
       -d '{"player_a": "Djokovic", "player_b": "Alcaraz", "num_simulations": 1000}' \
       | python3 -m json.tool | head -20
     ```

### Acceptance Criteria
```bash
# API health check:
curl -s http://localhost:8000/api/status | python3 -m json.tool
# Expected: {"ok": true, ...} with player names and win probabilities populated
```

---

## Task 7 — Latency Benchmarking

Confirm the pipeline is within target latency budgets.

### Steps

1. After a DEMO run with live matches, analyse the latency CSV:
   ```bash
   python3 -c "
   import pandas as pd
   df = pd.read_csv('latency_metrics.csv', header=None,
                    names=['event', 'status', 'win_a', 'win_b', 'latency_ms', 'cum_pnl'])
   live = df[df['event'] == 'point_won']
   print('Latency stats (ms):')
   print(live['latency_ms'].describe().round(2))
   print(f'p95: {live[\"latency_ms\"].quantile(0.95):.1f}ms')
   print(f'p99: {live[\"latency_ms\"].quantile(0.99):.1f}ms')
   "
   ```

2. Check the E2E order latency in the log:
   ```bash
   grep "LATENCY E2E" bot.log | awk '{print $NF}' | sort -n | tail -10
   ```

### Acceptance Criteria
| Metric | Target |
|--------|--------|
| Eval latency p95 | < 200 ms |
| E2E order latency (Kalshi round-trip) | < 500 ms |
| p99 eval latency | < 800 ms |

If latency is above target, investigate:
- `MarketMonitor` cache is being used (`[CACHE] Using MarketMonitor price` in logs)
- `available_balance` is being passed in (not re-fetched per tick)

---

## Task 8 — Production Deployment

> ⚠️ Only proceed after Tasks 3–7 are fully green in DEMO mode.

### Steps

1. Obtain **Production** Kalshi API keys from `https://api.elections.kalshi.com`

2. Update `kalshi_keys.json`:
   ```json
   {
     "api_key_id": "PROD-KEY-ID",
     "private_key_pem": "-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----",
     "max_bet_usdc": 50.0,
     "use_prod": true
   }
   ```
   - Start with `max_bet_usdc: 50.0` — scale up only after verifying P&L
   - `use_prod: true` switches **both** REST and WebSocket to production endpoints

3. Verify the endpoints have changed:
   ```bash
   python3 -c "
   from config import Config
   c = Config()
   print('API URL:', c.KALSHI_API_URL)
   print('Is PROD:', c.KALSHI_USE_PROD)
   "
   # Expected:
   # API URL: https://api.elections.kalshi.com/trade-api/v2
   # Is PROD: True
   ```

4. Run a single dry-run cycle against production endpoints to confirm market discovery works (no trades will be placed unless credentials are valid):
   ```bash
   KALSHI_USE_PROD=true python3 -c "
   from config import Config
   from kalshi_client import KalshiClient
   import asyncio

   async def test():
       c = Config()
       k = KalshiClient(c)
       markets = await k.get_atp_markets()
       print(f'Found {len(markets)} live tennis markets on PRODUCTION')
       for m in markets[:3]:
           print(f'  {m[\"player_a\"]} vs {m[\"player_b\"]}  — {m[\"ticker\"]}')
       await k.close()

   asyncio.run(test())
   "
   ```

5. Start the bot targeting production:
   ```bash
   source test_venv/bin/activate
   nohup python main.py >> bot.log 2>&1 &
   echo "Bot PID: $!"
   ```

6. Monitor the first 15 minutes closely:
   ```bash
   tail -f bot.log | grep -E "PLACING BUY|FILLED|SELL|ERROR|WARNING"
   ```

### Acceptance Criteria
```bash
# Confirm authenticated against PROD:
grep "AUTHENTICATED mode (PROD)" bot.log | head -1
# Expected: KalshiClient initialized in AUTHENTICATED mode (PROD).

# Confirm first real order used correct slippage price:
grep "PLACING BUY" bot.log | head -5
# Expected: prices between 5c and 95c — never 1c or 99c

# Confirm balance is real:
grep "Balance:" bot.log | head -5
# Expected: matches your actual Kalshi account balance
```

---

## Task 9 — Process Management (Keep Bot Running)

For long-running production use, manage the bot as a system service.

### Using `screen` (simplest):
```bash
screen -S tennisbot
source test_venv/bin/activate
python main.py

# Detach: Ctrl+A then D
# Reattach: screen -r tennisbot
```

### Using `systemd` (recommended for servers):

Create `/etc/systemd/system/tennisbot.service`:
```ini
[Unit]
Description=TennisBot77 Kalshi Betting Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/TennisBot77-main
ExecStart=/path/to/TennisBot77-main/test_venv/bin/python main.py
Restart=on-failure
RestartSec=30
StandardOutput=append:/path/to/TennisBot77-main/bot.log
StandardError=append:/path/to/TennisBot77-main/bot.log

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable tennisbot
sudo systemctl start tennisbot
sudo systemctl status tennisbot
```

### Dashboard service (companion):
```bash
# Add a second service or run alongside:
uvicorn server:app --host 0.0.0.0 --port 8000 &
```

### Acceptance Criteria
```bash
# Bot is running and healthy:
sudo systemctl status tennisbot
# Expected: Active: active (running)

# Log is being written:
tail -5 bot.log
# Expected: recent timestamps (within last 10 seconds in HF mode)
```

---

## Task 10 — Trading Mode Controls

The bot supports two trading modes switchable at runtime without a restart.

| Mode | Edge threshold | Kelly fraction | Max bet | Poll interval |
|------|---------------|----------------|---------|---------------|
| `normal` | 2.0% | 25% | `max_bet_usdc` | 2.0 s |
| `hf` (high-frequency) | 0.3% | 5% | $15 | 0.8 s |

### Switching modes at runtime:
```bash
# Switch to HF mode:
echo '{"mode": "hf"}' > trading_mode.json

# Switch back to normal:
echo '{"mode": "normal"}' > trading_mode.json
```

HF mode **auto-reverts** to normal after `$50` cumulative profit (`HF_PROFIT_THRESHOLD` in `main.py`).

### Acceptance Criteria
```bash
grep "\[HF MODE\]" bot.log | head -3
# Expected: [HF MODE] Edge=0.3%  Odds=[0.01,0.99]  Div=0.99
# (appears within one poll cycle of writing trading_mode.json)
```

---

## Task 11 — Log Monitoring & Alerting

Set up ongoing monitoring so errors are surfaced quickly.

### Real-time error watch:
```bash
tail -f bot.log | grep --line-buffered -E "\[ERROR\]|\[WARNING\]|Traceback|Exception"
```

### Daily summary report:
```bash
# Count errors and key events from today's log:
python3 -c "
import re, collections
from datetime import date

today = str(date.today())
counts = collections.Counter()
with open('bot.log') as f:
    for line in f:
        if today not in line:
            continue
        if '[ERROR]' in line:
            m = re.search(r'\[ERROR\] (\w+):', line)
            counts['ERROR:' + (m.group(1) if m else '?')] += 1
        elif '[FILLED]' in line:
            counts['FILLED'] += 1
        elif 'SELL' in line and 'PnL' in line:
            counts['SELL'] += 1
        elif 'EXIT MODEL REVERSAL' in line:
            counts['REVERSAL_EXIT'] += 1

for k, v in sorted(counts.items()):
    print(f'{k:35s} {v}')
"
```

### Key metrics to monitor:
| Metric | Healthy | Investigate if |
|--------|---------|----------------|
| `[ERROR]` lines per hour | 0 | > 5 |
| `[WS] orderbook stream error` | 0 | any |
| STALE WARNING per hour | < 10 | > 50 |
| `EXIT MODEL REVERSAL` per session | < 5 | > 20 |
| E2E latency p99 | < 500ms | > 1000ms |

---

## Task 12 — Config Tuning (Post Go-Live)

Once the bot has made 20+ live trades, tune these constants in `config.py`:

| Constant | Current | What it does | Tune if |
|----------|---------|--------------|---------|
| `KALSHI_FEE_RATE` | `0.07` | Subtracted from raw edge before entry | Kalshi changes fee schedule |
| `MODEL_REVERSAL_EXIT_EDGE` | `0.015` | Exit when edge drops 1.5% below zero | Too many premature exits → raise; holding losers too long → lower |
| `KELLY_FRACTION` | `1.0` | Global Kelly multiplier across all tiers | Drawdown too large → reduce to 0.5; underdeployed → raise to 1.5 |
| `MARKOV_SERVE_SCALE` | `0.25` | H2H win rate → Markov p_serve scaling | Model p vs market p consistently off → adjust by ±0.05 |
| `MAX_BET_USDC` | `250.0` | Hard cap per bet | Scale with bankroll — never exceed 5% of total |
| `MAX_CONCURRENT_MATCHES` | `3` | Parallel match workers | Server CPU/latency allows more → increase |

### Acceptance Criteria
After any config change, run the backtester to confirm the change improves simulated P&L:
```bash
python advanced_backtester.py 2>&1 | tail -30
# Compare Sharpe ratio and win rate before/after change
```

---

## Summary Checklist

| Task | Description | Blocker for PROD? |
|------|-------------|-------------------|
| Task 1 | Environment setup & imports | Yes |
| Task 2 | DEMO credentials | Yes |
| Task 3 | Fix scale-out double-sell bug | Yes |
| Task 4 | Full test suite green | Yes |
| Task 5 | DEMO smoke test (10 min) | Yes |
| Task 6 | Dashboard running | No |
| Task 7 | Latency benchmarks in budget | Recommended |
| Task 8 | Switch to production keys | — |
| Task 9 | Process management (systemd) | Recommended |
| Task 10 | Trading mode controls understood | No |
| Task 11 | Log monitoring active | Recommended |
| Task 12 | Config tuning post go-live | No (ongoing) |

---

## Known Issues Already Fixed (Do Not Re-Apply)

These were fixed on **2026-04-16** — do not apply them again:

| Fix | File | Description |
|-----|------|-------------|
| WS DRY-RUN 401 | `kalshi_client.py` | `stream_orderbook` now returns early if no private key |
| STALE score spam | `live_score_scraper.py` | `_is_scheduled` flag persisted across poll iterations |
| Flashscore log noise | `flashscore_pipeline/discovery.py` | "No match found" downgraded from WARNING to DEBUG |
| YES mapping | `bet_manager.py` | Already implemented — fuzzy last-name match against market title |
| Live Markov params | `main.py` | Already implemented — `update_params()` called every 10 serve points |
| Fee subtraction | `bet_manager.py` | Already implemented — `KALSHI_FEE_RATE` subtracted from edge |
| Balance pass-through | `main.py` / `bet_manager.py` | Already implemented — `get_balance()` called once per poll cycle |
