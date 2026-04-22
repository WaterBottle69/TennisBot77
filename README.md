# TennisBot — ATP Prediction & Kalshi Betting Bot

Automated tennis match prediction and live betting on [Kalshi](https://kalshi.com) prediction markets.  
A hybrid ML model (Gradient Boosting + LSTM neural network) trained on 50+ years of ATP data finds edges against market-implied probabilities, placing and managing positions autonomously while matches are in play.

---

## How It Works — Full System Flow

### 1. Market Discovery (Kalshi)

On startup, the bot scans Kalshi's entire open event catalogue (up to 6,000 events across 30 pages) looking for tennis head-to-head markets. There are no hardcoded player names — discovery is fully dynamic.

Each event is scored for "tennis-ness":
- `+100` pts if the ticker contains ATP / WTA / ITF / CHALLENGER
- `+80` pts if the title matches a `"Player A vs Player B"` pattern
- `+70` pts if broader tennis hints appear (Wimbledon, Roland Garros, etc.)
- `−200` pts if non-tennis keywords appear (Golf, NBA, NFL, FIFA, etc.)

Markets scoring above zero and containing extractable player names are kept. They are sorted by **soonest closing time first**, so the most imminent (and often live) matches are prioritised. Up to 48 unique markets are queued per session.

> Kalshi opens markets hours before a match starts, so the queue will contain a mix of pre-game and live matches at any given time.

---

### 2. Concurrency Model

```
run_game_session()
    └── 48 markets → asyncio.Queue
            └── 3 concurrent workers (asyncio, not threads)
                    └── Each worker: process_match() → next match when done
```

All three workers share the same Python event loop, cooperatively yielding at every `await` (API calls, sleeps). A shared `asyncio.Lock` prevents concurrent workers from double-spending the same balance when sizing bets.

---

### 3. Pre-Game Prediction

When a worker picks up a match, it immediately runs the full ML pipeline **once**:

**a) Player stats** — `HistoricalAnalyzer` scrapes ATP data for both players: ranking, height, age, serve %, handedness, surface win rate, H2H win rate, recent form (last 10 matches).

**b) Hybrid ML model** — `ml_engine.predict_win_prob()` runs two models in parallel:

| Model | Input | Notes |
|---|---|---|
| XGBoost (Gradient Boosting) | 21 static features | Rank diff, ELO diff, height diff, surface, serve stats |
| LSTM Neural Network | Last 10 matches as time-series sequences | Learns momentum and form patterns |

The outputs are blended (60% NN / 40% XGBoost by default, or via a trained logistic blender) into a single `hybrid_prob` — the pre-game win probability for Player A.

**c) Markov engine** — `LiveMatchState` is initialised with `p_serve` and `p_return` derived from the ML probability. This models every point, game, and set using dynamic programming to compute exact win probabilities at any match state.

---

### 4. Live Score Polling — Pre-Game and Live in the Same Loop

The bot enters a polling loop that runs every **2 seconds** (0.8s in High-Frequency mode). There is no separate pre-game vs. live mode — it is the same loop throughout. The only thing that changes is whether the Markov engine gets updated:

```
Every 2 seconds:
    │
    ├── Poll LiveScore CDN API (live matches globally)
    │       │
    │       ├── Match IS live → Markov engine updates with real points/games/sets
    │       │                   win_prob changes point by point
    │       │
    │       ├── Match NOT live, found in today's schedule → pre-game
    │       │   Markov engine not updated (no score yet)
    │       │   win_prob holds at ML pre-game estimate
    │       │   Bot still evaluates edge and can place pre-game bets
    │       │
    │       └── Match not found anywhere → stale
    │           Uses last known win_prob, logs warning every 10 misses
    │
    ├── Fetch current Kalshi YES/NO prices for this market
    │
    └── BetManager.evaluate_and_act()
            ├── Check exits on any open positions
            └── Check entry: is edge > threshold? → buy
```

The moment LiveScore sees the first point of a match, `is_live` flips to `True` and the Markov engine takes over from the static ML estimate — no restart, just a flag change mid-loop.

---

### 5. Edge Detection & Bet Sizing

```
edge_yes = model_probability_A  −  kalshi_yes_price
edge_no  = model_probability_B  −  kalshi_no_price
```

A bet is placed when edge exceeds the minimum threshold (1.5% normal, 0.3% HF mode) **and** the Kelly fraction is positive.

**Kelly sizing is tiered by model conviction:**

| Model probability | Kelly fraction used |
|---|---|
| ≥ 70% | 40% of full Kelly |
| ≥ 60% | 25% of full Kelly |
| ≥ 55% | 12% of full Kelly |
| < 55% | 5% of full Kelly |

A `_buy_lock` serialises balance checks and order placement so concurrent workers never exceed the available collateral.

---

### 6. Position Management

Open positions are checked **every poll cycle** for three exit conditions:

| Condition | Action |
|---|---|
| Price moved 15–25% in our favour | Take profit — sell |
| Edge reversed (model now disagrees with our position) | Cut loss — sell |
| Market price ≥ 0.92 (near resolution) | Lock in value — sell |

A 90-second cooldown prevents re-entering the same side of a market immediately after a sell.

---

### 7. High-Frequency Mode

Toggled via `trading_mode.json` at runtime. Lowers edge threshold to 0.3%, reduces Kelly to 5%, caps bets at $15, and polls every 0.8 seconds. Automatically reverts to normal mode after $50 cumulative profit.

---

## Architecture & Component Breakdown

TennisBot77 is built as a modular system where each component handles a specific part of the alpha generation and execution pipeline.

| Core Component | Implementation File | Role & Responsibilities |
|---|---|---|
| **Strategy Orchestrator** | `main.py` | The main entry point. Handles market discovery on Kalshi, manages the task queue, and spawns concurrent workers to process matches. |
| **Hybrid ML Engine** | `ml_engine.py` | A dual-model inference system blending **XGBoost** (for static features like rank and H2H) and **LSTM** (for sequential form and momentum) to generate pre-game win probabilities. |
| **Markov State Engine** | `markov_engine.py` | Uses dynamic programming to model point-by-point transitions. It computes exact live win probabilities based on point-level match states. |
| **Execution Manager** | `bet_manager.py` | Implements the alpha surface. Calculates Kelly-optimal bet sizes, manages orders on Kalshi, and executes tiered entry/exit logic. |
| **Kalshi Connectivity** | `kalshi_client.py` | A robust wrapper for the Kalshi API, handling authentication, market filters, and order placement. |
| **Real-time Telemetry** | `live_score_scraper.py` | Polls low-latency score APIs to provide the Markov engine with the exact current match state (points, games, sets). |
| **Data Orchestator** | `historical_analyzer.py` | Scrapes and pre-processes 50+ years of ATP data to build feature vectors for the ML models. |
| **Adaptive Controller** | `adaptive_controller.py` | Monitors live telemetry and model performance to adjust risk parameters (e.g., slippage protection, bankroll scaling) in real-time. |
| **User Dashboard** | `server.py` | A FastAPI-based backend that serves the live visualization dashboard and provides management endpoints. |

---

## Live Dashboard

The bot includes a premium web dashboard for real-time monitoring and manual intervention.

### Running the Dashboard Locally

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the FastAPI Server**:
   ```bash
   uvicorn server:app --host 0.0.0.0 --port 8000
   ```

3. **Access the Interface**:
   Open your browser and navigate to `http://localhost:8000`.

The dashboard features life-time model confidence visualizations, real-time log streaming, and controls to toggle between **Normal** and **High-Frequency (HF)** trading modes.

---

## Performance & Optimization

### Walk-Forward Optimization (WFO)
The system undergoes regular walk-forward optimization (see `wfo.py`) to prevent look-ahead bias. This process involves:
- **Rolling Training Windows**: Training on past seasons and testing on the subsequent one.
- **Isotonic Calibration**: Calibrating model outputs to ensure "60% confidence" actually maps to a 60% win rate.
- **Ablation Studies**: Regularly testing the importance of features like `Rank_Diff`, `Elo_Diff`, and `Surface_Win_Rate`.

### Risk Management
- **Kelly Tiering**: Bets are sized proportionally to the model's conviction edge.
- **Alpha Exits**: Positions are automatically closed if the model-implied edge reverses or price targets are hit.
- **Concurrency Locks**: A centralized async lock prevents race conditions in balance handling during high-frequency volatility.

---

## Data Source
[Jeff Sackmann — tennis_atp](https://github.com/JeffSackmann/tennis_atp) — 195,000+ ATP matches from 1968–2026.
