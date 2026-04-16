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

## Architecture

| File | Role |
|---|---|
| `main.py` | Orchestrator — discovery, queue, worker pool, polling loop |
| `kalshi_client.py` | Kalshi REST API — market discovery, order placement, auth |
| `live_score_scraper.py` | LiveScore CDN polling — live scores + daily schedule |
| `ml_engine.py` | Hybrid XGBoost + LSTM inference engine |
| `markov_engine.py` | Point-by-point win probability via dynamic programming |
| `elo_engine.py` | ELO rating engine with live event multipliers |
| `bet_manager.py` | Kelly sizing, entry/exit logic, alpha surface |
| `historical_analyzer.py` | ATP player stat scraping |
| `tennis_scraper.py` | Player profile scraping (height, rank, serve %) |
| `backtester_kelly.py` | Walk-forward backtest with Kelly betting simulation |
| `ml_trainer.py` | Model training pipeline (GBC + TimeSeriesSplit GridSearch) |
| `server.py` | WebSocket bridge + REST API for the dashboard |
| `static/index.html` | Live neural network visualiser dashboard |

---

## Backtest Results — Out-of-Sample 2022–2026

Walk-forward evaluation on **9,034 ATP matches** never seen during training.  
Model trained on 1968–2021. Test: **2022–2026**.  
Flat $10 bets, 8% vig, ≥4pp edge threshold vs. ELO-implied market odds.

### Baseline (14 features, pre-trained model)

| Metric | Value |
|---|---|
| Accuracy | 64.17% |
| ROC-AUC | 0.7035 |
| Brier Score | 0.2180 |
| Betting ROI | +7.6% |
| Bets placed | 3,236 |

### Expanded (21 features, retrained + isotonic calibration)

| Metric | Value | Delta |
|---|---|---|
| Accuracy | 65.39% | +1.22pp |
| ROC-AUC | 0.7197 | +0.0162 |
| Brier Score | 0.2136 | −0.0045 |
| Betting ROI | +36.4% | +28.8pp |
| Bets placed | 2,107 | Fewer, higher quality |

**New features added (Round 1):** `Height_Diff`, `Age_Diff`, `Rank_Diff`, `Rank_Points_Diff`, `Elo_Diff`, `Surface_Height_Grass` (height × grass interaction), `Age_Diff_Sq` (nonlinear age gap).

> The ROI figure uses ELO-implied odds as the "market" — a weak baseline. Real Kalshi markets are more efficient. Treat ROI as a theoretical upper bound; accuracy and Brier score are the reliable metrics.

---

## Quick Start

```bash
# Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Add Kalshi API keys (copy example and fill in)
cp kalshi_keys.json.example kalshi_keys.json

# Run the bot
source venv/bin/activate
python main.py >> bot.log 2>&1 &

# Watch the log
tail -f bot.log
```

## Re-running the Backtest

```bash
source venv/bin/activate
python backtester_kelly.py
```

Outputs both baseline (pre-trained model) and expanded (inline retrained) results side-by-side.

---

## Data Source

[Jeff Sackmann — tennis_atp](https://github.com/JeffSackmann/tennis_atp) — 195,000+ ATP matches from 1968–2026.

---

## Baseball Statistics — Ablation by Win Probability Weight

Research synthesis across the Samford Sports Analytics correlation study, PMC feature-selection study (65.75% ML accuracy across 30 teams), and SABR Pythagorean research. Stats are ranked by their measured correlation with team win percentage and their contribution to single-game win probability.

---

### Tier 1 — Core Predictors (r > 0.70) — Highest Weight

Drop any one of these and model accuracy collapses.

| Statistic | Correlation with Win% | What it measures |
|---|---|---|
| **Run Differential** | r = 0.910 | Runs scored − Runs allowed. Basis of Pythagorean win expectation: `W% ≈ RS² / (RS² + RA²)` |
| **ERA** | r ≈ 0.85 | Earned runs allowed per 9 innings — direct run prevention |
| **FIP** (Fielding Independent Pitching) | r ≈ 0.82 | Pitcher K, BB, HBP, HR only. Strips defensive noise — better future predictor than ERA |
| **Rolling Win%** | Selected in 100% of 30-team datasets | Current team record — captures everything the box score misses |

> 11 of the 19 most correlated statistics are pitching-related. Preventing runs is more predictive than scoring them.

---

### Tier 2 — Strong Secondary Predictors (r = 0.50–0.70)

Adding these yields ~2–4% accuracy lift over Tier 1 alone.

| Statistic | Correlation | What it measures |
|---|---|---|
| **LOB%** (Left On Base %) | r ≈ 0.68 | % of baserunners stranded — top positive pitching variable in ablation studies |
| **WHIP** | r ≈ 0.65 | (Walks + Hits) / IP — combines contact and command |
| **Pitching WAR** | r ≈ 0.64 | Total pitcher value above replacement — best single pitching aggregate |
| **wRC+** | r ≈ 0.62 | Park/league-adjusted run creation. 100 = league avg, 120 = 20% above avg |
| **wOBA** | r ≈ 0.60 | Run-weighted on-base average — singles/doubles/HR weighted by actual run value |
| **Offensive WAR** | r ≈ 0.58 | Total offensive value above replacement |
| **H/9** | r ≈ 0.57 | Hits allowed per 9 innings — contact prevention |
| **BAA** | r ≈ 0.55 | Opponents' batting average vs pitcher |

---

### Tier 3 — Moderate Predictors (r = 0.29–0.50)

Worth including but not individually load-bearing.

| Statistic | Notes |
|---|---|
| **K/9** | Key positive pitching factor — strikeouts directly remove run-scoring chances |
| **BB/9** | Strongly negative — free baserunners cost runs. Best control proxy |
| **HR/9** | Negative predictor — hardest thing for a pitcher to control |
| **xFIP** | FIP using league-average HR/FB rate — most stable pitcher skill estimate |
| **SIERA** | Linear regression to ERA using K%, GB%, BB% — most sophisticated rate stat |
| **ISO** (Isolated Power) | Extra bases per AB — power correlates with run scoring |
| **BB%** (Walk Rate) | Positive offensive predictor — patient hitters get on base |
| **K%** (Strikeout Rate) | Largest negative offensive factor in ablation studies |
| **OBP** | Root of run scoring — historically undervalued, now well understood |
| **Bullpen ERA** | Starters cover ~5 innings now; bullpen impact is increasingly critical |
| **Home Field Advantage** | ~54% historical win rate at home — worth a +4pp prior adjustment |
| **Starter Matchup (FIP diff)** | Per-game FIP differential between starting pitchers |

---

### Tier 4 — Negligible / Drop These (r < 0.29)

Ablation studies show removing these can actually improve accuracy.

| Statistic | Correlation | Why it fails |
|---|---|---|
| **Stolen Bases** | r = 0.003 | SB attempt costs offset gains unless >75% success rate |
| **Team Speed** | r = 0.006 | No meaningful team-level connection to wins |
| **Total Errors** | r = 0.007 | Rare and inconsistent — most runs allowed are not errors |
| **Fielding %** | r = 0.004 | Misses range, positioning, arm strength entirely |
| **Fastball Velocity** | r = 0.099 | Velocity without command or secondary pitches means little |
| **Pitch-type %** (slider, curve) | r = 0.000 | Completely uncorrelated with wins at team level |
| **Batting Average** | Low | Treats all hits equally — wOBA replaces this cleanly |
| **RBI** | Situational | Dependent on teammates getting on base — not a clean skill measure |

---

### Composite Weighting for a Game Win Probability Model

```
Win Probability =
    0.35 × Run Differential component  (Pythagorean W%)
  + 0.20 × Pitching quality            (FIP / xFIP / SIERA)
  + 0.15 × Offense                     (wRC+ / wOBA)
  + 0.12 × Rolling Win%                (last 15 games)
  + 0.08 × Bullpen ERA
  + 0.05 × Home field adjustment       (+4pp for home team)
  + 0.03 × Starter matchup             (starter FIP differential)
  + 0.02 × Rest / fatigue proxy        (days since last game, travel)
```

**Sources:** [Samford Sports Analytics](https://www.samford.edu/sports-analytics/fans/2022/MLB-Winning-Percentage-Breakdown-Which-Statistics-Help-Teams-Win-More-Games) · [PMC Feature Selection Study](https://pmc.ncbi.nlm.nih.gov/articles/PMC8871522/) · [Pythagorean Expectation — Wikipedia](https://en.wikipedia.org/wiki/Pythagorean_expectation) · [SABR Win Percentage Formula](https://sabr.org/journal/article/a-new-formula-to-predict-a-teams-winning-percentage/) · [FanGraphs wRC+](https://library.fangraphs.com/offense/wrc/)
