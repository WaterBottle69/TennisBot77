"""
trade_tracker.py — Comprehensive per-bet data logger.

Records EVERY available information point at entry and exit so that
post-session analysis can test any hypothesis without re-running the bot.

Output files (auto-rotated daily):
  trades_YYYYMMDD.csv   — one row per bet, outcome columns filled on exit
  trades_YYYYMMDD.jsonl — same data, full fidelity (no CSV truncation)
"""

import csv
import json
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

# ── Column order in CSV ────────────────────────────────────────────────────────

COLUMNS = [
    # ── Identity
    "trade_id",
    "timestamp_entry_utc",
    "timestamp_exit_utc",
    "ticker",
    "player_a",
    "player_b",

    # ── Tournament / venue
    "tournament",
    "venue_city",
    "venue_country",
    "court_surface",
    "court_type",
    "altitude_m",

    # ── Weather at match start
    "temp_celsius",
    "apparent_temp_c",
    "humidity_pct",
    "wind_speed_kmh",
    "wind_direction_deg",
    "precipitation_mm",

    # ── Sun / glare
    "sun_azimuth_deg",
    "sun_elevation_deg",
    "sun_glare_active",
    "sun_description",

    # ── Player A stats
    "p1_ranking",
    "p1_rank_points",
    "p1_age",
    "p1_height_cm",
    "p1_hand",
    "p1_win_rate",
    "p1_season_wins",
    "p1_season_losses",
    "p1_aces_per_match",
    "p1_bp_conversion_pct",
    "p1_first_serve_pct",

    # ── Player B stats
    "p2_ranking",
    "p2_rank_points",
    "p2_age",
    "p2_height_cm",
    "p2_hand",
    "p2_win_rate",
    "p2_season_wins",
    "p2_season_losses",
    "p2_aces_per_match",
    "p2_bp_conversion_pct",
    "p2_first_serve_pct",

    # ── Derived matchup features
    "rank_diff",
    "age_diff",
    "height_diff_cm",
    "pts_vs_rank_raw",
    "lh_net",
    "alt_x_ht_component",
    "alt_x_age_component",
    "lh_hard_component",
    "lh_clay_component",
    "pts_rank_component",

    # ── Model signal pipeline
    "model_prob_ml_base",
    "model_prob_nn",
    "model_prob_xgb",
    "logit_adj_age_temp",
    "model_prob_after_age_temp",
    "logit_adj_phys",
    "model_prob_after_phys",
    "model_prob_final_at_bet",

    # ── Markov state at bet time
    "markov_p_serve_initial",
    "markov_p_serve_at_bet",
    "markov_p_return_at_bet",
    "bayesian_posterior_at_bet",
    "serve_divergence",
    "pts_vs_rank_edge",
    "bayes_uncertainty",

    # ── ATP live stats at bet time
    "atp_first_serve_pct_a",
    "atp_pts_won_1st_a",
    "atp_first_serve_pct_b",
    "atp_pts_won_1st_b",
    "atp_break_pts_converted_a",
    "atp_live_ticks_elapsed",

    # ── Score state at bet time
    "score_sets_a",
    "score_sets_b",
    "score_games_a",
    "score_games_b",
    "score_points_a",
    "score_points_b",
    "p1_serving_at_bet",
    "total_live_ticks",

    # ── Market at bet time
    "betting_on",
    "yes_price_at_bet",
    "no_price_at_bet",
    "model_prob_for_side",
    "market_price_for_side",
    "edge_gross",
    "fee_estimate",
    "edge_net",
    "kelly_fraction_raw",
    "kelly_mult_convergence",
    "kelly_mult_adaptive",
    "kelly_mult_combined",

    # ── Flow signal at bet time
    "flow_direction",
    "flow_velocity_cents_per_s",
    "flow_z_score",
    "flow_vol_regime",

    # ── Execution
    "stake_usdc",
    "contracts_filled",
    "entry_price_actual",
    "order_id",
    "latency_ms",

    # ── Outcome (filled on exit)
    "exit_price",
    "exit_reason",
    "contracts_exited",
    "pnl_usdc",
    "roi_pct",
    "outcome",
    "kalshi_settlement_price",
    "match_winner_was_a",
]


class TradeTracker:
    """
    Thread-safe per-trade logger.

    Usage:
        tracker = TradeTracker(base_dir="/path/to/bot")

        # At entry:
        trade_id = tracker.record_entry(ticker="TNNS-...", betting_on="yes", ...)

        # At exit:
        tracker.record_exit(trade_id=trade_id, pnl_usdc=1.23, ...)
    """

    def __init__(self, base_dir: str = "."):
        self._base_dir = base_dir
        self._lock = threading.Lock()
        self._open: Dict[str, dict] = {}   # trade_id → entry row
        self._counter = 0
        self._date_str = ""
        self._csv_path = ""
        self._jsonl_path = ""
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None
        self._ensure_files()

    # ── File management ──────────────────────────────────────────────────────

    def _ensure_files(self):
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        if today == self._date_str:
            return
        self._date_str = today
        if self._csv_file:
            self._csv_file.close()
        if self._jsonl_file:
            self._jsonl_file.close()

        self._csv_path   = os.path.join(self._base_dir, f"trades_{today}.csv")
        self._jsonl_path = os.path.join(self._base_dir, f"trades_{today}.jsonl")

        write_header = not os.path.exists(self._csv_path)
        self._csv_file   = open(self._csv_path,   "a", newline="", encoding="utf-8")
        self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file, fieldnames=COLUMNS, extrasaction="ignore"
        )
        if write_header:
            self._csv_writer.writeheader()
            self._csv_file.flush()

    # ── Entry ────────────────────────────────────────────────────────────────

    def record_entry(
        self,
        *,
        ticker: str,
        player_a: str,
        player_b: str,
        betting_on: str,                    # "yes" or "no"
        yes_price_at_bet: float,
        no_price_at_bet: float,
        model_prob_for_side: float,
        market_price_for_side: float,
        edge_gross: float,
        fee_estimate: float,
        edge_net: float,
        kelly_fraction_raw: float,
        kelly_mult_convergence: float,
        kelly_mult_adaptive: float,
        kelly_mult_combined: float,
        stake_usdc: float,
        contracts_filled: int,
        entry_price_actual: float,
        order_id: str,
        latency_ms: float,
        # ── Signal pipeline
        model_prob_ml_base: float = float("nan"),
        model_prob_nn: float = float("nan"),
        model_prob_xgb: float = float("nan"),
        logit_adj_age_temp: float = 0.0,
        model_prob_after_age_temp: float = float("nan"),
        logit_adj_phys: float = 0.0,
        model_prob_after_phys: float = float("nan"),
        model_prob_final_at_bet: float = float("nan"),
        pts_vs_rank_edge: float = 0.0,
        serve_divergence: float = 0.0,
        # ── Markov state
        markov_p_serve_initial: float = float("nan"),
        markov_p_serve_at_bet: float = float("nan"),
        markov_p_return_at_bet: float = float("nan"),
        bayesian_posterior_at_bet: float = float("nan"),
        bayes_uncertainty: float = float("nan"),
        # ── Physical signal components
        pts_vs_rank_raw: float = 0.0,
        alt_x_ht_component: float = 0.0,
        alt_x_age_component: float = 0.0,
        lh_hard_component: float = 0.0,
        lh_clay_component: float = 0.0,
        pts_rank_component: float = 0.0,
        lh_net: float = 0.0,
        # ── Venue / weather
        tournament: str = "",
        venue_city: str = "",
        venue_country: str = "",
        court_surface: str = "",
        court_type: str = "",
        altitude_m: float = 0.0,
        temp_celsius: float = float("nan"),
        apparent_temp_c: float = float("nan"),
        humidity_pct: float = float("nan"),
        wind_speed_kmh: float = float("nan"),
        wind_direction_deg: float = float("nan"),
        precipitation_mm: float = float("nan"),
        # ── Sun
        sun_azimuth_deg: float = float("nan"),
        sun_elevation_deg: float = float("nan"),
        sun_glare_active: bool = False,
        sun_description: str = "",
        # ── Player A
        p1_ranking: int = 0,
        p1_rank_points: float = 0.0,
        p1_age: float = float("nan"),
        p1_height_cm: float = float("nan"),
        p1_hand: str = "",
        p1_win_rate: float = float("nan"),
        p1_season_wins: Any = None,
        p1_season_losses: Any = None,
        p1_aces_per_match: Any = None,
        p1_bp_conversion_pct: Any = None,
        p1_first_serve_pct: Any = None,
        # ── Player B
        p2_ranking: int = 0,
        p2_rank_points: float = 0.0,
        p2_age: float = float("nan"),
        p2_height_cm: float = float("nan"),
        p2_hand: str = "",
        p2_win_rate: float = float("nan"),
        p2_season_wins: Any = None,
        p2_season_losses: Any = None,
        p2_aces_per_match: Any = None,
        p2_bp_conversion_pct: Any = None,
        p2_first_serve_pct: Any = None,
        # ── Derived matchup
        rank_diff: float = 0.0,
        age_diff: float = 0.0,
        height_diff_cm: float = 0.0,
        # ── ATP live stats
        atp_first_serve_pct_a: float = float("nan"),
        atp_pts_won_1st_a: float = float("nan"),
        atp_first_serve_pct_b: float = float("nan"),
        atp_pts_won_1st_b: float = float("nan"),
        atp_break_pts_converted_a: float = float("nan"),
        atp_live_ticks_elapsed: int = 0,
        # ── Score state
        score_sets_a: int = 0,
        score_sets_b: int = 0,
        score_games_a: int = 0,
        score_games_b: int = 0,
        score_points_a: int = 0,
        score_points_b: int = 0,
        p1_serving_at_bet: Optional[bool] = None,
        total_live_ticks: int = 0,
        # ── Flow
        flow_direction: str = "NEUTRAL",
        flow_velocity_cents_per_s: float = 0.0,
        flow_z_score: float = 0.0,
        flow_vol_regime: str = "",
    ) -> str:
        with self._lock:
            self._ensure_files()
            self._counter += 1
            trade_id = f"{self._date_str}-{self._counter:04d}-{ticker[-8:]}-{betting_on.upper()}"

            row = {
                "trade_id":                   trade_id,
                "timestamp_entry_utc":        datetime.now(timezone.utc).isoformat(),
                "timestamp_exit_utc":         "",
                "ticker":                     ticker,
                "player_a":                   player_a,
                "player_b":                   player_b,
                "tournament":                 tournament,
                "venue_city":                 venue_city,
                "venue_country":              venue_country,
                "court_surface":              court_surface,
                "court_type":                 court_type,
                "altitude_m":                 altitude_m,
                "temp_celsius":               temp_celsius,
                "apparent_temp_c":            apparent_temp_c,
                "humidity_pct":               humidity_pct,
                "wind_speed_kmh":             wind_speed_kmh,
                "wind_direction_deg":         wind_direction_deg,
                "precipitation_mm":           precipitation_mm,
                "sun_azimuth_deg":            sun_azimuth_deg,
                "sun_elevation_deg":          sun_elevation_deg,
                "sun_glare_active":           sun_glare_active,
                "sun_description":            sun_description,
                "p1_ranking":                 p1_ranking,
                "p1_rank_points":             p1_rank_points,
                "p1_age":                     p1_age,
                "p1_height_cm":               p1_height_cm,
                "p1_hand":                    p1_hand,
                "p1_win_rate":                p1_win_rate,
                "p1_season_wins":             p1_season_wins,
                "p1_season_losses":           p1_season_losses,
                "p1_aces_per_match":          p1_aces_per_match,
                "p1_bp_conversion_pct":       p1_bp_conversion_pct,
                "p1_first_serve_pct":         p1_first_serve_pct,
                "p2_ranking":                 p2_ranking,
                "p2_rank_points":             p2_rank_points,
                "p2_age":                     p2_age,
                "p2_height_cm":               p2_height_cm,
                "p2_hand":                    p2_hand,
                "p2_win_rate":                p2_win_rate,
                "p2_season_wins":             p2_season_wins,
                "p2_season_losses":           p2_season_losses,
                "p2_aces_per_match":          p2_aces_per_match,
                "p2_bp_conversion_pct":       p2_bp_conversion_pct,
                "p2_first_serve_pct":         p2_first_serve_pct,
                "rank_diff":                  rank_diff,
                "age_diff":                   age_diff,
                "height_diff_cm":             height_diff_cm,
                "pts_vs_rank_raw":            pts_vs_rank_raw,
                "lh_net":                     lh_net,
                "alt_x_ht_component":         alt_x_ht_component,
                "alt_x_age_component":        alt_x_age_component,
                "lh_hard_component":          lh_hard_component,
                "lh_clay_component":          lh_clay_component,
                "pts_rank_component":         pts_rank_component,
                "model_prob_ml_base":         model_prob_ml_base,
                "model_prob_nn":              model_prob_nn,
                "model_prob_xgb":             model_prob_xgb,
                "logit_adj_age_temp":         logit_adj_age_temp,
                "model_prob_after_age_temp":  model_prob_after_age_temp,
                "logit_adj_phys":             logit_adj_phys,
                "model_prob_after_phys":      model_prob_after_phys,
                "model_prob_final_at_bet":    model_prob_final_at_bet,
                "markov_p_serve_initial":     markov_p_serve_initial,
                "markov_p_serve_at_bet":      markov_p_serve_at_bet,
                "markov_p_return_at_bet":     markov_p_return_at_bet,
                "bayesian_posterior_at_bet":  bayesian_posterior_at_bet,
                "serve_divergence":           serve_divergence,
                "pts_vs_rank_edge":           pts_vs_rank_edge,
                "bayes_uncertainty":          bayes_uncertainty,
                "atp_first_serve_pct_a":      atp_first_serve_pct_a,
                "atp_pts_won_1st_a":          atp_pts_won_1st_a,
                "atp_first_serve_pct_b":      atp_first_serve_pct_b,
                "atp_pts_won_1st_b":          atp_pts_won_1st_b,
                "atp_break_pts_converted_a":  atp_break_pts_converted_a,
                "atp_live_ticks_elapsed":     atp_live_ticks_elapsed,
                "score_sets_a":               score_sets_a,
                "score_sets_b":               score_sets_b,
                "score_games_a":              score_games_a,
                "score_games_b":              score_games_b,
                "score_points_a":             score_points_a,
                "score_points_b":             score_points_b,
                "p1_serving_at_bet":          p1_serving_at_bet,
                "total_live_ticks":           total_live_ticks,
                "betting_on":                 betting_on,
                "yes_price_at_bet":           yes_price_at_bet,
                "no_price_at_bet":            no_price_at_bet,
                "model_prob_for_side":        model_prob_for_side,
                "market_price_for_side":      market_price_for_side,
                "edge_gross":                 edge_gross,
                "fee_estimate":               fee_estimate,
                "edge_net":                   edge_net,
                "kelly_fraction_raw":         kelly_fraction_raw,
                "kelly_mult_convergence":     kelly_mult_convergence,
                "kelly_mult_adaptive":        kelly_mult_adaptive,
                "kelly_mult_combined":        kelly_mult_combined,
                "flow_direction":             flow_direction,
                "flow_velocity_cents_per_s":  flow_velocity_cents_per_s,
                "flow_z_score":               flow_z_score,
                "flow_vol_regime":            flow_vol_regime,
                "stake_usdc":                 stake_usdc,
                "contracts_filled":           contracts_filled,
                "entry_price_actual":         entry_price_actual,
                "order_id":                   order_id,
                "latency_ms":                 latency_ms,
                # outcome fields — blank at entry
                "exit_price":                 "",
                "exit_reason":                "",
                "contracts_exited":           "",
                "pnl_usdc":                   "",
                "roi_pct":                    "",
                "outcome":                    "",
                "kalshi_settlement_price":    "",
                "match_winner_was_a":         "",
            }

            self._open[trade_id] = row
            self._csv_writer.writerow(row)
            self._csv_file.flush()
            self._jsonl_file.write(json.dumps({"event": "entry", **row}) + "\n")
            self._jsonl_file.flush()

            log.info("[TRACKER] Entry recorded: %s  side=%s  stake=%.2f  edge=%+.4f",
                     trade_id, betting_on, stake_usdc, edge_net)
            return trade_id

    # ── Exit ─────────────────────────────────────────────────────────────────

    def record_exit(
        self,
        *,
        trade_id: str,
        exit_price: float,
        exit_reason: str,
        contracts_exited: int,
        pnl_usdc: float,
        kalshi_settlement_price: float = float("nan"),
        match_winner_was_a: Optional[bool] = None,
    ):
        with self._lock:
            self._ensure_files()
            entry = self._open.pop(trade_id, None)
            if entry is None:
                log.warning("[TRACKER] record_exit: unknown trade_id=%s", trade_id)
                return

            stake = float(entry.get("stake_usdc") or 0)
            roi_pct = (pnl_usdc / stake * 100) if stake > 0 else float("nan")
            outcome = "WIN" if pnl_usdc > 0 else ("LOSS" if pnl_usdc < 0 else "PUSH")

            exit_fields = {
                "timestamp_exit_utc":      datetime.now(timezone.utc).isoformat(),
                "exit_price":              exit_price,
                "exit_reason":             exit_reason,
                "contracts_exited":        contracts_exited,
                "pnl_usdc":                round(pnl_usdc, 4),
                "roi_pct":                 round(roi_pct, 2),
                "outcome":                 outcome,
                "kalshi_settlement_price": kalshi_settlement_price,
                "match_winner_was_a":      match_winner_was_a,
            }
            entry.update(exit_fields)
            self._jsonl_file.write(json.dumps({"event": "exit", **entry}) + "\n")
            self._jsonl_file.flush()

            log.info("[TRACKER] Exit recorded: %s  outcome=%s  pnl=%.4f  roi=%.1f%%",
                     trade_id, outcome, pnl_usdc, roi_pct)

    # ── Convenience: rewrite CSV on exit (full row update) ───────────────────

    def flush_all(self):
        """Rewrite the full CSV including any outstanding open positions. Call on shutdown."""
        with self._lock:
            for tid, row in self._open.items():
                row["exit_reason"] = "OPEN_AT_SHUTDOWN"
            log.info("[TRACKER] flush_all: %d open positions marked at shutdown", len(self._open))

    def __del__(self):
        try:
            if self._csv_file:
                self._csv_file.close()
            if self._jsonl_file:
                self._jsonl_file.close()
        except Exception:
            pass
