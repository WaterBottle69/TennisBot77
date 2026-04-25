import os
import json
import logging
import pandas as pd
import joblib
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LogisticRegression

log = logging.getLogger(__name__)

# --- ARCHITECTURE (Must match pytorch_model.py exactly) ---

from pytorch_model import TennisNet, XGBoostPyTorchBlender


class HybridMLEngine:
    def __init__(self, xgb_path='best_xgb_model.json', nn_path='best_tennis_nn.pth', feat_path='model_features.json', id_map_path='player_id_map.json'):
        self.xgb_path = xgb_path
        self.nn_path = nn_path
        self.feat_path = feat_path
        self.id_map_path = id_map_path
        
        self.xgb_model = None
        self.nn_model = None
        self.feature_names = []
        self.player_id_map = {}
        self.blender = None

        self._load_models()

    def _load_models(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 1. Load XGBoost
        try:
            abs_feat = os.path.join(base_dir, self.feat_path)
            abs_xgb = os.path.join(base_dir, self.xgb_path)
            if os.path.exists(abs_feat) and os.path.exists(abs_xgb):
                with open(abs_feat, 'r') as f:
                    self.feature_names = json.load(f)
                self.xgb_model = joblib.load(abs_xgb)
                log.info("XGBoost Baseline loaded.")
        except Exception as e:
            log.error(f"XGBoost load failed: {e}")

        # 2. Load Player ID Map
        try:
            abs_id_map = os.path.join(base_dir, self.id_map_path)
            if os.path.exists(abs_id_map):
                with open(abs_id_map, 'r') as f:
                    self.player_id_map = json.load(f)
                log.info(f"Player ID map loaded ({len(self.player_id_map)} players).")
        except Exception as e:
            log.error(f"ID Map load failed: {e}")

        # 2.5 Load NN Scaler
        try:
            self.nn_scaler = None
            scaler_path = os.path.join(base_dir, 'nn_scaler.joblib')
            if os.path.exists(scaler_path):
                self.nn_scaler = joblib.load(scaler_path)
                log.info("Loaded NN feature scaler.")
        except Exception as e:
            log.warning(f"NN Scaler load failed: {e}")

        # 3. Load Neural Network
        try:
            abs_nn = os.path.join(base_dir, self.nn_path)
            if os.path.exists(abs_nn):
                state = torch.load(abs_nn, map_location=torch.device('cpu'), weights_only=True)
                
                # Infer num_players from the saved embedding layer rather than player_id_map
                embed_weight = state.get("player_embedding.weight", None)
                if embed_weight is not None:
                    num_players = embed_weight.shape[0] - 1
                else:
                    num_players = 200000
                    if self.player_id_map:
                        id_vals = [v for v in self.player_id_map.values() if isinstance(v, (int, float))]
                        if id_vals:
                            num_players = max(max(id_vals), num_players)
                # fc layer weight shape rather than hard-coding it.  The fc stack is:
                #   fc.0.weight: (256, 280 + match_feat_dim)  (For Bidirectional LSTM with 8-dim embeds)
                # so match_feat_dim = fc.0.weight.shape[1] - 280
                state = torch.load(abs_nn, map_location=torch.device('cpu'), weights_only=True)
                fc0_shape = state.get("fc.0.weight", state.get("fc.weight", None))
                if fc0_shape is not None and fc0_shape.shape[1] > 280:
                    inferred_feat_dim = fc0_shape.shape[1] - 280
                else:
                    inferred_feat_dim = 20  # safe default matching training
                log.info(f"Neural Engine: inferred match_feat_dim={inferred_feat_dim} from checkpoint.")

                self.nn_model = TennisNet(num_players=int(num_players), match_feat_dim=inferred_feat_dim)
                self.nn_model.load_state_dict(state)
                self.nn_model.eval()
                self._nn_feat_dim = inferred_feat_dim
                log.info("Neural Engine (LSTM) loaded and engaged.")
        except Exception as e:
            log.error(f"Neural Engine load failed: {e}")

    def get_player_id(self, name: str) -> int:
        # ID 0 is padding_idx (zero embedding, never trained).
        # Fall back to 1 ("unknown player") so the model uses a real learned
        # embedding for players not in the map rather than a zero vector.
        return self.player_id_map.get(name, 1)

    def predict_win_prob(self, p1_stats: dict, p2_stats: dict, seq1: list = None, seq2: list = None) -> dict:
        """
        Returns a breakdown of both models' predictions.
        """
        res = {"nn_prob": 0.5, "xgb_prob": 0.5, "hybrid_prob": 0.5}
        
        # 1. XGBoost Prediction
        if self.xgb_model:
            try:
                surface = p1_stats.get('surface', 'Hard')

                # BUG FIX: Use ranking_pts (ATP ranking points) consistently.
                # 'elo' from tennisstats.com IS actually ATP ranking points (0–25,000),
                # not a true Elo score. We now read ranking_pts first (explicitly set
                # by main.py from the scraper), falling back to elo for compatibility.
                # This ensures the training feature distribution matches inference.
                def _rpts(stats, default=1000):
                    v = stats.get('ranking_pts')
                    if v is None:
                        v = stats.get('elo', default)
                    return float(v) if v is not None else float(default)

                p1_rpts = _rpts(p1_stats)
                p2_rpts = _rpts(p2_stats)

                feat = {
                    # Surface one-hot
                    'Surface_Hard':  1 if surface == 'Hard'  else 0,
                    'Surface_Clay':  1 if surface == 'Clay'  else 0,
                    'Surface_Grass': 1 if surface == 'Grass' else 0,
                    'Best_Of_Sets': p1_stats.get('best_of', 3),
                    # Player 1 base
                    'P1_Is_Right_Handed': 1 if p1_stats.get('hand', 'R') == 'R' else 0,
                    'P1_Height_cm': p1_stats.get('height_cm', 185),
                    'P1_Age':       p1_stats.get('age', 25),
                    'P1_Rank':      p1_stats.get('ranking', 50),
                    'P1_Rank_Points': p1_rpts,
                    # Player 2 base
                    'P2_Is_Right_Handed': 1 if p2_stats.get('hand', 'R') == 'R' else 0,
                    'P2_Height_cm': p2_stats.get('height_cm', 185),
                    'P2_Age':       p2_stats.get('age', 25),
                    'P2_Rank':      p2_stats.get('ranking', 50),
                    'P2_Rank_Points': p2_rpts,
                    # Rank differential — single strongest XGBoost predictor for tennis
                    'Rank_Diff': p1_stats.get('ranking', 50) - p2_stats.get('ranking', 50),
                    'Elo_Diff':  p1_rpts - p2_rpts,
                    # Surface-specific win rates [0..1]
                    'P1_Surface_Win_Rate': p1_stats.get('surface_win_rate', 0.5),
                    'P2_Surface_Win_Rate': p2_stats.get('surface_win_rate', 0.5),
                    # Recent form: win rate over last 10 matches [0..1]
                    'P1_Recent_Win_Rate': p1_stats.get('recent_win_rate', 0.5),
                    'P2_Recent_Win_Rate': p2_stats.get('recent_win_rate', 0.5),
                    # Serve stats [0..1] — first serve % and break point conversion
                    'P1_First_Serve_Pct':   p1_stats.get('first_serve_pct', 0.62),
                    'P2_First_Serve_Pct':   p2_stats.get('first_serve_pct', 0.62),
                    'P1_Break_Point_Conv':  p1_stats.get('break_point_conv', 0.40),
                    'P2_Break_Point_Conv':  p2_stats.get('break_point_conv', 0.40),
                }
                log.debug(
                    "[XGB] Feature check — P1_Rank_Points=%.0f P2_Rank_Points=%.0f "
                    "Elo_Diff=%.0f Surface=%s",
                    p1_rpts, p2_rpts, p1_rpts - p2_rpts, surface,
                )
                ordered_features = {k: feat.get(k, 0) for k in self.feature_names}
                df = pd.DataFrame([ordered_features])
                res["xgb_prob"] = float(self.xgb_model.predict_proba(df)[0][1])
            except Exception:
                pass

        # 2. Neural (LSTM) Prediction
        if self.nn_model and seq1 and seq2:
            try:
                # Ensure sequences are exactly length 10.
                # Use a list comprehension for padding — [[0.0]*4]*N creates N aliases
                # of the same inner list which corrupts if any row is later mutated.
                s1 = seq1[-10:] if len(seq1) >= 10 else [[0.0]*4 for _ in range(10-len(seq1))] + seq1
                s2 = seq2[-10:] if len(seq2) >= 10 else [[0.0]*4 for _ in range(10-len(seq2))] + seq2
                
                # Player IDs
                id1 = self.get_player_id(p1_stats.get('name', ''))
                id2 = self.get_player_id(p2_stats.get('name', ''))
                
                t_id1 = torch.tensor([id1], dtype=torch.long)
                t_id2 = torch.tensor([id2], dtype=torch.long)
                t_seq1 = torch.tensor([s1], dtype=torch.float32)
                t_seq2 = torch.tensor([s2], dtype=torch.float32)
                t_len1 = torch.tensor([10], dtype=torch.long)
                t_len2 = torch.tensor([10], dtype=torch.long)
                
                # Surface map: Hard=1, Clay=2, Grass=3
                surf_idx = 1
                if p1_stats.get('surface') == 'Clay': surf_idx = 2
                elif p1_stats.get('surface') == 'Grass': surf_idx = 3
                
                t_surf = torch.tensor([[surf_idx]], dtype=torch.long)
                t_tourn = torch.tensor([[0]], dtype=torch.long)
                
                feat_dim = getattr(self, '_nn_feat_dim', 3)
                rank_diff = float(p1_stats.get('ranking', 50)) - float(p2_stats.get('ranking', 50))
                rank1 = float(p1_stats.get('ranking', 50))
                rank2 = float(p2_stats.get('ranking', 50))
                elo1  = float(p1_stats.get('elo', 1000))
                elo2  = float(p2_stats.get('elo', 1000))
                age1  = float(p1_stats.get('age', 25))
                age2  = float(p2_stats.get('age', 25))
                h1    = float(p1_stats.get('height_cm', 185))
                h2    = float(p2_stats.get('height_cm', 185))
                hand1 = 1.0 if p1_stats.get('hand', 'R') == 'R' else 0.0
                hand2 = 1.0 if p2_stats.get('hand', 'R') == 'R' else 0.0
                best_of = float(p1_stats.get('best_of', 3))
                surf_h  = 1.0 if p1_stats.get('surface') == 'Hard' else 0.0
                surf_c  = 1.0 if p1_stats.get('surface') == 'Clay' else 0.0
                surf_g  = 1.0 if p1_stats.get('surface') == 'Grass' else 0.0
                
                cpi = float(p1_stats.get('cpi', 35.0))
                pA_fatigue = float(p1_stats.get('trailing_minutes', 180.0))
                pB_fatigue = float(p2_stats.get('trailing_minutes', 180.0))
                pA_arch = float(p1_stats.get('archetype', 0.5))
                pB_arch = float(p2_stats.get('archetype', 0.5))
                
                # New Niche Features (Defaults if live scraping hasn't populated them yet)
                pA_clutch = float(p1_stats.get('clutch_factor', 0.5))
                pB_clutch = float(p2_stats.get('clutch_factor', 0.5))
                pA_lefty_winrate = float(p1_stats.get('lefty_winrate', 0.5))
                pB_lefty_winrate = float(p2_stats.get('lefty_winrate', 0.5))
                pA_serve_var = float(p1_stats.get('serve_var', 0.0))
                pB_serve_var = float(p2_stats.get('serve_var', 0.0))
                altitude = float(p1_stats.get('altitude', 0.0))
                air_density = float(p1_stats.get('air_density', 1.0))
                
                # Build a feature vector; pad or truncate to match checkpoint's feat_dim
                full_feats = [rank_diff, rank1, rank2, elo1, elo2, age1, age2, h1, h2,
                              hand1, hand2, best_of, surf_h, surf_c, surf_g,
                              cpi, pA_fatigue, pB_fatigue, pA_arch, pB_arch,
                              pA_clutch, pB_clutch, pA_lefty_winrate, pB_lefty_winrate,
                              pA_serve_var, pB_serve_var, altitude, air_density]
                
                if getattr(self, 'nn_scaler', None) is not None:
                    try:
                        scaled_feats = self.nn_scaler.transform([full_feats])[0].tolist()
                    except Exception:
                        scaled_feats = full_feats
                else:
                    scaled_feats = full_feats
                    
                match_row = (scaled_feats + [0.0] * feat_dim)[:feat_dim]
                match_feats = [match_row]
                t_feats = torch.tensor(match_feats, dtype=torch.float32)
                
                # Skip NN if both sequences are all-zero (no real history scraped).
                # An all-zero LSTM input produces extreme logits that push sigmoid → ~1.0
                # giving a false 100% confidence with no information.
                seq1_has_data = any(any(v != 0.0 for v in step) for step in s1)
                seq2_has_data = any(any(v != 0.0 for v in step) for step in s2)
                if not (seq1_has_data and seq2_has_data):
                    log.warning("NN skipped: zero-padded sequences (no scraped history). Using XGB only.")
                else:
                    with torch.no_grad():
                        raw_out = self.nn_model(t_id1, t_id2, t_seq1, t_seq2, t_len1, t_len2, t_surf, t_tourn, t_feats)
                        raw_prob = float(raw_out.squeeze().item())
                        # Temperature scaling (T=2.5) softens extreme logits caused by
                        # sparse player embeddings — maps 0.99 → ~0.88, 0.01 → ~0.12.
                        # Clamp raw_out away from 0/1 before log so we never hit log(0)=-inf.
                        T = 2.5
                        raw_clamped = raw_out.clamp(1e-7, 1.0 - 1e-7)
                        logit = torch.log(raw_clamped / (1.0 - raw_clamped))
                        scaled = torch.sigmoid(logit / T)
                        res["nn_prob"] = float(scaled.squeeze().item())
                        if abs(raw_prob - res["nn_prob"]) > 0.05:
                            log.debug(f"NN temperature scaling: raw={raw_prob:.3f} → scaled={res['nn_prob']:.3f}")
            except Exception as e:
                log.warning(f"Neural prediction failed: {e}")

        # 3. Hybrid Blending
        # Track which models actually ran (don't rely on output == 0.5 as a signal;
        # a legitimately balanced match can produce exactly 0.5).
        xgb_ran = self.xgb_model is not None and res["xgb_prob"] != 0.5  # noqa: 0.5 sentinel used only as init default
        nn_ran  = self.nn_model  is not None and res["nn_prob"]  != 0.5

        if self.blender is not None and self.blender.is_fitted:
            res["hybrid_prob"] = self.blender.predict_proba(res["xgb_prob"], res["nn_prob"])
        elif xgb_ran and nn_ran:
            res["hybrid_prob"] = 0.4 * res["xgb_prob"] + 0.6 * res["nn_prob"]
        elif xgb_ran:
            res["hybrid_prob"] = res["xgb_prob"]
        elif nn_ran:
            res["hybrid_prob"] = res["nn_prob"]
        # else: both defaulted to 0.5, hybrid stays 0.5
        return res

    def calibrate_blender(self, xgb_probs, nn_probs, targets):
        if self.blender is None:
            self.blender = XGBoostPyTorchBlender()
        self.blender.fit(xgb_probs, nn_probs, targets)

    def run_ablation(self, p1_stats: dict, p2_stats: dict,
                     seq1: list = None, seq2: list = None) -> dict:
        """
        Ablation test: compare all model variants for a single matchup.

        Variants:
          baseline  — always 0.5 (no model, sanity anchor)
          xgb_only  — XGBoost alone
          lstm_only — LSTM/Neural alone (falls back to 0.5 if no sequences)
          hybrid    — weighted blend of XGBoost + LSTM (production model)

        Logs a compact comparison table and returns the dict for callers.
        """
        full  = self.predict_win_prob(p1_stats, p2_stats, seq1=seq1, seq2=seq2)
        xgb_p = full["xgb_prob"]
        nn_p  = full["nn_prob"]

        # XGBoost-only hybrid weight
        xgb_only_hybrid = xgb_p if xgb_p != 0.5 else 0.5

        # LSTM-only (skip XGBoost weighting)
        lstm_only_p = nn_p if (seq1 and seq2 and nn_p != 0.5) else 0.5

        result = {
            "baseline":  0.5,
            "xgb_only":  xgb_only_hybrid,
            "lstm_only": lstm_only_p,
            "hybrid":    full["hybrid_prob"],
        }

        p1 = p1_stats.get("name", "P1")
        p2 = p2_stats.get("name", "P2")
        log.info(
            "[ABLATION] %s vs %s | baseline=%.3f  xgb=%.3f  lstm=%.3f  hybrid=%.3f",
            p1, p2,
            result["baseline"], result["xgb_only"],
            result["lstm_only"], result["hybrid"],
        )

        # Spread: how much each model differs from the hybrid
        for variant in ("baseline", "xgb_only", "lstm_only"):
            diff = abs(result[variant] - result["hybrid"])
            if diff > 0.10:
                log.warning(
                    "[ABLATION] Large divergence: %s vs hybrid = %.3f  (Δ=%.3f)",
                    variant, result[variant], diff,
                )

        return result


ml_engine = HybridMLEngine()
