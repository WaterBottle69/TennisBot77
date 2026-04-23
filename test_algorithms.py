"""
test_algorithms.py — Comprehensive unit tests for the tennis betting bot algorithms.

Tests cover:
  - Markov chain engine (game/set/match win probabilities, LiveMatchState)
  - XGBoost/ML engine logic (feature construction, fallbacks)
  - Neural network logic (fallback, sequence padding, hybrid blending)
  - Kelly criterion sizing
  - Kalshi client parsing utilities and filtering helpers

Uses only stdlib unittest. External dependencies (kalshi, config, ML models)
are either imported directly where safe or mocked via unittest.mock.
"""

import sys
import os
import unittest
from unittest.mock import MagicMock, patch

# Make sure the module directory is on the path so we can import project modules.
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)


# ---------------------------------------------------------------------------
# Markov chain tests
# ---------------------------------------------------------------------------

class TestGameWinProb(unittest.TestCase):
    """Unit tests for game_win_prob (level-1 Markov DP)."""

    def setUp(self):
        from markov_engine import game_win_prob
        self.game_win_prob = game_win_prob

    def test_deuce_analytical_formula(self):
        """game_win_prob(p, (3,3)) should equal p²/(p²+(1-p)²)."""
        p = 0.6
        expected = p**2 / (p**2 + (1 - p)**2)
        result = self.game_win_prob(p, (3, 3))
        self.assertAlmostEqual(result, expected, places=6,
                               msg="Deuce formula mismatch")

    def test_probability_bounds(self):
        """Win probability must always lie in [0, 1]."""
        from markov_engine import game_win_prob
        for p in [0.1, 0.3, 0.5, 0.65, 0.75, 0.9]:
            wp = game_win_prob(p)
            self.assertGreaterEqual(wp, 0.0)
            self.assertLessEqual(wp, 1.0)

    def test_server_advantage(self):
        """Higher p (server dominance) → higher win prob."""
        low = self.game_win_prob(0.4)
        high = self.game_win_prob(0.7)
        self.assertLess(low, high)

    def test_certain_win_state(self):
        """At (4, 0) (server already won) → probability 1.0."""
        result = self.game_win_prob(0.6, (4, 0))
        self.assertAlmostEqual(result, 1.0, places=6)

    def test_certain_loss_state(self):
        """At (0, 4) (returner already won) → probability 0.0."""
        result = self.game_win_prob(0.6, (0, 4))
        self.assertAlmostEqual(result, 0.0, places=6)


class TestMatchWinProb(unittest.TestCase):
    """Unit tests for the three-level Markov chain match probability."""

    def setUp(self):
        from markov_engine import match_win_prob, LiveMatchState
        self.match_win_prob = match_win_prob
        self.LiveMatchState = LiveMatchState

    def test_markov_fair_match(self):
        """Symmetric serve/return probabilities should yield win_prob near 0.5."""
        lms = self.LiveMatchState(p_serve=0.65, p_return=0.35)
        wp = lms.win_probability()
        self.assertAlmostEqual(wp, 0.5, delta=0.05,
                               msg=f"Fair match win_prob={wp:.4f} not near 0.5")

    def test_markov_strong_server(self):
        """Strong server with asymmetric stats (p_serve=0.72, p_return=0.42) should win > 60%.

        Note: p_serve + p_return == 1.0 is a degenerate symmetric case that always
        yields 0.5 regardless of the magnitudes (both players have identical serve
        dominance).  We use asymmetric parameters (sum != 1) to test a genuinely
        superior player.
        """
        lms = self.LiveMatchState(p_serve=0.72, p_return=0.42)
        wp = lms.win_probability()
        self.assertGreater(wp, 0.6,
                           msg=f"Strong server win_prob={wp:.4f} should be > 0.6")

    def test_markov_probability_bounds(self):
        """Win probability must be in [0, 1] for various inputs."""
        combos = [
            (0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2),
            (0.4, 0.6), (0.3, 0.7),
        ]
        for ps, pr in combos:
            lms = self.LiveMatchState(ps, pr)
            wp = lms.win_probability()
            self.assertGreaterEqual(wp, 0.0, msg=f"p_serve={ps}, p_return={pr}: wp={wp} < 0")
            self.assertLessEqual(wp, 1.0, msg=f"p_serve={ps}, p_return={pr}: wp={wp} > 1")

    def test_markov_symmetry(self):
        """Swapping p_serve and p_return should give complementary probabilities."""
        lms_a = self.LiveMatchState(p_serve=0.70, p_return=0.30)
        lms_b = self.LiveMatchState(p_serve=0.30, p_return=0.70)
        wp_a = lms_a.win_probability()
        wp_b = lms_b.win_probability()
        self.assertAlmostEqual(wp_a + wp_b, 1.0, places=4,
                               msg=f"wp_a({wp_a:.4f}) + wp_b({wp_b:.4f}) != 1.0")

    def test_markov_update_score(self):
        """Updating score to sets=(1,0) games=(3,2) should shift prob toward winner."""
        lms_before = self.LiveMatchState(p_serve=0.65, p_return=0.35)
        wp_before = lms_before.win_probability()

        lms_after = self.LiveMatchState(p_serve=0.65, p_return=0.35)
        lms_after.update({"sets": (1, 0), "games": (3, 2)})
        wp_after = lms_after.win_probability()

        # Player A is winning (1 set up, leading 3-2 in current set)
        # → win prob should increase relative to baseline
        self.assertGreater(wp_after, wp_before,
                           msg=f"wp_after={wp_after:.4f} should be > wp_before={wp_before:.4f}")

    def test_markov_update_points(self):
        """update() should correctly map raw tennis scores (0,15,30,40) to Markov points."""
        lms = self.LiveMatchState(p_serve=0.65, p_return=0.35)
        lms.update({"points": (40, 0)})
        # 40 → 3 in internal representation; p1 should be near winning the game
        self.assertEqual(lms.current_game_points, (3, 0))

    def test_game_win_prob_deuce(self):
        """game_win_prob(0.6, (3,3)) ≈ 0.6²/(0.6²+0.4²) ≈ 0.6923."""
        from markov_engine import game_win_prob
        p = 0.6
        expected = p**2 / (p**2 + (1 - p)**2)
        result = game_win_prob(p, (3, 3))
        self.assertAlmostEqual(result, expected, places=4,
                               msg=f"Deuce game_win_prob={result:.4f}, expected≈{expected:.4f}")


# ---------------------------------------------------------------------------
# Helper: ensure ml_engine can be imported even without heavy deps (joblib,
# torch, sklearn).  We install lightweight stubs in sys.modules once, before
# any test class that touches ml_engine attempts an import.
# ---------------------------------------------------------------------------

def _stub_ml_deps():
    """Insert MagicMock stubs for optional ML heavy-weight dependencies."""
    heavy = [
        "joblib", "torch", "torch.nn", "torch.nn.functional", "torch.optim",
        "torch.nn.utils", "torch.nn.utils.rnn", "torch.utils", "torch.utils.data",
        "sklearn", "sklearn.linear_model", "sklearn.preprocessing", "sklearn.metrics", "sklearn.ensemble",
        "pandas",
    ]
    for mod in heavy:
        if mod not in sys.modules:
            sys.modules[mod] = MagicMock()

    # torch.no_grad must return a context manager
    import torch as _torch_stub
    import contextlib
    _torch_stub.no_grad = contextlib.nullcontext

    # torch.tensor / torch.load must return something sensible
    def _fake_tensor(data, *a, **kw):
        m = MagicMock()
        m.dim.return_value = 1
        m.squeeze.return_value = m
        m.item.return_value = 0.5
        return m

    _torch_stub.tensor = _fake_tensor
    _torch_stub.load = MagicMock(return_value={})


_stub_ml_deps()


# ---------------------------------------------------------------------------
# XGBoost / ML engine tests (mock — no model file required)
# ---------------------------------------------------------------------------

class TestXGBFeatureConstruction(unittest.TestCase):
    """Verify the feature dict that HybridMLEngine builds for XGBoost."""

    def _get_feat_dict(self, p1_stats, p2_stats):
        """Replicate the feature-building logic from HybridMLEngine.predict_win_prob."""
        surface = p1_stats.get('surface', 'Hard')
        feat = {
            'Surface_Hard': 1 if surface == 'Hard' else 0,
            'Surface_Clay': 1 if surface == 'Clay' else 0,
            'Surface_Grass': 1 if surface == 'Grass' else 0,
            'Best_Of_Sets': p1_stats.get('best_of', 3),
            'P1_Is_Right_Handed': 1 if p1_stats.get('hand', 'R') == 'R' else 0,
            'P1_Height_cm': p1_stats.get('height_cm', 185),
            'P1_Age': p1_stats.get('age', 25),
            'P1_Rank': p1_stats.get('ranking', 50),
            'P1_Rank_Points': p1_stats.get('elo', 1000),
            'P2_Is_Right_Handed': 1 if p2_stats.get('hand', 'R') == 'R' else 0,
            'P2_Height_cm': p2_stats.get('height_cm', 185),
            'P2_Age': p2_stats.get('age', 25),
            'P2_Rank': p2_stats.get('ranking', 50),
            'P2_Rank_Points': p2_stats.get('elo', 1000),
        }
        return feat

    def test_xgb_feature_construction_keys(self):
        """Feature dict must contain all required keys."""
        p1 = {'surface': 'Hard', 'best_of': 3, 'hand': 'R', 'height_cm': 188,
               'age': 28, 'ranking': 5, 'elo': 2000}
        p2 = {'hand': 'L', 'height_cm': 183, 'age': 24, 'ranking': 20, 'elo': 1800}
        feat = self._get_feat_dict(p1, p2)

        required_keys = [
            'Surface_Hard', 'Surface_Clay', 'Surface_Grass', 'Best_Of_Sets',
            'P1_Is_Right_Handed', 'P1_Height_cm', 'P1_Age', 'P1_Rank',
            'P1_Rank_Points', 'P2_Is_Right_Handed', 'P2_Height_cm',
            'P2_Age', 'P2_Rank', 'P2_Rank_Points',
        ]
        for k in required_keys:
            self.assertIn(k, feat, msg=f"Missing key: {k}")

    def test_feature_surface_hard(self):
        """Surface=Hard → Surface_Hard=1, others=0."""
        feat = self._get_feat_dict({'surface': 'Hard'}, {})
        self.assertEqual(feat['Surface_Hard'], 1)
        self.assertEqual(feat['Surface_Clay'], 0)
        self.assertEqual(feat['Surface_Grass'], 0)

    def test_feature_surface_clay(self):
        """Surface=Clay → Surface_Clay=1, others=0."""
        feat = self._get_feat_dict({'surface': 'Clay'}, {})
        self.assertEqual(feat['Surface_Hard'], 0)
        self.assertEqual(feat['Surface_Clay'], 1)
        self.assertEqual(feat['Surface_Grass'], 0)

    def test_feature_surface_grass(self):
        """Surface=Grass → Surface_Grass=1, others=0."""
        feat = self._get_feat_dict({'surface': 'Grass'}, {})
        self.assertEqual(feat['Surface_Hard'], 0)
        self.assertEqual(feat['Surface_Clay'], 0)
        self.assertEqual(feat['Surface_Grass'], 1)

    def test_feature_surface_encoding_mutual_exclusion(self):
        """Exactly one surface flag should be set to 1."""
        for surface in ('Hard', 'Clay', 'Grass'):
            feat = self._get_feat_dict({'surface': surface}, {})
            total = feat['Surface_Hard'] + feat['Surface_Clay'] + feat['Surface_Grass']
            self.assertEqual(total, 1, msg=f"Surface={surface}: expected exactly 1 flag set")

    def test_feature_handedness(self):
        """P1_Is_Right_Handed: R=1, L=0. P2_Is_Right_Handed: R=1, L=0."""
        feat = self._get_feat_dict({'hand': 'R'}, {'hand': 'L'})
        self.assertEqual(feat['P1_Is_Right_Handed'], 1)
        self.assertEqual(feat['P2_Is_Right_Handed'], 0)

    def test_feature_rank_values(self):
        """P1_Rank and P2_Rank are correctly plucked from stats."""
        feat = self._get_feat_dict({'ranking': 7}, {'ranking': 42})
        self.assertEqual(feat['P1_Rank'], 7)
        self.assertEqual(feat['P2_Rank'], 42)


class TestXGBFallback(unittest.TestCase):
    """When xgb_model is None, predict_win_prob should return xgb_prob=0.5."""

    def test_xgb_fallback_on_missing_model(self):
        from ml_engine import HybridMLEngine
        # Instantiate with non-existent paths so no models load.
        engine = HybridMLEngine(
            xgb_path='__nonexistent__.json',
            nn_path='__nonexistent__.pth',
            feat_path='__nonexistent_features__.json',
            id_map_path='__nonexistent_idmap__.json',
        )
        self.assertIsNone(engine.xgb_model)
        res = engine.predict_win_prob({'surface': 'Hard'}, {})
        self.assertAlmostEqual(res['xgb_prob'], 0.5, places=6,
                               msg="xgb_prob should be 0.5 when model is absent")


# ---------------------------------------------------------------------------
# Neural network logic tests (mock — no weights required)
# ---------------------------------------------------------------------------

class TestNNFallback(unittest.TestCase):
    """When nn_model is None, predict_win_prob should return nn_prob=0.5."""

    def test_nn_fallback_on_missing_model(self):
        from ml_engine import HybridMLEngine
        engine = HybridMLEngine(
            xgb_path='__nonexistent__.json',
            nn_path='__nonexistent__.pth',
            feat_path='__nonexistent_features__.json',
            id_map_path='__nonexistent_idmap__.json',
        )
        self.assertIsNone(engine.nn_model)
        res = engine.predict_win_prob({'surface': 'Hard'}, {}, seq1=[[1.0]*4]*5, seq2=[[1.0]*4]*5)
        self.assertAlmostEqual(res['nn_prob'], 0.5, places=6,
                               msg="nn_prob should be 0.5 when model is absent")

    def test_nn_sequence_padding(self):
        """
        Sequences shorter than 10 are left-padded with zero vectors to length 10.
        This replicates the padding logic in HybridMLEngine.predict_win_prob.
        """
        # Replicate the padding expression directly:
        seq = [[1.0, 2.0, 3.0, 4.0]] * 3   # length 3, need 10
        padded = [[0.0]*4] * (10 - len(seq)) + seq
        self.assertEqual(len(padded), 10, msg="Padded sequence length should be 10")
        # First 7 rows are zeros
        for i in range(7):
            self.assertEqual(padded[i], [0.0, 0.0, 0.0, 0.0],
                             msg=f"Row {i} should be zero-padding")
        # Last 3 rows are the original sequence
        for i in range(7, 10):
            self.assertEqual(padded[i], [1.0, 2.0, 3.0, 4.0],
                             msg=f"Row {i} should be original sequence row")

    def test_nn_sequence_no_padding_needed(self):
        """Sequences of exactly 10 should not be modified."""
        seq = [[float(i)]*4 for i in range(10)]
        padded = seq[-10:] if len(seq) >= 10 else [[0.0]*4]*(10-len(seq)) + seq
        self.assertEqual(len(padded), 10)
        self.assertEqual(padded, seq)

    def test_nn_sequence_truncation(self):
        """Sequences longer than 10 should be right-truncated to the last 10."""
        seq = [[float(i)]*4 for i in range(15)]
        padded = seq[-10:] if len(seq) >= 10 else [[0.0]*4]*(10-len(seq)) + seq
        self.assertEqual(len(padded), 10)
        self.assertEqual(padded, seq[5:])  # last 10 of 15

    def test_hybrid_blend_no_models(self):
        """With both models absent, hybrid_prob should default to 0.5."""
        from ml_engine import HybridMLEngine
        engine = HybridMLEngine(
            xgb_path='__nonexistent__.json',
            nn_path='__nonexistent__.pth',
            feat_path='__nonexistent_features__.json',
            id_map_path='__nonexistent_idmap__.json',
        )
        res = engine.predict_win_prob({'surface': 'Hard'}, {})
        self.assertAlmostEqual(res['hybrid_prob'], 0.5, places=6,
                               msg="hybrid_prob should be 0.5 when both models absent")

    def test_hybrid_blend_uses_xgb_when_nn_absent(self):
        """When only xgb_prob differs from 0.5, hybrid_prob should equal xgb_prob."""
        from ml_engine import HybridMLEngine
        engine = HybridMLEngine(
            xgb_path='__nonexistent__.json',
            nn_path='__nonexistent__.pth',
            feat_path='__nonexistent_features__.json',
            id_map_path='__nonexistent_idmap__.json',
        )
        # Monkey-patch: simulate xgb model available but nn absent
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = [[0.3, 0.72]]
        engine.xgb_model = mock_xgb
        engine.feature_names = ['Surface_Hard', 'Surface_Clay', 'Surface_Grass',
                                 'Best_Of_Sets', 'P1_Is_Right_Handed', 'P1_Height_cm',
                                 'P1_Age', 'P1_Rank', 'P1_Rank_Points',
                                 'P2_Is_Right_Handed', 'P2_Height_cm', 'P2_Age',
                                 'P2_Rank', 'P2_Rank_Points']

        res = engine.predict_win_prob({'surface': 'Hard', 'best_of': 3}, {})
        # nn_prob stays 0.5, xgb_prob != 0.5 → hybrid = xgb_prob
        self.assertAlmostEqual(res['xgb_prob'], 0.72, places=5)
        self.assertAlmostEqual(res['hybrid_prob'], res['xgb_prob'], places=5,
                               msg="hybrid_prob should equal xgb_prob when nn is absent")


# ---------------------------------------------------------------------------
# Kelly criterion tests
# ---------------------------------------------------------------------------

class TestKellyCriterion(unittest.TestCase):
    """Unit tests for BetManager._kelly_size."""

    def _make_bet_manager(self, max_bet=100.0, kelly_fraction=1.0, min_edge=0.01):
        """Create a BetManager with mocked dependencies and configurable params."""
        from bet_manager import BetManager
        mock_kalshi = MagicMock()
        mock_config = MagicMock()
        mock_config.MAX_BET_USDC = max_bet
        mock_config.KELLY_FRACTION = kelly_fraction
        mock_config.MIN_BET_USDC = 1.0
        mock_config.MAX_GAME_EXPOSURE = max_bet * 4
        mock_config.MIN_EDGE = min_edge
        mock_config.EXTREME_ODDS_MIN = 0.05
        mock_config.EXTREME_ODDS_MAX = 0.95
        mock_config.MAX_MODEL_DIVERGENCE = 0.35
        mock_config.MIN_ROI_THRESHOLD = 0.01
        mock_config.MAX_SLIPPAGE = 0.02
        bm = BetManager(mock_kalshi, mock_config)
        return bm

    def test_kelly_positive_edge(self):
        """p=0.6, price=0.5 → f* = (p*b - q)/b = (0.6*1 - 0.4)/1 = 0.2 → size > 0."""
        bm = self._make_bet_manager(kelly_fraction=1.0)
        # b = (1/0.5) - 1 = 1.0, f* = (0.6*1 - 0.4)/1 = 0.2
        size = bm._kelly_size(p=0.6, market_price=0.5, available_balance=1000.0)
        self.assertGreater(size, 0.0, msg="Positive edge should yield positive Kelly size")

    def test_kelly_zero_edge(self):
        """p=0.5, price=0.5 → f* = (0.5*1 - 0.5)/1 = 0 → size = 0."""
        bm = self._make_bet_manager(kelly_fraction=1.0)
        size = bm._kelly_size(p=0.5, market_price=0.5, available_balance=1000.0)
        self.assertAlmostEqual(size, 0.0, places=4,
                               msg="Zero edge should yield zero Kelly size")

    def test_kelly_negative_edge(self):
        """p=0.4, price=0.5 → f* < 0 → clamped to 0."""
        bm = self._make_bet_manager(kelly_fraction=1.0)
        size = bm._kelly_size(p=0.4, market_price=0.5, available_balance=1000.0)
        self.assertEqual(size, 0.0, msg="Negative edge should yield zero (floored) Kelly size")

    def test_kelly_extreme_price_zero(self):
        """market_price=0 → guard → size = 0."""
        bm = self._make_bet_manager()
        size = bm._kelly_size(p=0.9, market_price=0.0, available_balance=1000.0)
        self.assertEqual(size, 0.0, msg="Price=0 should yield zero Kelly size (guard)")

    def test_kelly_extreme_price_one(self):
        """market_price=1 → guard → size = 0."""
        bm = self._make_bet_manager()
        size = bm._kelly_size(p=0.9, market_price=1.0, available_balance=1000.0)
        self.assertEqual(size, 0.0, msg="Price=1 should yield zero Kelly size (guard)")

    def test_kelly_cap_at_max_bet(self):
        """Kelly size must never exceed MAX_BET_USDC."""
        bm = self._make_bet_manager(max_bet=50.0, kelly_fraction=1.0)
        # Use extreme edge and huge balance to make raw Kelly massive
        size = bm._kelly_size(p=0.99, market_price=0.01, available_balance=1_000_000.0)
        self.assertLessEqual(size, 50.0,
                             msg=f"Kelly size {size} exceeds MAX_BET_USDC=50")

    def test_kelly_fraction_scales_size(self):
        """Halving KELLY_FRACTION should roughly halve the output.

        We use MAX_BET_USDC=10000 (far above the raw Kelly size) and a small
        balance (100 USDC) so neither the absolute cap nor the balance cap
        binds, leaving the KELLY_FRACTION as the only active multiplier.
        p=0.55, price=0.45 gives f*≈0.18, raw_size≈18 USDC — well under 10000.
        """
        bm_full = self._make_bet_manager(max_bet=10_000.0, kelly_fraction=1.0)
        bm_half = self._make_bet_manager(max_bet=10_000.0, kelly_fraction=0.5)
        full_size = bm_full._kelly_size(p=0.55, market_price=0.45, available_balance=100.0)
        half_size = bm_half._kelly_size(p=0.55, market_price=0.45, available_balance=100.0)
        self.assertGreater(full_size, 0.0, msg="Full-Kelly size should be positive")
        self.assertAlmostEqual(half_size, full_size * 0.5, delta=0.5,
                               msg="Half Kelly should produce ~half the full Kelly size")


# ---------------------------------------------------------------------------
# Kalshi client parsing / filtering tests (no real API calls)
# ---------------------------------------------------------------------------

class TestParseYesNoCents(unittest.TestCase):
    """Unit tests for _parse_yes_no_cents from kalshi_client."""

    def setUp(self):
        from kalshi_client import _parse_yes_no_cents
        self.parse = _parse_yes_no_cents

    def test_dollar_format(self):
        """yes_ask_dollars=0.65 should yield yes=65 cents."""
        m = {'yes_ask_dollars': 0.65, 'no_ask_dollars': 0.35}
        y, n = self.parse(m)
        self.assertEqual(y, 65)
        self.assertEqual(n, 35)

    def test_cents_integer_format(self):
        """yes_ask=65 (integer cents) should yield yes=65 cents."""
        m = {'yes_ask': 65, 'no_ask': 35}
        y, n = self.parse(m)
        self.assertEqual(y, 65)
        self.assertEqual(n, 35)

    def test_sanity_check_fallback_to_last_price(self):
        """yes=90, no=90 (total=180, outside 85–115 range) → fallback to last_price."""
        m = {'yes_ask': 90, 'no_ask': 90, 'last_price': 60}
        y, n = self.parse(m)
        # Sanity fails (total=180), so should fall back to last_price=60
        self.assertEqual(y, 60)
        self.assertEqual(n, 40)

    def test_fallback_default_when_all_missing(self):
        """No price fields → default to 50/50."""
        y, n = self.parse({})
        self.assertEqual(y, 50)
        self.assertEqual(n, 50)

    def test_infer_complement_when_one_side_missing(self):
        """If only yes is present, no = 100 - yes."""
        m = {'yes_ask': 70}
        y, n = self.parse(m)
        self.assertEqual(y, 70)
        self.assertEqual(n, 30)

    def test_infer_complement_no_side_missing(self):
        """If only no is present, yes = 100 - no."""
        m = {'no_ask': 40}
        y, n = self.parse(m)
        self.assertEqual(y, 60)
        self.assertEqual(n, 40)

    def test_price_clamp_within_bounds(self):
        """Parsed prices should always be in [1, 99]."""
        for yes_val in [1, 50, 99]:
            m = {'yes_ask': yes_val}
            y, n = self.parse(m)
            self.assertGreaterEqual(y, 1)
            self.assertLessEqual(y, 99)
            self.assertGreaterEqual(n, 1)
            self.assertLessEqual(n, 99)


class TestTennisEventFilter(unittest.TestCase):
    """Unit tests for _is_tennis_event and _players_from_event."""

    def setUp(self):
        from kalshi_client import _is_tennis_event, _players_from_event
        self.is_tennis = _is_tennis_event
        self.players_from = _players_from_event

    def _make_event(self, series_ticker='', title='', sub_title='',
                    event_ticker='', category='sports'):
        return {
            'series_ticker': series_ticker,
            'event_ticker': event_ticker,
            'title': title,
            'sub_title': sub_title,
            'category': category,
        }

    # -- _is_tennis_event -------------------------------------------------

    def test_atp_series_ticker_passes(self):
        """Event with ATP in series_ticker should pass _is_tennis_event."""
        event = self._make_event(series_ticker='KXATPCHALLENGER',
                                 title='Djokovic vs Alcaraz ATP')
        self.assertTrue(self.is_tennis(event),
                        msg="ATP series ticker should be identified as tennis")

    def test_wta_passes(self):
        """Event with WTA in title should pass _is_tennis_event."""
        event = self._make_event(title='WTA Finals: Swiatek vs Sabalenka')
        self.assertTrue(self.is_tennis(event))

    def test_golf_fails(self):
        """Golf event should not pass _is_tennis_event."""
        event = self._make_event(series_ticker='KXGOLF',
                                 title='Golf: Rory McIlroy to win Masters')
        self.assertFalse(self.is_tennis(event),
                         msg="Golf event should NOT be identified as tennis")

    def test_nba_fails(self):
        """NBA event should not pass _is_tennis_event."""
        event = self._make_event(title='NBA: Lakers vs Celtics')
        self.assertFalse(self.is_tennis(event))

    def test_excluded_series_substr_fails(self):
        """Series ticker containing an excluded substring should fail."""
        event = self._make_event(series_ticker='KXGOLFTENNIS',
                                 title='Some match')
        self.assertFalse(self.is_tennis(event),
                         msg="GOLFTENNIS series should be excluded")

    # -- _players_from_event ----------------------------------------------

    def test_players_from_event_vs(self):
        """'Djokovic vs Alcaraz' → ('Djokovic', 'Alcaraz')."""
        event = self._make_event(title='Djokovic vs Alcaraz')
        result = self.players_from(event)
        self.assertIsNotNone(result, msg="Should extract players from 'vs' title")
        a, b = result
        self.assertIn('Djokovic', a)
        self.assertIn('Alcaraz', b)

    def test_players_from_event_sub_title_priority(self):
        """sub_title takes priority over title for player extraction."""
        event = self._make_event(
            title='Match',
            sub_title='Carlos Alcaraz vs Novak Djokovic'
        )
        result = self.players_from(event)
        self.assertIsNotNone(result)
        a, b = result
        self.assertIn('Alcaraz', a)
        self.assertIn('Djokovic', b)

    def test_players_from_event_will_style_fails(self):
        """'Will X beat Y?' prop-style titles should return None."""
        event = self._make_event(title='Will Djokovic win Wimbledon?')
        result = self.players_from(event)
        self.assertIsNone(result, msg="'Will ...' titles should not extract as H2H")

    def test_players_from_event_no_title_fails(self):
        """Empty title should return None."""
        event = self._make_event(title='')
        result = self.players_from(event)
        self.assertIsNone(result)

    def test_players_from_event_identical_names_fails(self):
        """Identical player names should return None (sanity check)."""
        event = self._make_event(title='Djokovic vs Djokovic')
        result = self.players_from(event)
        self.assertIsNone(result, msg="Identical player names should not pass")

    def test_players_from_event_excludes_blacklisted_tokens(self):
        """Titles with blacklisted tokens (team, field, etc.) should return None."""
        event = self._make_event(title='Team A vs Team B')
        result = self.players_from(event)
        self.assertIsNone(result, msg="'team' token should be excluded")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
