"""
feature_engineering.py — Research-paper-derived feature engineering primitives.

Implements:
  1. CircadianDeficitScore  — exponentially decaying travel fatigue by direction
  2. PlayerArchetypeClustering — K-Means grouping on serve stats
  3. EWMAMicroStats — EWMA volatility / efficiency trackers
  4. EnvironmentalModifiers — court + weather coefficients on serve probability

These features are consumed by the ML pipeline (ml_trainer.py / ml_engine.py).
All classes are pure-python / numpy / scikit-learn only.  Graceful fallbacks
are provided for optional dependencies.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    KMeans = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Circadian Deficit Score
# ──────────────────────────────────────────────────────────────────────────────

class CircadianDeficitScore:
    """
    Exponentially decaying travel-fatigue score.

    Research finding:
      - Eastward travel: -24.5 min/timezone crossed  (harder to recover)
      - Westward travel: +30.0 min/timezone crossed  (easier to recover)

    The penalty decays exponentially after `half_life_days` days of acclimation.
    A negative score indicates *deficit* (fatigue); a positive score is
    acclimation tailwind.

    Example:
        cds = CircadianDeficitScore(half_life_days=1.5)
        score = cds.compute(timezones_crossed=-7, days_since_travel=2.0)
    """

    EASTWARD_PENALTY_MIN: float = -24.5
    WESTWARD_PENALTY_MIN: float = 30.0

    def __init__(self, half_life_days: float = 1.5) -> None:
        if half_life_days <= 0:
            raise ValueError("half_life_days must be positive.")
        self.half_life_days = float(half_life_days)
        # exp(-lambda * t_halflife) = 0.5  →  lambda = ln(2) / t_halflife
        self._lambda = math.log(2.0) / self.half_life_days

    def compute(
        self,
        timezones_crossed: float,
        days_since_travel: float,
    ) -> float:
        """
        Args:
            timezones_crossed: signed integer.
                negative = eastward (e.g. NYC → London = -5)
                positive = westward (e.g. London → NYC = +5)
            days_since_travel: days since arrival at the venue.

        Returns:
            Fatigue score in minutes of sleep debt (signed).
            Negative values indicate performance drag.
        """
        if days_since_travel < 0:
            days_since_travel = 0.0

        if timezones_crossed < 0:
            raw = self.EASTWARD_PENALTY_MIN * abs(timezones_crossed)
        else:
            raw = self.WESTWARD_PENALTY_MIN * abs(timezones_crossed)

        decay = math.exp(-self._lambda * days_since_travel)
        return raw * decay

    def batch_compute(
        self,
        records: Iterable[Tuple[float, float]],
    ) -> List[float]:
        """Vectorised version for dataset enrichment."""
        return [self.compute(tz, d) for tz, d in records]


# ──────────────────────────────────────────────────────────────────────────────
# 2. Player Archetype Clustering
# ──────────────────────────────────────────────────────────────────────────────

ARCHETYPE_LABELS: Tuple[str, str, str] = ("BigServer", "AllCourter", "Grinder")


@dataclass
class ArchetypeResult:
    label: str
    cluster_id: int
    one_hot_interactions: Dict[str, int] = field(default_factory=dict)


class PlayerArchetypeClustering:
    """
    K-Means clustering on (first-serve %, ace %, second-serve win %).

    Produces one of three labels: BigServer / AllCourter / Grinder and
    emits interaction features of the form
        Server_vs_Grinder_Clay
        AllCourter_vs_BigServer_HardCourt
    for downstream model consumption.
    """

    FEATURE_COLUMNS: Tuple[str, str, str] = (
        "first_serve_pct",
        "ace_pct",
        "second_serve_win_pct",
    )

    def __init__(self, n_clusters: int = 3, random_state: int = 42) -> None:
        self.n_clusters = n_clusters
        self.random_state = random_state
        self._model: Optional["KMeans"] = None
        self._label_map: Dict[int, str] = {}

    # .....................................................................
    def fit(self, feature_matrix: np.ndarray) -> "PlayerArchetypeClustering":
        """
        Args:
            feature_matrix: shape (n_players, 3) — columns matching FEATURE_COLUMNS.
        """
        if not _SKLEARN_AVAILABLE:
            log.warning("scikit-learn not available — using heuristic archetype rules.")
            return self._fit_heuristic(feature_matrix)

        X = np.asarray(feature_matrix, dtype=float)
        if X.shape[1] != 3:
            raise ValueError(f"Expected 3 feature columns, got {X.shape[1]}")

        self._model = KMeans(
            n_clusters=self.n_clusters,
            n_init=10,
            random_state=self.random_state,
        )
        self._model.fit(X)

        # Map cluster centroid -> human label based on (ace%, serve%) profile.
        # Highest ace% → BigServer; lowest → Grinder; middle → AllCourter.
        centroids = self._model.cluster_centers_
        ace_ranks = np.argsort(centroids[:, 1])  # ascending by ace_pct
        self._label_map = {
            int(ace_ranks[0]): "Grinder",
            int(ace_ranks[1]): "AllCourter",
            int(ace_ranks[2]): "BigServer",
        }
        return self

    def _fit_heuristic(self, X: np.ndarray) -> "PlayerArchetypeClustering":
        """Fallback when sklearn is unavailable — tercile split on ace_pct."""
        X = np.asarray(X, dtype=float)
        ace_col = X[:, 1]
        q33, q66 = np.quantile(ace_col, [0.33, 0.66])
        self._heuristic_thresholds = (q33, q66)
        self._model = None
        self._label_map = {0: "Grinder", 1: "AllCourter", 2: "BigServer"}
        return self

    # .....................................................................
    def predict_label(self, features: Sequence[float]) -> str:
        X = np.asarray(features, dtype=float).reshape(1, -1)
        if self._model is not None:
            cluster_id = int(self._model.predict(X)[0])
            return self._label_map.get(cluster_id, "AllCourter")

        # Heuristic path
        if not hasattr(self, "_heuristic_thresholds"):
            return "AllCourter"
        q33, q66 = self._heuristic_thresholds
        ace_pct = float(features[1])
        if ace_pct <= q33:
            return "Grinder"
        if ace_pct >= q66:
            return "BigServer"
        return "AllCourter"

    # .....................................................................
    def classify(
        self,
        features: Sequence[float],
        opponent_label: Optional[str] = None,
        surface: Optional[str] = None,
    ) -> ArchetypeResult:
        """
        Returns the archetype label plus one-hot interaction features.
        Interaction features have the form:
            `<self>_vs_<opponent>_<surface>`  -> 1
        and all other combinations -> 0.
        """
        label = self.predict_label(features)
        cluster_id = -1
        if self._model is not None:
            cluster_id = int(
                self._model.predict(np.asarray(features, dtype=float).reshape(1, -1))[0]
            )

        interactions: Dict[str, int] = {}
        if opponent_label and surface:
            key = f"{label}_vs_{opponent_label}_{surface}"
            interactions[key] = 1
        return ArchetypeResult(
            label=label,
            cluster_id=cluster_id,
            one_hot_interactions=interactions,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. EWMA Micro-Stats
# ──────────────────────────────────────────────────────────────────────────────

class EWMAMicroStats:
    """
    Exponentially weighted moving averages (and variances) of:
      - serve-velocity volatility
      - break efficiency (break points won / break points faced)

    Useful as short-term form signals fed into the probabilistic match model.
    """

    def __init__(self, alpha: float = 0.3) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        self.alpha = float(alpha)
        self._mean_vol: Optional[float] = None
        self._var_vol: float = 0.0
        self._break_eff: Optional[float] = None

    # .....................................................................
    def update_velocity(self, velocity_sample: float) -> float:
        """Update serve velocity volatility tracker and return current mean."""
        v = float(velocity_sample)
        if self._mean_vol is None:
            self._mean_vol = v
            self._var_vol = 0.0
            return self._mean_vol
        delta = v - self._mean_vol
        self._mean_vol += self.alpha * delta
        self._var_vol = (1.0 - self.alpha) * (self._var_vol + self.alpha * delta * delta)
        return self._mean_vol

    def velocity_volatility(self) -> float:
        return math.sqrt(max(self._var_vol, 0.0))

    # .....................................................................
    def update_break_efficiency(self, won: int, faced: int) -> float:
        """Update break-point efficiency with a new game summary."""
        if faced <= 0:
            return self._break_eff if self._break_eff is not None else 0.0
        sample = float(won) / float(faced)
        if self._break_eff is None:
            self._break_eff = sample
        else:
            self._break_eff = self.alpha * sample + (1.0 - self.alpha) * self._break_eff
        return self._break_eff

    # .....................................................................
    def snapshot(self) -> Dict[str, float]:
        return {
            "ewma_velocity_mean":       float(self._mean_vol or 0.0),
            "ewma_velocity_volatility": self.velocity_volatility(),
            "ewma_break_efficiency":    float(self._break_eff or 0.0),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 4. Environmental Modifiers
# ──────────────────────────────────────────────────────────────────────────────

class EnvironmentalModifiers:
    """
    Scales baseline serve-win probability by measurable environmental factors.

    Research-derived approximations:
      - Indoor matches favour the server by +1.5 percentage points.
      - Each +10C above 20C reduces serve probability by ~0.8 pp (heat/fatigue).
      - Each +20% humidity above 40% reduces serve probability by ~0.4 pp
        (slower ball through the air).
    """

    INDOOR_BONUS: float = 0.015
    TEMP_COEF_PER_C: float = -0.0008
    TEMP_BASELINE_C: float = 20.0
    HUMIDITY_COEF_PER_PCT: float = -0.0002
    HUMIDITY_BASELINE_PCT: float = 40.0

    def __init__(
        self,
        indoor_bonus: float = INDOOR_BONUS,
        temp_coef: float = TEMP_COEF_PER_C,
        humidity_coef: float = HUMIDITY_COEF_PER_PCT,
    ) -> None:
        self.indoor_bonus = float(indoor_bonus)
        self.temp_coef = float(temp_coef)
        self.humidity_coef = float(humidity_coef)

    def apply(
        self,
        base_serve_prob: float,
        *,
        indoor: bool = False,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> float:
        p = float(base_serve_prob)
        if indoor:
            p += self.indoor_bonus
        if temperature_c is not None:
            p += self.temp_coef * (float(temperature_c) - self.TEMP_BASELINE_C)
        if humidity_pct is not None:
            p += self.humidity_coef * (float(humidity_pct) - self.HUMIDITY_BASELINE_PCT)
        # Clamp to sane probabilistic range.
        return max(0.01, min(0.99, p))

    def to_feature_dict(
        self,
        *,
        indoor: bool = False,
        temperature_c: Optional[float] = None,
        humidity_pct: Optional[float] = None,
    ) -> Dict[str, float]:
        return {
            "env_indoor_flag":    1.0 if indoor else 0.0,
            "env_temperature_c":  float(temperature_c) if temperature_c is not None else 0.0,
            "env_humidity_pct":   float(humidity_pct)   if humidity_pct   is not None else 0.0,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Module-level convenience
# ──────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ARCHETYPE_LABELS",
    "ArchetypeResult",
    "CircadianDeficitScore",
    "EWMAMicroStats",
    "EnvironmentalModifiers",
    "PlayerArchetypeClustering",
]
