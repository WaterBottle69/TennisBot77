"""
bayesian_updater.py — Bayesian live-state fusion for serve probabilities.

A Beta(alpha, beta) conjugate model is maintained per player-surface context.
Each point outcome is absorbed with the standard Beta-Binomial update.
A small Gaussian drift term models fatigue and on-court momentum swings
between games.

The posterior mean / variance are consumed by the Markov engine
(markov_engine.py) and the risk manager to feed uncertainty-aware Kelly
sizing.
"""

from __future__ import annotations

import logging
import math
from typing import Tuple

import numpy as np

log = logging.getLogger(__name__)


class BayesianServeProbUpdater:
    """
    Online Beta(alpha, beta) estimator for P(point won on serve).

    Given a pre-match prior mean (e.g. from ATP stats) and a concentration
    parameter (effective prior sample size), each observed point updates the
    posterior.

    Example:
        up = BayesianServeProbUpdater(prior_mean=0.65, concentration=50)
        up.update(points_won=3, points_total=4)
        mu = up.get_posterior_mean()
    """

    def __init__(self, prior_mean: float, concentration: float = 50.0) -> None:
        if not (0.0 < prior_mean < 1.0):
            raise ValueError("prior_mean must be in (0, 1).")
        if concentration <= 0:
            raise ValueError("concentration must be positive.")

        self._prior_mean = float(prior_mean)
        self._concentration = float(concentration)

        # Beta(a, b) with a + b = concentration, a / (a+b) = prior_mean
        self.alpha_pre = prior_mean * concentration
        self.beta_pre = (1.0 - prior_mean) * concentration

        self.alpha = self.alpha_pre
        self.beta = self.beta_pre

    # .....................................................................
    def update(self, points_won: int, points_total: int) -> float:
        """
        Conjugate Beta-Binomial update.

        Args:
            points_won:   successes in this batch.
            points_total: total points observed in this batch.

        Returns:
            Posterior mean after the update.
        """
        if points_total < 0 or points_won < 0 or points_won > points_total:
            raise ValueError("Invalid update counts.")
        self.alpha += float(points_won)
        self.beta += float(points_total - points_won)
        return self.get_posterior_mean()

    # .....................................................................
    def get_posterior(self) -> Tuple[float, float]:
        """Returns `(alpha_post, beta_post)`."""
        return self.alpha, self.beta

    def get_posterior_mean(self) -> float:
        denom = self.alpha + self.beta
        return self.alpha / denom if denom > 0 else 0.5

    def get_uncertainty(self) -> float:
        """Variance of the current Beta posterior."""
        a, b = self.alpha, self.beta
        denom = (a + b) ** 2 * (a + b + 1.0)
        return (a * b) / denom if denom > 0 else 0.0

    def get_credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """Approximate (low, high) credible interval via normal approximation."""
        mu = self.get_posterior_mean()
        sigma = math.sqrt(self.get_uncertainty())
        # z-score for two-sided level
        if level >= 0.99:
            z = 2.576
        elif level >= 0.95:
            z = 1.96
        elif level >= 0.90:
            z = 1.645
        else:
            z = 1.0
        return max(0.0, mu - z * sigma), min(1.0, mu + z * sigma)

    # .....................................................................
    def reset(self) -> None:
        """Reset to the original prior."""
        self.alpha = self.alpha_pre
        self.beta = self.beta_pre


# ──────────────────────────────────────────────────────────────────────────────
# Skill drift
# ──────────────────────────────────────────────────────────────────────────────

class GaussianSkillDrift:
    """
    Applies a small Gaussian random walk to Beta parameters between games to
    model fatigue / momentum drift.  The update preserves the posterior mean
    in expectation but re-inflates variance, reflecting reduced confidence
    as time passes.
    """

    def __init__(
        self,
        drift_std: float = 0.5,
        concentration_decay: float = 0.98,
        rng_seed: int = 42,
    ) -> None:
        if drift_std < 0:
            raise ValueError("drift_std must be non-negative.")
        if not (0.0 < concentration_decay <= 1.0):
            raise ValueError("concentration_decay must be in (0, 1].")
        self.drift_std = float(drift_std)
        self.concentration_decay = float(concentration_decay)
        self._rng = np.random.default_rng(rng_seed)

    def apply(self, updater: BayesianServeProbUpdater) -> None:
        """
        Mutates `updater` in place:
          1. Shrinks (alpha + beta) by `concentration_decay` (more uncertainty).
          2. Adds symmetric Gaussian noise to alpha, balancing it from beta.
        """
        total = updater.alpha + updater.beta
        mean = updater.get_posterior_mean()

        # Variance inflation via concentration decay.
        new_total = max(2.0, total * self.concentration_decay)

        # Small random perturbation on the mean.
        noise = self._rng.normal(0.0, self.drift_std)
        new_mean = max(0.01, min(0.99, mean + noise / max(new_total, 1.0)))

        updater.alpha = new_mean * new_total
        updater.beta = (1.0 - new_mean) * new_total


__all__ = ["BayesianServeProbUpdater", "GaussianSkillDrift"]
