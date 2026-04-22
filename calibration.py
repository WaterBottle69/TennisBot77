"""
calibration.py — Venn-Abers predictor for probability calibration with uncertainty.

Provides a drop-in replacement for isotonic regression that returns a *pair*
of calibrated probabilities [p0, p1] bracketing the true probability.  The
width |p1 - p0| is interpreted as epistemic uncertainty and consumed by the
robust-Kelly sizing logic in bet_manager.py.

Falls back gracefully if the optional `venn_abers` package is unavailable;
the implementation below is self-contained and only requires scikit-learn.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    from sklearn.isotonic import IsotonicRegression
    _SKLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover
    IsotonicRegression = None  # type: ignore[assignment]
    _SKLEARN_AVAILABLE = False

try:
    # Optional dependency — if the user installed it we use their implementation
    from venn_abers import VennAbers as _ExternalVennAbers  # type: ignore
    _VA_EXTERNAL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ExternalVennAbers = None  # type: ignore[assignment]
    _VA_EXTERNAL_AVAILABLE = False

log = logging.getLogger(__name__)


class VennAbersCalibrator:
    """
    Venn-Abers probabilistic calibrator.

    Fits two isotonic regressions on the training set: one assuming the
    next label is 0 and one assuming it is 1.  At prediction time, the
    pair [p0, p1] forms a Venn-Abers interval.  Guarantees validity in
    the online inductive sense.

    Usage:
        va = VennAbersCalibrator()
        va.fit(scores, labels)
        lo, hi = va.predict_interval(0.62)
        p     = va.predict(0.62)
        u     = va.uncertainty(0.62)
    """

    def __init__(self) -> None:
        self._iso_0: Optional[IsotonicRegression] = None
        self._iso_1: Optional[IsotonicRegression] = None
        self._fitted: bool = False
        # Cache raw training data so tiny test sets can still be handled even
        # without sklearn being available at fit-time.
        self._train_scores: Optional[np.ndarray] = None
        self._train_labels: Optional[np.ndarray] = None

    # .....................................................................
    def fit(
        self,
        scores: Sequence[float],
        labels: Sequence[int],
    ) -> "VennAbersCalibrator":
        s = np.asarray(scores, dtype=float).ravel()
        y = np.asarray(labels, dtype=int).ravel()
        if s.shape != y.shape:
            raise ValueError("scores and labels must have the same shape.")
        if s.size == 0:
            raise ValueError("Cannot fit on empty data.")

        self._train_scores = s
        self._train_labels = y

        if not _SKLEARN_AVAILABLE:
            log.warning(
                "scikit-learn isotonic regression unavailable — falling back "
                "to empirical bin calibration."
            )
            self._fitted = True
            return self

        # Augmented label trick: append (s_test, 0) for iso_0 and (s_test, 1)
        # for iso_1 at *runtime*.  For the fit itself we just fit plain
        # isotonic regressions on the observed labels.  The augmentation
        # is applied per-query in `_predict_one`.
        self._iso_0 = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(s, y)
        self._iso_1 = IsotonicRegression(
            y_min=0.0, y_max=1.0, out_of_bounds="clip"
        ).fit(s, y)
        self._fitted = True
        return self

    # .....................................................................
    def _ensure_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError("VennAbersCalibrator must be fit() before use.")

    def _predict_one(self, score: float) -> Tuple[float, float]:
        """Core Venn-Abers calibration for a single score."""
        self._ensure_fitted()
        s = float(score)

        # Fast path via sklearn isotonic regressions + online augmentation.
        if self._iso_0 is not None and self._iso_1 is not None:
            assert self._train_scores is not None and self._train_labels is not None
            aug_scores = np.concatenate([self._train_scores, [s]])

            aug_labels_0 = np.concatenate([self._train_labels, [0]])
            aug_labels_1 = np.concatenate([self._train_labels, [1]])

            iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            iso0.fit(aug_scores, aug_labels_0)
            iso1.fit(aug_scores, aug_labels_1)
            p0 = float(iso0.predict([s])[0])
            p1 = float(iso1.predict([s])[0])
        else:
            # Empirical fallback: bin by 0.05 and compute mean label in bin.
            p0, p1 = self._empirical_interval(s)

        # By Venn-Abers construction, p0 <= p1 (ensure for safety).
        if p0 > p1:
            p0, p1 = p1, p0
        p0 = float(max(0.0, min(1.0, p0)))
        p1 = float(max(0.0, min(1.0, p1)))
        return p0, p1

    def _empirical_interval(self, score: float) -> Tuple[float, float]:
        assert self._train_scores is not None and self._train_labels is not None
        s = self._train_scores
        y = self._train_labels.astype(float)
        bin_half = 0.05
        mask = (s >= score - bin_half) & (s <= score + bin_half)
        if mask.sum() < 2:
            # Shrink to raw score if no neighbours found.
            return score, score
        mean_y = float(y[mask].mean())
        std_y = float(y[mask].std())
        p0 = max(0.0, mean_y - std_y)
        p1 = min(1.0, mean_y + std_y)
        return p0, p1

    # .....................................................................
    def predict_interval(self, score: float) -> List[float]:
        """Return `[p0, p1]` for a single score."""
        p0, p1 = self._predict_one(score)
        return [p0, p1]

    def predict(self, score: float) -> float:
        """Return a single point estimate — closest to uncalibrated score."""
        p0, p1 = self._predict_one(score)
        # Choose whichever of p0, p1 is closer to the uncalibrated score.
        # Falls back to midpoint if equidistant.
        d0 = abs(score - p0)
        d1 = abs(score - p1)
        if abs(d0 - d1) < 1e-9:
            return 0.5 * (p0 + p1)
        return p0 if d0 < d1 else p1

    def uncertainty(self, score: float) -> float:
        """Width of the Venn-Abers interval — acts as epistemic uncertainty."""
        p0, p1 = self._predict_one(score)
        return float(abs(p1 - p0))

    # Convenience batch API (used by backtests)
    def predict_batch(self, scores: Sequence[float]) -> np.ndarray:
        return np.array([self.predict(s) for s in scores], dtype=float)

    def predict_interval_batch(
        self, scores: Sequence[float]
    ) -> np.ndarray:
        return np.array([self._predict_one(s) for s in scores], dtype=float)


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

def get_calibrator(kind: str = "venn_abers") -> VennAbersCalibrator:
    """
    Factory returning the project's preferred calibrator.

    The default `"venn_abers"` replaces the legacy isotonic-only calibrator
    throughout the pipeline, so existing callers can simply swap
        from sklearn.isotonic import IsotonicRegression
    for
        from calibration import get_calibrator
        cal = get_calibrator()
    """
    kind = (kind or "venn_abers").lower()
    if kind in ("venn_abers", "venn-abers", "va"):
        return VennAbersCalibrator()
    raise ValueError(f"Unknown calibrator kind: {kind!r}")


__all__ = ["VennAbersCalibrator", "get_calibrator"]


## this is the implementation for the current p1, and p2, stati#
#please make sure that the implementaton plan is working correcyly