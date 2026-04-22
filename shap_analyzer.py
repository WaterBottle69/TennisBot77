"""
shap_analyzer.py — SHAP feature importance for XGBoost and LSTM models.

Generates:
  1. Global bar plot — mean |SHAP| per feature (which stats matter most overall)
  2. Beeswarm plot  — distribution + direction of each feature's impact
  3. Waterfall plot — per-prediction breakdown (why this specific prob was assigned)

Run standalone:
    python shap_analyzer.py

Or import and call analyze_xgb(model, X_test) for programmatic use.
"""

import os
import logging
import warnings
import numpy as np
import json

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(BASE_DIR, "best_xgb_model.json")
FEATURES_PATH = os.path.join(BASE_DIR, "model_features.json")
SHAP_OUT_DIR  = os.path.join(BASE_DIR, "shap_outputs")


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_xgb_model():
    """Load the production XGBoost model."""
    try:
        import xgboost as xgb
        model = xgb.XGBClassifier()
        model.load_model(MODEL_PATH)
        log.info(f"Loaded XGBoost model from {MODEL_PATH}")
        return model
    except Exception as e:
        log.error(f"Failed to load XGBoost model: {e}")
        return None


def load_feature_names() -> list:
    """Load feature names from model_features.json."""
    try:
        with open(FEATURES_PATH) as f:
            feats = json.load(f)
        if isinstance(feats, list):
            return feats
        if isinstance(feats, dict):
            return list(feats.keys())
        return []
    except Exception:
        return []


def analyze_xgb(model=None, X_test=None, feature_names=None, max_display=20):
    """
    Run SHAP TreeExplainer on the XGBoost model.

    Args:
        model:         XGBoost model (loaded if None)
        X_test:        numpy array or DataFrame of test features
        feature_names: list of feature column names
        max_display:   how many features to show in plots

    Returns:
        shap_values numpy array, or None on failure
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as e:
        log.error(f"SHAP or matplotlib not installed: {e}")
        return None

    if model is None:
        model = load_xgb_model()
    if model is None:
        return None

    if feature_names is None:
        feature_names = load_feature_names()

    if X_test is None:
        log.warning("No test data provided — generating synthetic sample for demo.")
        n_features = len(feature_names) if feature_names else 14
        X_test = np.random.rand(100, n_features)

    _ensure_dir(SHAP_OUT_DIR)

    try:
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        # Handle binary classifier output (may return list of 2 arrays)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values

        # ── 1. Global Bar Plot ────────────────────────────────────────────────
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            sv, X_test,
            feature_names=feature_names or None,
            plot_type="bar",
            max_display=max_display,
            show=False,
        )
        plt.title("Global Feature Importance (Mean |SHAP|)")
        plt.tight_layout()
        bar_path = os.path.join(SHAP_OUT_DIR, "global_bar_plot.png")
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved global bar plot → {bar_path}")

        # ── 2. Beeswarm Plot ─────────────────────────────────────────────────
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv, X_test,
            feature_names=feature_names or None,
            plot_type="dot",
            max_display=max_display,
            show=False,
        )
        plt.title("Beeswarm: Feature Value vs SHAP Impact")
        plt.tight_layout()
        bee_path = os.path.join(SHAP_OUT_DIR, "beeswarm_plot.png")
        plt.savefig(bee_path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved beeswarm plot → {bee_path}")

        # ── 3. Waterfall Plot (first prediction in X_test) ────────────────────
        try:
            expected_value = (
                explainer.expected_value[1]
                if isinstance(explainer.expected_value, (list, np.ndarray))
                else explainer.expected_value
            )
            explanation = shap.Explanation(
                values=sv[0],
                base_values=expected_value,
                data=X_test[0] if hasattr(X_test, "__getitem__") else X_test.iloc[0].values,
                feature_names=feature_names or None,
            )
            plt.figure(figsize=(10, 6))
            shap.waterfall_plot(explanation, max_display=max_display, show=False)
            plt.title("Waterfall: Why Did Model Predict This Probability?")
            plt.tight_layout()
            wf_path = os.path.join(SHAP_OUT_DIR, "waterfall_plot.png")
            plt.savefig(wf_path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Saved waterfall plot → {wf_path}")
        except Exception as e:
            log.warning(f"Waterfall plot failed: {e}")

        log.info("SHAP analysis complete. Outputs in: %s", SHAP_OUT_DIR)
        return sv

    except Exception as e:
        log.error(f"SHAP analysis failed: {e}")
        return None


def walk_forward_ablation(
    feature_groups: dict,
    X_full,
    y_full,
    train_end_idx: int,
    feature_names: list = None,
):
    """
    Walk-Forward Ablation Study.

    Trains the XGBoost model on data[:train_end_idx], tests on data[train_end_idx:].
    Systematically removes each feature group and measures Brier Score degradation.

    Per information.md: uses Walk-Forward Validation to prevent look-ahead bias.
    K-fold cross-validation is NOT used here (it leaks future data in time-series).

    Args:
        feature_groups: dict of {group_name: [col_indices]}
            e.g. {"serve_pct": [0, 1], "break_pts": [2, 3]}
        X_full:       full feature matrix (n_samples x n_features)
        y_full:       full label vector (n_samples,)
        train_end_idx: index splitting train vs test
        feature_names: optional list for reporting

    Returns:
        dict of {group_name: {"brier_base": float, "brier_ablated": float, "delta": float}}
    """
    try:
        import xgboost as xgb
        from sklearn.metrics import brier_score_loss, log_loss
    except ImportError as e:
        log.error(f"Required packages missing: {e}")
        return {}

    X_train, X_test = X_full[:train_end_idx], X_full[train_end_idx:]
    y_train, y_test = y_full[:train_end_idx], y_full[train_end_idx:]

    if len(X_test) == 0 or len(X_train) == 0:
        log.error("Insufficient data for walk-forward split.")
        return {}

    # Baseline model (all features)
    baseline_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
        verbosity=0,
    )
    baseline_model.fit(X_train, y_train)
    baseline_probs = baseline_model.predict_proba(X_test)[:, 1]
    brier_base     = brier_score_loss(y_test, baseline_probs)
    logloss_base   = log_loss(y_test, baseline_probs)
    log.info(f"[ABLATION] Baseline Brier={brier_base:.5f}  LogLoss={logloss_base:.5f}")

    results = {}
    for group_name, col_indices in feature_groups.items():
        # Zero out the feature group columns
        X_ablated = X_full.copy()
        X_ablated[:, col_indices] = 0.0

        X_tr_ab = X_ablated[:train_end_idx]
        X_te_ab = X_ablated[train_end_idx:]

        model_ab = xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
            verbosity=0,
        )
        model_ab.fit(X_tr_ab, y_train)
        probs_ab  = model_ab.predict_proba(X_te_ab)[:, 1]
        brier_ab  = brier_score_loss(y_test, probs_ab)
        delta     = brier_ab - brier_base   # positive = ablation HURT the model

        results[group_name] = {
            "brier_base":    round(brier_base, 6),
            "brier_ablated": round(brier_ab,   6),
            "delta":         round(delta,       6),
            "verdict":       "KEEP" if delta > 0.001 else "REMOVE",
        }
        log.info(
            f"[ABLATION] {group_name:30s} Brier={brier_ab:.5f} "
            f"Δ={delta:+.5f} → {results[group_name]['verdict']}"
        )

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("Running SHAP analysis on production XGBoost model...")
    model = load_xgb_model()
    feature_names = load_feature_names()
    shap_vals = analyze_xgb(model=model, feature_names=feature_names)
    if shap_vals is not None:
        log.info(f"SHAP complete. Shape: {shap_vals.shape}")
    else:
        log.warning("SHAP analysis did not produce output. Check model path and dependencies.")
