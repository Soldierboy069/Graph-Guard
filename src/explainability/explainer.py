"""
GraphGuard: Phase 8 — Explainable AI
Module: explainer.py

SHAP-based feature attribution for individual flow predictions.
Produces human-readable explanations for why a flow was flagged as an attack.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphGuardExplainer:
    """
    Wraps SHAP TreeExplainer / KernelExplainer for flow-level explanations.

    Designed to work with the GraphGuard model's edge feature representation.
    Uses a surrogate approach: SHAP on the edge feature portion of predictions,
    since the full GNN is not directly interpretable with standard SHAP.

    For per-flow explanations:
    - Fits a surrogate LinearModel/TreeModel on GraphGuard's edge predictions
    - Computes SHAP values on the edge traffic features
    - Returns top-k features with their attribution scores
    """

    def __init__(
        self,
        feature_names: List[str],
        label_encoder: Dict[str, int],
        background_samples: int = 100,
    ) -> None:
        """
        Args:
            feature_names: Names of edge traffic features.
            label_encoder: String→int label mapping.
            background_samples: Number of background samples for SHAP masker.
        """
        self.feature_names = feature_names
        self.label_encoder = label_encoder
        self.int_to_label = {v: k for k, v in label_encoder.items()}
        self.background_samples = background_samples
        self._explainer = None
        self._surrogate = None

    def fit(
        self,
        X_background: np.ndarray,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> None:
        """
        Fit a surrogate model and SHAP explainer on background data.

        Args:
            X_background: Edge features for background samples [n, d].
            y_true: True labels.
            y_pred: GraphGuard predicted labels (used as surrogate targets).
        """
        try:
            import shap
            from sklearn.ensemble import RandomForestClassifier

            # Sample background data for fast and robust surrogate fitting
            sample_limit = 10000
            if len(X_background) > sample_limit:
                sub_idx = np.random.choice(len(X_background), sample_limit, replace=False)
                X_fit = X_background[sub_idx]
                y_fit = y_pred[sub_idx]
            else:
                X_fit = X_background
                y_fit = y_pred

            n = min(self.background_samples, len(X_fit))
            idx = np.random.choice(len(X_fit), n, replace=False)
            X_bg = X_fit[idx]

            # Fit lightweight surrogate
            logger.info("Fitting surrogate RandomForestClassifier for SHAP (n_fit=%d, n_bg=%d)...", len(X_fit), n)
            self._surrogate = RandomForestClassifier(
                n_estimators=50, max_depth=6, n_jobs=-1, random_state=42
            )
            self._surrogate.fit(X_fit, y_fit)

            # SHAP TreeExplainer on surrogate
            self._explainer = shap.TreeExplainer(
                self._surrogate,
                data=X_bg,
                feature_names=self.feature_names,
            )
            logger.info("SHAP explainer ready.")
        except ImportError:
            logger.error("SHAP not installed. Run: pip install shap")
            raise

    def explain_flow(
        self,
        edge_features: np.ndarray,
        predicted_class: int,
        top_k: int = 15,
    ) -> Dict[str, Any]:
        """
        Explain a single flow prediction.

        Args:
            edge_features: 1D array of edge feature values.
            predicted_class: Integer predicted class.
            top_k: Number of top features to return.

        Returns:
            Explanation dict with feature attributions.
        """
        if self._explainer is None:
            raise RuntimeError("Call fit() before explain_flow().")

        import shap
        x = edge_features.reshape(1, -1)
        shap_values = self._explainer.shap_values(x)

        # Handle multiclass outputs (list or 3D numpy array)
        if isinstance(shap_values, list):
            sv = shap_values[predicted_class][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            sv = shap_values[0, :, predicted_class]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            sv = shap_values[0]
        else:
            sv = np.asarray(shap_values).flatten()

        # Top-k features by absolute SHAP value
        abs_sv = np.abs(sv)
        top_idx = np.argsort(abs_sv)[::-1][:top_k]

        top_features = [
            {
                "feature": self.feature_names[i],
                "shap_value": float(sv[i]),
                "abs_shap": float(abs_sv[i]),
                "feature_value": float(edge_features[i]),
            }
            for i in top_idx
        ]

        return {
            "predicted_class": predicted_class,
            "predicted_label": self.int_to_label.get(predicted_class, str(predicted_class)),
            "top_features": top_features,
            "all_shap_values": sv.tolist(),
        }

    def explain_batch(
        self,
        edge_features: np.ndarray,
        predicted_classes: np.ndarray,
        top_k: int = 15,
    ) -> List[Dict[str, Any]]:
        """Explain a batch of flows."""
        results = []
        for i in range(len(edge_features)):
            results.append(
                self.explain_flow(edge_features[i], int(predicted_classes[i]), top_k)
            )
        return results

    def plot_explanation(
        self,
        explanation: Dict[str, Any],
        save_path: Path,
        flow_id: Optional[str] = None,
    ) -> None:
        """
        Save a horizontal bar chart of top feature attributions.

        Args:
            explanation: Output from explain_flow().
            save_path: Where to save the PNG.
            flow_id: Optional flow identifier for the chart title.
        """
        top_features = explanation["top_features"]
        feat_names = [f["feature"] for f in top_features]
        shap_vals = [f["shap_value"] for f in top_features]

        colors = ["#d73027" if v > 0 else "#4575b4" for v in shap_vals]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(feat_names[::-1], shap_vals[::-1], color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        label = explanation["predicted_label"]
        title = f"Flow Explanation — Predicted: {label}"
        if flow_id:
            title = f"{title} | Flow: {flow_id}"
        ax.set_title(title)
        plt.tight_layout()

        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Explanation plot saved to %s", save_path)

    def save_explanation(self, explanation: Dict[str, Any], path: Path) -> None:
        """Save explanation dict to JSON."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(explanation, f, indent=2)
