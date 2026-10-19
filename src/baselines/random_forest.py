"""
GraphGuard: Phase 6 — Baseline Models
Module: random_forest.py

Random Forest classifier on raw traffic-flow features (no graph context).
Implements the same train/test split as GraphGuard for fair comparison.
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_random_forest(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label_multiclass",
    cfg: Dict[str, Any] = None,
) -> Tuple[RandomForestClassifier, dict]:
    """
    Train a Random Forest classifier on tabular flow features.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        feature_cols: List of numeric feature columns.
        label_col: Target label column.
        cfg: Config dict.

    Returns:
        Tuple of (fitted model, validation metrics dict).
    """
    rf_cfg = cfg.get("baselines", {}).get("random_forest", {}) if cfg else {}

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values

    logger.info("Training Random Forest: n_estimators=%d, n_jobs=%d ...",
                rf_cfg.get("n_estimators", 200), rf_cfg.get("n_jobs", -1))

    clf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", None),
        n_jobs=rf_cfg.get("n_jobs", -1),
        class_weight=rf_cfg.get("class_weight", "balanced"),
        random_state=rf_cfg.get("random_state", 42),
    )
    clf.fit(X_train, y_train)

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    logger.info("Random Forest Val F1 (macro): %.4f", val_f1)

    return clf, {"val_f1_macro": val_f1}


def save_rf_model(clf: RandomForestClassifier, path: Path) -> None:
    """Persist fitted Random Forest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(clf, f)
    logger.info("Random Forest saved to %s", path)


def load_rf_model(path: Path) -> RandomForestClassifier:
    """Load a persisted Random Forest."""
    with open(path, "rb") as f:
        clf = pickle.load(f)
    logger.info("Random Forest loaded from %s", path)
    return clf
