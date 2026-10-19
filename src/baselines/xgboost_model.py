"""
GraphGuard: Phase 6 — Baseline Models
Module: xgboost_model.py

XGBoost classifier on raw traffic-flow features (no graph context).
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_xgboost(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list,
    label_col: str = "label_multiclass",
    cfg: Dict[str, Any] = None,
) -> Tuple[XGBClassifier, dict]:
    """
    Train an XGBoost classifier on tabular flow features.

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        feature_cols: List of numeric feature columns.
        label_col: Target label column.
        cfg: Config dict.

    Returns:
        Tuple of (fitted model, validation metrics dict).
    """
    xgb_cfg = cfg.get("baselines", {}).get("xgboost", {}) if cfg else {}

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values.astype(int)
    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values.astype(int)

    logger.info("Training XGBoost: n_estimators=%d ...", xgb_cfg.get("n_estimators", 200))

    n_classes = int(y_train.max()) + 1
    # Use binary:logistic for 2-class, multi:softprob for >2
    if n_classes == 2:
        objective = "binary:logistic"
        eval_metric = "logloss"
    else:
        objective = "multi:softprob"
        eval_metric = xgb_cfg.get("eval_metric", "mlogloss")

    clf = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 200),
        max_depth=xgb_cfg.get("max_depth", 6),
        learning_rate=xgb_cfg.get("learning_rate", 0.1),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        n_jobs=xgb_cfg.get("n_jobs", -1),
        objective=objective,
        eval_metric=eval_metric,
        random_state=xgb_cfg.get("random_state", 42),
        tree_method="hist",
        device="cpu",
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = clf.predict(X_val)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    logger.info("XGBoost Val F1 (macro): %.4f", val_f1)

    return clf, {"val_f1_macro": val_f1}


def save_xgb_model(clf: XGBClassifier, path: Path) -> None:
    """Save XGBoost model using pickle."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pkl_path = path.with_suffix(".pkl")
    import pickle
    with open(pkl_path, "wb") as f:
        pickle.dump(clf, f)
    logger.info("XGBoost saved to %s", pkl_path)


def load_xgb_model(path: Path) -> XGBClassifier:
    """Load XGBoost model from pickle."""
    import pickle
    pkl_path = path.with_suffix(".pkl") if path.suffix != ".pkl" else path
    with open(pkl_path, "rb") as f:
        clf = pickle.load(f)
    logger.info("XGBoost loaded from %s", pkl_path)
    return clf
