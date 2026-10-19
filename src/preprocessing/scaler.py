"""
GraphGuard: Phase 2 — Data Preprocessing
Module: scaler.py

Feature scaling — fit only on training data to prevent data leakage.
Persists fitted scaler for use during inference and dashboard display.
"""
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

SCALER_CLASSES = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "robust": RobustScaler,
}


def get_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Identify numeric feature columns, excluding metadata/label columns.

    Args:
        df: DataFrame to inspect.
        exclude_cols: Column names to exclude (metadata, labels, etc.).

    Returns:
        List of numeric feature column names suitable for scaling.
    """
    if exclude_cols is None:
        exclude_cols = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]
    return feature_cols


def fit_scaler(
    df: pd.DataFrame,
    feature_cols: List[str],
    scaler_type: str = "standard",
) -> object:
    """
    Fit a scaler on the TRAINING data only.

    Args:
        df: Training DataFrame.
        feature_cols: Columns to fit.
        scaler_type: One of 'standard', 'minmax', 'robust'.

    Returns:
        Fitted scaler object.
    """
    if scaler_type not in SCALER_CLASSES:
        raise ValueError(f"Unknown scaler type '{scaler_type}'. Choose from {list(SCALER_CLASSES)}")

    scaler = SCALER_CLASSES[scaler_type]()
    scaler.fit(df[feature_cols].values)
    logger.info("Fitted %s scaler on %d features", scaler_type, len(feature_cols))
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler: object,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Apply a pre-fitted scaler to transform a DataFrame in-place.

    Args:
        df: DataFrame to transform.
        scaler: Pre-fitted scaler object.
        feature_cols: Columns to transform (must match those used to fit).

    Returns:
        DataFrame with scaled feature columns.
    """
    df = df.copy()
    df[feature_cols] = scaler.transform(df[feature_cols].values)
    return df


def save_scaler(scaler: object, path: Path) -> None:
    """Persist fitted scaler to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info("Scaler saved to %s", path)


def load_scaler(path: Path) -> object:
    """Load a previously fitted scaler from disk."""
    with open(path, "rb") as f:
        scaler = pickle.load(f)
    logger.info("Scaler loaded from %s", path)
    return scaler
