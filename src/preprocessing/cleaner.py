"""
GraphGuard: Phase 2 — Data Preprocessing
Module: cleaner.py

Data cleaning: handles missing values, infinite values,
duplicate rows, and invalid rows in CICIDS2017 chunks.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def remove_inf_values(df: pd.DataFrame, fill_value: float = np.nan) -> Tuple[pd.DataFrame, int]:
    """
    Replace +inf and -inf values with fill_value (default NaN for later dropping).

    Args:
        df: Input DataFrame.
        fill_value: Value to replace infinities with.

    Returns:
        Tuple of (cleaned DataFrame, count of replaced values).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    inf_mask = np.isinf(df[numeric_cols])
    count = int(inf_mask.sum().sum())
    if count > 0:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], fill_value)
        logger.debug("Replaced %d inf values", count)
    return df, count


def drop_missing_label(df: pd.DataFrame, label_col: str = "Label") -> Tuple[pd.DataFrame, int]:
    """
    Drop rows where the label column is missing or empty.

    Args:
        df: Input DataFrame.
        label_col: Name of the label column.

    Returns:
        Tuple of (filtered DataFrame, number of rows dropped).
    """
    if label_col not in df.columns:
        logger.warning("Label column '%s' not found in DataFrame.", label_col)
        return df, 0

    before = len(df)
    df = df.dropna(subset=[label_col])
    # Also drop empty strings
    df = df[df[label_col].astype(str).str.strip() != ""]
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with missing label", dropped)
    return df, dropped


def fill_numeric_nan(df: pd.DataFrame, fill_value: float = 0.0) -> Tuple[pd.DataFrame, int]:
    """
    Fill remaining NaN values in numeric columns with fill_value.

    Args:
        df: Input DataFrame.
        fill_value: Value to fill NaN with.

    Returns:
        Tuple of (filled DataFrame, count of filled values).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    nan_count = int(df[numeric_cols].isna().sum().sum())
    if nan_count > 0:
        df[numeric_cols] = df[numeric_cols].fillna(fill_value)
        logger.debug("Filled %d NaN values with %s", nan_count, fill_value)
    return df, nan_count


def drop_duplicates(
    df: pd.DataFrame, subset: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, int]:
    """
    Drop exact duplicate rows.

    Args:
        df: Input DataFrame.
        subset: Columns to consider for duplicates. If None, use all columns.

    Returns:
        Tuple of (deduplicated DataFrame, number of rows dropped).
    """
    before = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d duplicate rows", dropped)
    return df, dropped


def validate_numeric_ranges(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Drop rows where numeric features have clearly invalid values (e.g., negative packet counts).

    Applies lightweight sanity checks; does not filter legitimate large values.

    Args:
        df: Input DataFrame.

    Returns:
        Tuple of (validated DataFrame, number of rows dropped).
    """
    before = len(df)
    # Remove rows where Flow Duration is negative
    if "Flow Duration" in df.columns:
        df = df[~(df["Flow Duration"] < 0)]

    # Remove rows where packet counts are negative
    for col in ["Total Fwd Packets", "Total Backward Packets"]:
        if col in df.columns:
            df = df[~(df[col] < 0)]

    dropped = before - len(df)
    if dropped > 0:
        logger.debug("Dropped %d rows with invalid numeric ranges", dropped)
    return df, dropped


def clean_chunk(
    df: pd.DataFrame,
    label_col: str = "Label",
    fill_nan_value: float = 0.0,
) -> Tuple[pd.DataFrame, dict]:
    """
    Apply the full cleaning pipeline to a single DataFrame chunk.

    Pipeline order:
    1. Replace inf values with NaN
    2. Drop rows with missing labels
    3. Fill remaining NaN with fill_value
    4. Validate numeric ranges

    Args:
        df: Raw DataFrame chunk.
        label_col: Name of the label column.
        fill_nan_value: Replacement value for NaN after inf removal.

    Returns:
        Tuple of (cleaned DataFrame, stats dict with counts of each operation).
    """
    stats = {"inf_replaced": 0, "missing_label_dropped": 0, "nan_filled": 0, "invalid_dropped": 0}

    df, stats["inf_replaced"] = remove_inf_values(df)
    df, stats["missing_label_dropped"] = drop_missing_label(df, label_col)
    df, stats["nan_filled"] = fill_numeric_nan(df, fill_nan_value)
    df, stats["invalid_dropped"] = validate_numeric_ranges(df)

    return df, stats
