"""
GraphGuard: Phase 2 — Data Preprocessing
Module: splitter.py

Time-aware train/validation/test split to prevent temporal data leakage.
Sorts by Timestamp and allocates chronologically ordered partitions.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_timestamps(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    timestamp_format: str = "%m/%d/%Y %H:%M",
) -> pd.DataFrame:
    """
    Parse the Timestamp column to datetime, correctly mapping CICIDS2017 working hours.
    Working hours 8-12 are AM/noon, hours 1-5 are PM (13-17).

    Args:
        df: DataFrame with raw timestamp strings.
        timestamp_col: Name of the timestamp column.
        timestamp_format: Fallback strptime format string.

    Returns:
        DataFrame with Timestamp converted to datetime64.
    """
    if timestamp_col not in df.columns:
        logger.warning("Timestamp column '%s' not found.", timestamp_col)
        return df

    df = df.copy()
    import re

    def _convert_ts(val):
        if pd.isna(val):
            return pd.NaT
        s = str(val).strip()
        m = re.match(r"(\d+)/(\d+)/(\d+)\s+(\d+):(\d+)(?::(\d+))?", s)
        if m:
            d, month, y, h, mn = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
            sec = int(m.group(6)) if m.group(6) else 0
            if y < 100:
                y += 2000
            # CICIDS2017 working hours: 1 to 5 are PM (13 to 17)
            if 1 <= h <= 5:
                h += 12
            return pd.Timestamp(year=y, month=month, day=d, hour=h, minute=mn, second=sec)
        return pd.to_datetime(s, format=timestamp_format, errors="coerce")

    # Fast check: apply conversion
    try:
        df[timestamp_col] = df[timestamp_col].apply(_convert_ts)
    except Exception as e:
        logger.warning("Regex timestamp parse failed (%s), falling back to pd.to_datetime", e)
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], format=timestamp_format, errors="coerce")

    nat_count = df[timestamp_col].isna().sum()
    if nat_count > 0:
        logger.warning("%d rows could not be parsed as timestamps and will be NaT.", nat_count)
    return df


def time_aware_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    timestamp_col: str = "Timestamp",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame chronologically to avoid temporal leakage.

    Sorts rows by Timestamp, then splits by position ratio.
    Rows with NaT timestamps are appended to training set.

    Args:
        df: DataFrame with parsed Timestamp column.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.
        test_ratio: Fraction for test.
        timestamp_col: Datetime column name.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    if timestamp_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        # Separate NaT rows
        nat_mask = df[timestamp_col].isna()
        nat_df = df[nat_mask]
        valid_df = df[~nat_mask].sort_values(timestamp_col).reset_index(drop=True)
    else:
        logger.warning("No datetime Timestamp found; falling back to random-order split.")
        nat_df = pd.DataFrame()
        valid_df = df.reset_index(drop=True)

    n = len(valid_df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train_df = valid_df.iloc[:train_end].copy()
    val_df = valid_df.iloc[train_end:val_end].copy()
    test_df = valid_df.iloc[val_end:].copy()

    # Append NaT rows to training (they have no temporal information)
    if len(nat_df) > 0:
        train_df = pd.concat([train_df, nat_df], ignore_index=True)

    logger.info(
        "Time-aware split: train=%d | val=%d | test=%d",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


def report_split_distribution(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    label_col: str = "Label",
) -> Dict[str, pd.Series]:
    """
    Log and return label distribution for each split.

    Args:
        train, val, test: Split DataFrames.
        label_col: Label column name.

    Returns:
        Dict with 'train', 'val', 'test' label counts.
    """
    distributions = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        if label_col in split.columns:
            dist = split[label_col].value_counts()
            distributions[name] = dist
            logger.info("  [%s] %d rows | Classes: %s", name, len(split), dist.to_dict())
    return distributions


def campaign_window_temporal_split(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    w1_range: Tuple[str, str] = ("2017-07-04 09:00:00", "2017-07-04 10:35:00"),
    w2_range: Tuple[str, str] = ("2017-07-04 14:00:00", "2017-07-04 15:20:00"),
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Campaign-window chronological split:
    Within each campaign independently, flows are partitioned chronologically
    into 70% training, 15% validation, and 15% testing.

    Args:
        df: DataFrame with parsed Timestamp column.
        timestamp_col: Datetime column name.
        train_ratio, val_ratio, test_ratio: Partition fractions (sum to 1.0).
        w1_range: (start, end) strings for FTP-Patator campaign.
        w2_range: (start, end) strings for SSH-Patator campaign.

    Returns:
        Tuple of (train_df, val_df, test_df, provenance_dict).
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Campaign 1: FTP-Patator
    w1_mask = (df[timestamp_col] >= pd.to_datetime(w1_range[0])) & (df[timestamp_col] <= pd.to_datetime(w1_range[1]))
    w1 = df[w1_mask].sort_values(timestamp_col).reset_index(drop=True)
    w1["_campaign"] = "FTP-Patator-Morning"

    n1 = len(w1)
    w1_tr_end = int(n1 * train_ratio)
    w1_va_end = w1_tr_end + int(n1 * val_ratio)

    w1_train = w1.iloc[:w1_tr_end].copy()
    w1_val = w1.iloc[w1_tr_end:w1_va_end].copy()
    w1_test = w1.iloc[w1_va_end:].copy()

    # Campaign 2: SSH-Patator
    w2_mask = (df[timestamp_col] >= pd.to_datetime(w2_range[0])) & (df[timestamp_col] <= pd.to_datetime(w2_range[1]))
    w2 = df[w2_mask].sort_values(timestamp_col).reset_index(drop=True)
    w2["_campaign"] = "SSH-Patator-Afternoon"

    n2 = len(w2)
    w2_tr_end = int(n2 * train_ratio)
    w2_va_end = w2_tr_end + int(n2 * val_ratio)

    w2_train = w2.iloc[:w2_tr_end].copy()
    w2_val = w2.iloc[w2_tr_end:w2_va_end].copy()
    w2_test = w2.iloc[w2_va_end:].copy()

    # Combine partitions
    train_df = pd.concat([w1_train, w2_train], ignore_index=True).sort_values(timestamp_col).reset_index(drop=True)
    val_df = pd.concat([w1_val, w2_val], ignore_index=True).sort_values(timestamp_col).reset_index(drop=True)
    test_df = pd.concat([w1_test, w2_test], ignore_index=True).sort_values(timestamp_col).reset_index(drop=True)

    provenance = {
        "FTP_Patator_Campaign": {
            "window": [w1_range[0], w1_range[1]],
            "total_flows": n1,
            "train": {
                "flows": len(w1_train),
                "start": str(w1_train[timestamp_col].min()),
                "end": str(w1_train[timestamp_col].max()),
                "labels": w1_train["Label"].value_counts().to_dict() if "Label" in w1_train else {},
            },
            "val": {
                "flows": len(w1_val),
                "start": str(w1_val[timestamp_col].min()),
                "end": str(w1_val[timestamp_col].max()),
                "labels": w1_val["Label"].value_counts().to_dict() if "Label" in w1_val else {},
            },
            "test": {
                "flows": len(w1_test),
                "start": str(w1_test[timestamp_col].min()),
                "end": str(w1_test[timestamp_col].max()),
                "labels": w1_test["Label"].value_counts().to_dict() if "Label" in w1_test else {},
            },
        },
        "SSH_Patator_Campaign": {
            "window": [w2_range[0], w2_range[1]],
            "total_flows": n2,
            "train": {
                "flows": len(w2_train),
                "start": str(w2_train[timestamp_col].min()),
                "end": str(w2_train[timestamp_col].max()),
                "labels": w2_train["Label"].value_counts().to_dict() if "Label" in w2_train else {},
            },
            "val": {
                "flows": len(w2_val),
                "start": str(w2_val[timestamp_col].min()),
                "end": str(w2_val[timestamp_col].max()),
                "labels": w2_val["Label"].value_counts().to_dict() if "Label" in w2_val else {},
            },
            "test": {
                "flows": len(w2_test),
                "start": str(w2_test[timestamp_col].min()),
                "end": str(w2_test[timestamp_col].max()),
                "labels": w2_test["Label"].value_counts().to_dict() if "Label" in w2_test else {},
            },
        },
        "Combined": {
            "train": {
                "flows": len(train_df),
                "labels": train_df["Label"].value_counts().to_dict() if "Label" in train_df else {},
            },
            "val": {
                "flows": len(val_df),
                "labels": val_df["Label"].value_counts().to_dict() if "Label" in val_df else {},
            },
            "test": {
                "flows": len(test_df),
                "labels": test_df["Label"].value_counts().to_dict() if "Label" in test_df else {},
            },
        },
    }

    logger.info("Campaign-window chronological split completed:")
    logger.info("  Train: %d flows | Val: %d flows | Test: %d flows", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df, provenance


def audit_and_deduplicate_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list,
    tuple_cols: list = None,
    deduplicate_features: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Audit exact duplicates and 5-tuple collisions across splits.
    Optionally drops exact feature duplicates in Val/Test that exist in Train.

    Args:
        train_df, val_df, test_df: Split DataFrames.
        feature_cols: List of traffic feature columns.
        tuple_cols: List of network 5-tuple column names.
        deduplicate_features: If True, drops exact feature duplicates from Val/Test.

    Returns:
        Tuple of (clean_train, clean_val, clean_test, audit_report).
    """
    if tuple_cols is None:
        tuple_cols = ["Source IP", "Destination IP", "Source Port", "Destination Port", "Protocol"]

    existing_tuple_cols = [c for c in tuple_cols if c in train_df.columns]
    existing_feat_cols = [c for c in feature_cols if c in train_df.columns]

    report = {"initial_counts": {"train": len(train_df), "val": len(val_df), "test": len(test_df)}}

    # 5-tuple collisions
    if len(existing_tuple_cols) >= 2:
        train_tuples = set(map(tuple, train_df[existing_tuple_cols].fillna("").values))
        val_tuples = set(map(tuple, val_df[existing_tuple_cols].fillna("").values))
        test_tuples = set(map(tuple, test_df[existing_tuple_cols].fillna("").values))

        report["5_tuple_collisions"] = {
            "train_val_overlap": len(train_tuples & val_tuples),
            "train_test_overlap": len(train_tuples & test_tuples),
            "val_test_overlap": len(val_tuples & test_tuples),
        }

    # Exact feature duplicates
    train_feat_hashes = set(map(tuple, train_df[existing_feat_cols].fillna(0).values))
    val_feat_mask = val_df[existing_feat_cols].fillna(0).apply(tuple, axis=1).isin(train_feat_hashes)
    test_feat_mask = test_df[existing_feat_cols].fillna(0).apply(tuple, axis=1).isin(train_feat_hashes)

    report["exact_feature_duplicates"] = {
        "val_in_train": int(val_feat_mask.sum()),
        "test_in_train": int(test_feat_mask.sum()),
    }

    if deduplicate_features:
        logger.info(
            "Dropping exact feature duplicates: %d from Val, %d from Test",
            val_feat_mask.sum(), test_feat_mask.sum()
        )
        val_df = val_df[~val_feat_mask].reset_index(drop=True)
        test_df = test_df[~test_feat_mask].reset_index(drop=True)

    report["final_counts"] = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    return train_df, val_df, test_df, report

