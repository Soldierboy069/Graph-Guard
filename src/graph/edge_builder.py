"""
GraphGuard: Phase 3 — Graph Construction
Module: edge_builder.py

Constructs directed edge index and edge feature matrix from flow records.
Each flow becomes a directed edge: Source IP → Destination IP.
"""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_edge_index(
    df: pd.DataFrame,
    ip_to_node: Dict[str, int],
    src_col: str = "Source IP",
    dst_col: str = "Destination IP",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build directed edge index arrays from flow records.

    Args:
        df: DataFrame with flow records.
        ip_to_node: IP address → node ID mapping.
        src_col: Source IP column name.
        dst_col: Destination IP column name.

    Returns:
        Tuple of:
        - edge_src: 1D int array of source node IDs (length = num_edges)
        - edge_dst: 1D int array of destination node IDs (length = num_edges)
        - valid_mask: boolean mask of rows that had valid IP mappings
    """
    src_ids = df[src_col].map(ip_to_node)
    dst_ids = df[dst_col].map(ip_to_node)

    valid_mask = src_ids.notna() & dst_ids.notna()
    n_invalid = (~valid_mask).sum()
    if n_invalid > 0:
        logger.warning("Skipping %d flows with unmapped IPs.", n_invalid)

    edge_src = src_ids[valid_mask].astype(int).values
    edge_dst = dst_ids[valid_mask].astype(int).values

    logger.info("Edge index: %d directed edges", len(edge_src))
    return edge_src, edge_dst, valid_mask.values


def get_edge_feature_columns(
    df: pd.DataFrame,
    exclude_cols: Optional[List[str]] = None,
) -> List[str]:
    """
    Return list of numeric columns to use as edge features.

    Excludes metadata, label, and non-feature columns.

    Args:
        df: Flow DataFrame.
        exclude_cols: Columns to explicitly exclude.

    Returns:
        List of feature column names.
    """
    if exclude_cols is None:
        exclude_cols = []

    # Standard metadata columns to always exclude
    always_exclude = {
        "Flow ID", "Source IP", "Source Port", "Destination IP",
        "Destination Port", "Protocol", "Timestamp", "Label",
        "label_multiclass", "label_binary",
    }
    always_exclude.update(exclude_cols)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in always_exclude]
    return feature_cols


def build_edge_features(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    feature_cols: List[str],
) -> np.ndarray:
    """
    Extract edge feature matrix for valid (mapped) flow rows.

    Args:
        df: Flow DataFrame.
        valid_mask: Boolean mask from build_edge_index (rows with valid IP mappings).
        feature_cols: Numeric columns to use as edge features.

    Returns:
        Float32 array of shape (num_edges, num_features).
    """
    edge_features = df[feature_cols][valid_mask].values.astype(np.float32)
    logger.info("Edge features shape: %s", edge_features.shape)
    return edge_features


def build_edge_labels(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    label_col: str = "label_multiclass",
) -> np.ndarray:
    """
    Extract integer edge labels for valid flows.

    Args:
        df: Flow DataFrame with encoded label columns.
        valid_mask: Boolean mask from build_edge_index.
        label_col: Name of integer label column.

    Returns:
        Int64 array of shape (num_edges,).
    """
    edge_labels = df[label_col][valid_mask].values.astype(np.int64)
    logger.info("Edge labels shape: %s | unique: %s", edge_labels.shape, np.unique(edge_labels))
    return edge_labels


def build_edge_timestamps(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
    timestamp_col: str = "Timestamp",
) -> Optional[np.ndarray]:
    """
    Extract per-edge timestamps as float (Unix epoch seconds).

    Args:
        df: Flow DataFrame with parsed Timestamp column.
        valid_mask: Boolean mask.
        timestamp_col: Datetime column name.

    Returns:
        Float64 array of Unix timestamps, or None if column unavailable.
    """
    if timestamp_col not in df.columns:
        return None
    ts_series = df[timestamp_col][valid_mask]
    if not pd.api.types.is_datetime64_any_dtype(ts_series):
        return None
    # Convert to Unix seconds (float) — pandas 2.x compatible
    timestamps = ts_series.astype(np.int64).values / 1e9  # nanoseconds → seconds
    logger.info("Edge timestamps extracted: %d values", len(timestamps))
    return timestamps.astype(np.float64)


def get_flow_metadata(
    df: pd.DataFrame,
    valid_mask: np.ndarray,
) -> pd.DataFrame:
    """
    Extract metadata columns for valid flows (for dashboard lookups).

    Args:
        df: Flow DataFrame.
        valid_mask: Boolean mask.

    Returns:
        DataFrame with metadata columns for each edge.
    """
    meta_cols = [c for c in [
        "Flow ID", "Source IP", "Source Port", "Destination IP",
        "Destination Port", "Protocol", "Timestamp", "Label",
        "label_multiclass", "label_binary",
    ] if c in df.columns]
    return df[meta_cols][valid_mask].reset_index(drop=True)
