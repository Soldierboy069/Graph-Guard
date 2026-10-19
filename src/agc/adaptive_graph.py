"""
GraphGuard: Phase 4 — Adaptive Graph Construction (AGC)
Module: adaptive_graph.py

Implements a practical temporal windowing AGC mechanism.
Divides the timeline into fixed-width windows (e.g., 5-minute intervals).
For each window, identifies new/updated/stale edges and computes adaptive weights.

This is NOT claimed to be a novel algorithm — it is a documented, reproducible
temporal graph update strategy based on observed traffic data.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class AdaptiveGraphConstructor:
    """
    Temporal AGC: builds a series of graph snapshots over time windows.

    For each time window:
    - Identifies active edges (flows within the window)
    - Tracks new nodes/edges (not seen before)
    - Updates edge weights based on activity frequency
    - Decays weights of inactive edges
    - Prunes stale edges below minimum weight threshold

    Adaptive edge weight formula:
        w(e, t) = freq(e, t) / max_freq(t)   (recency component)
        w(e, t) = alpha * w(e, t-1) + (1-alpha) * w_current  (EMA update)

    where alpha = weight_decay (configured in config.yaml).
    """

    def __init__(
        self,
        window_minutes: int = 5,
        stale_threshold_windows: int = 3,
        weight_decay: float = 0.5,
        min_edge_weight: float = 0.01,
    ) -> None:
        """
        Initialize AGC parameters.

        Args:
            window_minutes: Duration of each time window in minutes.
            stale_threshold_windows: Number of windows without activity before pruning.
            weight_decay: EMA decay factor for inactive edge weights (0-1).
            min_edge_weight: Minimum weight before an edge is pruned.
        """
        self.window_minutes = window_minutes
        self.stale_threshold_windows = stale_threshold_windows
        self.weight_decay = weight_decay
        self.min_edge_weight = min_edge_weight

        # State tracking across windows
        self._edge_weights: Dict[Tuple[int, int], float] = {}
        self._edge_last_active: Dict[Tuple[int, int], int] = {}
        self._all_nodes: set = set()
        self._all_edges: set = set()
        self._window_index: int = 0

    def reset(self) -> None:
        """Reset all internal state (start fresh for a new graph)."""
        self._edge_weights.clear()
        self._edge_last_active.clear()
        self._all_nodes.clear()
        self._all_edges.clear()
        self._window_index = 0

    def process_window(
        self,
        window_df: pd.DataFrame,
        ip_to_node: Dict[str, int],
        src_col: str = "Source IP",
        dst_col: str = "Destination IP",
        label_col: str = "Label",
        window_start: Optional[pd.Timestamp] = None,
        window_end: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """
        Process a single time window and return AGC statistics.

        Args:
            window_df: DataFrame slice for this time window.
            ip_to_node: IP→node_id mapping.
            src_col, dst_col: Column names for source/dest IPs.
            label_col: Label column.
            window_start, window_end: Timestamps for this window.

        Returns:
            Dict with AGC statistics for this window.
        """
        w_idx = self._window_index

        # Map IPs to node IDs for this window
        src_ids = window_df[src_col].map(ip_to_node).dropna().astype(int)
        dst_ids = window_df[dst_col].map(ip_to_node).dropna().astype(int)
        valid = src_ids.index.intersection(dst_ids.index)
        edges_this_window = list(zip(src_ids[valid].values, dst_ids[valid].values))
        nodes_this_window = set(src_ids[valid].values) | set(dst_ids[valid].values)

        # New nodes (not seen in any previous window)
        new_nodes = nodes_this_window - self._all_nodes
        self._all_nodes |= nodes_this_window

        # Edge frequencies in this window
        edge_freq: Dict[Tuple[int, int], int] = {}
        for e in edges_this_window:
            edge_freq[e] = edge_freq.get(e, 0) + 1

        # Max frequency for normalization
        max_freq = max(edge_freq.values()) if edge_freq else 1

        # New edges (not seen before)
        new_edges = set(edge_freq.keys()) - self._all_edges
        self._all_edges |= set(edge_freq.keys())

        # Update edge weights (EMA)
        updated_edges = []
        for e, freq in edge_freq.items():
            w_current = freq / max_freq
            if e in self._edge_weights:
                # EMA update
                old_w = self._edge_weights[e]
                new_w = (1 - self.weight_decay) * old_w + self.weight_decay * w_current
                self._edge_weights[e] = new_w
                updated_edges.append(e)
            else:
                self._edge_weights[e] = w_current
            self._edge_last_active[e] = w_idx

        # Decay inactive edges
        stale_edges = []
        to_prune = []
        for e, last_active in list(self._edge_last_active.items()):
            if e in edge_freq:
                continue  # already updated above
            windows_inactive = w_idx - last_active
            if windows_inactive >= self.stale_threshold_windows:
                stale_edges.append(e)
                to_prune.append(e)
            else:
                # Decay weight
                self._edge_weights[e] *= self.weight_decay

        # Prune stale edges
        for e in to_prune:
            del self._edge_weights[e]
            del self._edge_last_active[e]
            self._all_edges.discard(e)

        # Also prune edges that fell below minimum weight
        below_min = [e for e, w in self._edge_weights.items() if w < self.min_edge_weight]
        for e in below_min:
            del self._edge_weights[e]
            self._edge_last_active.pop(e, None)

        # Attack ratio this window
        attack_count = 0
        if label_col in window_df.columns:
            attack_count = int((window_df[label_col] != "BENIGN").sum())

        result = {
            "window_index": w_idx,
            "window_start": str(window_start) if window_start else None,
            "window_end": str(window_end) if window_end else None,
            "flows_in_window": len(window_df),
            "attack_flows": attack_count,
            "benign_flows": len(window_df) - attack_count,
            "active_nodes": len(nodes_this_window),
            "new_nodes": len(new_nodes),
            "active_edges": len(edge_freq),
            "new_edges": len(new_edges),
            "updated_edges": len(updated_edges),
            "stale_edges_removed": len(stale_edges),
            "total_tracked_edges": len(self._edge_weights),
            "total_tracked_nodes": len(self._all_nodes),
            "avg_edge_weight": float(np.mean(list(self._edge_weights.values()))) if self._edge_weights else 0.0,
            "max_edge_weight": float(max(self._edge_weights.values())) if self._edge_weights else 0.0,
        }

        self._window_index += 1
        return result

    def get_current_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """Return current edge weights for all active edges."""
        return dict(self._edge_weights)


def build_temporal_windows(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    window_minutes: int = 5,
) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]:
    """
    Divide DataFrame into time-ordered windows.

    Args:
        df: DataFrame with datetime Timestamp column.
        timestamp_col: Name of datetime column.
        window_minutes: Window size in minutes.

    Returns:
        List of (window_start, window_end, window_df) tuples.
    """
    if timestamp_col not in df.columns or not pd.api.types.is_datetime64_any_dtype(df[timestamp_col]):
        logger.warning("Timestamp column not available or not datetime. Cannot build temporal windows.")
        return []

    df_sorted = df.dropna(subset=[timestamp_col]).sort_values(timestamp_col)
    t_min = df_sorted[timestamp_col].min()
    t_max = df_sorted[timestamp_col].max()

    windows = []
    window_delta = pd.Timedelta(minutes=window_minutes)
    current = t_min

    while current <= t_max:
        next_t = current + window_delta
        mask = (df_sorted[timestamp_col] >= current) & (df_sorted[timestamp_col] < next_t)
        window_df = df_sorted[mask]
        if len(window_df) > 0:
            windows.append((current, next_t, window_df))
        current = next_t

    logger.info("Built %d temporal windows of %d minutes each", len(windows), window_minutes)
    return windows
