"""
GraphGuard: Phase 9 — Threat Propagation Score
Module: tps.py

Computes a normalized Threat Propagation Score (TPS) for each node/flow.
TPS is an alert-prioritization score, not a prediction of future attacks.

Formula (documented weights in config.yaml):
    TPS(node) = w1 * attack_probability
              + w2 * suspicious_neighbor_ratio
              + w3 * recent_attack_frequency
              + w4 * degree_centrality
              + w5 * betweenness_proxy
    → normalized to [0, 1]

where:
    attack_probability       = fraction of outbound flows from node that are predicted attacks
    suspicious_neighbor_ratio = fraction of direct neighbors with any attack flow
    recent_attack_frequency  = fraction of flows in recent time window that are attacks
    degree_centrality        = in+out degree / max_degree (normalized)
    betweenness_proxy        = simplified hub score (out_degree / total_degree)
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


THREAT_LEVELS = {
    "critical": 0.75,
    "high": 0.50,
    "medium": 0.25,
    "low": 0.0,
}


def get_threat_level(tps: float, thresholds: Dict[str, float] = None) -> str:
    """Return threat level string for a given TPS value."""
    if thresholds is None:
        thresholds = THREAT_LEVELS
    if tps >= thresholds.get("critical", 0.75):
        return "CRITICAL"
    elif tps >= thresholds.get("high", 0.50):
        return "HIGH"
    elif tps >= thresholds.get("medium", 0.25):
        return "MEDIUM"
    else:
        return "LOW"


def compute_tps(
    flow_metadata: pd.DataFrame,
    predictions: np.ndarray,
    ip_to_node: Dict[str, int],
    label_encoder: Dict[str, int],
    weights: Optional[Dict[str, float]] = None,
    recent_window_minutes: int = 30,
) -> pd.DataFrame:
    """
    Compute Threat Propagation Score for every node (IP address).

    Args:
        flow_metadata: Per-flow metadata DataFrame with Source IP, Dest IP, Timestamp, Label.
        predictions: Predicted integer labels for each flow (same order as flow_metadata).
        ip_to_node: IP→node_id mapping.
        label_encoder: String→int label mapping.
        weights: Component weights dict. Defaults to equal weights.
        recent_window_minutes: Time window for recent attack frequency.

    Returns:
        DataFrame with columns: [node_id, ip, tps, threat_level, component scores...]
    """
    if weights is None:
        weights = {
            "attack_probability": 0.35,
            "suspicious_neighbor_ratio": 0.25,
            "recent_attack_frequency": 0.20,
            "degree_centrality": 0.10,
            "betweenness_proxy": 0.10,
        }

    node_to_ip = {v: k for k, v in ip_to_node.items()}
    n_nodes = len(ip_to_node)

    benign_int = label_encoder.get("BENIGN", 0)
    is_attack = (predictions != benign_int).astype(float)

    src_col = "Source IP"
    dst_col = "Destination IP"
    ts_col = "Timestamp"

    src_ids = flow_metadata[src_col].map(ip_to_node).fillna(-1).astype(int)
    dst_ids = flow_metadata[dst_col].map(ip_to_node).fillna(-1).astype(int)

    # ── Component 1: Attack probability per source node ──────────────
    attack_prob = np.zeros(n_nodes)
    total_out = np.zeros(n_nodes)
    attack_out = np.zeros(n_nodes)
    for i, (si, atk) in enumerate(zip(src_ids.values, is_attack)):
        if si >= 0:
            total_out[si] += 1
            attack_out[si] += atk
    safe_tot = np.where(total_out > 0, total_out, 1)
    attack_prob = attack_out / safe_tot

    # ── Component 2: Suspicious neighbor ratio ────────────────────────
    # For each node, what fraction of its neighbors have any attack flow?
    node_has_attack = (attack_out > 0).astype(float)

    # Build adjacency: neighbors of each node
    neighbor_attack_ratio = np.zeros(n_nodes)
    neighbor_count = np.zeros(n_nodes)

    for si, di in zip(src_ids.values, dst_ids.values):
        if si >= 0 and di >= 0:
            neighbor_count[si] += 1
            if node_has_attack[di]:
                neighbor_attack_ratio[si] += 1

    safe_nb = np.where(neighbor_count > 0, neighbor_count, 1)
    suspicious_neighbor_ratio = neighbor_attack_ratio / safe_nb

    # ── Component 3: Recent attack frequency ──────────────────────────
    recent_attack_freq = np.zeros(n_nodes)
    if ts_col in flow_metadata.columns and pd.api.types.is_datetime64_any_dtype(flow_metadata[ts_col]):
        t_max = flow_metadata[ts_col].max()
        t_recent = t_max - pd.Timedelta(minutes=recent_window_minutes)
        recent_mask = (flow_metadata[ts_col] >= t_recent).values

        recent_attack = np.zeros(n_nodes)
        recent_total = np.zeros(n_nodes)
        for i, (si, atk) in enumerate(zip(src_ids.values, is_attack)):
            if si >= 0 and recent_mask[i]:
                recent_total[si] += 1
                recent_attack[si] += atk
        safe_rec = np.where(recent_total > 0, recent_total, 1)
        recent_attack_freq = recent_attack / safe_rec
    else:
        # Fall back to overall attack probability
        recent_attack_freq = attack_prob.copy()

    # ── Component 4: Degree centrality ────────────────────────────────
    out_degree = np.bincount(src_ids[src_ids >= 0].values, minlength=n_nodes).astype(float)
    in_degree = np.bincount(dst_ids[dst_ids >= 0].values, minlength=n_nodes).astype(float)
    total_degree = out_degree + in_degree
    max_deg = total_degree.max() if total_degree.max() > 0 else 1
    degree_centrality = total_degree / max_deg

    # ── Component 5: Betweenness proxy (out_degree / total_degree) ────
    safe_total_deg = np.where(total_degree > 0, total_degree, 1)
    betweenness_proxy = out_degree / safe_total_deg

    # ── Combine with weights ───────────────────────────────────────────
    w_atk = weights.get("attack_probability", 0.35)
    w_nb = weights.get("suspicious_neighbor_ratio", 0.25)
    w_rec = weights.get("recent_attack_frequency", 0.20)
    w_deg = weights.get("degree_centrality", 0.10)
    w_btw = weights.get("betweenness_proxy", weights.get("graph_betweenness", 0.10))

    tps = (
        w_atk * attack_prob
        + w_nb * suspicious_neighbor_ratio
        + w_rec * recent_attack_freq
        + w_deg * degree_centrality
        + w_btw * betweenness_proxy
    )

    # Normalize to [0, 1]
    tps_min, tps_max = tps.min(), tps.max()
    if tps_max > tps_min:
        tps = (tps - tps_min) / (tps_max - tps_min)

    # Build result DataFrame
    records = []
    for node_id in range(n_nodes):
        ip = node_to_ip.get(node_id, f"node_{node_id}")
        t = float(tps[node_id])
        records.append({
            "node_id": node_id,
            "ip": ip,
            "tps": round(t, 4),
            "threat_level": get_threat_level(t),
            "attack_probability": round(float(attack_prob[node_id]), 4),
            "suspicious_neighbor_ratio": round(float(suspicious_neighbor_ratio[node_id]), 4),
            "recent_attack_frequency": round(float(recent_attack_freq[node_id]), 4),
            "degree_centrality": round(float(degree_centrality[node_id]), 4),
            "betweenness_proxy": round(float(betweenness_proxy[node_id]), 4),
            "out_degree": int(out_degree[node_id]),
            "in_degree": int(in_degree[node_id]),
        })

    tps_df = pd.DataFrame(records).sort_values("tps", ascending=False).reset_index(drop=True)

    logger.info(
        "TPS computed: %d nodes | CRITICAL=%d | HIGH=%d | MEDIUM=%d | LOW=%d",
        len(tps_df),
        (tps_df["threat_level"] == "CRITICAL").sum(),
        (tps_df["threat_level"] == "HIGH").sum(),
        (tps_df["threat_level"] == "MEDIUM").sum(),
        (tps_df["threat_level"] == "LOW").sum(),
    )
    return tps_df
