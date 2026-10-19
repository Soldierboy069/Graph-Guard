"""
GraphGuard: Phase 3 — Graph Construction
Module: node_builder.py

Maps unique IP addresses to integer node IDs.
Computes meaningful node features aggregated from flow statistics.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def build_ip_to_node_mapping(
    df: pd.DataFrame,
    src_col: str = "Source IP",
    dst_col: str = "Destination IP",
) -> Dict[str, int]:
    """
    Create a mapping from unique IP address strings to integer node IDs.

    Both source and destination IPs are included.

    Args:
        df: DataFrame with flow records.
        src_col: Column name for source IPs.
        dst_col: Column name for destination IPs.

    Returns:
        Dict mapping IP string → integer node ID (0-indexed).
    """
    src_ips = df[src_col].dropna().unique()
    dst_ips = df[dst_col].dropna().unique()
    all_ips = sorted(set(src_ips.tolist() + dst_ips.tolist()))
    ip_to_node = {ip: idx for idx, ip in enumerate(all_ips)}
    logger.info("Built IP→node mapping: %d unique IPs", len(ip_to_node))
    return ip_to_node


def compute_node_features(
    df: pd.DataFrame,
    ip_to_node: Dict[str, int],
    src_col: str = "Source IP",
    dst_col: str = "Destination IP",
    label_col: str = "Label",
) -> pd.DataFrame:
    """
    Compute aggregated node features for each IP address from flow statistics.

    Node features capture structural and traffic-based properties:
    - connection_count: total flows involving this IP (in or out)
    - out_degree: number of unique destinations this IP talks to
    - in_degree: number of unique sources communicating to this IP
    - out_flow_bytes_mean: mean forward bytes in outbound flows
    - in_flow_bytes_mean: mean forward bytes in inbound flows
    - out_packet_count_mean: mean packet count in outbound flows
    - in_packet_count_mean: mean packet count in inbound flows
    - out_flow_duration_mean: mean flow duration in outbound flows
    - in_flow_duration_mean: mean flow duration in inbound flows
    - unique_neighbors: total distinct peer IPs this node connects to or receives from

    Args:
        df: DataFrame with flow records.
        ip_to_node: IP→node_id mapping.
        src_col: Source IP column.
        dst_col: Destination IP column.

    Returns:
        DataFrame indexed by node_id with feature columns.
    """
    n_nodes = len(ip_to_node)

    # Initialize feature arrays
    features = {
        "node_id": list(ip_to_node.values()),
        "ip": list(ip_to_node.keys()),
        "connection_count": np.zeros(n_nodes, dtype=np.float32),
        "out_degree": np.zeros(n_nodes, dtype=np.float32),
        "in_degree": np.zeros(n_nodes, dtype=np.float32),
        "out_flow_bytes_mean": np.zeros(n_nodes, dtype=np.float32),
        "in_flow_bytes_mean": np.zeros(n_nodes, dtype=np.float32),
        "out_packet_count_mean": np.zeros(n_nodes, dtype=np.float32),
        "in_packet_count_mean": np.zeros(n_nodes, dtype=np.float32),
        "out_flow_duration_mean": np.zeros(n_nodes, dtype=np.float32),
        "in_flow_duration_mean": np.zeros(n_nodes, dtype=np.float32),
        "unique_neighbors": np.zeros(n_nodes, dtype=np.float32),
    }

    ip_to_idx = {ip: idx for idx, ip in enumerate(ip_to_node.keys())}
    peer_sets: Dict[int, set] = {idx: set() for idx in range(n_nodes)}

    # Outbound aggregations (group by Source IP)
    bytes_col = "Total Length of Fwd Packets"
    pkts_col = "Total Fwd Packets"
    dur_col = "Flow Duration"

    def safe_col(col):
        return col if col in df.columns else None

    bc = safe_col(bytes_col)
    pc = safe_col(pkts_col)
    dc = safe_col(dur_col)

    logger.info("Computing outbound node features...")
    out_grp = df.groupby(src_col)

    for ip, grp in out_grp:
        if ip not in ip_to_idx:
            continue
        idx = ip_to_idx[ip]
        features["connection_count"][idx] += len(grp)
        features["out_degree"][idx] = grp[dst_col].nunique()
        peer_sets[idx].update(grp[dst_col].dropna().unique())
        if bc:
            features["out_flow_bytes_mean"][idx] = grp[bc].mean()
        if pc:
            features["out_packet_count_mean"][idx] = grp[pc].mean()
        if dc:
            features["out_flow_duration_mean"][idx] = grp[dc].mean()

    logger.info("Computing inbound node features...")
    in_grp = df.groupby(dst_col)

    for ip, grp in in_grp:
        if ip not in ip_to_idx:
            continue
        idx = ip_to_idx[ip]
        features["connection_count"][idx] += len(grp)
        features["in_degree"][idx] = grp[src_col].nunique()
        peer_sets[idx].update(grp[src_col].dropna().unique())
        if bc:
            features["in_flow_bytes_mean"][idx] = grp[bc].mean()
        if pc:
            features["in_packet_count_mean"][idx] = grp[pc].mean()
        if dc:
            features["in_flow_duration_mean"][idx] = grp[dc].mean()

    # Total distinct unique communication partners
    for idx in range(n_nodes):
        features["unique_neighbors"][idx] = float(len(peer_sets[idx]))

    node_df = pd.DataFrame(features)
    node_df = node_df.set_index("node_id").sort_index()

    logger.info("Node feature matrix shape: %s", node_df.shape)
    return node_df
