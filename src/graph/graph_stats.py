"""
GraphGuard: Phase 3 — Graph Construction
Module: graph_stats.py

Computes and reports comprehensive statistics about the constructed graph.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_graph_stats(
    n_nodes: int,
    edge_src: np.ndarray,
    edge_dst: np.ndarray,
    edge_labels: np.ndarray,
    label_encoder: Dict[str, int],
) -> Dict[str, Any]:
    """
    Compute comprehensive graph statistics.

    Args:
        n_nodes: Total number of unique IP nodes.
        edge_src: Source node IDs for each edge.
        edge_dst: Destination node IDs for each edge.
        edge_labels: Integer label for each edge.
        label_encoder: String→int label mapping.

    Returns:
        Dictionary of graph statistics.
    """
    n_edges = len(edge_src)

    # Degree calculations
    out_degree = np.bincount(edge_src, minlength=n_nodes)
    in_degree = np.bincount(edge_dst, minlength=n_nodes)
    total_degree = out_degree + in_degree

    # Edge label distribution
    int_to_label = {v: k for k, v in label_encoder.items()}
    label_counts = {}
    for label_int in np.unique(edge_labels):
        label_str = int_to_label.get(int(label_int), f"unknown_{label_int}")
        label_counts[label_str] = int(np.sum(edge_labels == label_int))

    benign_count = label_counts.get("BENIGN", 0)
    attack_count = n_edges - benign_count

    stats = {
        "num_nodes": n_nodes,
        "num_edges": n_edges,
        "avg_out_degree": float(out_degree.mean()),
        "avg_in_degree": float(in_degree.mean()),
        "max_out_degree": int(out_degree.max()),
        "max_in_degree": int(in_degree.max()),
        "avg_total_degree": float(total_degree.mean()),
        "max_total_degree": int(total_degree.max()),
        "benign_edges": benign_count,
        "attack_edges": attack_count,
        "attack_ratio": round(attack_count / n_edges, 4) if n_edges > 0 else 0,
        "label_distribution": label_counts,
        "num_isolated_nodes": int(np.sum(total_degree == 0)),
    }

    logger.info("Graph Statistics:")
    logger.info("  Nodes:           %d", stats["num_nodes"])
    logger.info("  Edges:           %d", stats["num_edges"])
    logger.info("  Avg Out-Degree:  %.2f", stats["avg_out_degree"])
    logger.info("  Max Out-Degree:  %d", stats["max_out_degree"])
    logger.info("  Benign edges:    %d (%.1f%%)", benign_count, 100 * benign_count / max(n_edges, 1))
    logger.info("  Attack edges:    %d (%.1f%%)", attack_count, 100 * attack_count / max(n_edges, 1))
    logger.info("  Label dist:      %s", label_counts)

    return stats


def save_graph_stats(stats: Dict[str, Any], path: Path) -> None:
    """Save graph statistics to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Graph stats saved to %s", path)
