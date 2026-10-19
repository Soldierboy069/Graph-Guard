"""
GraphGuard: Unit tests for Phase 3 (Graph Construction) and Phase 4 (AGC)
"""
import pytest
import numpy as np
import pandas as pd

from src.graph.node_builder import build_ip_to_node_mapping, compute_node_features
from src.graph.edge_builder import build_edge_index, build_edge_features, build_edge_labels
from src.graph.graph_stats import compute_graph_stats
from src.agc.adaptive_graph import AdaptiveGraphConstructor, build_temporal_windows


def test_ip_to_node_mapping():
    df = pd.DataFrame({
        "Source IP": ["192.168.1.1", "192.168.1.2", "192.168.1.3"],
        "Destination IP": ["192.168.1.2", "192.168.1.3", "192.168.1.4"],
    })
    mapping = build_ip_to_node_mapping(df)
    assert len(mapping) == 4
    assert set(mapping.values()) == {0, 1, 2, 3}


def test_edge_building():
    df = pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.2"],
        "Destination IP": ["10.0.0.2", "10.0.0.3"],
        "Flow Duration": [100, 200],
        "label_multiclass": [0, 1],
    })
    mapping = {"10.0.0.1": 0, "10.0.0.2": 1, "10.0.0.3": 2}
    src, dst, mask = build_edge_index(df, mapping)
    assert len(src) == 2
    assert list(src) == [0, 1]
    assert list(dst) == [1, 2]

    attr = build_edge_features(df, mask, ["Flow Duration"])
    assert attr.shape == (2, 1)

    labels = build_edge_labels(df, mask, "label_multiclass")
    assert list(labels) == [0, 1]


def test_graph_stats():
    src = np.array([0, 1, 0])
    dst = np.array([1, 2, 2])
    labels = np.array([0, 1, 0])
    encoder = {"BENIGN": 0, "ATTACK": 1}

    stats = compute_graph_stats(3, src, dst, labels, encoder)
    assert stats["num_nodes"] == 3
    assert stats["num_edges"] == 3
    assert stats["benign_edges"] == 2
    assert stats["attack_edges"] == 1


def test_agc_window_processing():
    ip_to_node = {"10.0.0.1": 0, "10.0.0.2": 1}
    df1 = pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.1"],
        "Destination IP": ["10.0.0.2", "10.0.0.2"],
        "Label": ["BENIGN", "BENIGN"],
    })
    agc = AdaptiveGraphConstructor(window_minutes=5)
    stats1 = agc.process_window(df1, ip_to_node)
    assert stats1["new_nodes"] == 2
    assert stats1["new_edges"] == 1
    assert stats1["flows_in_window"] == 2


def test_compute_node_features():
    df = pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.1", "10.0.0.2"],
        "Destination IP": ["10.0.0.2", "10.0.0.3", "10.0.0.3"],
        "Total Length of Fwd Packets": [100.0, 200.0, 300.0],
        "Total Fwd Packets": [2.0, 4.0, 6.0],
        "Flow Duration": [1000.0, 2000.0, 3000.0],
    })
    ip_to_node = {"10.0.0.1": 0, "10.0.0.2": 1, "10.0.0.3": 2}
    node_df = compute_node_features(df, ip_to_node)
    assert len(node_df) == 3
    assert "unique_neighbors" in node_df.columns
    assert "attack_neighbor_ratio" not in node_df.columns
    # 10.0.0.1 talks to 10.0.0.2 and 10.0.0.3 -> 2 unique neighbors
    assert node_df.loc[0, "unique_neighbors"] == 2.0
    # 10.0.0.2 has incoming from 10.0.0.1 and outgoing to 10.0.0.3 -> 2 unique neighbors
    assert node_df.loc[1, "unique_neighbors"] == 2.0
    # 10.0.0.3 has incoming from 10.0.0.1 and 10.0.0.2 -> 2 unique neighbors
    assert node_df.loc[2, "unique_neighbors"] == 2.0

