#!/usr/bin/env python3
"""
GraphGuard — Script 02: Graph Construction
==========================================
Loads preprocessed Parquet files and constructs a PyTorch Geometric
HeteroData / Data graph with:
  - node features (aggregated per-IP statistics)
  - directed edge index (Source IP → Destination IP)
  - edge features (CICIDS2017 traffic flow features)
  - edge labels (multiclass + binary)
  - edge timestamps
  - flow metadata (for dashboard lookups)

Saves:
  data/processed/graph_data.pt        ← PyG Data object
  results/graphs/graph_stats.json     ← graph statistics
  results/graphs/flow_metadata.parquet ← per-edge metadata

Usage:
    python scripts/02_build_graph.py              # uses full processed dataset
    python scripts/02_build_graph.py --sample     # expects sample parquet files
    python scripts/02_build_graph.py --split train  # build from train split only
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.graph.edge_builder import (
    build_edge_features,
    build_edge_index,
    build_edge_labels,
    build_edge_timestamps,
    get_edge_feature_columns,
    get_flow_metadata,
)
from src.graph.graph_stats import compute_graph_stats, save_graph_stats
from src.graph.node_builder import build_ip_to_node_mapping, compute_node_features
from src.preprocessing.encoder import load_label_encoder
from src.utils.config import load_config, resolve_path, get_project_root
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphGuard: Graph Construction")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="all",
                        help="Which split to build graph from: train | val | test | all")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    root = get_project_root(cfg)
    processed_dir = resolve_path(cfg, "data", "processed")
    graphs_dir = resolve_path(cfg, "results", "graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------
    # Load processed splits
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 3: Graph Construction")
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # Load processed splits independently
    # ----------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 3: Inductive Graph Construction")
    logger.info("=" * 60)

    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Loaded splits: Train=%d, Val=%d, Test=%d", len(train_df), len(val_df), len(test_df))

    # Combined DataFrame ONLY for discovering full IP universe mapping
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    ip_to_node = build_ip_to_node_mapping(full_df)
    n_nodes = len(ip_to_node)
    logger.info("Total IP node universe: %d nodes", n_nodes)

    # Load label encoder and feature columns
    encoder_path = processed_dir / "label_encoder.json"
    label_encoder = load_label_encoder(encoder_path)

    feat_col_path = processed_dir / "feature_columns.json"
    if feat_col_path.exists():
        with open(feat_col_path) as f:
            feature_cols = json.load(f)
        feature_cols = [c for c in feature_cols if c in train_df.columns]
    else:
        feature_cols = get_edge_feature_columns(train_df)
    logger.info("Traffic feature columns: %d features", len(feature_cols))

    # ----------------------------------------------------------------
    # 1. TRAINING GRAPH (Constructed STRICTLY from TRAIN flows)
    # ----------------------------------------------------------------
    logger.info("Building TRAINING graph (G_train) from TRAIN flows only...")
    train_node_df = compute_node_features(train_df, ip_to_node)
    node_feat_cols = [c for c in train_node_df.columns if c != "ip"]

    x_train = torch.tensor(train_node_df[node_feat_cols].values.astype(np.float32), dtype=torch.float32)
    tr_src, tr_dst, tr_mask = build_edge_index(train_df, ip_to_node)
    edge_index_train = torch.tensor(np.stack([tr_src, tr_dst], axis=0), dtype=torch.long)
    edge_attr_train = torch.tensor(build_edge_features(train_df, tr_mask, feature_cols), dtype=torch.float32)
    y_train_multi = torch.tensor(build_edge_labels(train_df, tr_mask, "label_multiclass"), dtype=torch.long)
    y_train_binary = torch.tensor(build_edge_labels(train_df, tr_mask, "label_binary"), dtype=torch.long)
    train_timestamps = build_edge_timestamps(train_df, tr_mask, "Timestamp")

    logger.info("G_train assembled: %d nodes, %d edges", n_nodes, edge_index_train.shape[1])

    # ----------------------------------------------------------------
    # 2. VALIDATION QUERY EDGES (Target edge exclusion: evaluated against G_train)
    # ----------------------------------------------------------------
    logger.info("Building VALIDATION query edges...")
    va_src, va_dst, va_mask = build_edge_index(val_df, ip_to_node)
    val_edge_index = torch.tensor(np.stack([va_src, va_dst], axis=0), dtype=torch.long)
    val_edge_attr = torch.tensor(build_edge_features(val_df, va_mask, feature_cols), dtype=torch.float32)
    val_y_multi = torch.tensor(build_edge_labels(val_df, va_mask, "label_multiclass"), dtype=torch.long)
    val_y_binary = torch.tensor(build_edge_labels(val_df, va_mask, "label_binary"), dtype=torch.long)

    # ----------------------------------------------------------------
    # 3. HISTORICAL EVALUATION GRAPH (G_hist = Train + Val flows up to test time)
    # ----------------------------------------------------------------
    logger.info("Building HISTORICAL context graph (G_hist) from Train + Val flows...")
    hist_df = pd.concat([train_df, val_df], ignore_index=True)
    hist_node_df = compute_node_features(hist_df, ip_to_node)
    x_hist = torch.tensor(hist_node_df[node_feat_cols].values.astype(np.float32), dtype=torch.float32)

    hi_src, hi_dst, hi_mask = build_edge_index(hist_df, ip_to_node)
    edge_index_hist = torch.tensor(np.stack([hi_src, hi_dst], axis=0), dtype=torch.long)
    edge_attr_hist = torch.tensor(build_edge_features(hist_df, hi_mask, feature_cols), dtype=torch.float32)

    logger.info("G_hist assembled: %d nodes, %d edges (test edges strictly excluded)", n_nodes, edge_index_hist.shape[1])

    # ----------------------------------------------------------------
    # 4. TEST QUERY EDGES (Target edge exclusion: evaluated against G_hist)
    # ----------------------------------------------------------------
    logger.info("Building TEST query edges...")
    te_src, te_dst, te_mask = build_edge_index(test_df, ip_to_node)
    test_edge_index = torch.tensor(np.stack([te_src, te_dst], axis=0), dtype=torch.long)
    test_edge_attr = torch.tensor(build_edge_features(test_df, te_mask, feature_cols), dtype=torch.float32)
    test_y_multi = torch.tensor(build_edge_labels(test_df, te_mask, "label_multiclass"), dtype=torch.long)
    test_y_binary = torch.tensor(build_edge_labels(test_df, te_mask, "label_binary"), dtype=torch.long)

    # ----------------------------------------------------------------
    # Assemble Inductive PyG Data Object
    # ----------------------------------------------------------------
    from torch_geometric.data import Data

    data = Data(
        x=x_train,
        edge_index=edge_index_train,
        edge_attr=edge_attr_train,
        y=y_train_multi,
        y_binary=y_train_binary,
    )

    # Inductive training graph
    data.x_train = x_train
    data.edge_index_train = edge_index_train
    data.edge_attr_train = edge_attr_train
    data.y_train = y_train_multi
    data.y_train_binary = y_train_binary

    # Validation query edges (evaluated against G_train)
    data.val_edge_index = val_edge_index
    data.val_edge_attr = val_edge_attr
    data.val_y = val_y_multi
    data.val_y_binary = val_y_binary

    # Historical context graph (for test inference)
    data.x_hist = x_hist
    data.edge_index_hist = edge_index_hist
    data.edge_attr_hist = edge_attr_hist

    # Test query edges (evaluated against G_hist, never in message passing)
    data.test_edge_index = test_edge_index
    data.test_edge_attr = test_edge_attr
    data.test_y = test_y_multi
    data.test_y_binary = test_y_binary

    # Metadata
    data.feature_names = feature_cols
    data.node_feature_names = node_feat_cols
    data.inductive_mode = True

    # ----------------------------------------------------------------
    # Save outputs
    # ----------------------------------------------------------------
    graph_path = processed_dir / "graph_data.pt"
    torch.save(data, graph_path)
    logger.info("Inductive graph data saved to %s", graph_path)

    # Save flow metadata for test set
    test_flow_metadata = get_flow_metadata(test_df, te_mask)
    test_flow_metadata["_split"] = "test"
    test_flow_metadata.to_parquet(graphs_dir / "flow_metadata.parquet", index=False)
    logger.info("Test flow metadata saved to %s", graphs_dir / "flow_metadata.parquet")

    # Save node-to-IP mapping
    with open(graphs_dir / "ip_to_node.json", "w") as f:
        json.dump(ip_to_node, f, indent=2)

    # Graph statistics on G_train
    stats = compute_graph_stats(
        n_nodes=n_nodes,
        edge_src=tr_src,
        edge_dst=tr_dst,
        edge_labels=y_train_multi.numpy(),
        label_encoder=label_encoder,
    )
    stats["inductive_protocol"] = {
        "G_train_edges": int(edge_index_train.shape[1]),
        "G_hist_edges": int(edge_index_hist.shape[1]),
        "val_query_edges": int(val_edge_index.shape[1]),
        "test_query_edges": int(test_edge_index.shape[1]),
        "target_edge_exclusion": True,
    }
    save_graph_stats(stats, graphs_dir / "graph_stats.json")

    logger.info("=" * 60)
    logger.info("Inductive Graph construction complete.")
    logger.info("  Total IP Nodes:       %d", n_nodes)
    logger.info("  G_train Edges:        %d", edge_index_train.shape[1])
    logger.info("  G_hist Edges:         %d", edge_index_hist.shape[1])
    logger.info("  Val Query Edges:      %d", val_edge_index.shape[1])
    logger.info("  Test Query Edges:     %d", test_edge_index.shape[1])
    logger.info("  Node Feats (x):       %d", x_train.shape[1])
    logger.info("  Edge Feats:           %d", edge_attr_train.shape[1])
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
