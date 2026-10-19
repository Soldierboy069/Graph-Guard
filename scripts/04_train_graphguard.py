#!/usr/bin/env python3
"""
GraphGuard — Script 04: Train GraphGuard (GraphSAGE + EdgeClassifier)
======================================================================
Loads the preprocessed graph and trains the full GraphGuard model.

Usage:
    python scripts/04_train_graphguard.py
    python scripts/04_train_graphguard.py --experiment full_graphguard
    python scripts/04_train_graphguard.py --no-graph-context   # ablation
    python scripts/04_train_graphguard.py --binary             # binary classification

Output:
    models/{experiment_name}_best.pt
    results/metrics/{experiment_name}_training_history.json
    results/predictions/{experiment_name}_test_predictions.parquet
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.models.edge_classifier import EdgeClassifier, GraphGuardModel
from src.models.graphsage import GraphSAGEEncoder
from src.models.trainer import get_device, train_graphguard
from src.utils.config import load_config, resolve_path, get_project_root
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphGuard: Train GNN model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="full_graphguard",
                        help="Experiment name for checkpointing")
    parser.add_argument("--binary", action="store_true",
                        help="Use binary labels (BENIGN vs ATTACK)")
    parser.add_argument("--no-graph-context", action="store_true",
                        help="Ablation: skip node embeddings, use edge features only")
    return parser.parse_args()


def run_train(
    experiment_name: str = "full_graphguard",
    binary: bool = False,
    no_graph_context: bool = False,
    cfg_override: dict = None,
) -> dict:
    cfg = cfg_override or load_config()
    set_seed(cfg["project"]["seed"])

    processed_dir = resolve_path(cfg, "data", "processed")
    models_dir = resolve_path(cfg, "models")
    metrics_dir = resolve_path(cfg, "results", "metrics")
    preds_dir = resolve_path(cfg, "results", "predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)

    graph_path = processed_dir / "graph_data.pt"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph not found at {graph_path}. Run 02_build_graph.py first.")

    logger.info("=" * 60)
    logger.info("GraphGuard — Training GNN: %s", experiment_name)
    logger.info("=" * 60)

    data = torch.load(graph_path, map_location="cpu", weights_only=False)

    if binary:
        if hasattr(data, "y_train_binary"):
            data.y_train = data.y_train_binary
            data.val_y = data.val_y_binary
            data.test_y = data.test_y_binary
        data.y = getattr(data, "y_binary", data.y)
        num_classes = 2
        logger.info("Using BINARY classification (2 classes: BENIGN vs ATTACK)")
    else:
        num_classes = int(data.y_train.max().item() + 1) if hasattr(data, "y_train") else int(data.y.max().item() + 1)
        logger.info("Using MULTICLASS classification (%d classes)", num_classes)

    gnn_cfg = cfg["model"]["graphsage"]
    clf_cfg = cfg["model"]["edge_classifier"]
    device = get_device(cfg["model"]["training"].get("use_mps", True))

    node_feat_dim = data.x_train.shape[1] if hasattr(data, "x_train") else data.x.shape[1]
    edge_feat_dim = data.edge_attr_train.shape[1] if hasattr(data, "edge_attr_train") else data.edge_attr.shape[1]
    embedding_dim = gnn_cfg["embedding_dim"]

    if no_graph_context or experiment_name == "no_graph_context":
        logger.info("ABLATION: no-graph-context mode — zeroing node embeddings (edge features only).")

        class ZeroEncoder(torch.nn.Module):
            def __init__(self, embed_dim):
                super().__init__()
                self.embed_dim = embed_dim
            def forward(self, x, edge_index):
                return torch.zeros(x.shape[0], self.embed_dim, device=x.device)

        encoder = ZeroEncoder(embedding_dim)
    elif experiment_name == "no_graphsage":
        logger.info("ABLATION: no_graphsage mode — raw node features without neighborhood aggregation.")

        class RawNodeEncoder(torch.nn.Module):
            def __init__(self, in_dim, embed_dim):
                super().__init__()
                self.proj = torch.nn.Sequential(
                    torch.nn.Linear(in_dim, embed_dim),
                    torch.nn.BatchNorm1d(embed_dim),
                    torch.nn.ReLU()
                )
            def forward(self, x, edge_index):
                # Direct projection of node features without neighbor aggregation
                return self.proj(x)

        encoder = RawNodeEncoder(node_feat_dim, embedding_dim)
    elif experiment_name == "no_agc":
        logger.info("ABLATION: no_agc mode — static graph without temporal windowing or adaptive weights.")
        encoder = GraphSAGEEncoder(
            in_channels=node_feat_dim,
            hidden_dim=gnn_cfg["hidden_dim"],
            embedding_dim=embedding_dim,
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            normalize=gnn_cfg["normalize"],
            aggregator=gnn_cfg["aggregator"],
        )
    elif experiment_name == "no_adaptive_weights":
        logger.info("ABLATION: no_adaptive_weights mode — graph structure preserved, uniform edge weights.")
        encoder = GraphSAGEEncoder(
            in_channels=node_feat_dim,
            hidden_dim=gnn_cfg["hidden_dim"],
            embedding_dim=embedding_dim,
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            normalize=gnn_cfg["normalize"],
            aggregator=gnn_cfg["aggregator"],
        )
    else:
        logger.info("MODEL: Full GraphGuard (GraphSAGE + Inductive Context + EdgeClassifier)")
        encoder = GraphSAGEEncoder(
            in_channels=node_feat_dim,
            hidden_dim=gnn_cfg["hidden_dim"],
            embedding_dim=embedding_dim,
            num_layers=gnn_cfg["num_layers"],
            dropout=gnn_cfg["dropout"],
            normalize=gnn_cfg["normalize"],
            aggregator=gnn_cfg["aggregator"],
        )

    classifier = EdgeClassifier(
        node_embed_dim=embedding_dim,
        edge_feat_dim=edge_feat_dim,
        num_classes=num_classes,
        hidden_dims=clf_cfg["hidden_dims"],
        dropout=clf_cfg["dropout"],
    )

    model = GraphGuardModel(encoder, classifier)
    results = train_graphguard(
        model=model,
        data=data,
        cfg=cfg,
        model_save_dir=models_dir,
        metrics_save_dir=metrics_dir,
        device=device,
        experiment_name=experiment_name,
    )

    # Save test predictions
    test_meta_path = resolve_path(cfg, "results", "graphs") / "flow_metadata.parquet"
    if test_meta_path.exists():
        meta = pd.read_parquet(test_meta_path)
        test_meta = meta[meta.get("_split", pd.Series(["test"] * len(meta))) == "test"].reset_index(drop=True)
    else:
        test_meta = pd.DataFrame()

    preds_df = pd.DataFrame({
        "true_label": results["test_labels"],
        "predicted_label": results["test_predictions"],
    })
    if len(test_meta) == len(preds_df):
        preds_df = pd.concat([test_meta.reset_index(drop=True), preds_df], axis=1)

    preds_df.to_parquet(preds_dir / f"{experiment_name}_test_predictions.parquet", index=False)

    return results


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    results = run_train(
        experiment_name=args.experiment,
        binary=args.binary,
        no_graph_context=args.no_graph_context,
        cfg_override=cfg,
    )
    logger.info("Training complete: best val F1=%.4f, test F1=%.4f", results["best_val_f1"], results["test_f1"])


if __name__ == "__main__":
    main()
