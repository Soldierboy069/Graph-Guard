#!/usr/bin/env python3
"""
GraphGuard — Script 08: Calculate Threat Propagation Score
==========================================================
Computes TPS for every IP node using actual model predictions.

Output:
    results/metrics/tps_scores.csv
    results/metrics/tps_summary.json
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.preprocessing.encoder import load_label_encoder
from src.threat.tps import compute_tps
from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_dir="results/logs")


def main() -> None:
    cfg = load_config()
    processed_dir = resolve_path(cfg, "data", "processed")
    graphs_dir = resolve_path(cfg, "results", "graphs")
    preds_dir = resolve_path(cfg, "results", "predictions")
    metrics_dir = resolve_path(cfg, "results", "metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 9: Threat Propagation Score")
    logger.info("=" * 60)

    label_encoder = load_label_encoder(processed_dir / "label_encoder.json")

    # Load flow metadata
    meta_path = graphs_dir / "flow_metadata.parquet"
    if not meta_path.exists():
        logger.error("Flow metadata not found. Run 02_build_graph.py first.")
        sys.exit(1)
    flow_meta = pd.read_parquet(meta_path)

    # Load predictions directly from test predictions parquet if available
    pred_path = preds_dir / "full_graphguard_test_predictions.parquet"
    if pred_path.exists():
        logger.info("Loading test predictions from %s", pred_path)
        pred_df = pd.read_parquet(pred_path)
        predictions = pred_df["predicted_label"].values
        # Ensure flow_meta matches predictions
        if len(flow_meta) != len(predictions) and all(col in pred_df.columns for col in ["Source IP", "Destination IP"]):
            flow_meta = pred_df
        logger.info("Loaded %d test predictions (%d predicted attacks)", len(predictions), (predictions != 0).sum())
    else:
        # Fallback to checkpoint inference or metadata labels
        graph_path = processed_dir / "graph_data.pt"
        ckpt_path = resolve_path(cfg, "models") / "full_graphguard_best.pt"

        if graph_path.exists() and ckpt_path.exists():
            logger.info("Running inductive test inference with trained GraphGuard model...")
            import torch
            from src.models.graphsage import GraphSAGEEncoder
            from src.models.edge_classifier import EdgeClassifier, GraphGuardModel

            data = torch.load(graph_path, map_location="cpu", weights_only=False)
            num_classes = len(label_encoder)
            encoder = GraphSAGEEncoder(in_channels=data.x.shape[1], hidden_dim=128, embedding_dim=64, num_layers=2)
            classifier = EdgeClassifier(node_embed_dim=64, edge_feat_dim=data.edge_attr.shape[1], num_classes=num_classes, hidden_dims=[256, 128])
            model = GraphGuardModel(encoder, classifier)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            with torch.no_grad():
                x_hist = getattr(data, "x_hist", data.x)
                ei_hist = getattr(data, "edge_index_hist", data.edge_index)
                q_ei = getattr(data, "test_edge_index", data.edge_index)
                q_ea = getattr(data, "test_edge_attr", data.edge_attr)
                h = model.encoder(x_hist, ei_hist)
                logits = model.classifier(h, q_ei, q_ea)
                predictions = logits.argmax(dim=-1).numpy()
            logger.info("Inference complete: %d flows predicted (%d attacks)", len(predictions), (predictions != 0).sum())
        else:
            logger.warning("Predictions not found. Using flow metadata labels.")
            predictions = flow_meta["label_multiclass"].values if "label_multiclass" in flow_meta.columns else None

    if predictions is None:
        logger.error("No predictions available for TPS computation.")
        sys.exit(1)

    # Load IP→node mapping
    with open(graphs_dir / "ip_to_node.json") as f:
        ip_to_node = json.load(f)

    tps_weights = cfg["threat"]["tps"]["weights"]
    recent_window = cfg["threat"]["tps"]["recent_window_minutes"]

    tps_df = compute_tps(
        flow_metadata=flow_meta,
        predictions=predictions,
        ip_to_node=ip_to_node,
        label_encoder=label_encoder,
        weights=tps_weights,
        recent_window_minutes=recent_window,
    )

    tps_df.to_csv(metrics_dir / "tps_scores.csv", index=False)

    summary = {
        "total_nodes": len(tps_df),
        "critical_nodes": int((tps_df["threat_level"] == "CRITICAL").sum()),
        "high_nodes": int((tps_df["threat_level"] == "HIGH").sum()),
        "medium_nodes": int((tps_df["threat_level"] == "MEDIUM").sum()),
        "low_nodes": int((tps_df["threat_level"] == "LOW").sum()),
        "avg_tps": round(float(tps_df["tps"].mean()), 4),
        "max_tps": round(float(tps_df["tps"].max()), 4),
        "top_10_nodes": tps_df.head(10)[["ip", "tps", "threat_level"]].to_dict("records"),
        "formula_weights": tps_weights,
    }
    with open(metrics_dir / "tps_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("TPS complete: %d nodes scored", len(tps_df))
    logger.info("  Critical: %d | High: %d | Medium: %d | Low: %d",
                summary["critical_nodes"], summary["high_nodes"],
                summary["medium_nodes"], summary["low_nodes"])
    logger.info("Outputs: %s", metrics_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
