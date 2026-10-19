#!/usr/bin/env python3
"""
GraphGuard — Script 03: Run Adaptive Graph Construction (AGC)
=============================================================
Applies temporal windowing to build a series of graph snapshots.
Uses actual timestamps from processed data.

Output:
    results/graphs/agc_snapshots.json    ← per-window statistics
    results/graphs/agc_summary.csv       ← summary table
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.agc.adaptive_graph import AdaptiveGraphConstructor, build_temporal_windows
from src.graph.node_builder import build_ip_to_node_mapping
from src.utils.config import load_config, resolve_path, get_project_root
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    processed_dir = resolve_path(cfg, "data", "processed")
    graphs_dir = resolve_path(cfg, "results", "graphs")
    graphs_dir.mkdir(parents=True, exist_ok=True)

    agc_cfg = cfg["agc"]
    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 4: Adaptive Graph Construction")
    logger.info("Window size: %d minutes", agc_cfg["window_minutes"])
    logger.info("=" * 60)

    # Load training data (AGC uses training split only to avoid leakage)
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    df = pd.concat([train_df, val_df], ignore_index=True)
    logger.info("Loaded %d rows for AGC", len(df))

    # Build node mapping
    ip_to_node = build_ip_to_node_mapping(df)

    # Build temporal windows
    windows = build_temporal_windows(
        df,
        timestamp_col=cfg["preprocessing"]["timestamp_column"],
        window_minutes=agc_cfg["window_minutes"],
    )

    if not windows:
        logger.error("No temporal windows generated. Check that Timestamp column is valid.")
        sys.exit(1)

    # Run AGC
    agc = AdaptiveGraphConstructor(
        window_minutes=agc_cfg["window_minutes"],
        stale_threshold_windows=agc_cfg["stale_threshold_windows"],
        weight_decay=agc_cfg["weight_decay"],
        min_edge_weight=agc_cfg["min_edge_weight"],
    )

    all_stats = []
    for w_start, w_end, w_df in windows:
        stats = agc.process_window(
            window_df=w_df,
            ip_to_node=ip_to_node,
            window_start=w_start,
            window_end=w_end,
        )
        all_stats.append(stats)
        logger.info(
            "Window %03d [%s → %s]: flows=%d | new_nodes=%d | new_edges=%d | stale_removed=%d",
            stats["window_index"], w_start.strftime("%H:%M"), w_end.strftime("%H:%M"),
            stats["flows_in_window"], stats["new_nodes"],
            stats["new_edges"], stats["stale_edges_removed"],
        )

    # Save outputs
    with open(graphs_dir / "agc_snapshots.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(graphs_dir / "agc_summary.csv", index=False)

    logger.info("=" * 60)
    logger.info("AGC complete: %d windows processed", len(all_stats))
    logger.info("  Total new nodes discovered: %d", sum(s["new_nodes"] for s in all_stats))
    logger.info("  Total new edges discovered: %d", sum(s["new_edges"] for s in all_stats))
    logger.info("  Total stale edges removed:  %d", sum(s["stale_edges_removed"] for s in all_stats))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
