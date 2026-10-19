#!/usr/bin/env python3
"""
GraphGuard — Script 06: Evaluate All Models
============================================
Computes metrics for GraphGuard, Random Forest, and XGBoost on test set.
Generates confusion matrices, comparison table, and training curves.

Output:
    results/metrics/graphguard_metrics.json
    results/metrics/rf_metrics.json
    results/metrics/xgb_metrics.json
    results/metrics/model_comparison.csv
    results/figures/graphguard_confusion_matrix.png
    results/figures/rf_confusion_matrix.png
    results/figures/xgb_confusion_matrix.png
    results/figures/graphguard_training_curves.png
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.metrics import (
    compare_models,
    compute_metrics,
    plot_confusion_matrix,
    plot_training_curves,
    save_metrics,
)
from src.preprocessing.encoder import load_label_encoder
from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_dir="results/logs")


def load_predictions(path: Path) -> tuple:
    """Load predictions parquet and return (y_true, y_pred)."""
    df = pd.read_parquet(path)
    return df["true_label"].values, df["predicted_label"].values


def main() -> None:
    cfg = load_config()
    preds_dir = resolve_path(cfg, "results", "predictions")
    metrics_dir = resolve_path(cfg, "results", "metrics")
    figures_dir = resolve_path(cfg, "results", "figures")
    processed_dir = resolve_path(cfg, "data", "processed")

    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 7: Evaluation")
    logger.info("=" * 60)

    multi_encoder = load_label_encoder(processed_dir / "label_encoder.json")
    binary_encoder = {"BENIGN": 0, "ATTACK": 1}

    # ================================================================
    # PART A: MULTICLASS EVALUATION (BENIGN, FTP-Patator, SSH-Patator)
    # ================================================================
    logger.info("=== PART A: MULTICLASS EVALUATION ===")
    multi_metrics = []

    # 1. GraphGuard (Multiclass)
    gg_multi_path = preds_dir / "full_graphguard_test_predictions.parquet"
    if gg_multi_path.exists():
        y_true, y_pred = load_predictions(gg_multi_path)
        gg_m = compute_metrics(y_true, y_pred, multi_encoder, "GraphGuard (Multiclass)")
        save_metrics(gg_m, metrics_dir / "graphguard_multiclass_metrics.json")
        save_metrics(gg_m, metrics_dir / "graphguard_metrics.json")
        plot_confusion_matrix(gg_m, figures_dir / "graphguard_multiclass_confusion_matrix.png")
        multi_metrics.append(gg_m)

        history_path = metrics_dir / "full_graphguard_training_history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            plot_training_curves(history, figures_dir / "graphguard_training_curves.png")

    # 2. Random Forest (Multiclass)
    rf_multi_path = preds_dir / "rf_multiclass_test_predictions.parquet"
    if not rf_multi_path.exists():
        rf_multi_path = preds_dir / "rf_test_predictions.parquet"
    if rf_multi_path.exists():
        y_true, y_pred = load_predictions(rf_multi_path)
        rf_m = compute_metrics(y_true, y_pred, multi_encoder, "Random Forest (Multiclass)")
        save_metrics(rf_m, metrics_dir / "rf_multiclass_metrics.json")
        save_metrics(rf_m, metrics_dir / "rf_metrics.json")
        plot_confusion_matrix(rf_m, figures_dir / "rf_multiclass_confusion_matrix.png")
        multi_metrics.append(rf_m)

    # 3. XGBoost (Multiclass)
    xgb_multi_path = preds_dir / "xgb_multiclass_test_predictions.parquet"
    if not xgb_multi_path.exists():
        xgb_multi_path = preds_dir / "xgb_test_predictions.parquet"
    if xgb_multi_path.exists():
        y_true, y_pred = load_predictions(xgb_multi_path)
        xgb_m = compute_metrics(y_true, y_pred, multi_encoder, "XGBoost (Multiclass)")
        save_metrics(xgb_m, metrics_dir / "xgb_multiclass_metrics.json")
        save_metrics(xgb_m, metrics_dir / "xgb_metrics.json")
        plot_confusion_matrix(xgb_m, figures_dir / "xgb_multiclass_confusion_matrix.png")
        multi_metrics.append(xgb_m)

    if multi_metrics:
        compare_models(multi_metrics, metrics_dir / "model_comparison_multiclass.csv")
        compare_models(multi_metrics, metrics_dir / "model_comparison.csv")

    # ================================================================
    # PART B: BINARY EVALUATION (BENIGN vs ATTACK)
    # ================================================================
    logger.info("=== PART B: BINARY EVALUATION ===")
    binary_metrics = []

    # 1. GraphGuard (Binary)
    gg_bin_path = preds_dir / "full_graphguard_binary_test_predictions.parquet"
    if gg_bin_path.exists():
        y_true, y_pred = load_predictions(gg_bin_path)
        gg_b = compute_metrics(y_true, y_pred, binary_encoder, "GraphGuard (Binary)")
        save_metrics(gg_b, metrics_dir / "graphguard_binary_metrics.json")
        plot_confusion_matrix(gg_b, figures_dir / "graphguard_binary_confusion_matrix.png")
        binary_metrics.append(gg_b)

    # 2. Random Forest (Binary)
    rf_bin_path = preds_dir / "rf_binary_test_predictions.parquet"
    if rf_bin_path.exists():
        y_true, y_pred = load_predictions(rf_bin_path)
        rf_b = compute_metrics(y_true, y_pred, binary_encoder, "Random Forest (Binary)")
        save_metrics(rf_b, metrics_dir / "rf_binary_metrics.json")
        plot_confusion_matrix(rf_b, figures_dir / "rf_binary_confusion_matrix.png")
        binary_metrics.append(rf_b)

    # 3. XGBoost (Binary)
    xgb_bin_path = preds_dir / "xgb_binary_test_predictions.parquet"
    if xgb_bin_path.exists():
        y_true, y_pred = load_predictions(xgb_bin_path)
        xgb_b = compute_metrics(y_true, y_pred, binary_encoder, "XGBoost (Binary)")
        save_metrics(xgb_b, metrics_dir / "xgb_binary_metrics.json")
        plot_confusion_matrix(xgb_b, figures_dir / "xgb_binary_confusion_matrix.png")
        binary_metrics.append(xgb_b)

    if binary_metrics:
        compare_models(binary_metrics, metrics_dir / "model_comparison_binary.csv")

    logger.info("=" * 60)
    logger.info("Evaluation complete. Outputs in: %s", metrics_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
