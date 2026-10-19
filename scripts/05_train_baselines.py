#!/usr/bin/env python3
"""
GraphGuard — Script 05: Train Baseline Models
=============================================
Trains Random Forest and XGBoost on the same train/test split as GraphGuard.
Uses tabular traffic-flow features only (no graph context).

Usage:
    python scripts/05_train_baselines.py

Output:
    models/rf_model.pkl
    models/xgb_model.json
    results/predictions/rf_test_predictions.parquet
    results/predictions/xgb_test_predictions.parquet
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.baselines.random_forest import save_rf_model, train_random_forest
from src.baselines.xgboost_model import save_xgb_model, train_xgboost
from src.utils.config import load_config, resolve_path, get_project_root
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    processed_dir = resolve_path(cfg, "data", "processed")
    models_dir = resolve_path(cfg, "models")
    preds_dir = resolve_path(cfg, "results", "predictions")
    preds_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 6: Baseline Model Training")
    logger.info("=" * 60)

    # Load splits
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    val_df = pd.read_parquet(processed_dir / "val.parquet")
    test_df = pd.read_parquet(processed_dir / "test.parquet")

    logger.info("Train: %d | Val: %d | Test: %d", len(train_df), len(val_df), len(test_df))

    # Load feature columns
    with open(processed_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)
    feature_cols = [c for c in feature_cols if c in train_df.columns]
    logger.info("Feature columns: %d", len(feature_cols))

    # ----------------------------------------------------------------
    # MULTICLASS BASELINES (BENIGN, FTP-Patator, SSH-Patator)
    # ----------------------------------------------------------------
    logger.info("=== MULTICLASS BASELINES ===")
    label_col_multi = "label_multiclass"

    # Random Forest (Multiclass)
    logger.info("--- Random Forest (Multiclass) ---")
    rf_clf_multi, rf_val_m = train_random_forest(train_df, val_df, feature_cols, label_col_multi, cfg)
    save_rf_model(rf_clf_multi, models_dir / "rf_model_multiclass.pkl")
    rf_preds_multi = rf_clf_multi.predict(test_df[feature_cols].values)

    pd.DataFrame({
        "true_label": test_df[label_col_multi].values,
        "predicted_label": rf_preds_multi,
    }).to_parquet(preds_dir / "rf_multiclass_test_predictions.parquet", index=False)
    # Also save as default rf_test_predictions for backward compatibility
    pd.DataFrame({
        "true_label": test_df[label_col_multi].values,
        "predicted_label": rf_preds_multi,
    }).to_parquet(preds_dir / "rf_test_predictions.parquet", index=False)

    rf_test_f1_multi = f1_score(test_df[label_col_multi].values, rf_preds_multi, average="macro", zero_division=0)
    logger.info("RF Multiclass Test F1 (macro): %.4f", rf_test_f1_multi)

    # XGBoost (Multiclass)
    logger.info("--- XGBoost (Multiclass) ---")
    xgb_clf_multi, xgb_val_m = train_xgboost(train_df, val_df, feature_cols, label_col_multi, cfg)
    save_xgb_model(xgb_clf_multi, models_dir / "xgb_model_multiclass")
    xgb_preds_multi = xgb_clf_multi.predict(test_df[feature_cols].values)

    pd.DataFrame({
        "true_label": test_df[label_col_multi].values,
        "predicted_label": xgb_preds_multi,
    }).to_parquet(preds_dir / "xgb_multiclass_test_predictions.parquet", index=False)
    # Also save as default xgb_test_predictions for backward compatibility
    pd.DataFrame({
        "true_label": test_df[label_col_multi].values,
        "predicted_label": xgb_preds_multi,
    }).to_parquet(preds_dir / "xgb_test_predictions.parquet", index=False)

    xgb_test_f1_multi = f1_score(test_df[label_col_multi].values, xgb_preds_multi, average="macro", zero_division=0)
    logger.info("XGBoost Multiclass Test F1 (macro): %.4f", xgb_test_f1_multi)

    # ----------------------------------------------------------------
    # BINARY BASELINES (BENIGN vs ATTACK)
    # ----------------------------------------------------------------
    logger.info("=== BINARY BASELINES ===")
    label_col_bin = "label_binary"

    # Random Forest (Binary)
    logger.info("--- Random Forest (Binary) ---")
    rf_clf_bin, _ = train_random_forest(train_df, val_df, feature_cols, label_col_bin, cfg)
    save_rf_model(rf_clf_bin, models_dir / "rf_model_binary.pkl")
    rf_preds_bin = rf_clf_bin.predict(test_df[feature_cols].values)

    pd.DataFrame({
        "true_label": test_df[label_col_bin].values,
        "predicted_label": rf_preds_bin,
    }).to_parquet(preds_dir / "rf_binary_test_predictions.parquet", index=False)

    rf_test_f1_bin = f1_score(test_df[label_col_bin].values, rf_preds_bin, average="macro", zero_division=0)
    logger.info("RF Binary Test F1 (macro): %.4f", rf_test_f1_bin)

    # XGBoost (Binary)
    logger.info("--- XGBoost (Binary) ---")
    xgb_clf_bin, _ = train_xgboost(train_df, val_df, feature_cols, label_col_bin, cfg)
    save_xgb_model(xgb_clf_bin, models_dir / "xgb_model_binary")
    xgb_preds_bin = xgb_clf_bin.predict(test_df[feature_cols].values)

    pd.DataFrame({
        "true_label": test_df[label_col_bin].values,
        "predicted_label": xgb_preds_bin,
    }).to_parquet(preds_dir / "xgb_binary_test_predictions.parquet", index=False)

    xgb_test_f1_bin = f1_score(test_df[label_col_bin].values, xgb_preds_bin, average="macro", zero_division=0)
    logger.info("XGBoost Binary Test F1 (macro): %.4f", xgb_test_f1_bin)

    logger.info("=" * 60)
    logger.info("Baseline training complete:")
    logger.info("  Multiclass -> RF Macro F1: %.4f | XGBoost Macro F1: %.4f", rf_test_f1_multi, xgb_test_f1_multi)
    logger.info("  Binary     -> RF Macro F1: %.4f | XGBoost Macro F1: %.4f", rf_test_f1_bin, xgb_test_f1_bin)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
