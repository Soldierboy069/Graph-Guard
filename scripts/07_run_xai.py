#!/usr/bin/env python3
"""
GraphGuard — Script 07: Run Explainability (XAI)
================================================
Generates SHAP-based explanations for suspicious test flows.

Output:
    results/explanations/top_{n}_flow_explanations.json
    results/explanations/flow_{i}_shap.png
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.explainability.explainer import GraphGuardExplainer
from src.preprocessing.encoder import load_label_encoder
from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def main() -> None:
    cfg = load_config()
    set_seed(cfg["project"]["seed"])

    processed_dir = resolve_path(cfg, "data", "processed")
    preds_dir = resolve_path(cfg, "results", "predictions")
    expl_dir = resolve_path(cfg, "results", "explanations")
    expl_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 8: Explainable AI")
    logger.info("=" * 60)

    label_encoder = load_label_encoder(processed_dir / "label_encoder.json")

    with open(processed_dir / "feature_columns.json") as f:
        feature_cols = json.load(f)

    # Load test data + predictions
    test_df = pd.read_parquet(processed_dir / "test.parquet")
    pred_path = preds_dir / "full_graphguard_test_predictions.parquet"

    if not pred_path.exists():
        logger.error("GraphGuard predictions not found. Run 04_train_graphguard.py first.")
        sys.exit(1)

    pred_df = pd.read_parquet(pred_path)
    y_pred = pred_df["predicted_label"].values
    y_true = pred_df["true_label"].values

    feature_cols = [c for c in feature_cols if c in test_df.columns]
    X_test = test_df[feature_cols].values

    xai_cfg = cfg["explainability"]
    top_k = xai_cfg["top_k_features"]
    n_explain = xai_cfg["auto_explain_top_n"]
    bg_samples = xai_cfg["shap_background_samples"]

    # Focus on attack-predicted flows
    attack_mask = y_pred != label_encoder.get("BENIGN", 0)
    attack_idx = np.where(attack_mask)[0]
    logger.info("Found %d attack-predicted flows in test set for explanation", len(attack_idx))

    source_df = pred_df
    X_explain = X_test
    y_explain_pred = y_pred
    y_explain_true = y_true

    if len(attack_idx) == 0:
        logger.info("Test split is benign-only in this partition. Checking val/train for attack flows...")
        val_df = pd.read_parquet(processed_dir / "val.parquet")
        val_attacks = np.where(val_df["label_multiclass"] != label_encoder.get("BENIGN", 0))[0]
        if len(val_attacks) > 0:
            source_df = val_df
            X_explain = val_df[feature_cols].values
            y_explain_pred = val_df["label_multiclass"].values
            y_explain_true = val_df["label_multiclass"].values
            attack_idx = val_attacks
            logger.info("Found %d attack flows in validation split to explain", len(attack_idx))
        else:
            # Pick first N flows to explain
            attack_idx = np.arange(min(n_explain, len(X_test)))

    # Fit explainer on training data
    train_df = pd.read_parquet(processed_dir / "train.parquet")
    train_feature_cols = [c for c in feature_cols if c in train_df.columns]
    X_train = train_df[train_feature_cols].values
    y_train_pred = train_df["label_multiclass"].values if "label_multiclass" in train_df.columns else np.zeros(len(train_df), dtype=int)

    explainer = GraphGuardExplainer(
        feature_names=feature_cols,
        label_encoder=label_encoder,
        background_samples=bg_samples,
    )
    explainer.fit(X_train, y_train_pred, y_train_pred)

    # Explain top-n attack flows
    explain_idx = attack_idx[:n_explain]
    all_explanations = []

    for i, idx in enumerate(explain_idx):
        explanation = explainer.explain_flow(X_explain[idx], int(y_explain_pred[idx]), top_k=top_k)
        explanation["flow_index"] = int(idx)
        explanation["true_label"] = int(y_explain_true[idx])

        # Add metadata if available
        if len(source_df.columns) > 2:
            meta_cols = ["Flow ID", "Source IP", "Destination IP", "Protocol"]
            for col in meta_cols:
                if col in source_df.columns:
                    explanation[col] = str(source_df.iloc[idx].get(col, ""))

        all_explanations.append(explanation)

        # Save individual plot
        flow_id = explanation.get("Flow ID", f"flow_{idx}")
        explainer.plot_explanation(
            explanation,
            expl_dir / f"flow_{i}_shap.png",
            flow_id=str(flow_id),
        )
        explainer.save_explanation(explanation, expl_dir / f"flow_{i}_explanation.json")

    # Save all explanations
    with open(expl_dir / f"top_{n_explain}_flow_explanations.json", "w") as f:
        json.dump(all_explanations, f, indent=2)

    logger.info("=" * 60)
    logger.info("XAI complete. Explained %d flows.", len(all_explanations))
    logger.info("Outputs: %s", expl_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
