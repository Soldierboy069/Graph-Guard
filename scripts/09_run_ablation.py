#!/usr/bin/env python3
"""
GraphGuard — Script 09: Run Ablation Study
==========================================
Trains 4 ablation variants using the same split as the full model.
All results come from actual training runs — nothing is fabricated.

Variants:
    1. full_graphguard        — Full model (AGC + graph context + adaptive weights)
    2. no_agc                 — Static graph, no adaptive weighting
    3. no_graph_context       — Edge features only (no node embeddings)
    4. no_adaptive_weights    — No EMA weight updates in AGC

Output:
    results/metrics/ablation.csv
    results/figures/ablation_comparison.png
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.metrics import compute_metrics
from src.preprocessing.encoder import load_label_encoder
from src.utils.config import load_config, resolve_path
from src.utils.logger import setup_logger

logger = setup_logger(__name__, log_dir="results/logs")

import importlib.util
spec = importlib.util.spec_from_file_location("script_04", str(PROJECT_ROOT / "scripts" / "04_train_graphguard.py"))
script_04 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script_04)
run_train = script_04.run_train


def run_experiment(experiment_name: str, no_graph_context: bool = False) -> bool:
    """Run a single GraphGuard training experiment in-process."""
    try:
        logger.info("Running experiment: %s", experiment_name)
        run_train(
            experiment_name=experiment_name,
            no_graph_context=no_graph_context,
        )
        return True
    except Exception as e:
        logger.error("Experiment %s failed: %s", experiment_name, e)
        return False


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="GraphGuard Ablation Study")
    parser.add_argument("--force", action="store_true", help="Force retrain all ablation variants")
    args = parser.parse_args()

    cfg = load_config()
    preds_dir = resolve_path(cfg, "results", "predictions")
    metrics_dir = resolve_path(cfg, "results", "metrics")
    figures_dir = resolve_path(cfg, "results", "figures")
    processed_dir = resolve_path(cfg, "data", "processed")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 11: Ablation Study")
    logger.info("=" * 60)

    label_encoder = load_label_encoder(processed_dir / "label_encoder.json")

    ablation_experiments = cfg["ablation"]["experiments"]
    results = []

    for exp in ablation_experiments:
        name = exp["name"]
        logger.info("Ablation: %s — %s", name, exp["description"])

        pred_path = preds_dir / f"{name}_test_predictions.parquet"
        if args.force or not pred_path.exists():
            no_graph = not exp.get("use_graph_context", True)
            success = run_experiment(name, no_graph_context=no_graph)
        else:
            logger.info("Using existing predictions for %s", name)
            success = True

        if success and pred_path.exists():
            pred_df = pd.read_parquet(pred_path)
            y_true = pred_df["true_label"].values
            y_pred = pred_df["predicted_label"].values
            m = compute_metrics(y_true, y_pred, label_encoder, name)
            row = {
                "experiment": name,
                "description": exp["description"],
                "accuracy": m["accuracy"],
                "f1_macro": m["f1_macro"],
                "f1_weighted": m["f1_weighted"],
                "precision_macro": m["precision_macro"],
                "recall_macro": m["recall_macro"],
                "fpr_macro": m["fpr_macro"],
            }
            results.append(row)
        else:
            results.append({
                "experiment": name,
                "description": exp["description"],
                "accuracy": None,
                "f1_macro": None,
                "f1_weighted": None,
                "precision_macro": None,
                "recall_macro": None,
                "fpr_macro": None,
            })

    # Save ablation table
    ablation_df = pd.DataFrame(results)
    ablation_df.to_csv(metrics_dir / "ablation.csv", index=False)
    logger.info("Ablation results:\n%s", ablation_df.to_string(index=False))

    # Plot comparison
    valid = ablation_df.dropna(subset=["f1_macro"])
    if len(valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(valid))
        colors = ["#2166ac", "#4393c3", "#92c5de", "#f4a582", "#d6604d", "#7fbc41"][:len(valid)]
        bars = ax.bar(x, valid["f1_macro"], color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(valid["experiment"], rotation=20, ha="right", fontsize=9)
        ax.set_ylabel("F1 Score (Macro)")
        ax.set_title("GraphGuard Ablation Study — F1 Score Comparison")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, valid["f1_macro"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(figures_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    logger.info("=" * 60)
    logger.info("Ablation study complete.")
    logger.info("Results: %s", metrics_dir / "ablation.csv")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
