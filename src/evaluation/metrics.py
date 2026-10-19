"""
GraphGuard: Phase 7 — Evaluation
Module: metrics.py

Comprehensive evaluation metrics for binary and multiclass classification.
All metrics computed from actual predictions — never hardcoded.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_encoder: Dict[str, int],
    model_name: str = "model",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Args:
        y_true: Ground-truth integer labels.
        y_pred: Predicted integer labels.
        label_encoder: String→int label mapping (to recover class names).
        model_name: Name for logging.

    Returns:
        Dict with accuracy, precision, recall, F1, macro_F1, weighted_F1, FPR, confusion_matrix.
    """
    int_to_label = {v: k for k, v in label_encoder.items()}
    classes = sorted(set(y_true) | set(y_pred))
    class_names = [int_to_label.get(c, str(c)) for c in classes]

    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # False Positive Rate (macro average over classes)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    fpr_per_class = []
    for i in range(len(classes)):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denom = fp + tn
        fpr_per_class.append(fp / denom if denom > 0 else 0.0)
    fpr_macro = float(np.mean(fpr_per_class))

    # Per-class report
    report = classification_report(
        y_true, y_pred, labels=classes, target_names=class_names,
        output_dict=True, zero_division=0
    )

    metrics = {
        "model": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision_macro), 4),
        "recall_macro": round(float(recall_macro), 4),
        "f1_macro": round(float(f1_macro), 4),
        "precision_weighted": round(float(precision_weighted), 4),
        "recall_weighted": round(float(recall_weighted), 4),
        "f1_weighted": round(float(f1_weighted), 4),
        "fpr_macro": round(float(fpr_macro), 4),
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "per_class_report": report,
        "n_samples": len(y_true),
    }

    logger.info("--- %s ---", model_name)
    logger.info("  Accuracy:         %.4f", accuracy)
    logger.info("  F1 (macro):       %.4f", f1_macro)
    logger.info("  F1 (weighted):    %.4f", f1_weighted)
    logger.info("  Precision (mac):  %.4f", precision_macro)
    logger.info("  Recall (macro):   %.4f", recall_macro)
    logger.info("  FPR (macro):      %.4f", fpr_macro)

    return metrics


def save_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """Save metrics dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", path)


def plot_confusion_matrix(
    metrics: Dict[str, Any],
    save_path: Path,
    figsize: tuple = (10, 8),
) -> None:
    """
    Plot and save confusion matrix as PNG.

    Args:
        metrics: Metrics dict containing 'confusion_matrix' and 'class_names'.
        save_path: Path to save PNG.
        figsize: Figure size tuple.
    """
    cm = np.array(metrics["confusion_matrix"])
    class_names = metrics["class_names"]
    model_name = metrics.get("model", "Model")

    # Normalize per row (true class)
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = np.where(row_sums > 0, cm_norm / row_sums, 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Normalized proportion")

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(class_names, fontsize=9)

    # Annotate cells
    thresh = cm_norm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(
                j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center", va="center", fontsize=7,
                color="white" if cm_norm[i, j] > thresh else "black",
            )

    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix", fontsize=13)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


def plot_training_curves(
    history: Dict[str, list],
    save_path: Path,
    experiment_name: str = "GraphGuard",
) -> None:
    """Plot training/validation loss and F1 curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], label="Train Loss")
    ax1.plot(epochs, history["val_loss"], label="Val Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{experiment_name} — Training/Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_f1"], label="Train F1")
    ax2.plot(epochs, history["val_f1"], label="Val F1")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 (Macro)")
    ax2.set_title(f"{experiment_name} — Training/Validation F1")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved to %s", save_path)


def compare_models(
    metrics_list: List[Dict[str, Any]],
    save_path: Path,
) -> pd.DataFrame:
    """
    Create a side-by-side comparison table of all models.

    Args:
        metrics_list: List of metrics dicts (one per model).
        save_path: Path to save comparison CSV.

    Returns:
        Comparison DataFrame.
    """
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Accuracy": m["accuracy"],
            "Precision (Macro)": m["precision_macro"],
            "Recall (Macro)": m["recall_macro"],
            "F1 (Macro)": m["f1_macro"],
            "F1 (Weighted)": m["f1_weighted"],
            "FPR (Macro)": m["fpr_macro"],
            "Samples": m["n_samples"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    logger.info("Model comparison saved to %s", save_path)
    logger.info("\n%s", df.to_string(index=False))
    return df
