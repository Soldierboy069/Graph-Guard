"""
GraphGuard: Phase 5 — Graph Neural Network
Module: trainer.py

Training loop for GraphGuard (GraphSAGE + EdgeClassifier).
Handles:
  - Mini-batch edge training via NeighborLoader
  - Class imbalance with weighted cross-entropy loss
  - Early stopping and model checkpointing
  - Training/validation logging
  - Apple Silicon MPS support
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader

from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__)


def get_device(use_mps: bool = True) -> torch.device:
    """
    Select compute device: MPS (Apple Silicon) > CPU.

    Args:
        use_mps: Whether to attempt MPS device.

    Returns:
        torch.device
    """
    if use_mps and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        logger.info("Using Apple Silicon MPS device.")
        return torch.device("mps")
    logger.info("Using CPU device.")
    return torch.device("cpu")


def compute_class_weights(
    labels: np.ndarray, num_classes: int, device: torch.device
) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for weighted cross-entropy loss.

    Args:
        labels: Integer label array.
        num_classes: Total number of classes.
        device: Torch device.

    Returns:
        Tensor of shape [num_classes] with per-class weights.
    """
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    # Smooth to avoid division by zero for missing classes
    counts = np.where(counts == 0, 1.0, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


class EarlyStopping:
    """Early stopping based on validation metric with minimum improvement threshold."""

    def __init__(self, patience: int = 15, min_delta: float = 0.0001, mode: str = "max") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, score: float) -> bool:
        """
        Update state based on new score. Returns True if training should stop.

        Args:
            score: Validation metric value (higher is better if mode='max').

        Returns:
            True if early stopping criterion is met.
        """
        if self.best_score is None:
            self.best_score = score
            return False

        improved = (
            (score > self.best_score + self.min_delta)
            if self.mode == "max"
            else (score < self.best_score - self.min_delta)
        )

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        return False


def train_one_epoch(
    model: nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[float, float]:
    """
    Train the model for one epoch using edge mini-batching.

    Args:
        model: GraphGuardModel.
        data: PyG Data object.
        train_mask: Boolean mask for training edges.
        optimizer: PyTorch optimizer.
        criterion: Loss function.
        device: Compute device.
        batch_size: Number of edges per batch.

    Returns:
        Tuple of (mean_loss, macro_f1) over the epoch.
    """
    model.train()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    y = data.y.to(device)

    train_indices = torch.where(train_mask)[0]
    n_batches = max(1, len(train_indices) // batch_size)
    perm = torch.randperm(len(train_indices))
    train_indices = train_indices[perm]

    all_preds = []
    all_labels = []
    total_loss = 0.0

    for i in range(n_batches):
        batch_idx = train_indices[i * batch_size: (i + 1) * batch_size]
        if len(batch_idx) == 0:
            continue

        optimizer.zero_grad()
        logits = model(x, edge_index, edge_attr)  # [num_edges, num_classes]
        batch_logits = logits[batch_idx]
        batch_y = y[batch_idx]

        loss = criterion(batch_logits, batch_y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = batch_logits.argmax(dim=-1).detach().cpu().numpy()
        labels = batch_y.detach().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    mean_loss = total_loss / n_batches
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return mean_loss, macro_f1


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data: Data,
    mask: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on a subset of edges (val or test).

    Args:
        model: GraphGuardModel.
        data: PyG Data object.
        mask: Boolean edge mask.
        criterion: Loss function.
        device: Compute device.

    Returns:
        Tuple of (loss, macro_f1, predictions, true_labels).
    """
    model.eval()
    x = data.x.to(device)
    edge_index = data.edge_index.to(device)
    edge_attr = data.edge_attr.to(device)
    y = data.y.to(device)

    logits = model(x, edge_index, edge_attr)
    eval_logits = logits[mask]
    eval_y = y[mask]

    loss = criterion(eval_logits, eval_y).item()
    preds = eval_logits.argmax(dim=-1).cpu().numpy()
    labels = eval_y.cpu().numpy()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)

    return loss, macro_f1, preds, labels


@torch.no_grad()
def evaluate_inductive(
    model: nn.Module,
    x_context: torch.Tensor,
    edge_index_context: torch.Tensor,
    query_edge_index: torch.Tensor,
    query_edge_attr: torch.Tensor,
    query_y: torch.Tensor,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Inductive target-edge-excluded evaluation.
    1. Node embeddings are generated by message passing strictly on context edges.
    2. Query edges are classified via [h_u | h_v | edge_attr] without being in the graph.
    """
    model.eval()
    x_c = x_context.to(device)
    ei_c = edge_index_context.to(device)
    q_ei = query_edge_index.to(device)
    q_ea = query_edge_attr.to(device)
    q_y = query_y.to(device)

    # Compute node embeddings from historical context
    if hasattr(model, "encoder") and hasattr(model, "classifier"):
        node_embed = model.encoder(x_c, ei_c)
        logits = model.classifier(node_embed, q_ei, q_ea)
    else:
        logits = model(x_c, q_ei, q_ea)

    loss = criterion(logits, q_y).item()
    preds = logits.argmax(dim=-1).cpu().numpy()
    labels = q_y.cpu().numpy()
    macro_f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return loss, macro_f1, preds, labels


def train_graphguard(
    model: nn.Module,
    data: Data,
    cfg: Dict[str, Any],
    model_save_dir: Path,
    metrics_save_dir: Path,
    device: torch.device,
    experiment_name: str = "full_graphguard",
) -> Dict[str, Any]:
    """
    Full training loop for GraphGuard supporting both Inductive and Transductive modes.
    """
    train_cfg = cfg["model"]["training"]
    epochs = train_cfg["epochs"]
    lr = train_cfg["learning_rate"]
    wd = train_cfg["weight_decay"]
    patience = train_cfg["patience"]
    min_delta = train_cfg["min_delta"]
    batch_size = train_cfg["batch_size"]

    model = model.to(device)
    is_inductive = getattr(data, "inductive_mode", False)

    if is_inductive:
        train_labels = data.y_train.numpy()
        num_classes = int(data.y_train.max().item()) + 1
        x_tr = data.x_train
        ei_tr = data.edge_index_train
        ea_tr = data.edge_attr_train
        y_tr = data.y_train
    else:
        train_labels = data.y[data.train_edge_mask].numpy()
        num_classes = int(data.y.max().item()) + 1

    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )

    early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode="max")

    best_val_f1 = -1.0
    best_epoch = 0
    checkpoint_path = model_save_dir / f"{experiment_name}_best.pt"
    model_save_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": [], "lr": []}

    logger.info(
        "Starting training: %d epochs | device=%s | classes=%d | inductive=%s",
        epochs, device, num_classes, is_inductive
    )

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        if is_inductive:
            model.train()
            train_mask = torch.ones(len(y_tr), dtype=torch.bool)
            train_data = Data(x=x_tr, edge_index=ei_tr, edge_attr=ea_tr, y=y_tr)
            train_loss, train_f1 = train_one_epoch(
                model, train_data, train_mask, optimizer, criterion, device, batch_size
            )
            # Inductive validation: evaluate val query edges against G_train
            val_loss, val_f1, val_preds, val_labels = evaluate_inductive(
                model, x_tr, ei_tr, data.val_edge_index, data.val_edge_attr, data.val_y, criterion, device
            )
        else:
            train_loss, train_f1 = train_one_epoch(
                model, data, data.train_edge_mask, optimizer, criterion, device, batch_size
            )
            val_loss, val_f1, val_preds, val_labels = evaluate(
                model, data, data.val_edge_mask, criterion, device
            )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %03d/%03d | loss=%.4f/%.4f | F1_macro=%.4f/%.4f | lr=%.6f | %.1fs",
            epoch, epochs, train_loss, val_loss, train_f1, val_f1, current_lr, elapsed
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_loss": val_loss,
            }, checkpoint_path)
            logger.info("  ✓ New best val F1=%.4f — checkpoint saved.", val_f1)

        if early_stopper.step(val_f1):
            logger.info("Early stopping triggered at epoch %d.", epoch)
            break

    # Load best model for test evaluation
    logger.info("Loading best checkpoint from epoch %d...", best_epoch)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    if is_inductive:
        # Inductive test: evaluate test query edges against G_hist (test edges strictly excluded)
        test_loss, test_f1, test_preds, test_labels = evaluate_inductive(
            model, data.x_hist, data.edge_index_hist, data.test_edge_index, data.test_edge_attr, data.test_y, criterion, device
        )
    else:
        test_loss, test_f1, test_preds, test_labels = evaluate(
            model, data, data.test_edge_mask, criterion, device
        )

    logger.info("Test results: loss=%.4f | F1_macro=%.4f", test_loss, test_f1)

    metrics_save_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_save_dir / f"{experiment_name}_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch,
        "test_f1": test_f1,
        "test_loss": test_loss,
        "test_predictions": test_preds,
        "test_labels": test_labels,
        "history": history,
        "checkpoint_path": str(checkpoint_path),
    }
