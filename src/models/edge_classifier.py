"""
GraphGuard: Phase 5 — Graph Neural Network
Module: edge_classifier.py

MLP-based edge classifier that concatenates:
  [source_node_embedding | destination_node_embedding | edge_traffic_features]
→ Predicts attack class per communication flow.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EdgeClassifier(nn.Module):
    """
    Edge-level MLP classifier for flow/intrusion detection.

    For each directed edge (src→dst):
        input = concat(node_embed[src], node_embed[dst], edge_features)
        → Linear → BN → ReLU → Dropout → ... → Linear → logits

    This ensures graph-structural context (node embeddings) is combined
    with raw traffic features for every flow classification decision.
    """

    def __init__(
        self,
        node_embed_dim: int,
        edge_feat_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.3,
    ) -> None:
        """
        Args:
            node_embed_dim: Dimension of node embeddings from GraphSAGE.
            edge_feat_dim: Number of edge (traffic flow) features.
            num_classes: Number of output classes.
            hidden_dims: List of hidden layer sizes. Default: [256, 128].
            dropout: Dropout rate.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        # Input = src_embed + dst_embed + edge_features
        in_dim = 2 * node_embed_dim + edge_feat_dim

        layers = []
        prev_dim = in_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ])
            prev_dim = h_dim

        layers.append(nn.Linear(prev_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

        logger.info(
            "EdgeClassifier: in=%d → %s → %d classes (dropout=%.2f)",
            in_dim, hidden_dims, num_classes, dropout
        )

    def forward(
        self,
        node_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Classify each edge based on node embeddings + edge features.

        Args:
            node_embeddings: [num_nodes, embed_dim] node embedding matrix.
            edge_index: [2, num_edges] source/dest node indices.
            edge_attr: [num_edges, edge_feat_dim] edge feature matrix.

        Returns:
            logits: [num_edges, num_classes] raw class scores.
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        src_embed = node_embeddings[src_idx]   # [num_edges, embed_dim]
        dst_embed = node_embeddings[dst_idx]   # [num_edges, embed_dim]

        # Concatenate: graph context + traffic features
        x = torch.cat([src_embed, dst_embed, edge_attr], dim=-1)
        logits = self.mlp(x)
        return logits

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        for layer in self.mlp:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


class GraphGuardModel(nn.Module):
    """
    Full GraphGuard model: GraphSAGE encoder + EdgeClassifier.

    Wraps both components for convenient single-call forward pass.
    """

    def __init__(
        self,
        graphsage_encoder: nn.Module,
        edge_classifier: nn.Module,
    ) -> None:
        super().__init__()
        self.encoder = graphsage_encoder
        self.classifier = edge_classifier

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass.

        Args:
            x: Node features [num_nodes, node_feat_dim].
            edge_index: [2, num_edges].
            edge_attr: [num_edges, edge_feat_dim].

        Returns:
            logits: [num_edges, num_classes].
        """
        node_embeddings = self.encoder(x, edge_index)
        logits = self.classifier(node_embeddings, edge_index, edge_attr)
        return logits

    def get_node_embeddings(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return node embeddings without classification (for visualization/TPS)."""
        with torch.no_grad():
            return self.encoder(x, edge_index)
