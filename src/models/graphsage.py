"""
GraphGuard: Phase 5 — Graph Neural Network
Module: graphsage.py

GraphSAGE node encoder using PyTorch Geometric SAGEConv layers.
Produces node embeddings that are subsequently used for edge classification.
"""
import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE node encoder.

    Architecture:
        Input node features
            → SAGEConv(in_dim → hidden_dim) + BatchNorm + ReLU + Dropout
            → SAGEConv(hidden_dim → embedding_dim) + BatchNorm + (optional L2 norm)

    The produced node embeddings are used by the EdgeClassifier downstream.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        embedding_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        normalize: bool = True,
        aggregator: str = "mean",
    ) -> None:
        """
        Args:
            in_channels: Number of input node feature dimensions.
            hidden_dim: Hidden layer dimension.
            embedding_dim: Output embedding dimension.
            num_layers: Number of SAGEConv layers (≥ 2).
            dropout: Dropout probability.
            normalize: If True, L2-normalize output embeddings.
            aggregator: Aggregation method for SAGEConv ('mean', 'max', 'lstm').
        """
        super().__init__()
        self.normalize = normalize
        self.dropout = dropout

        # Build conv layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        dims = [in_channels] + [hidden_dim] * (num_layers - 1) + [embedding_dim]
        for i in range(num_layers):
            self.convs.append(SAGEConv(dims[i], dims[i + 1], aggr=aggregator))
            self.bns.append(nn.BatchNorm1d(dims[i + 1]))

        logger.info(
            "GraphSAGE encoder: in=%d → %s → %d (layers=%d, dropout=%.2f)",
            in_channels, hidden_dim, embedding_dim, num_layers, dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAGE encoder.

        Args:
            x: Node feature matrix [num_nodes, in_channels].
            edge_index: Graph connectivity [2, num_edges].

        Returns:
            Node embedding matrix [num_nodes, embedding_dim].
        """
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)

        if self.normalize:
            x = F.normalize(x, p=2, dim=-1)

        return x

    def reset_parameters(self) -> None:
        """Re-initialize all learnable parameters."""
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
