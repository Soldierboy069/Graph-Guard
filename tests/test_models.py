"""
GraphGuard: Unit tests for Phase 5 (GNN Models) and Phase 9 (Threat Propagation)
"""
import pytest
import torch
import numpy as np
import pandas as pd

from src.models.graphsage import GraphSAGEEncoder
from src.models.edge_classifier import EdgeClassifier, GraphGuardModel
from src.threat.tps import compute_tps, get_threat_level


def test_graphsage_encoder_forward():
    num_nodes = 20
    in_dim = 10
    hidden_dim = 32
    embed_dim = 16

    x = torch.randn(num_nodes, in_dim)
    edge_index = torch.randint(0, num_nodes, (2, 50))

    encoder = GraphSAGEEncoder(
        in_channels=in_dim,
        hidden_dim=hidden_dim,
        embedding_dim=embed_dim,
        num_layers=2,
    )
    embeddings = encoder(x, edge_index)
    assert embeddings.shape == (num_nodes, embed_dim)
    # Check L2 normalization (norm should be ~1.0)
    norms = torch.norm(embeddings, p=2, dim=-1)
    np.testing.assert_allclose(norms.detach().numpy(), 1.0, atol=1e-5)


def test_edge_classifier_forward():
    num_nodes = 20
    num_edges = 30
    embed_dim = 16
    edge_feat_dim = 25
    num_classes = 3

    node_embeddings = torch.randn(num_nodes, embed_dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attr = torch.randn(num_edges, edge_feat_dim)

    classifier = EdgeClassifier(
        node_embed_dim=embed_dim,
        edge_feat_dim=edge_feat_dim,
        num_classes=num_classes,
        hidden_dims=[32, 16],
    )
    logits = classifier(node_embeddings, edge_index, edge_attr)
    assert logits.shape == (num_edges, num_classes)


def test_tps_computation():
    flow_meta = pd.DataFrame({
        "Source IP": ["10.0.0.1", "10.0.0.2", "10.0.0.1"],
        "Destination IP": ["10.0.0.2", "10.0.0.3", "10.0.0.3"],
        "Label": ["BENIGN", "BENIGN", "FTP-Patator"],
    })
    predictions = np.array([0, 0, 1])
    ip_to_node = {"10.0.0.1": 0, "10.0.0.2": 1, "10.0.0.3": 2}
    label_encoder = {"BENIGN": 0, "FTP-Patator": 1}

    tps_df = compute_tps(flow_meta, predictions, ip_to_node, label_encoder)
    assert len(tps_df) == 3
    assert "tps" in tps_df.columns
    assert "threat_level" in tps_df.columns
    assert (tps_df["tps"] >= 0.0).all() and (tps_df["tps"] <= 1.0).all()


def test_threat_levels():
    assert get_threat_level(0.85) == "CRITICAL"
    assert get_threat_level(0.60) == "HIGH"
    assert get_threat_level(0.35) == "MEDIUM"
    assert get_threat_level(0.10) == "LOW"
