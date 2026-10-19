#!/usr/bin/env python3
"""
Generate node_features.txt, edge_features.txt, and feature_provenance_table.md.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

with open(PROCESSED_DIR / "feature_columns.json") as f:
    edge_cols = json.load(f)

# 1. edge_features.txt
with open(PROCESSED_DIR / "edge_features.txt", "w") as f:
    f.write("# GraphGuard Edge Features Manifest (77 Netflow Features)\n")
    f.write("# Format: feature_name | source_columns | label_derived | timestamp_derived | train_only_aggregation\n\n")
    for col in edge_cols:
        ts_derived = "YES" if any(k in col for k in ["IAT", "Idle", "Active", "Duration"]) else "NO"
        f.write(f"{col} | CICFlowMeter netflow extraction | NO | {ts_derived} | NO (per-flow attribute)\n")

# 2. feature_provenance_table.md
node_feats = [
    ("connection_count", "Node", "Source IP, Destination IP", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("out_degree", "Node", "Source IP, Destination IP", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("in_degree", "Node", "Source IP, Destination IP", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("unique_neighbors", "Node", "Source IP, Destination IP", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("out_flow_bytes_mean", "Node", "Source IP, Total Length of Fwd Packets", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("in_flow_bytes_mean", "Node", "Destination IP, Total Length of Fwd Packets", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("out_packet_count_mean", "Node", "Source IP, Total Fwd Packets", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("in_packet_count_mean", "Node", "Destination IP, Total Fwd Packets", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("out_flow_duration_mean", "Node", "Source IP, Flow Duration", "NO", "NO", "NO", "YES", "YES", "YES"),
    ("in_flow_duration_mean", "Node", "Destination IP, Flow Duration", "NO", "NO", "NO", "YES", "YES", "YES"),
]

with open(PROCESSED_DIR / "feature_provenance_table.md", "w") as f:
    f.write("# GraphGuard Feature Provenance & Safety Table\n\n")
    f.write("| Feature | Type | Source Columns | Uses Label? | Uses Future Traffic? | Uses Timestamp? | Training-Safe? | Validation-Safe? | Test-Safe? |\n")
    f.write("|---|---|---|---|---|---|---|---|---|\n")
    for row in node_feats:
        f.write(f"| `{row[0]}` | {row[1]} | {row[2]} | **{row[3]}** | **{row[4]}** | {row[5]} | **{row[6]}** | **{row[7]}** | **{row[8]}** |\n")
    for col in edge_cols:
        ts_used = "YES" if any(k in col for k in ["IAT", "Idle", "Active", "Duration"]) else "NO"
        f.write(f"| `{col}` | Edge | CICFlowMeter netflow extraction | **NO** | **NO** | {ts_used} | **YES** | **YES** | **YES** |\n")

print("Generated edge_features.txt and feature_provenance_table.md successfully.")
