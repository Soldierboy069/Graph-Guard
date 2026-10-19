"""
GraphGuard: Phase 2 — Data Preprocessing
Module: encoder.py

Label normalization and encoding for CICIDS2017 dataset.
Supports both binary (BENIGN vs ATTACK) and multiclass classification.
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Known label normalization aliases (handle whitespace/dash variants from dataset)
LABEL_NORMALIZATION: Dict[str, str] = {
    "benign": "BENIGN",
    "ftp-patator": "FTP-Patator",
    "ssh-patator": "SSH-Patator",
    "dos hulk": "DoS Hulk",
    "dos goldeneye": "DoS GoldenEye",
    "dos slowloris": "DoS slowloris",
    "dos slowhttptest": "DoS Slowhttptest",
    "heartbleed": "Heartbleed",
    "web attack \x96 brute force": "Web Attack – Brute Force",
    "web attack – brute force": "Web Attack – Brute Force",
    "web attack - brute force": "Web Attack – Brute Force",
    "web attack \x96 xss": "Web Attack – XSS",
    "web attack – xss": "Web Attack – XSS",
    "web attack - xss": "Web Attack – XSS",
    "web attack \x96 sql injection": "Web Attack – Sql Injection",
    "web attack – sql injection": "Web Attack – Sql Injection",
    "web attack - sql injection": "Web Attack – Sql Injection",
    "infiltration": "Infiltration",
    "bot": "Bot",
    "ddos": "DDoS",
    "portscan": "PortScan",
}


def normalize_labels(series: pd.Series) -> pd.Series:
    """
    Normalize raw label strings from the dataset to canonical forms.

    Handles:
    - Leading/trailing whitespace
    - Case variations
    - Windows-1252 special dash characters (\\x96)
    - Unicode dash variants

    Args:
        series: Raw label column as a pandas Series.

    Returns:
        Normalized label Series.
    """
    normalized = (
        series.astype(str)
        .str.strip()
        # Replace Windows-1252 em-dash (0x96) with standard –
        .str.replace("\x96", "–", regex=False)
        # Map to canonical form
        .map(lambda x: LABEL_NORMALIZATION.get(x.lower(), x))
    )
    return normalized


def get_unique_labels(series: pd.Series) -> List[str]:
    """
    Return sorted list of unique label values.

    Args:
        series: Normalized label Series.

    Returns:
        Sorted list of unique label strings.
    """
    return sorted(series.dropna().unique().tolist())


def build_label_encoder(labels: List[str]) -> Dict[str, int]:
    """
    Build a deterministic label-to-integer mapping.
    BENIGN is always index 0; attack classes follow in sorted order.

    Args:
        labels: List of unique label strings.

    Returns:
        Dict mapping label string → integer index.
    """
    attack_labels = sorted([l for l in labels if l != "BENIGN"])
    encoder = {"BENIGN": 0}
    for i, label in enumerate(attack_labels, start=1):
        encoder[label] = i
    logger.info("Label encoder built: %d classes — %s", len(encoder), list(encoder.keys()))
    return encoder


def build_binary_encoder() -> Dict[str, int]:
    """Return binary label encoder: BENIGN=0, all other=1."""
    return {"BENIGN": 0, "__ATTACK__": 1}


def encode_labels(
    series: pd.Series,
    encoder: Dict[str, int],
    binary: bool = False,
) -> pd.Series:
    """
    Encode normalized label strings to integers.

    Args:
        series: Normalized label Series.
        encoder: Label-to-int mapping from build_label_encoder().
        binary: If True, encode as BENIGN=0, ATTACK=1.

    Returns:
        Integer-encoded label Series. Unknown labels get -1.
    """
    if binary:
        return series.map(lambda x: 0 if x == "BENIGN" else 1)
    else:
        return series.map(lambda x: encoder.get(x, -1))


def save_label_encoder(encoder: Dict[str, int], path: Path) -> None:
    """Save label encoder to JSON for later use in dashboard/inference."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(encoder, f, indent=2)
    logger.info("Label encoder saved to %s", path)


def load_label_encoder(path: Path) -> Dict[str, int]:
    """Load label encoder from JSON."""
    with open(path, "r") as f:
        encoder = json.load(f)
    logger.info("Label encoder loaded from %s (%d classes)", path, len(encoder))
    return encoder


def get_class_distribution(series: pd.Series) -> pd.Series:
    """
    Return class distribution as a count Series sorted by frequency descending.

    Args:
        series: Label Series (string or integer).

    Returns:
        Series of label counts.
    """
    return series.value_counts().sort_values(ascending=False)
