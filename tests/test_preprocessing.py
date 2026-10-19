"""
GraphGuard: Unit tests for Phase 2 (Preprocessing)
"""
import pytest
import numpy as np
import pandas as pd

from src.preprocessing.cleaner import (
    remove_inf_values,
    drop_missing_label,
    fill_numeric_nan,
    drop_duplicates,
    clean_chunk,
)
from src.preprocessing.encoder import (
    normalize_labels,
    build_label_encoder,
    encode_labels,
)
from src.preprocessing.scaler import (
    fit_scaler,
    apply_scaler,
    get_feature_columns,
)
from src.preprocessing.splitter import time_aware_split


def test_remove_inf_values():
    df = pd.DataFrame({
        "a": [1.0, np.inf, 3.0, -np.inf],
        "b": ["x", "y", "z", "w"],
    })
    cleaned, count = remove_inf_values(df)
    assert count == 2
    assert not np.isinf(cleaned["a"]).any()


def test_drop_missing_label():
    df = pd.DataFrame({
        "feature": [1, 2, 3, 4],
        "Label": ["BENIGN", np.nan, "", "DDoS"],
    })
    cleaned, dropped = drop_missing_label(df, "Label")
    assert dropped == 2
    assert len(cleaned) == 2
    assert list(cleaned["Label"]) == ["BENIGN", "DDoS"]


def test_clean_chunk():
    df = pd.DataFrame({
        "Flow Duration": [100, -5, np.inf, 200],
        "Total Fwd Packets": [10, 5, 2, 3],
        "Label": ["BENIGN", "BENIGN", "FTP-Patator", "BENIGN"],
    })
    cleaned, stats = clean_chunk(df)
    assert stats["inf_replaced"] == 1
    assert stats["invalid_dropped"] == 1
    assert len(cleaned) == 3


def test_label_normalization_and_encoding():
    raw_labels = pd.Series(["BENIGN", "FTP-Patator", "Web Attack \x96 Brute Force", "BENIGN"])
    norm = normalize_labels(raw_labels)
    assert norm.iloc[2] == "Web Attack – Brute Force"

    encoder = build_label_encoder(sorted(norm.unique()))
    assert encoder["BENIGN"] == 0
    assert len(encoder) == 3

    encoded = encode_labels(norm, encoder, binary=False)
    assert encoded.iloc[0] == 0
    assert encoded.iloc[1] > 0

    binary_encoded = encode_labels(norm, encoder, binary=True)
    assert list(binary_encoded) == [0, 1, 1, 0]


def test_scaler_fit_apply():
    df_train = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "f2": [10.0, 20.0, 30.0, 40.0, 50.0],
        "Label": ["BENIGN"] * 5,
    })
    feat_cols = ["f1", "f2"]
    scaler = fit_scaler(df_train, feat_cols, "standard")
    scaled_train = apply_scaler(df_train, scaler, feat_cols)

    np.testing.assert_allclose(scaled_train[feat_cols].mean(), [0.0, 0.0], atol=1e-6)


def test_time_aware_split():
    dates = pd.date_range("2017-07-04 09:00", periods=10, freq="min")
    df = pd.DataFrame({
        "Timestamp": dates,
        "feat": range(10),
        "Label": ["BENIGN"] * 10,
    })
    train, val, test = time_aware_split(df, 0.6, 0.2, 0.2, "Timestamp")
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    assert train["Timestamp"].max() <= val["Timestamp"].min()
    assert val["Timestamp"].max() <= test["Timestamp"].min()
