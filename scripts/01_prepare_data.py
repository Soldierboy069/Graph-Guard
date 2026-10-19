#!/usr/bin/env python3
"""
GraphGuard — Script 01: Data Preparation
=========================================
Reads CICIDS2017 GeneratedLabelledFlows dataset from ZIP in memory-safe chunks.
Applies cleaning, label normalization, feature scaling, and time-aware split.
Saves processed Parquet files for downstream graph construction.

Usage:
    python scripts/01_prepare_data.py                  # full dataset
    python scripts/01_prepare_data.py --sample         # Tuesday only (100k rows)
    python scripts/01_prepare_data.py --config path/to/config.yaml

Output files (data/processed/):
    train.parquet
    val.parquet
    test.parquet
    label_encoder.json
    scaler.pkl
    preprocessing_stats.json
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.preprocessing.cleaner import clean_chunk
from src.preprocessing.csv_loader import load_all_chunks
from src.preprocessing.encoder import (
    build_label_encoder,
    encode_labels,
    get_class_distribution,
    get_unique_labels,
    normalize_labels,
    save_label_encoder,
)
from src.preprocessing.scaler import (
    apply_scaler,
    fit_scaler,
    get_feature_columns,
    save_scaler,
)
from src.preprocessing.splitter import (
    audit_and_deduplicate_splits,
    campaign_window_temporal_split,
    parse_timestamps,
    report_split_distribution,
    time_aware_split,
)
from src.utils.config import load_config, resolve_path, get_project_root
from src.utils.logger import setup_logger
from src.utils.seeds import set_seed

logger = setup_logger(__name__, log_dir="results/logs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GraphGuard: Data Preparation Pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample mode: process only the sample_file specified in config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])

    root = get_project_root(cfg)
    processed_dir = resolve_path(cfg, "data", "processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    zip_path = root / cfg["paths"]["datasets"]["generated_labelled_flows_zip"]
    subfolder = cfg["paths"]["datasets"]["generated_labelled_flows_dir"]
    chunk_size = cfg["preprocessing"]["chunk_size"]
    label_col = cfg["preprocessing"]["label_column"]
    ts_col = cfg["preprocessing"]["timestamp_column"]
    ts_fmt = cfg["preprocessing"]["timestamp_format"]
    fill_nan = cfg["preprocessing"]["fill_nan"]
    scaler_type = cfg["preprocessing"]["scaler"]

    metadata_cols = cfg["preprocessing"]["metadata_columns"]
    split_cfg = cfg["preprocessing"]["split"]

    sample_file = cfg["sample_file"] if args.sample else None
    max_rows = cfg["sample_max_rows"] if args.sample else None

    logger.info("=" * 60)
    logger.info("GraphGuard — Phase 2: Data Preparation")
    logger.info("Mode: %s", "SAMPLE" if args.sample else "FULL DATASET")
    logger.info("Source ZIP: %s", zip_path)
    logger.info("=" * 60)

    # ----------------------------------------------------------------
    # PASS 1: Stream all chunks to collect data and discover labels
    # ----------------------------------------------------------------
    all_chunks = []
    cleaning_totals = defaultdict(int)
    total_rows = 0

    logger.info("Pass 1: Reading and cleaning chunks...")
    for filename, chunk in load_all_chunks(
        zip_path=zip_path,
        subfolder=subfolder,
        chunk_size=chunk_size,
        encoding=cfg["preprocessing"]["encoding"],
        encoding_errors=cfg["preprocessing"]["encoding_errors"],
        sample_file=sample_file,
        max_rows=max_rows,
    ):
        chunk, stats = clean_chunk(chunk, label_col=label_col, fill_nan_value=fill_nan)
        if len(chunk) == 0:
            continue

        # Normalize labels immediately
        chunk[label_col] = normalize_labels(chunk[label_col])

        # Parse timestamps
        chunk = parse_timestamps(chunk, ts_col, ts_fmt)

        for k, v in stats.items():
            cleaning_totals[k] += v
        total_rows += len(chunk)
        all_chunks.append(chunk)

    if not all_chunks:
        logger.error("No data loaded. Check dataset path and ZIP contents.")
        sys.exit(1)

    logger.info("Total rows after cleaning: %d", total_rows)
    logger.info("Cleaning stats: %s", dict(cleaning_totals))

    # Concatenate all chunks
    logger.info("Concatenating all chunks...")
    df = pd.concat(all_chunks, ignore_index=True)
    del all_chunks

    # ----------------------------------------------------------------
    # Label discovery and encoding
    # ----------------------------------------------------------------
    logger.info("Discovering labels...")
    unique_labels = get_unique_labels(df[label_col])
    logger.info("Unique labels found (%d): %s", len(unique_labels), unique_labels)

    label_dist = get_class_distribution(df[label_col])
    logger.info("Class distribution:\n%s", label_dist.to_string())

    multiclass_encoder = build_label_encoder(unique_labels)
    save_label_encoder(multiclass_encoder, processed_dir / "label_encoder.json")

    # Add integer label columns
    df["label_multiclass"] = encode_labels(df[label_col], multiclass_encoder, binary=False)
    df["label_binary"] = encode_labels(df[label_col], multiclass_encoder, binary=True)

    # Drop rows where label could not be encoded (label=-1)
    invalid_label_mask = df["label_multiclass"] == -1
    if invalid_label_mask.sum() > 0:
        logger.warning("Dropping %d rows with unknown labels.", invalid_label_mask.sum())
        df = df[~invalid_label_mask].reset_index(drop=True)

    # ----------------------------------------------------------------
    # Campaign-window chronological split
    # ----------------------------------------------------------------
    logger.info("Applying campaign-window chronological split (train=%.0f%% val=%.0f%% test=%.0f%%)...",
                split_cfg["train"] * 100, split_cfg["val"] * 100, split_cfg["test"] * 100)

    train_df, val_df, test_df, provenance = campaign_window_temporal_split(
        df,
        timestamp_col=ts_col,
        train_ratio=split_cfg["train"],
        val_ratio=split_cfg["val"],
        test_ratio=split_cfg["test"],
    )

    # Save campaign provenance
    with open(processed_dir / "campaign_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2)

    split_dists = report_split_distribution(train_df, val_df, test_df, label_col)

    # ----------------------------------------------------------------
    # Feature scaling — fit ONLY on train set
    # ----------------------------------------------------------------
    logger.info("Fitting scaler on training data...")
    feature_cols = get_feature_columns(
        train_df,
        exclude_cols=metadata_cols + ["label_multiclass", "label_binary", "_campaign"],
    )
    logger.info("Feature columns (%d): %s", len(feature_cols), feature_cols[:10])

    scaler = fit_scaler(train_df, feature_cols, scaler_type)
    save_scaler(scaler, processed_dir / "scaler.pkl")

    train_df = apply_scaler(train_df, scaler, feature_cols)
    val_df = apply_scaler(val_df, scaler, feature_cols)
    test_df = apply_scaler(test_df, scaler, feature_cols)

    # ----------------------------------------------------------------
    # Duplicate Audit & Filtering (Drop exact feature duplicates in Val/Test)
    # ----------------------------------------------------------------
    logger.info("Running duplicate flow audit across splits...")
    train_df, val_df, test_df, dup_report = audit_and_deduplicate_splits(
        train_df, val_df, test_df, feature_cols=feature_cols, deduplicate_features=True
    )
    with open(processed_dir / "duplicate_audit.json", "w") as f:
        json.dump(dup_report, f, indent=2)

    # ----------------------------------------------------------------
    # Save to Parquet
    # ----------------------------------------------------------------
    logger.info("Saving processed splits to Parquet...")
    train_df.to_parquet(processed_dir / "train.parquet", index=False)
    val_df.to_parquet(processed_dir / "val.parquet", index=False)
    test_df.to_parquet(processed_dir / "test.parquet", index=False)

    # Also save the feature column list
    with open(processed_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)

    # ----------------------------------------------------------------
    # Generate statistics CSVs for Train, Val, and Test
    # ----------------------------------------------------------------
    def generate_stats_csv(split_data: pd.DataFrame, split_name: str, target_path: Path):
        t_start = split_data[ts_col].min()
        t_end = split_data[ts_col].max()
        f_count = len(split_data)
        b_count = (split_data[label_col] == "BENIGN").sum()
        a_count = f_count - b_count
        val_counts = split_data[label_col].value_counts()
        a_classes = [c for c in val_counts.index if c != "BENIGN"]
        pcts = {c: round(cnt / f_count * 100, 2) for c, cnt in val_counts.items()}
        row = {
            "split": split_name,
            "time_range_start": str(t_start),
            "time_range_end": str(t_end),
            "flow_count": f_count,
            "benign_count": b_count,
            "attack_count": a_count,
            "attack_classes": ", ".join(a_classes) if a_classes else "None",
            "class_percentages": json.dumps(pcts),
        }
        pd.DataFrame([row]).to_csv(target_path, index=False)
        logger.info("Saved %s statistics to %s", split_name, target_path)

    generate_stats_csv(train_df, "TRAIN", processed_dir / "train_statistics.csv")
    generate_stats_csv(val_df, "VALIDATION", processed_dir / "val_statistics.csv")
    generate_stats_csv(test_df, "TEST", processed_dir / "test_statistics.csv")

    # ----------------------------------------------------------------
    # Save preprocessing statistics
    # ----------------------------------------------------------------
    stats_output = {
        "total_rows_raw": total_rows,
        "total_rows_after_cleaning": len(df),
        "cleaning_stats": dict(cleaning_totals),
        "unique_labels": unique_labels,
        "label_encoder": multiclass_encoder,
        "class_distribution": label_dist.to_dict(),
        "split": {
            "train": len(train_df),
            "val": len(val_df),
            "test": len(test_df),
        },
        "duplicate_audit": dup_report,
        "campaign_provenance": provenance,
        "feature_columns": feature_cols,
        "scaler_type": scaler_type,
        "mode": "sample" if args.sample else "full",
    }

    with open(processed_dir / "preprocessing_stats.json", "w") as f:
        json.dump(stats_output, f, indent=2, default=str)

    logger.info("=" * 60)
    logger.info("Preprocessing complete.")
    logger.info("  Train: %d rows", len(train_df))
    logger.info("  Val:   %d rows", len(val_df))
    logger.info("  Test:  %d rows", len(test_df))
    logger.info("  Saved to: %s", processed_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
