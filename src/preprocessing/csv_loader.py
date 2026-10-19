"""
GraphGuard: Phase 2 — Data Preprocessing
Module: csv_loader.py

Memory-safe chunked CSV discovery and loading from CICIDS2017 dataset ZIPs.
Handles both GeneratedLabelledFlows and MachineLearningCSV formats.
"""
import io
import logging
import zipfile
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple

import pandas as pd

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


# Canonical column name normalization map:
# Strips leading/trailing whitespace and applies known aliases.
COLUMN_ALIASES: Dict[str, str] = {
    "flow id": "Flow ID",
    "source ip": "Source IP",
    "source port": "Source Port",
    "destination ip": "Destination IP",
    "destination port": "Destination Port",
    "protocol": "Protocol",
    "timestamp": "Timestamp",
    "label": "Label",
    "flow duration": "Flow Duration",
    "total fwd packets": "Total Fwd Packets",
    "total backward packets": "Total Backward Packets",
    "total length of fwd packets": "Total Length of Fwd Packets",
    "total length of bwd packets": "Total Length of Bwd Packets",
}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strip whitespace from column names and apply canonical aliases.

    Args:
        df: DataFrame with raw column names.

    Returns:
        DataFrame with cleaned column names.
    """
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    for col in df.columns:
        lower = col.lower()
        if lower in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[lower]
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def list_csvs_in_zip(zip_path: Path, subfolder: str) -> List[str]:
    """
    List all CSV file names inside a ZIP archive's subfolder.

    Args:
        zip_path: Path to the ZIP archive.
        subfolder: Name of the folder inside the ZIP to search.

    Returns:
        List of archive member names that are CSV files.
    """
    csvs = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.startswith(subfolder) and name.lower().endswith(".csv"):
                csvs.append(name)
    logger.info("Found %d CSV file(s) in %s/%s", len(csvs), zip_path.name, subfolder)
    return csvs


def iter_chunks_from_zip(
    zip_path: Path,
    archive_member: str,
    chunk_size: int = 50_000,
    encoding: str = "utf-8",
    encoding_errors: str = "replace",
    max_rows: Optional[int] = None,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream a CSV from inside a ZIP in memory-safe chunks WITHOUT full extraction.

    Args:
        zip_path: Path to the ZIP archive.
        archive_member: Name of the CSV file inside the ZIP.
        chunk_size: Number of rows per chunk.
        encoding: Character encoding.
        encoding_errors: How to handle encoding errors ('replace', 'ignore', 'strict').
        max_rows: If set, stop after yielding this many total rows (for sampling).

    Yields:
        DataFrame chunks with normalized column names.
    """
    logger.info("Streaming %s from %s (chunk_size=%d)", archive_member, zip_path.name, chunk_size)
    total_rows = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(archive_member) as raw_file:
            # Wrap in TextIOWrapper for pandas
            text_file = io.TextIOWrapper(raw_file, encoding=encoding, errors=encoding_errors)
            reader = pd.read_csv(
                text_file,
                chunksize=chunk_size,
                low_memory=False,
                on_bad_lines="skip",
            )
            for chunk in reader:
                chunk = normalize_column_names(chunk)
                rows_in_chunk = len(chunk)

                if max_rows is not None:
                    remaining = max_rows - total_rows
                    if remaining <= 0:
                        break
                    if rows_in_chunk > remaining:
                        chunk = chunk.iloc[:remaining]

                total_rows += len(chunk)
                yield chunk

                if max_rows is not None and total_rows >= max_rows:
                    logger.info("Reached max_rows=%d — stopping early.", max_rows)
                    break

    logger.info("Finished streaming %s: %d rows total", archive_member, total_rows)


def load_all_chunks(
    zip_path: Path,
    subfolder: str,
    chunk_size: int = 50_000,
    encoding: str = "utf-8",
    encoding_errors: str = "replace",
    file_filter: Optional[List[str]] = None,
    sample_file: Optional[str] = None,
    max_rows: Optional[int] = None,
) -> Generator[Tuple[str, pd.DataFrame], None, None]:
    """
    Stream all CSV files inside a ZIP's subfolder, yielding (filename, chunk) pairs.

    Args:
        zip_path: Path to the ZIP archive.
        subfolder: Directory prefix inside the ZIP (e.g., 'TrafficLabelling ').
        chunk_size: Rows per chunk.
        encoding: Character encoding.
        encoding_errors: Encoding error handling strategy.
        file_filter: If provided, only process CSV files whose names appear in this list.
        sample_file: If provided, only process this one file (sample mode).
        max_rows: Maximum total rows to yield per file (for sampling).

    Yields:
        Tuples of (archive_member_name, DataFrame_chunk).
    """
    all_members = list_csvs_in_zip(zip_path, subfolder)

    if sample_file:
        # Match by basename
        all_members = [m for m in all_members if Path(m).name == sample_file]
        if not all_members:
            raise FileNotFoundError(
                f"Sample file '{sample_file}' not found in {zip_path.name}/{subfolder}"
            )
        logger.info("SAMPLE MODE: processing only %s", sample_file)

    elif file_filter:
        all_members = [m for m in all_members if Path(m).name in file_filter]

    for member in sorted(all_members):
        filename = Path(member).name
        logger.info("Processing: %s", filename)
        for chunk in iter_chunks_from_zip(
            zip_path, member, chunk_size, encoding, encoding_errors, max_rows
        ):
            yield filename, chunk


def get_column_schema(
    zip_path: Path, archive_member: str, encoding: str = "utf-8", encoding_errors: str = "replace"
) -> List[str]:
    """
    Read only the header row of a CSV inside a ZIP to determine column schema.

    Args:
        zip_path: Path to the ZIP archive.
        archive_member: Name of the CSV file inside the ZIP.
        encoding: Character encoding.
        encoding_errors: Encoding error handling strategy.

    Returns:
        List of column names after normalization.
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(archive_member) as raw_file:
            text_file = io.TextIOWrapper(raw_file, encoding=encoding, errors=encoding_errors)
            header = text_file.readline()
    cols = [c.strip() for c in header.strip().split(",")]
    return cols
