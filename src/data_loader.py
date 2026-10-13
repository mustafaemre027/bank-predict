from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CSV_SEP, RAW_DATA_PATH, TARGET

EXPECTED_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "contact",
    "day",
    "month",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "y",
]


def load_raw(path: str | Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw bank marketing dataset from the project data folder."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"Raw data file not found at '{file_path}'. Download it first with src/download_data.py."
        )

    return pd.read_csv(file_path, sep=CSV_SEP)


def validate_raw(df: pd.DataFrame) -> bool:
    """Validate the raw dataset against the project contract."""
    if df.shape != (45211, 17):
        raise ValueError(
            f"Expected raw dataset shape (45211, 17), but received {df.shape}."
        )

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(
            "Unexpected raw dataset columns. "
            f"Expected: {EXPECTED_COLUMNS}. Received: {list(df.columns)}."
        )

    target_values = set(df[TARGET].astype(str).unique())
    if not target_values.issubset({"yes", "no"}):
        raise ValueError(
            f"Target column contains unexpected values: {sorted(target_values)}. "
            "Only 'yes' and 'no' are allowed."
        )

    yes_count = int(df[TARGET].value_counts().get("yes", 0))
    if yes_count != 5289:
        raise ValueError(
            f"Expected 5289 'yes' target values, but found {yes_count}."
        )

    return True


__all__ = ["EXPECTED_COLUMNS", "load_raw", "validate_raw"]
