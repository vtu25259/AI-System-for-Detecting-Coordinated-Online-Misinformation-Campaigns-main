"""Utilities for loading and validating post data."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import pandas as pd

REQUIRED_COLUMNS = ["user_id", "post_text", "timestamp"]


def load_posts_from_csv(file_input: Union[str, Path, bytes]) -> pd.DataFrame:
    """
    Load posts from a CSV file path or file-like input.

    Parameters
    ----------
    file_input:
        CSV source (path, bytes, or file-like object used by Streamlit uploader).
    """
    df = pd.read_csv(file_input)
    return validate_posts_df(df)


def validate_posts_df(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and normalize timestamp column."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required columns: {missing_str}")

    clean_df = df.copy()
    clean_df["user_id"] = clean_df["user_id"].astype(str)
    clean_df["post_text"] = clean_df["post_text"].astype(str).str.strip()
    clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
    clean_df = clean_df.dropna(subset=["user_id", "post_text", "timestamp"])
    clean_df = clean_df.reset_index(drop=True)

    if clean_df.empty:
        raise ValueError("No valid rows found after validation.")

    return clean_df
