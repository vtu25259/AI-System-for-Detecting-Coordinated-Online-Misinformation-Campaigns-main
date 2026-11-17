"""Helpers to merge and normalize multi-source post data."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def merge_sources(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge source DataFrames into: post_text, timestamp, source."""
    valid = [f.copy() for f in frames if f is not None and not f.empty]
    if not valid:
        return pd.DataFrame(columns=["post_text", "timestamp", "source"])

    merged = pd.concat(valid, ignore_index=True)
    merged["post_text"] = merged["post_text"].astype(str).str.strip()
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce", utc=True)
    merged["source"] = merged["source"].fillna("unknown").astype(str)
    merged = merged.dropna(subset=["post_text", "timestamp"])
    merged = merged[merged["post_text"] != ""].reset_index(drop=True)
    return merged
