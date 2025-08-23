from __future__ import annotations
from dataclasses import dataclass
from typing import List
import numpy as np
import pandas as pd
import os

FREQ_MINUTES = 1  # expected sampling interval

@dataclass
class SplitMasks:
    train: np.ndarray
    analysis: np.ndarray

def _read_csv_resilient(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for Windows/pandas:
    1) Try fast C parser.
    2) Fallback to Python engine if C parser errors.
    3) Final fallback: chunked reading to cap memory usage.
    """
    # 1) Try fast C parser
    try:
        return pd.read_csv(
            path,
            parse_dates=["Time"],
            low_memory=False,
            memory_map=True,
            encoding="utf-8",
        )
    except Exception:
        pass

    # 2) Fallback: Python engine (more tolerant)
    try:
        return pd.read_csv(
            path,
            parse_dates=["Time"],
            low_memory=False,
            engine="python",
            on_bad_lines="skip",
            encoding_errors="ignore",
        )
    except Exception:
        pass

    # 3) Final fallback: chunked load (guards memory)
    chunks: List[pd.DataFrame] = []
    # Use a moderate chunksize; adjust if needed
    for ch in pd.read_csv(
        path,
        chunksize=100_000,
        engine="python",
        on_bad_lines="skip",
        encoding_errors="ignore",
    ):
        chunks.append(ch)
    df = pd.concat(chunks, ignore_index=True)

    # Ensure Time is parsed
    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column.")
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")
    # Drop rows where Time couldn't be parsed
    df = df[df["Time"].notna()]
    return df

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV, parse Time, sort; robust against parser memory errors."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = _read_csv_resilient(path)
    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column.")
    # Coerce non-numerics to numeric where possible to stabilize downstream ops
    for c in df.columns:
        if c != "Time":
            if not np.issubdtype(df[c].dtype, np.number):
                df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.sort_values("Time").reset_index(drop=True)
    return df

def ensure_regular_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1-min spacing; resample + interpolate numeric holes if needed."""
    deltas = df["Time"].diff().dropna().dt.total_seconds().values / 60.0
    if not np.allclose(deltas, FREQ_MINUTES, atol=1e-6):
        full_index = pd.date_range(df["Time"].min(), df["Time"].max(), freq=f"{FREQ_MINUTES}min")
        df = df.set_index("Time").reindex(full_index).rename_axis("Time").reset_index()
    # Interpolate numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    return df

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """All numeric columns except Time."""
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "Time"]

def make_masks(df: pd.DataFrame, train_start: str, train_end: str,
               anal_start: str, anal_end: str) -> SplitMasks:
    train = (df["Time"] >= pd.Timestamp(train_start)) & (df["Time"] <= pd.Timestamp(train_end))
    analysis = (df["Time"] >= pd.Timestamp(anal_start)) & (df["Time"] <= pd.Timestamp(anal_end))
    return SplitMasks(train.values, analysis.values)

def assert_min_training_hours(df: pd.DataFrame, masks: SplitMasks, min_hours: int = 72) -> None:
    """Guard: require at least 72h training data."""
    n = int(masks.train.sum())
    if n < min_hours * 60:
        raise ValueError(f"Need at least {min_hours} hours ({min_hours*60} rows) of training data; found {n}.")
