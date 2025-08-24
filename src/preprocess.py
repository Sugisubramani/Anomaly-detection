"""Preprocessing utilities for the anomaly detection pipeline.

Responsibilities
---------------
- Robust CSV loading with tolerant fallbacks.
- Time index validation, regularization to 1-minute spacing, interpolation.
- Feature column selection (numeric only, excluding 'Time').
- Construction and validation of train/analysis masks.
- (Optional) Temporal feature engineering: lags & rolling stats.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Iterable, Tuple, Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

# Expected sampling interval in minutes
FREQ_MINUTES: int = 1

BoolArray = npt.NDArray[np.bool_]


@dataclass
class SplitMasks:
    """Boolean masks for train and analysis slices over the full dataframe.

    Attributes
    ----------
    train
        Mask (len == len(df)) selecting rows in the training window.
    analysis
        Mask (len == len(df)) selecting rows in the analysis window.
    """
    train: BoolArray
    analysis: BoolArray


# ---------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------
def _read_csv_resilient(path: str) -> pd.DataFrame:
    """Load a CSV with progressive fallbacks and ensure a 'Time' column.

    Strategy
    --------
    1) Try fast C engine.
    2) Fall back to Python engine (tolerant; skips bad lines).
    3) Final fallback: chunked read to reduce peak memory.

    Returns
    -------
    pd.DataFrame
        Raw dataframe containing at least a 'Time' column.

    Raises
    ------
    ValueError
        If the file lacks a 'Time' column or timestamps are unparseable.
    """
    # 1) Fast C engine
    try:
        df = pd.read_csv(
            path,
            parse_dates=["Time"],
            low_memory=False,
            memory_map=True,
            encoding="utf-8",
        )
    except Exception:
        # 2) Python engine (more tolerant)
        try:
            df = pd.read_csv(
                path,
                parse_dates=["Time"],
                low_memory=False,
                engine="python",
                on_bad_lines="skip",
                encoding_errors="ignore",
            )
        except Exception:
            # 3) Chunked fallback
            chunks: List[pd.DataFrame] = []
            for ch in pd.read_csv(
                path,
                chunksize=100_000,
                engine="python",
                on_bad_lines="skip",
                encoding_errors="ignore",
            ):
                chunks.append(ch)
            df = pd.concat(chunks, ignore_index=True)

    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column.")

    # Coerce/validate timestamps
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce", utc=False)
    df = df[df["Time"].notna()].copy()
    if df.empty:
        raise ValueError("After parsing, no valid timestamps remained in 'Time'.")

    return df


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV, parse/validate 'Time', coerce numerics, and sort.

    Non-numeric columns (except 'Time') are coerced to numeric with `NaN`
    for unparseable values to stabilize downstream numeric ops.

    Raises
    ------
    FileNotFoundError
        If the input path does not exist.
    ValueError
        If 'Time' is missing or cannot be parsed to any valid timestamps.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = _read_csv_resilient(path)

    # Coerce non-numeric columns (excluding 'Time') to numeric where possible
    for c in df.columns:
        if c != "Time" and not np.issubdtype(df[c].dtype, np.number):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sort by time, drop exact duplicate timestamps (keep first for determinism)
    df = df.sort_values("Time").drop_duplicates(subset=["Time"], keep="first").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------
# Regularization & interpolation
# ---------------------------------------------------------------------
def ensure_regular_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1‑minute spacing; resample + interpolate numeric gaps if needed.

    Steps
    -----
    - Validate monotonic non-decreasing timestamps (sorted in `load_csv`).
    - If spacing is not ~1 minute, reindex to a complete 1-minute grid.
    - Linearly interpolate numeric columns; then forward/backward fill edges.

    Notes
    -----
    - Non-numeric columns (if any) are left as-is after reindexing; only
      numeric columns are interpolated/filled.

    Returns
    -------
    pd.DataFrame
        Regularized dataframe with a 'Time' column at 1-minute intervals.
    """
    if "Time" not in df.columns:
        raise ValueError("ensure_regular_and_interpolate: missing 'Time' column.")

    if df.empty:
        raise ValueError("ensure_regular_and_interpolate: empty dataframe.")

    deltas_min = df["Time"].diff().dropna().dt.total_seconds().to_numpy() / 60.0
    needs_resample = True
    if deltas_min.size > 0:
        # Consider it regular if all deltas are within a tiny tolerance of 1 minute
        needs_resample = not np.allclose(deltas_min, float(FREQ_MINUTES), atol=1e-6)

    if needs_resample:
        full_index = pd.date_range(df["Time"].iloc[0], df["Time"].iloc[-1], freq=f"{FREQ_MINUTES}min")
        df = (
            df.set_index("Time")
            .reindex(full_index)
            .rename_axis("Time")
            .reset_index()
            .rename(columns={"index": "Time"})
        )

    # Interpolate numeric columns only
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()

    return df


# ---------------------------------------------------------------------
# Feature selection & masks
# ---------------------------------------------------------------------
def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return all numeric feature columns, excluding 'Time'.

    Returns
    -------
    List[str]
        Names of numeric columns to be used as features.
    """
    return [c for c in df.select_dtypes(include=[np.number]).columns if c != "Time"]


def make_masks(
    df: pd.DataFrame,
    train_start: str,
    train_end: str,
    anal_start: str,
    anal_end: str,
) -> SplitMasks:
    """Create boolean masks for training and analysis windows.

    Parameters
    ----------
    df
        Regularized dataframe containing a 'Time' column.
    train_start, train_end, anal_start, anal_end
        Boundaries as ISO-like strings; inclusive endpoints.

    Returns
    -------
    SplitMasks
        Boolean masks aligned to `df` rows.
    """
    ts = pd.to_datetime(df["Time"])
    train = (ts >= pd.Timestamp(train_start)) & (ts <= pd.Timestamp(train_end))
    analysis = (ts >= pd.Timestamp(anal_start)) & (ts <= pd.Timestamp(anal_end))
    return SplitMasks(train.values.astype(bool), analysis.values.astype(bool))


def assert_min_training_hours(df: pd.DataFrame, masks: SplitMasks, min_hours: int = 72) -> None:
    """Validate that the training window has at least `min_hours` of data.

    Parameters
    ----------
    df
        Dataframe (regularized to 1-minute spacing).
    masks
        Output of `make_masks`.
    min_hours
        Minimum training hours required.

    Raises
    ------
    ValueError
        If the training slice is smaller than `min_hours` worth of rows.
    """
    if "Time" not in df.columns or df.empty:
        raise ValueError("assert_min_training_hours: invalid dataframe.")
    if masks.train.size != len(df):
        raise ValueError("assert_min_training_hours: mask length mismatch.")

    required_rows = int(min_hours * 60)
    n_rows = int(np.count_nonzero(masks.train))
    if n_rows < required_rows:
        raise ValueError(
            f"Need at least {min_hours} hours ({required_rows} rows) of training data; found {n_rows}."
        )


def validate_input_schema(df: pd.DataFrame, max_nan_frac: float = 0.2) -> None:
    """Validate dataframe schema before modeling.

    Checks
    ------
    - 'Time' column exists.
    - At least one numeric feature column remains (excluding 'Time').
    - No feature exceeds `max_nan_frac` missing values.
    - Warn if any column is constant.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    max_nan_frac : float, optional
        Maximum allowed fraction of NaN values per column (default=0.2).

    Raises
    ------
    ValueError
        If critical schema requirements are violated.
    """
    if "Time" not in df.columns:
        raise ValueError("validate_input_schema: missing 'Time' column.")

    feat_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Time"]
    if not feat_cols:
        raise ValueError("No numeric feature columns found after loading.")

    too_many_nans: List[tuple[str, float]] = []
    constant_cols: List[str] = []
    for c in feat_cols:
        nan_frac = float(df[c].isna().mean())
        if nan_frac > max_nan_frac:
            too_many_nans.append((c, nan_frac))
        # constant (or all-NaN which we already counted above)
        if df[c].nunique(dropna=True) <= 1:
            constant_cols.append(c)

    if too_many_nans:
        worst = ", ".join([f"{c}={p:.1%}" for c, p in too_many_nans[:4]])
        raise ValueError(
            "Some features exceed allowed NaN fraction "
            f"(>{max_nan_frac:.0%}): {worst}"
            + (" …" if len(too_many_nans) > 4 else "")
        )

    for c in constant_cols:
        print(f"WARNING: Feature '{c}' is constant or nearly constant.")


# ---------------------------------------------------------------------
# Temporal feature engineering
# ---------------------------------------------------------------------
def add_lag_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lags: int,
    *,
    time_col: str = "Time",
) -> Tuple[pd.DataFrame, List[str]]:
    """Add t-1..t-lag shifted copies of each feature (past-only; no leakage).

    Parameters
    ----------
    df : DataFrame with a monotonically increasing time column.
    feature_cols : columns to lag (numeric).
    lags : number of lag steps (minutes, since data is at 1-min freq).
    time_col : name of the timestamp column.

    Returns
    -------
    df_out, new_cols : updated dataframe and list of new lag column names.
    """
    if lags <= 0:
        return df, []

    df_out = df.copy()
    new_cols: List[str] = []
    for c in feature_cols:
        for k in range(1, int(lags) + 1):
            name = f"{c}_lag{k}"
            df_out[name] = df_out[c].shift(k)
            new_cols.append(name)

    # Fill initial NaNs introduced by lagging (safe: uses only past data)
    df_out[new_cols] = df_out[new_cols].ffill().bfill()
    return df_out, new_cols


def add_rolling_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    windows: Iterable[int],
) -> Tuple[pd.DataFrame, List[str]]:
    """Add rolling mean/std features for each window (past-only).

    Parameters
    ----------
    windows : iterable of window sizes in minutes (e.g., [5, 15, 60])

    Returns
    -------
    df_out, new_cols : updated dataframe and list of new rolling column names.
    """
    wins = [int(w) for w in windows if int(w) > 1]
    if not wins:
        return df, []

    df_out = df.copy()
    new_cols: List[str] = []
    for c in feature_cols:
        s = df_out[c]
        for w in wins:
            m_name = f"{c}_roll{w}m_mean"
            s_name = f"{c}_roll{w}m_std"
            df_out[m_name] = s.rolling(window=w, min_periods=w).mean()
            df_out[s_name] = s.rolling(window=w, min_periods=w).std()
            new_cols.extend([m_name, s_name])

    # Fill initial NaNs from warm-up windows
    df_out[new_cols] = df_out[new_cols].ffill().bfill()
    return df_out, new_cols


def build_temporal_features(
    df: pd.DataFrame,
    base_feature_cols: List[str],
    *,
    lags: int = 0,
    rolling_windows: Optional[Iterable[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Compose lag + rolling features and return updated df + all feature cols.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (already regularized).
    base_feature_cols : List[str]
        Original numeric feature columns (excluding 'Time').
    lags : int
        Number of lag steps to add per feature (t-1..t-lags). 0 disables.
    rolling_windows : Optional[Iterable[int]]
        Minutes for which to add rolling mean/std per feature (e.g., [5, 15]).

    Returns
    -------
    df2 : pd.DataFrame
        Dataframe with new temporal features added.
    all_cols : List[str]
        List of all feature columns to use downstream (base + derived).
    """
    all_cols = list(base_feature_cols)
    df2 = df

    if lags and int(lags) > 0:
        df2, lag_cols = add_lag_features(df2, base_feature_cols, int(lags))
        all_cols.extend(lag_cols)

    if rolling_windows:
        df2, roll_cols = add_rolling_features(df2, base_feature_cols, rolling_windows)
        all_cols.extend(roll_cols)

    return df2, all_cols