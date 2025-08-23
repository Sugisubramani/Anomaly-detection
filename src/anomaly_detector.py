#!/usr/bin/env python3
"""
Honeywell Hackathon – Multivariate Time-Series Anomaly Detector (PCA-based)

Outputs:
- Abnormality_score (0..100; percentile within analysis window)
- top_feature_1 .. top_feature_7  (feature names; only >1% contributors; pad with "")

Usage:
    python src/anomaly_detector.py --input <input_csv> --output <output_csv>
"""
from __future__ import annotations
import argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

# ------------------------------ Config ------------------------------
TRAIN_START = pd.Timestamp("2004-01-01 00:00:00")
TRAIN_END   = pd.Timestamp("2004-01-05 23:59:00")
ANAL_START  = pd.Timestamp("2004-01-06 00:00:00")
ANAL_END    = pd.Timestamp("2004-01-10 07:59:00")
FREQ_MINUTES = 1
ROLL_WINDOW = 5          # minutes for smoothing
N_TOP = 7
MIN_SHARE = 0.01         # 1% minimum contributor share
MAX_PCA_COMPONENTS = 20  # cap for speed

@dataclass
class SplitMasks:
    train: np.ndarray
    analysis: np.ndarray

def validate_regular_intervals_and_resample(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure 1-min spacing; if irregular, resample + interpolate."""
    df = df.sort_values("Time").reset_index(drop=True)
    deltas = df["Time"].diff().dropna().dt.total_seconds().values / 60.0
    if not np.allclose(deltas, FREQ_MINUTES, atol=1e-6):
        full_index = pd.date_range(df["Time"].min(), df["Time"].max(), freq=f"{FREQ_MINUTES}min")
        df = (
            df.set_index("Time")
              .reindex(full_index)
              .rename_axis("Time")
              .reset_index()
        )
    # Interpolate numeric holes (forward/back fill at ends)
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        df[num_cols] = df[num_cols].interpolate(method="linear").ffill().bfill()
    return df

def split_masks(df: pd.DataFrame) -> SplitMasks:
    train = (df["Time"] >= TRAIN_START) & (df["Time"] <= TRAIN_END)
    analysis = (df["Time"] >= ANAL_START) & (df["Time"] <= ANAL_END)
    return SplitMasks(train.values, analysis.values)

def pca_fit_reconstruct(X_train: np.ndarray, X_all: np.ndarray) -> Tuple[np.ndarray, PCA]:
    pca = PCA(n_components=min(MAX_PCA_COMPONENTS, X_train.shape[1]))
    pca.fit(X_train)
    X_proj = pca.inverse_transform(pca.transform(X_all))
    residuals = X_all - X_proj        # in scaled space
    err2 = residuals ** 2             # per-feature squared error contributions
    return err2, pca

def percentile_scores_within_analysis(raw: np.ndarray, mask_analysis: np.ndarray) -> np.ndarray:
    """Map raw scores to 0..100 by ranking within analysis window only."""
    analysis_values = raw[mask_analysis]
    if analysis_values.size == 0:
        return np.zeros_like(raw)
    sorted_vals = np.sort(analysis_values)
    ranks = np.searchsorted(sorted_vals, raw, side="right")
    pct = 100.0 * ranks / float(sorted_vals.size)
    # avoid exact zeros (perfect predictions)
    return np.maximum(pct, 0.05)

def compute_top_contributors(err2: np.ndarray, feat_names: List[str], k: int = 7, min_share: float = 0.01) -> List[List[str]]:
    sums = err2.sum(axis=1, keepdims=True)
    denom = np.where(sums == 0.0, 1.0, sums)  # guard
    shares = err2 / denom
    out: List[List[str]] = []
    fn = np.array(feat_names)
    for i in range(err2.shape[0]):
        mask = shares[i] > min_share
        idx = np.where(mask)[0]
        if idx.size == 0:
            out.append([""] * k)
            continue
        # sort by share desc, tie-break alphabetically
        order = np.lexsort((fn[idx], -shares[i, idx]))
        chosen = list(fn[idx][order][:k])
        if len(chosen) < k:
            chosen += [""] * (k - len(chosen))
        out.append(chosen)
    return out

def run(input_csv: str, output_csv: str) -> None:
    # 1) Load
    df = pd.read_csv(input_csv, parse_dates=["Time"])
    if "Time" not in df.columns:
        raise ValueError("CSV must contain a 'Time' column.")

    # 2) Preprocess
    df = validate_regular_intervals_and_resample(df)
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != "Time"]
    if len(feature_cols) == 0:
        raise ValueError("No numeric feature columns found.")

    # 3) Split
    masks = split_masks(df)

    # 4) Scale (fit on training only)
    scaler = RobustScaler()
    X_train = df.loc[masks.train, feature_cols].to_numpy()
    X_all   = df[feature_cols].to_numpy()
    Xtr_s = scaler.fit_transform(X_train)
    Xall_s = scaler.transform(X_all)

    # 5) PCA reconstruction error (per-feature)
    err2_all, _ = pca_fit_reconstruct(Xtr_s, Xall_s)

    # 6) Raw anomaly score (smoothed)
    raw = np.sqrt(err2_all.sum(axis=1))
    if ROLL_WINDOW > 1:
        raw = pd.Series(raw).rolling(ROLL_WINDOW, center=True, min_periods=1).median().to_numpy()

    # 7) 0-100 scoring via analysis percentiles
    scores = percentile_scores_within_analysis(raw, masks.analysis)
    scores = np.round(scores, 2)

    # 8) Top contributors (7)
    top7 = compute_top_contributors(err2_all, feature_cols, k=N_TOP, min_share=MIN_SHARE)

    # 9) Append outputs
    out = df.copy()
    out["Abnormality_score"] = scores
    for i in range(N_TOP):
        out[f"top_feature_{i+1}"] = [row[i] for row in top7]

    # 10) Save
    out.to_csv(output_csv, index=False)

    # 11) Print sanity
    train_scores = out.loc[masks.train, "Abnormality_score"]
    print(f"Training window score mean: {train_scores.mean():.2f}, max: {train_scores.max():.2f}")
    print(f"Saved: {output_csv}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to input CSV")
    p.add_argument("--output", required=True, help="Path to write modified CSV")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args.input, args.output)
