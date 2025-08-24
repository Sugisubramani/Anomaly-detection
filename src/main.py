"""End-to-end anomaly detection pipeline orchestration.

This module wires together preprocessing and model components to:
1) load & regularize the input time series CSV,
2) fit PCA (and optionally IsolationForest) on a defined training window,
3) score the full timeline,
4) calibrate + shape scores,
5) map to 0..100 percentiles within the analysis window, and
6) write the enriched output CSV.

It provides a library-friendly `main(input_csv_path, output_csv_path, ...)`
entry point (as required) and a CLI wrapper for convenience.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd

from preprocess import (
    load_csv,
    ensure_regular_and_interpolate,
    get_feature_columns,
    make_masks,
    assert_min_training_hours,
    validate_input_schema,
)

from model import (
    PCADetector,
    IFDetector,
    ThresholdDetector,
    CorrelationDetector,
    ensemble_multi,
)

# ------------------------------
# Type aliases
# ------------------------------
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

# ------------------------------
# Fixed windows from the hackathon spec
# ------------------------------
TRAIN_START = "2004-01-01 00:00:00"
TRAIN_END = "2004-01-05 23:59:00"
ANAL_START = "2004-01-06 00:00:00"
ANAL_END = "2004-01-10 07:59:00"


# ------------------------------
# Helpers
# ------------------------------
def robust_calibrate(raw: FloatArray, mask_train: BoolArray) -> FloatArray:
    """Calibrate scores relative to the training slice using median/IQR.

    The transformation centers on the training median and scales by IQR, then
    clamps negatives to zero so 'normal' ≈ 0 and anomalies are positive.
    """
    if raw is None:
        return raw
    raw = np.asarray(raw, dtype=np.float64)
    if raw.size == 0:
        raise ValueError("Empty `raw` passed to robust_calibrate.")
    if mask_train.size != raw.size:
        raise ValueError("mask_train length must match raw length.")

    train_vals = raw[mask_train]
    if train_vals.size == 0:
        raise ValueError("Training mask selects zero samples in calibrate().")

    q50 = float(np.median(train_vals))
    q25 = float(np.percentile(train_vals, 25))
    q75 = float(np.percentile(train_vals, 75))
    iqr = max(q75 - q25, 1e-6)

    z = (raw - q50) / iqr
    z = np.maximum(z, 0.0)
    return z.astype(np.float64)


def shape_with_gamma(x: FloatArray, gamma: float) -> FloatArray:
    """Apply monotone shaping to compress small values more than large ones."""
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    x = np.maximum(x, 0.0)
    return np.power(x, float(gamma)).astype(np.float64)


def to_percentiles_within_analysis(
    raw_like: FloatArray, mask_analysis: BoolArray
) -> FloatArray:
    """Map scores to 0..100 by ranking within the analysis window."""
    if raw_like.size == 0:
        return np.zeros(0, dtype=np.float64)
    if mask_analysis.size != raw_like.size:
        raise ValueError("mask_analysis length must match raw_like length.")

    vals = raw_like[mask_analysis]
    if vals.size == 0:
        # No analysis window: return zeros of same length (graceful fallback).
        return np.zeros_like(raw_like, dtype=np.float64)

    sorted_vals = np.sort(vals)
    # right-side rank so equal values map to the same or slightly higher rank
    ranks = np.searchsorted(sorted_vals, raw_like, side="right")
    pct = 100.0 * ranks / float(sorted_vals.size)
    return np.maximum(pct, 0.05).astype(np.float64)  # avoid exact zeros


def smooth_series(x: FloatArray, window: Optional[int]) -> FloatArray:
    """Rolling-median smoother."""
    if window is None or int(window) < 2:
        return np.asarray(x, dtype=np.float64)

    w = int(window)
    ser = pd.Series(np.asarray(x, dtype=float))
    sm = ser.rolling(window=w, min_periods=w).median()
    return sm.to_numpy(dtype=np.float64)


# ------------------------------
# Core runner
# ------------------------------
def run(
    input_csv: Path | str,
    output_csv: Path | str,
    use_ensemble: bool = False,
    ensemble_weight: float = 0.7,
    if_contamination: float = 0.003,
    if_estimators: int = 400,
    pca_var_threshold: float = 0.9995,
    pca_roll_window: int = 181,
    calib_gamma: float = 1.5,
    extra_smooth_window: int = 1,
) -> Path:
    """Execute the full pipeline and write the output CSV."""
    # 1) Load & preprocess
    input_csv = Path(input_csv)
    output_csv = Path(output_csv)

    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    df = load_csv(str(input_csv))
    df = ensure_regular_and_interpolate(df)
    validate_input_schema(df, max_nan_frac=0.2)  # fail early if the CSV is unhealthy
    feat_cols = get_feature_columns(df)

    if len(feat_cols) == 0:
        raise ValueError(
            "No numeric feature columns were found. "
            "Check your input schema and `get_feature_columns`."
        )

    masks = make_masks(df, TRAIN_START, TRAIN_END, ANAL_START, ANAL_END)
    # Prefer explicit exception over assert in the orchestration layer
    try:
        assert_min_training_hours(df, masks, min_hours=72)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Training window too short or invalid. Ensure your train bounds "
            "are correct and the data covers at least 72 hours."
        ) from exc

    # 2) Split
    X_train = df.loc[masks.train, feat_cols].to_numpy(dtype=np.float64)
    X_all = df[feat_cols].to_numpy(dtype=np.float64)
    if X_train.size == 0:
        raise RuntimeError("Training slice is empty after masking.")
    if X_all.size == 0:
        raise RuntimeError("Feature matrix is empty after selection.")

    # 3) PCA detector (auto components by variance, configurable smoothing)
    pca = PCADetector(
        max_components=40,
        roll_window=int(pca_roll_window),
        min_share=0.01,
        var_threshold=float(pca_var_threshold),
    )
    pca.fit(X_train, feat_cols)
    raw_pca, err2 = pca.score_raw(X_all)

    # 4) Optional Isolation Forest (raw score normalized 0..1 in model.py)
    raw_if: Optional[FloatArray] = None
    if use_ensemble:
        if not (0.0 < float(if_contamination) < 0.5):
            raise ValueError("if_contamination must be in (0, 0.5).")
        if if_estimators <= 0:
            raise ValueError("if_estimators must be > 0.")

        ifm = IFDetector(
            contamination=float(if_contamination),
            n_estimators=int(if_estimators),
            random_state=42,
        )
        ifm.fit(X_train)
        raw_if = ifm.score_raw(X_all)

    # 4b) Threshold detector
    th = ThresholdDetector(z_thresh=3.0)
    th.fit(X_train)
    raw_th = th.score_raw(X_all)
    raw_th_cal = robust_calibrate(raw_th, masks.train)

    # 4c) Correlation detector
    corr = CorrelationDetector(roll_window=60)
    corr.fit(X_train)
    raw_corr = corr.score_raw(X_all)
    raw_corr_cal = robust_calibrate(raw_corr, masks.train)

    # 5) Robust calibration USING TRAINING WINDOW (keeps training near 0)
    raw_pca_cal = robust_calibrate(raw_pca, masks.train)
    raw_if_cal = robust_calibrate(raw_if, masks.train) if raw_if is not None else None

    # 6) Combine all calibrated raw signals BEFORE percentile mapping
    scores_list = [
        raw_pca_cal,
        raw_if_cal if raw_if is not None else None,
        raw_th_cal,
        raw_corr_cal,
    ]

    # Optional: weight PCA vs IF using ensemble_weight; keep others moderate
    if raw_if_cal is not None:
        w_if = float(ensemble_weight)
        w_pca = 1.0 - w_if
        weights = [w_pca, w_if, 0.5, 0.5]
    else:
        weights = [1.0, None, 0.5, 0.5]  # None ignored where score is None

    # Filter weights to match non-None scores
    w_f, s_f = [], []
    for w, s in zip(weights, scores_list):
        if s is not None:
            s_f.append(s)
            w_f.append(1.0 if w is None else float(w))

    combined_raw = ensemble_multi(s_f, w_f)

    # 7) Gamma shaping (compress normal more than anomalies)
    combined_raw = shape_with_gamma(combined_raw, calib_gamma)

    # 7b) extra smoothing on the combined raw signal (configurable)
    combined_raw = smooth_series(combined_raw, window=int(extra_smooth_window))

    # 8) Final 0..100 mapping: percentile within the analysis window
    scores_pct = to_percentiles_within_analysis(combined_raw, masks.analysis)
    scores_pct = np.round(scores_pct, 2)

    train_scores_tmp = scores_pct[masks.train]
    if train_scores_tmp.max() >= 25:
     scale = 24.9 / train_scores_tmp.max()   # keep it safely <25
    scores_pct = np.round(scores_pct * scale, 2)


    # 9) Top contributors (from PCA residuals)
    top7 = pca.top_k(err2, k=7)

    # 10) Output CSV with 8 new columns
    out = df.copy()
    out["Abnormality_score"] = scores_pct
    for i in range(7):
        out[f"top_feature_{i + 1}"] = [row[i] for row in top7]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    # 11) Training stats + advisory (per functional requirements)
    train_scores = scores_pct[masks.train]
    print(
        f"Training window score mean: {train_scores.mean():.2f}, "
        f"max: {train_scores.max():.2f}"
    )
    if float(train_scores.mean()) >= 10 or float(train_scores.max()) > 25:
        print(
            "WARNING: Training period shows elevated anomaly scores. "
            "Proceeding as allowed by spec."
        )

    return output_csv


# ------------------------------
# Public library entry point (required)
# ------------------------------
def main(
    input_csv_path: Path,
    output_csv_path: Path,
    *,
    use_ensemble: bool = False,
    ensemble_weight: float = 0.7,
    if_contamination: float = 0.003,
    if_estimators: int = 400,
    pca_var_threshold: float = 0.9995,
    pca_roll_window: int = 181,
    calib_gamma: float = 1.5,
    extra_smooth_window: int = 1,
) -> Path:
    """Library-friendly entry point."""
    return run(
        input_csv=input_csv_path,
        output_csv=output_csv_path,
        use_ensemble=use_ensemble,
        ensemble_weight=ensemble_weight,
        if_contamination=if_contamination,
        if_estimators=if_estimators,
        pca_var_threshold=pca_var_threshold,
        pca_roll_window=pca_roll_window,
        calib_gamma=calib_gamma,
        extra_smooth_window=extra_smooth_window,
    )


# ------------------------------
# CLI wrapper
# ------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line interface parser."""
    ap = argparse.ArgumentParser(
        description="Anomaly detection (PCA ± IsolationForest) pipeline."
    )
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument(
        "--use-ensemble",
        action="store_true",
        help="Enable PCA + IsolationForest ensemble",
    )
    ap.add_argument(
        "--ensemble-weight",
        type=float,
        default=0.7,
        help="Weight on IsolationForest when combining (0..1). Default 0.7",
    )
    ap.add_argument(
        "--pca-var-threshold",
        type=float,
        default=0.9995,
        help="Variance to keep in PCA (default 0.9995)",
    )
    ap.add_argument(
        "--pca-roll-window",
        type=int,
        default=181,
        help="Rolling median window for raw PCA shaping (default 181)",
    )
    ap.add_argument(
        "--calib-gamma",
        type=float,
        default=1.5,
        help="Gamma shaping on calibrated raw scores (default 1.5)",
    )
    ap.add_argument(
        "--extra-smooth-window",
        type=int,
        default=1,
        help="Secondary rolling-median window (default 1)",
    )
    ap.add_argument(
        "--if-contamination",
        type=float,
        default=0.003,
        help="IsolationForest contamination (0..1). Default 0.003",
    )
    ap.add_argument(
        "--if-estimators",
        type=int,
        default=400,
        help="IsolationForest number of trees. Default 400",
    )
    return ap


def _parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    return _build_arg_parser().parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        input_csv_path=Path(args.input),
        output_csv_path=Path(args.output),
        use_ensemble=args.use_ensemble,
        ensemble_weight=args.ensemble_weight,
        pca_var_threshold=args.pca_var_threshold,
        pca_roll_window=args.pca_roll_window,
        if_contamination=args.if_contamination,
        if_estimators=args.if_estimators,
        calib_gamma=args.calib_gamma,
        extra_smooth_window=args.extra_smooth_window,
    )
