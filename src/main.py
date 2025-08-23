from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import numpy.typing as npt
import pandas as pd

from preprocess import (
    load_csv, ensure_regular_and_interpolate, get_feature_columns,
    make_masks, assert_min_training_hours
)
from model import PCADetector, IFDetector  # ensemble_percentiles not needed

# Type aliases
FloatArray = npt.NDArray[np.floating]

# Exact windows from the hackathon
TRAIN_START = "2004-01-01 00:00:00"
TRAIN_END   = "2004-01-05 23:59:00"
ANAL_START  = "2004-01-06 00:00:00"
ANAL_END    = "2004-01-10 07:59:00"


def robust_calibrate(raw: FloatArray, mask_train: npt.NDArray[np.bool_]) -> FloatArray:
    """
    Center and scale model outputs by the training distribution (median/IQR),
    then clamp negatives to 0 so 'normal' is near 0.
    """
    train_vals = raw[mask_train]
    q50 = np.median(train_vals)
    q25 = np.percentile(train_vals, 25)
    q75 = np.percentile(train_vals, 75)
    iqr = max(q75 - q25, 1e-6)
    z = (raw - q50) / iqr
    z = np.maximum(z, 0.0)
    return z.astype(np.float64)


def shape_with_gamma(x: FloatArray, gamma: float) -> FloatArray:
    """
    Monotonic shaping to compress small values more than large ones.
    For gamma > 1, small (training-like) scores shrink relative to big anomalies.
    """
    x = np.maximum(x, 0.0)
    return np.power(x, float(gamma)).astype(np.float64)


def to_percentiles_within_analysis(raw_like: FloatArray, mask_analysis: npt.NDArray[np.bool_]) -> FloatArray:
    """Map any raw array to 0..100 by ranking within the analysis window."""
    vals = raw_like[mask_analysis]
    if vals.size == 0:
        return np.zeros_like(raw_like, dtype=np.float64)
    sorted_vals = np.sort(vals)
    ranks = np.searchsorted(sorted_vals, raw_like, side="right")
    pct = 100.0 * ranks / float(sorted_vals.size)
    return np.maximum(pct, 0.05).astype(np.float64)  # avoid exact zeros


def smooth_series(x, window: int):
    import numpy as np, pandas as pd
    # NEW: guard for tiny windows
    if window is None or int(window) < 2:
        return np.asarray(x, dtype=float)
    window = int(window)
    return (
        pd.Series(x, dtype=float)
        .rolling(window=window, min_periods=window)  # or min_periods=max(1, window//2)
        .median()
        .to_numpy()
    )


def run(
    input_csv: str,
    output_csv: str,
    use_ensemble: bool = False,
    ensemble_weight: float = 0.7,
    if_contamination: float = 0.003,
    if_estimators: int = 400,
    pca_var_threshold: float = 0.9995,
    pca_roll_window: int = 181,
    calib_gamma: float = 1.5,
    extra_smooth_window: int = 1,
) -> None:
    # 1) Load & preprocess
    df = load_csv(input_csv)
    df = ensure_regular_and_interpolate(df)
    feat_cols = get_feature_columns(df)
    masks = make_masks(df, TRAIN_START, TRAIN_END, ANAL_START, ANAL_END)
    assert_min_training_hours(df, masks, min_hours=72)

    # 2) Split
    X_train = df.loc[masks.train, feat_cols].to_numpy(dtype=np.float64)
    X_all   = df[feat_cols].to_numpy(dtype=np.float64)

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
    raw_if: FloatArray | None = None
    if use_ensemble:
        ifm = IFDetector(
            contamination=float(if_contamination),
            n_estimators=int(if_estimators),
            random_state=42,
        )
        ifm.fit(X_train)
        raw_if = ifm.score_raw(X_all)

    # 5) Robust calibration USING TRAINING WINDOW (keeps training near 0)
    raw_pca_cal = robust_calibrate(raw_pca, masks.train)
    if raw_if is not None:
        raw_if_cal = robust_calibrate(raw_if, masks.train)
    else:
        raw_if_cal = None

    # 6) Combine calibrated raw signals BEFORE percentile mapping
    if raw_if_cal is not None:
        w = max(0.0, min(1.0, float(ensemble_weight)))
        combined_raw = (1.0 - w) * raw_pca_cal + w * raw_if_cal
    else:
        combined_raw = raw_pca_cal

    # 7) Gamma shaping (compress normal more than anomalies)
    combined_raw = shape_with_gamma(combined_raw, calib_gamma)

    # 7b) extra smoothing on the combined raw signal (configurable)
    combined_raw = smooth_series(combined_raw, window=int(extra_smooth_window))

    # 8) Final 0..100 mapping: percentile within the analysis window (per spec)
    scores_pct = to_percentiles_within_analysis(combined_raw, masks.analysis)
    scores_pct = np.round(scores_pct, 2)

    # 9) Top contributors (from PCA residuals)
    top7 = pca.top_k(err2, k=7)

    # 10) Output CSV with 8 new columns
    out = df.copy()
    out["Abnormality_score"] = scores_pct
    for i in range(7):
        out[f"top_feature_{i+1}"] = [row[i] for row in top7]

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)

    # 11) Training stats + warning (per spec)
    train_scores = out.loc[masks.train, "Abnormality_score"]
    print(f"Training window score mean: {train_scores.mean():.2f}, max: {train_scores.max():.2f}")
    if train_scores.mean() >= 10 or train_scores.max() >= 25:
        print("WARNING: Training period shows elevated anomaly scores. Proceeding as allowed by spec.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input CSV")
    ap.add_argument("--output", required=True, help="Path to output CSV")
    ap.add_argument("--use-ensemble", action="store_true", help="Enable PCA + IsolationForest ensemble")
    ap.add_argument("--ensemble-weight", type=float, default=0.7,
                    help="Weight on IsolationForest when combining (0..1). Default 0.7")
    ap.add_argument("--pca-var-threshold", type=float, default=0.9995,
                help="Variance to keep in PCA (default 0.9995)")
    ap.add_argument("--pca-roll-window", type=int, default=181,
                help="Rolling median window for smoothing raw PCA score (default 181)")
    ap.add_argument("--calib-gamma", type=float, default=1.5,
                help="Gamma shaping on calibrated raw scores (default 1.5)")
    ap.add_argument("--extra-smooth-window", type=int, default=1,
                help="Secondary rolling-median window (default 1)")

    ap.add_argument("--if-contamination", type=float, default=0.003,
                    help="IsolationForest contamination (0..1). Default 0.003")
    ap.add_argument("--if-estimators", type=int, default=400,
                    help="IsolationForest number of trees. Default 400")
    
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.input,
        args.output,
        use_ensemble=args.use_ensemble,
        ensemble_weight=args.ensemble_weight,
        pca_var_threshold=args.pca_var_threshold,
        pca_roll_window=args.pca_roll_window,
        if_contamination=args.if_contamination,
        if_estimators=args.if_estimators,
        calib_gamma=args.calib_gamma,
        extra_smooth_window=args.extra_smooth_window,
    )
