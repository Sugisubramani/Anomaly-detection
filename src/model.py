"""Model components for anomaly detection.

This module provides:
- `PCADetector`: PCA reconstruction-error detector with per-feature attribution.
- `IFDetector`: IsolationForest-based anomaly scorer (normalized to [0, 1]).
- `AnomalyOutputs`: (optional) container for downstream packaging.

Design notes
------------
- Both detectors keep their own `RobustScaler` fit **only** on the training data.
- `PCADetector` selects components by cumulative variance threshold
  (`var_threshold`) and caps by `max_components`.
- Raw PCA scores are optionally smoothed via a rolling median (`roll_window`).
- `IFDetector` inverts `decision_function` so higher means "more anomalous"
  and min-max normalizes to [0, 1] with a stable denominator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
import warnings

# ------------------------------
# Type aliases
# ------------------------------
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


@dataclass
class AnomalyOutputs:
    """Container for anomaly scoring outputs.

    Attributes
    ----------
    scores_raw
        Uncalibrated raw anomaly scores (one per sample).
    scores_pct
        Percentile-mapped scores in [0, 100] (one per sample).
    err2_per_feature
        Per-sample, per-feature squared reconstruction errors (PCA).
    top7_names
        Top-`k` (default 7) feature names per sample by contribution share.
    """
    scores_raw: FloatArray
    scores_pct: FloatArray
    err2_per_feature: FloatArray
    top7_names: List[List[str]]


class PCADetector:
    """PCA reconstruction-error detector with per-feature attribution.

    Pipeline
    --------
    1) Fit `RobustScaler` on training data only.
    2) Choose PCA components to reach `var_threshold` (capped by `max_components`).
    3) Score raw samples via reconstruction RMSE of scaled features.
    4) Optionally smooth raw scores by rolling median of length `roll_window`.

    Parameters
    ----------
    max_components
        Maximum number of PCA components to retain.
    roll_window
        Rolling median window applied to raw scores (if > 1).
    min_share
        Minimum per-feature contribution share to consider when selecting top-k.
    var_threshold
        Target cumulative variance explained (0..1) for component selection.

    Notes
    -----
    - Call `fit(X_train, feature_names)` before `score_raw`, `top_k`.
    """

    def __init__(
        self,
        max_components: int = 40,
        roll_window: int = 11,
        min_share: float = 0.01,
        var_threshold: float = 0.995,
    ) -> None:
        self.max_components: int = int(max(1, max_components))
        self.roll_window: int = int(max(1, roll_window))
        self.min_share: float = float(min_share)
        if not (0.0 < var_threshold <= 1.0):
            raise ValueError("var_threshold must be in (0, 1].")
        self.var_threshold: float = float(var_threshold)

        self.scaler: RobustScaler = RobustScaler()
        self.pca: Optional[PCA] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, X_train: FloatArray, feature_names: Sequence[str]) -> None:
        """Fit scaler and PCA on the training data.

        Parameters
        ----------
        X_train
            Training matrix of shape (n_samples, n_features).
        feature_names
            Names of the features in `X_train`; length must equal n_features.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2 or X_train.size == 0:
            raise ValueError("X_train must be a non-empty 2D array.")
        if len(feature_names) != X_train.shape[1]:
            raise ValueError(
                "feature_names length must match number of columns in X_train."
            )
        self.feature_names = list(map(str, feature_names))

        Xs = self.scaler.fit_transform(X_train)

        # Determine number of components to reach the variance threshold
        pca_full = PCA().fit(Xs)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        ncomp = int(np.searchsorted(cum, self.var_threshold) + 1)
        ncomp = max(1, min(ncomp, min(self.max_components, Xs.shape[1])))

        self.pca = PCA(n_components=ncomp).fit(Xs)

    def _recon_err2(self, X_all: FloatArray) -> FloatArray:
        """Compute per-feature squared reconstruction errors on all samples."""
        if self.pca is None:
            raise RuntimeError("PCADetector is not fitted. Call fit() first.")
        X_all = np.asarray(X_all, dtype=np.float64)
        if X_all.ndim != 2 or X_all.size == 0:
            raise ValueError("X_all must be a non-empty 2D array.")

        Xs = self.scaler.transform(X_all)
        X_proj = self.pca.inverse_transform(self.pca.transform(Xs))
        resid = Xs - X_proj
        return resid ** 2  # (n_samples, n_features)

    def score_raw(self, X_all: FloatArray) -> Tuple[FloatArray, FloatArray]:
        """Compute raw anomaly scores and per-feature errors.

        The raw score is the RMSE of the per-feature scaled residuals. If
        `roll_window > 1`, applies a centered rolling median (min_periods=1).

        Parameters
        ----------
        X_all
            Full matrix to score (n_samples, n_features).

        Returns
        -------
        raw : FloatArray
            Raw anomaly score per sample (length n_samples).
        err2 : FloatArray
            Per-feature squared errors, shape (n_samples, n_features).
        """
        err2 = self._recon_err2(X_all)
        raw = np.sqrt(err2.sum(axis=1))
        if self.roll_window > 1:
            raw = (
                pd.Series(raw)
                .rolling(self.roll_window, center=True, min_periods=1)
                .median()
                .to_numpy(dtype=np.float64)
            )
        else:
            raw = raw.astype(np.float64, copy=False)
        return raw, err2

    @staticmethod
    def to_percentiles(raw: FloatArray, mask_analysis: BoolArray) -> FloatArray:
        """Map scores to percentiles [0, 100] within the analysis window.

        Returns zeros if the analysis window is empty.
        """
        raw = np.asarray(raw, dtype=np.float64)
        if mask_analysis.size != raw.size:
            raise ValueError("mask_analysis length must match raw length.")
        vals = raw[mask_analysis]
        if vals.size == 0:
            return np.zeros_like(raw)
        sorted_vals = np.sort(vals)
        ranks = np.searchsorted(sorted_vals, raw, side="right")
        pct = 100.0 * ranks / float(sorted_vals.size)
        return np.maximum(pct, 0.05)

    def top_k(self, err2: FloatArray, k: int = 7) -> List[List[str]]:
        """Return top-`k` contributing feature names per sample.

        Contribution is computed as each feature's share of total squared error
        for the sample; features below `min_share` are filtered out.

        Parameters
        ----------
        err2
            Per-feature squared errors, shape (n_samples, n_features).
        k
            Number of names to return per sample (defaults to 7).

        Returns
        -------
        List[List[str]]
            For each sample, list of up to `k` feature names (padded with "" if
            fewer than `k` pass the `min_share` threshold).
        """
        if self.feature_names is None:
            raise RuntimeError("PCADetector.feature_names is not set. Call fit().")
        if k <= 0:
            return [[] for _ in range(err2.shape[0])]

        err2 = np.asarray(err2, dtype=np.float64)
        if err2.ndim != 2 or err2.size == 0:
            raise ValueError("err2 must be a non-empty 2D array.")

        sums = err2.sum(axis=1, keepdims=True)
        denom = np.where(sums == 0.0, 1.0, sums)
        shares = err2 / denom

        names = np.array(self.feature_names)
        out: List[List[str]] = []
        for i in range(err2.shape[0]):
            mask = shares[i] > self.min_share
            idx = np.where(mask)[0]
            if idx.size == 0:
                out.append([""] * k)
                continue
            # Sort primarily by share (desc), then by name (asc) for stability
            order = np.lexsort((names[idx], -shares[i, idx]))
            chosen = list(names[idx][order][:k])
            if len(chosen) < k:
                chosen += [""] * (k - len(chosen))
            out.append(chosen)
        return out


class IFDetector:
    """IsolationForest-based anomaly scorer.

    The raw `decision_function` is inverted so that higher values mean
    "more anomalous"; then we min-max normalize to [0, 1].

    Parameters
    ----------
    contamination
        Proportion of expected outliers in the training set, in (0, 0.5).
    n_estimators
        Number of trees.
    random_state
        Seed for reproducibility.
    """

    def __init__(
        self,
        contamination: float = 0.003,
        n_estimators: int = 400,
        random_state: int = 42,
    ) -> None:
        if not (0.0 < contamination < 0.5):
            raise ValueError("contamination must be in (0, 0.5).")
        if n_estimators <= 0:
            raise ValueError("n_estimators must be > 0.")

        self.scaler: RobustScaler = RobustScaler()
        self.iforest: IsolationForest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: FloatArray) -> None:
        """Fit scaler and IsolationForest on the training data."""
        X_train = np.asarray(X_train, dtype=np.float64)
        if X_train.ndim != 2 or X_train.size == 0:
            raise ValueError("X_train must be a non-empty 2D array.")
        Xs = self.scaler.fit_transform(X_train)
        self.iforest.fit(Xs)

    def score_raw(self, X_all: FloatArray) -> FloatArray:
        """Score all samples and normalize to [0, 1]."""
        X_all = np.asarray(X_all, dtype=np.float64)
        if X_all.ndim != 2 or X_all.size == 0:
            raise ValueError("X_all must be a non-empty 2D array.")
        Xs = self.scaler.transform(X_all)
        s = -self.iforest.decision_function(Xs)  # invert so higher = more abnormal
        s_min = float(np.min(s))
        s_max = float(np.max(s))
        rng = s_max - s_min
        if rng <= 1e-12:
            # All identical scores; return zeros to avoid NaNs.
            return np.zeros_like(s, dtype=np.float64)
        return ((s - s_min) / rng).astype(np.float64)


def ensemble_percentiles(
    pca_pct: FloatArray, if_pct: Optional[FloatArray], weight_if: float = 0.6
) -> FloatArray:
    """Weighted average on percentile scales.

    Parameters
    ----------
    pca_pct
        PCA-based percentile scores in [0, 100].
    if_pct
        IsolationForest-based percentile scores in [0, 100], or None.
    weight_if
        Weight on IF in [0, 1]; 0 => PCA only, 1 => IF only.

    Returns
    -------
    FloatArray
        Weighted percentile scores in [0, 100].
    """
    if if_pct is None:
        return np.asarray(pca_pct, dtype=np.float64)
    w = max(0.0, min(1.0, float(weight_if)))
    p = np.asarray(pca_pct, dtype=np.float64)
    q = np.asarray(if_pct, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError("pca_pct and if_pct must have the same shape.")
    return ((1.0 - w) * p + w * q).astype(np.float64) 

class ThresholdDetector:
    """Simple z-score threshold anomaly detector.

    Flags samples where one or more features exceed mean ± 3σ (from training).
    Score is normalized to [0, 1] based on max z-score observed.
    """

    def __init__(self, z_thresh: float = 3.0):
        self.z_thresh = float(z_thresh)
        self.means_: Optional[np.ndarray] = None
        self.stds_: Optional[np.ndarray] = None
        self.max_z_: float = 1.0

    def fit(self, X_train: FloatArray) -> None:
        X = np.asarray(X_train, dtype=np.float64)
        self.means_ = X.mean(axis=0)
        self.stds_ = np.where(X.std(axis=0) == 0, 1.0, X.std(axis=0))

    def score_raw(self, X_all: FloatArray) -> FloatArray:
        if self.means_ is None or self.stds_ is None:
            raise RuntimeError("ThresholdDetector not fitted.")
        X = np.asarray(X_all, dtype=np.float64)
        zscores = np.abs((X - self.means_) / self.stds_)
        max_z = zscores.max(axis=1)
        self.max_z_ = max(self.max_z_, max_z.max())
        return (max_z / self.max_z_).clip(0, 1).astype(np.float64)

class CorrelationDetector:
    """Detect anomalies via correlation drift from the training baseline.

    - Fits baseline correlation on training data.
    - For each timestamp, computes rolling-window correlation and compares
      to baseline; the mean absolute difference is the raw score.
    - Returns a 0..1 normalized series.
    """

    def __init__(self, roll_window: int = 60):
        self.roll_window = int(roll_window)
        self.base_corr_: Optional[np.ndarray] = None
        self.n_features_: Optional[int] = None

    def fit(self, X_train: FloatArray) -> None:
        X = np.asarray(X_train, dtype=np.float64)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            base = np.corrcoef(X, rowvar=False)
        self.base_corr_ = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
        self.n_features_ = X.shape[1]

    def score_raw(self, X_all: FloatArray) -> FloatArray:
        if self.base_corr_ is None:
            raise RuntimeError("CorrelationDetector not fitted.")
        X = np.asarray(X_all, dtype=np.float64)
        n = X.shape[0]
        scores = np.zeros(n, dtype=np.float64)

        for i in range(n):
            start = max(0, i - self.roll_window // 2)
            end = min(n, i + self.roll_window // 2)
            window = X[start:end]
            if window.shape[0] < 5:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                corr = np.corrcoef(window, rowvar=False)
            corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
            diff = np.abs(corr - self.base_corr_)
            scores[i] = np.nanmean(diff)

        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        mx = scores.max()
        if mx > 0:
            scores = scores / mx
        return scores



def ensemble_multi(
    scores: List[Optional[FloatArray]], weights: Optional[List[float]] = None
) -> FloatArray:
    """Generalized ensemble fusion for multiple detectors.

    Parameters
    ----------
    scores : list of FloatArray or None
        Each detector's percentile scores in [0, 100].
    weights : list of float, optional
        Weights (default equal).

    Returns
    -------
    FloatArray
        Weighted average scores in [0, 100].
    """
    valid_scores = [np.asarray(s, dtype=np.float64) for s in scores if s is not None]
    if not valid_scores:
        return np.array([], dtype=np.float64)
    if weights is None:
        weights = [1.0] * len(valid_scores)
    else:
        weights = [max(0.0, w) for w in weights]
    weights = np.array(weights, dtype=np.float64)
    weights /= weights.sum()
    stacked = np.vstack(valid_scores)
    return (weights @ stacked).astype(np.float64)
