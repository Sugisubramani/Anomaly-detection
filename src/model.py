from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


@dataclass
class AnomalyOutputs:
    scores_raw: np.ndarray
    scores_pct: np.ndarray
    err2_per_feature: np.ndarray
    top7_names: List[List[str]]


class PCADetector:
    """
    PCA reconstruction-error detector with per-feature attribution.
    - RobustScaler fit on training only
    - PCA components auto-chosen by cumulative variance threshold (var_threshold),
      capped by max_components
    - Rolling median smoothing over raw scores (roll_window minutes)
    """
    def __init__(
        self,
        max_components: int = 40,
        roll_window: int = 11,
        min_share: float = 0.01,
        var_threshold: float = 0.995,  # fraction of variance to keep (0..1)
    ):
        self.max_components = max_components
        self.roll_window = roll_window
        self.min_share = min_share
        self.var_threshold = var_threshold
        self.scaler = RobustScaler()
        self.pca: Optional[PCA] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, X_train: np.ndarray, feature_names: List[str]) -> None:
        self.feature_names = feature_names
        Xs = self.scaler.fit_transform(X_train)

        # Choose number of components to reach var_threshold, capped.
        pca_full = PCA().fit(Xs)
        cum = np.cumsum(pca_full.explained_variance_ratio_)
        ncomp = int(np.searchsorted(cum, self.var_threshold) + 1)
        ncomp = max(1, min(ncomp, min(self.max_components, Xs.shape[1])))

        self.pca = PCA(n_components=ncomp).fit(Xs)

    def _recon_err2(self, X_all: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_all)
        X_proj = self.pca.inverse_transform(self.pca.transform(Xs))
        resid = Xs - X_proj
        return resid ** 2  # per-feature squared error

    def score_raw(self, X_all: np.ndarray):
        err2 = self._recon_err2(X_all)
        raw = np.sqrt(err2.sum(axis=1))
        if self.roll_window > 1:
            raw = pd.Series(raw).rolling(self.roll_window, center=True, min_periods=1).median().to_numpy()
        return raw, err2

    @staticmethod
    def to_percentiles(raw: np.ndarray, mask_analysis: np.ndarray) -> np.ndarray:
        vals = raw[mask_analysis]
        if vals.size == 0:
            return np.zeros_like(raw)
        sorted_vals = np.sort(vals)
        ranks = np.searchsorted(sorted_vals, raw, side="right")
        pct = 100.0 * ranks / float(sorted_vals.size)
        return np.maximum(pct, 0.05)  # avoid exact zeros

    def top_k(self, err2: np.ndarray, k: int = 7) -> List[List[str]]:
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
            order = np.lexsort((names[idx], -shares[i, idx]))  # sort by share desc, tie alphabetical
            chosen = list(names[idx][order][:k])
            if len(chosen) < k:
                chosen += [""] * (k - len(chosen))
            out.append(chosen)
        return out


class IFDetector:
    """
    IsolationForest detector for an ensemble score.
    We invert decision_function (higher => more abnormal), then min-max normalize to [0,1].
    """
    def __init__(self, contamination: float = 0.003, n_estimators: int = 400, random_state: int = 42):
        self.scaler = RobustScaler()
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray) -> None:
        Xs = self.scaler.fit_transform(X_train)
        self.iforest.fit(Xs)

    def score_raw(self, X_all: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X_all)
        s = -self.iforest.decision_function(Xs)  # invert so higher = more abnormal
        # normalize to 0..1 for stability
        s = (s - s.min()) / (s.max() - s.min() + 1e-12)
        return s


def ensemble_percentiles(pca_pct: np.ndarray, if_pct: Optional[np.ndarray], weight_if: float = 0.6) -> np.ndarray:
    """
    Average of two percentile scales with a weight on IsolationForest.
    weight_if in [0,1]; 0 => PCA only, 1 => IF only.
    """
    if if_pct is None:
        return pca_pct
    w = max(0.0, min(1.0, float(weight_if)))
    return (1.0 - w) * pca_pct + w * if_pct
