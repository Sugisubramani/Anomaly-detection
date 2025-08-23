# Submission Outline

## 1. Proposed Solution
We model the normal multivariate behavior with PCA on a robustly scaled training window (Jan 1–5, 2004). For each row we compute the per-feature reconstruction error; their L2 norm is the raw anomaly score. We smooth with a 5-minute rolling median and convert scores to 0–100 using percentile ranking within the analysis period (Jan 6–10, 2004). Top contributors are features whose squared-error shares exceed 1%, ranked by magnitude (tie-break alphabetical), padded to seven.

**How it addresses the problem**
- Threshold violations -> large residuals for that tag.
- Relationship changes -> PCA captures normal correlations; breaking them increases residual.
- Pattern deviations -> multivariate structure + smoothing highlight sustained deviations.

**Innovation & uniqueness**
- Transparent, per-feature attribution from squared residuals.
- Percentile scaling yields interpretable 0–100.
- Fast, single-file, modular, PEP8-friendly.

## 2. Technical Approach
- **Tech:** Python, pandas, NumPy, scikit-learn.
- **Method:** Load → validate timestamps & interpolate → split windows → RobustScaler on training → PCA on training → per-feature residuals → smooth → percentile map (analysis) → top-7 contributors (>1%) → write CSV.
- **Flow chart:** data → preprocess → train PCA → score → percentile mapping → contributors → output.

## 3. Feasibility & Viability
- Runtime < 15 minutes on typical datasets.
- Handles missing data, constant features, single-feature edge cases, and avoids exact zeros.
- Modular code with docstrings and type hints.
