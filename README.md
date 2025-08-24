# Honeywell Hackathon – Multivariate Time-Series Anomaly Detection

## What it does
I built a PCA-based anomaly detector for multivariate time-series data.  
It learns the "normal" behavior from the **training window** (Jan 1–5, 2004),  
then scores the **analysis window** (Jan 6–10, 2004) to find anomalies.

**Output:**  
- Original dataset columns  
- + 8 new columns:  
  - `Abnormality_score` (0–100 severity)  
  - `top_feature_1 … top_feature_7` (main contributing features per row, padded with `""` if fewer than 7 contribute)

## Why this works
- **PCA (Principal Component Analysis):** captures the normal correlations among variables.  
- **Reconstruction error:** if variables deviate from their usual relationships, the error grows → anomaly.  
- **Percentile scaling (0–100):** makes scores comparable and interpretable.  
- **Top contributors:** we report the features that explain >1% of the anomaly, so you can see *why* a row is abnormal.

This covers all three anomaly types in the spec:
- Threshold violations (large deviations in a single variable)  
- Relationship changes (correlations break down)  
- Pattern deviations (temporal structure shifts)

## How to run
Requirements:
pip install -r requirements.txt

Then run:
python src/main.py --input data/TEP_Train_Test.csv --output outputs/TEP_with_scores.csv
This produces outputs/TEP_with_scores.csv with the 8 new columns.