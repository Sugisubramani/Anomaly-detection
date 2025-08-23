# Honeywell Hackathon – Multivariate Time-Series Anomaly Detection

## Quick Start
```bash
# 1) Create & activate virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Run
python src/main.py --input data/TEP_Train_Test.csv --output outputs/TEP_with_scores.csv
```

## What it does
- Trains on 2004-01-01 00:00 to 2004-01-05 23:59 (normal period).
- Scores 2004-01-06 00:00 to 2004-01-10 07:59 (analysis).
- Outputs 8 new columns: Abnormality_score + top_feature_1..7.

See `docs/outline.md` for the write-up you can paste into the submission form.

