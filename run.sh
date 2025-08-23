#!/usr/bin/env bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/anomaly_detector.py --input data/TEP_Train_Test.csv --output outputs/TEP_with_scores.csv
