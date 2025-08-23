@echo off
python -m venv .venv
call .venv\Scripts\activate
pip install -r requirements.txt
python src\main.py --input data\TEP_Train_Test.csv --output outputs\TEP_with_scores.csv --use-ensemble
