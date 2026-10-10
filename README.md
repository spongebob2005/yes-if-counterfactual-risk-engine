# yes, if — Counterfactual Payment Risk Engine

A small Streamlit prototype based on the supplied build plan. It demonstrates
three interactions: a sample live decision, an adjustable counterfactual
explorer, and a synthetic policy-cost comparison.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

The project uses the logo files in `files/` directly. Keep those assets in place
when moving or deploying the app.

## Phase 1: prepare IEEE-CIS data

The build plan recommends IEEE-CIS Fraud Detection. Access is through Kaggle's
competition page and requires a Kaggle account and acceptance of that
competition's rules. After downloading and extracting the competition files,
place `train_transaction.csv` and `train_identity.csv` in `data/raw/`, then run:

```bash
python -m scripts.prepare_ieee_data
```

The preparation script joins on `TransactionID`, sorts by `TransactionDT`,
creates 60/15/10/15 chronological train/validation/calibration/test splits,
computes prior-only card/address history once across the sorted timeline,
derives row-level features, and applies category levels learned from train only.
It saves four Parquet splits and `category_levels.json` under `data/processed/`.
Downloaded and processed data are excluded from version control.

## Demo limits

All transactions, scores, actions, mitigations, and policy curves are synthetic.
The risk score is a hand-written heuristic, not a calibrated probability or
trained model. Counterfactual changes are illustrative and do not establish
causality. Do not use this prototype to approve, decline, or investigate real
payments.

## Tests

```bash
python -m unittest discover -s tests
```
