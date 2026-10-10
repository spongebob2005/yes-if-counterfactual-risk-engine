![yes, if — The cheapest way to yes](files/yes-if_1-primary.png)

<h1 align="center">Counterfactual Payment Risk Engine</h1>

<p align="center">
  A risk-exploration demo that asks: <strong>what is the smallest, lowest-friction step that could turn a risky payment into a yes?</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/status-prototype-0b9d7a" alt="Prototype">
  <img src="https://img.shields.io/badge/Python-3.10%2B-172335" alt="Python 3.10 or newer">
  <img src="https://img.shields.io/badge/UI-Streamlit-ff4b4b" alt="Streamlit">
</p>

---

## The idea

Traditional payment decisions can stop at a risk score. **yes, if** explores the next question: what practical change might lower the score, and how much friction would it add?

This branded Streamlit prototype brings that idea to life with synthetic examples. It is designed to make the concept tangible—not to make real payment decisions.

## Explore the demo

| Workspace | What you can explore |
| --- | --- |
| **Live decision** | Inspect an illustrative transaction score, suggested action, and the signals behind it. |
| **Counterfactual explorer** | Adjust transaction amount and timing, then compare simulated “yes, if…” options. |
| **Checkout plan** | See which PayPal sandbox calls each decision (and each “yes, if…” step) would trigger. A dry run: it never contacts PayPal. |
| **Policy battle** | Compare synthetic cumulative-cost curves across four illustrative policies. |

## Run locally

**Requirements:** Python 3.10 or newer.

```bash
git clone https://github.com/spongebob2005/yes-if-counterfactual-risk-engine.git
cd yes-if-counterfactual-risk-engine

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Streamlit prints a local URL when the dashboard is ready.

## Optional: prepare IEEE-CIS data

The included preparation pipeline is a separate research step; the dashboard itself runs on built-in synthetic examples. To use the pipeline, download the IEEE-CIS Fraud Detection files from Kaggle (Kaggle account and competition rules apply), then install its additional data dependencies:

```bash
python -m pip install pandas numpy pyarrow
mkdir -p data/raw
# Place train_transaction.csv and train_identity.csv in data/raw/
python -m scripts.prepare_ieee_data
```

The script joins the files by `TransactionID`, sorts transactions chronologically, builds prior-only history features, and creates 60/15/10/15 train, validation, calibration, and test splits. Processed Parquet files and category levels are written to `data/processed/`; raw and processed data are excluded from version control.

## Optional: PayPal sandbox smoke test

The risk engine is designed to sit between PayPal order approval and payment capture. This standalone script exercises the PayPal **sandbox** flow it will control: create an order with intent `AUTHORIZE`, approve it as a sandbox buyer, authorize, then capture, partially capture, or void. It is not yet wired into the dashboard, and it only talks to the sandbox.

1. Create a free account at [developer.paypal.com](https://developer.paypal.com) and open **Apps & Credentials** in **Sandbox** mode.
2. Copy `.env.example` to `.env` and fill in your sandbox Client ID and Secret. `.env` is git-ignored; never commit real keys.
3. Install the extra dependencies and run:

```bash
python -m pip install requests python-dotenv
python scripts/paypal_smoke_test.py --action partial   # capture 60% now; the rest stays authorized
python scripts/paypal_smoke_test.py --action void      # void the authorization
python scripts/paypal_smoke_test.py --action full      # capture in full
```

The script prints an approval link. Open it in a private window and sign in with your **sandbox personal** account, then press Enter in the terminal.

### How decisions map to PayPal calls

The client lives in `src/paypal/client.py` (sandbox only; it refuses non-sandbox URLs) and the decision-to-call mapping in `src/paypal/gateway.py`. Orders must be created with intent `AUTHORIZE` so every option below stays available.

| Engine decision | PayPal / app behavior |
| --- | --- |
| ALLOW | Capture the full authorization |
| “Reduce the amount by 25%” | Partial capture (`final_capture: false`); the remainder stays authorized |
| “Wait 30 minutes” | Keep the authorization, then re-score and capture later |
| STEP UP | Our app asks for a verification, then captures (a PayPal call is only made after it passes) |
| HOLD | Leave the authorization uncaptured while the payment is reviewed |
| BLOCK | Void the authorization |

Step-up and hold are app-level behavior, not PayPal features. The PayPal calls are covered by mocked unit tests (`tests/test_paypal.py`); only the smoke test talks to the real sandbox.

## Run tests

```bash
python -m unittest discover -s tests
```

## Important: demo only

Every transaction, score, action, mitigation, and policy curve shown in the dashboard is synthetic. The score is based on hand-written heuristics; it is **not** a trained model, calibrated probability, causal estimate, or validated fraud signal. Do not use this prototype to approve, decline, or investigate real payments.

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">
  <sub>yes, if · The cheapest way to yes · Prototype for exploration only</sub>
</p>
