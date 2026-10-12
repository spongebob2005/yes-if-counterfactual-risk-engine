"""Streamlit dashboard (M8): the full flow, offline, from the same artifacts the API serves.

    streamlit run dashboard/app.py

The hosted website's console (/app) shows the same pages; this one needs no network.
"""

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from risk_engine import RiskEngine  # noqa: E402
from src.paypal import plan_for_action, plan_for_change  # noqa: E402

LOGO = ROOT / "files" / "yes-if_1-primary.png"
st.set_page_config(page_title="yes, if - Payment Risk Engine", page_icon=str(LOGO), layout="wide")


@st.cache_resource
def engine() -> RiskEngine:
    return RiskEngine.load()


@st.cache_data
def results() -> dict:
    return json.loads((ROOT / "reports" / "results.json").read_text())


@st.cache_data
def samples() -> list[dict]:
    return json.loads((ROOT / "models" / "samples.json").read_text())


def pick_sample(key: str) -> dict:
    options = samples()
    chosen = st.selectbox("Payment", options, key=key,
                          format_func=lambda s: f"{s['id']} · {s['title']} · ${s['transaction']['TransactionAmt']:,.2f}")
    return chosen


def render_plan(plan) -> None:
    for number, step in enumerate(plan.steps, start=1):
        where = "PayPal API" if step.is_paypal_call else "our app"
        amount = f" · ${step.amount:,.2f}" if step.amount is not None else ""
        st.markdown(f"{number}. **{step.kind.replace('_', ' ').upper()}** ({where}){amount}: {step.description}")


with st.sidebar:
    if LOGO.exists():
        st.image(str(LOGO), width=170)
    page = st.radio("Page", ("Live decision", "Counterfactual explorer", "Policy battle", "Adversary arena", "Network view"))
    st.divider()
    meta = engine().meta
    st.caption(f"Model trained {meta['trained_at']} on {meta['data_source']} data "
               f"({meta['deployed_model']}). PR-AUC {meta['metrics']['pr_auc']:.3f}.")
    st.caption("Synthetic data and simulated friction outcomes. Not for real payment decisions.")

eng = engine()
tau = eng.thresholds["tau_allow"]

if page == "Live decision":
    st.title("Score it, explain it, find the yes.")
    sample = pick_sample("decision")
    result = eng.decide(sample["transaction"], "analyst")
    tx = result["transaction"]
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        a, b = st.columns(2)
        a.metric("Fraud probability", f"{result['risk']:.1%}", help=f"Approval line {tau:.2%}")
        b.metric("Action", result["action"].replace("_", " "))
        st.caption(f"Buyer sees: {result['guidance']}")
        st.subheader("Why")
        for reason in result["top_reasons"]:
            arrow = "▲" if reason["direction"] == "raises" else "▼"
            st.markdown(f"{arrow} **{reason['reason']}** ({reason['impact']:+.2f} log-odds)")
        st.subheader("PayPal plan")
        render_plan(plan_for_action(result["action"], tx["TransactionAmt"]))
    with right:
        st.subheader("yes, if…")
        if result["action"] == "ALLOW":
            st.success("Already approved: no change needed.")
        elif not result["counterfactuals"]:
            st.info("No allowed change gets this under the approval line.")
        for option in result["counterfactuals"]:
            with st.container(border=True):
                st.markdown(f"**{option['label']}**  \nrisk {option['risk']:.2%} · change cost {option['cost']:.2f}")
                render_plan(plan_for_change(option["change"], tx["TransactionAmt"], tx["card6"]))

elif page == "Counterfactual explorer":
    st.title("What would make this a yes?")
    sample = pick_sample("explorer")
    base = eng.complete(sample["transaction"])
    controls, out = st.columns([1, 1], gap="large")
    with controls:
        paid = st.slider("Paid now (%)", 10, 100, 100, 5)
        delay = st.slider("Goes through later (hours)", 0, 24, 0)
        card6 = st.radio("Funding source", ("debit", "credit"), index=0 if base["card6"] == "debit" else 1, horizontal=True)
    changed = dict(base, TransactionAmt=round(base["TransactionAmt"] * paid / 100, 2),
                   TransactionDT=base["TransactionDT"] + delay * 3600, card6=card6)
    before, after = eng.score([base, changed])
    with out:
        a, b = st.columns(2)
        a.metric("Original", f"{before:.2%}", eng.choose_action(before, base))
        b.metric("With your changes", f"{after:.2%}", f"{after - before:+.2%}", delta_color="inverse")
        st.caption(f"Approval line {tau:.2%}. Counterfactuals describe the model, not causal truth.")
        for option in eng.counterfactuals(base):
            st.markdown(f"- yes, if the amount is {option['change']['amount_mult']:.0%}, "
                        f"{option['change']['delay_h']} h later, by {option['change']['card6']} → {option['risk']:.2%}")

elif page == "Policy battle":
    st.title("When the world changes, friction matters.")
    e5 = results()["E5_policy"]
    st.line_chart({name: p["curve"] for name, p in e5["policies"].items()}, height=380)
    st.caption(f"Step-up's catch rate drops from 85% to 40% at payment {e5['drift_index']:,} of {e5['stream_length']:,}.")
    st.dataframe([{"policy": k, "total": v["total"], "after drift": v["after_drift"], **v["breakdown"]}
                  for k, v in sorted(e5["policies"].items(), key=lambda kv: kv[1]["after_drift"])], hide_index=True)

elif page == "Adversary arena":
    st.title("An attacker that keeps probing.")
    r = results()
    left, right = st.columns(2)
    with left:
        st.subheader("Evasion by retraining round")
        st.bar_chart({f"Round {x['round']}": x["evasion"] for x in r["E6_adversary"]["rounds"]})
        st.caption("Clean PR-AUC: " + ", ".join(f"{x['pr_auc']:.3f}" for x in r["E6_adversary"]["rounds"]))
    with right:
        st.subheader("Evasion vs query budget")
        e7 = r["E7_defense"]
        st.line_chart({"open scores": e7["open_scores"], "coarse output": e7["coarse_output"],
                       "coarse + rate limit": e7["rate_limited"]})
        st.caption("Budgets: " + ", ".join(map(str, e7["budgets"])))

else:
    st.title("One device, many cards.")
    ring = results()["ring_example"]
    st.markdown(f"**{ring['device']}**: {len({e['card'] for e in ring['edges']})} distinct cards, "
                f"{len(ring['edges'])} payments, {sum(e['fraud'] for e in ring['edges'])} fraud.")
    st.dataframe(ring["edges"], hide_index=True)
