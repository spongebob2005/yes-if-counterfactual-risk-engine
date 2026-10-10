from pathlib import Path

import streamlit as st

from risk_engine import (
    SAMPLE_TRANSACTIONS,
    Transaction,
    action_for_score,
    counterfactual_options,
    policy_cost_series,
    score_transaction,
)
from src.paypal import plan_for_action, plan_for_counterfactual


ROOT = Path(__file__).resolve().parents[1]
PRIMARY_LOGO = ROOT / "files" / "yes-if_1-primary.png"
INVERSE_LOGO = ROOT / "files" / "yes-if_2-inverse.png"

st.set_page_config(
    page_title="yes, if — Payment Risk Engine",
    page_icon="🟢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #101a2b;
        --ink: #172335;
        --muted: #657386;
        --green: #0b9d7a;
        --mint: #e7f7f1;
        --paper: #f5f7fa;
        --line: #e5eaf0;
    }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: var(--navy); }
    [data-testid="stSidebar"] * { color: #f3f7fb; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 10px;
        padding: 7px 10px;
    }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1420px; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.035em; }
    .eyebrow {
        color: var(--green); text-transform: uppercase; letter-spacing: .14em;
        font-size: .72rem; font-weight: 800; margin: 0 0 8px;
    }
    .lede { color: var(--muted); font-size: 1rem; margin-top: -.4rem; }
    .panel {
        background: #fff; border: 1px solid var(--line); border-radius: 16px;
        padding: 20px 22px; box-shadow: 0 8px 24px rgba(16, 26, 43, .035);
    }
    .panel-label {
        color: var(--muted); text-transform: uppercase; letter-spacing: .1em;
        font-size: .7rem; font-weight: 750;
    }
    .big-value { color: var(--ink); font-size: 2rem; font-weight: 750; line-height: 1.1; }
    .action-pill {
        display: inline-block; background: var(--mint); color: #087d63;
        border-radius: 100px; padding: 6px 11px; font-size: .76rem;
        font-weight: 800; letter-spacing: .04em;
    }
    .notice {
        background: #fff8e8; color: #73531b; border: 1px solid #f2dfb7;
        border-radius: 12px; padding: 12px 15px; font-size: .86rem;
    }
    .option {
        background: #fff; border: 1px solid var(--line); border-radius: 13px;
        padding: 15px 17px; margin: 9px 0;
    }
    .option-title { font-size: .98rem; font-weight: 750; color: var(--ink); }
    .option-detail { font-size: .83rem; color: var(--muted); margin-top: 4px; }
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid var(--line); border-radius: 14px;
        padding: 14px 17px;
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    div[data-testid="stProgress"] > div > div { background-color: var(--green); }
    hr { border-color: var(--line); }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    if INVERSE_LOGO.exists():
        st.image(str(INVERSE_LOGO), width="stretch")
    st.markdown("### Payment Risk Engine")
    st.caption("A clearer path from risk signals to a safer yes.")
    page = st.radio(
        "Workspace",
        ("Live decision", "Counterfactual explorer", "Checkout plan", "Policy battle"),
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**DEMO MODE**")
    st.caption(
        "This prototype uses synthetic examples and hand-written heuristics. "
        "It is not a trained or validated fraud model."
    )
    st.caption("No real payment data is processed.")

if PRIMARY_LOGO.exists():
    st.image(str(PRIMARY_LOGO), width=250)

st.markdown(
    '<div class="eyebrow">Counterfactual payment risk engine · prototype</div>',
    unsafe_allow_html=True,
)


def transaction_selector() -> Transaction:
    return st.selectbox(
        "Choose a sample transaction",
        SAMPLE_TRANSACTIONS,
        format_func=lambda txn: (
            f"{txn.transaction_id}  ·  ${txn.amount:,.2f}"
        ),
    )


def risk_label(score: float) -> str:
    if score < 0.24:
        return "Lower"
    if score < 0.48:
        return "Guarded"
    if score < 0.70:
        return "Elevated"
    return "High"


if page == "Live decision":
    st.title("A decision with a way forward.")
    st.markdown(
        '<p class="lede">Explore what the demo engine would do with a sample payment — '
        "and why.</p>",
        unsafe_allow_html=True,
    )
    txn = transaction_selector()
    score, reasons = score_transaction(txn)
    action = action_for_score(score)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="panel-label">Illustrative assessment</div>', unsafe_allow_html=True)
        metric, decision = st.columns(2)
        metric.metric("Heuristic risk score", f"{score:.0%}", risk_label(score))
        decision.metric("Suggested action", action)
        st.progress(score, text="Relative score · not a calibrated probability")
        st.markdown("</div>", unsafe_allow_html=True)
        st.subheader("Signals behind this result")
        if reasons:
            for name, weight in reasons[:4]:
                st.markdown(f"- **{name}** · illustrative contribution {weight:.0%}")
        else:
            st.markdown("No elevated signals in this sample.")
    with right:
        st.subheader("Transaction snapshot")
        st.markdown(
            f"""
            <div class="panel">
            <div class="panel-label">Sample · {txn.transaction_id}</div>
            <h2 style="margin:.45rem 0 1rem">${txn.amount:,.2f}</h2>
            <div>⏱ &nbsp;{txn.minutes_since_last} min since previous transaction</div>
            <div style="margin-top:.55rem">⌖ &nbsp;{txn.distance_km:,} km from usual location</div>
            <div style="margin-top:.55rem">▦ &nbsp;{txn.velocity_1h} transactions in the last hour</div>
            <div style="margin-top:.55rem">◉ &nbsp;{"New" if txn.new_device else "Known"} device ·
            {"International" if txn.international else "Domestic"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("")
        st.markdown(
            '<div class="notice">Demo only: the score is a hand-authored heuristic, '
            "not a real-world fraud probability or production decision.</div>",
            unsafe_allow_html=True,
        )

elif page == "Counterfactual explorer":
    st.title("What would make this a yes?")
    st.markdown(
        '<p class="lede">Change amount or timing to explore how the illustrative score responds, '
        "then compare low-friction next steps.</p>",
        unsafe_allow_html=True,
    )
    txn = transaction_selector()
    controls, result = st.columns([0.9, 1.1], gap="large")
    with controls:
        st.subheader("Adjust the transaction")
        amount = st.slider(
            "Transaction amount",
            min_value=0.0,
            max_value=max(500.0, float(txn.amount) * 1.5),
            value=float(txn.amount),
            step=10.0,
            format="$%.2f",
        )
        delay = st.slider(
            "Minutes since previous transaction",
            min_value=0,
            max_value=120,
            value=min(txn.minutes_since_last, 120),
            step=1,
        )
        baseline, _ = score_transaction(txn)
        adjusted, _ = score_transaction(
            txn, amount=amount, minutes_since_last=delay
        )
        st.metric(
            "Illustrative score",
            f"{adjusted:.0%}",
            f"{(adjusted - baseline):+.0%} vs. sample",
            delta_color="inverse",
        )
        st.progress(adjusted, text="Relative score · not a calibrated probability")
    with result:
        st.subheader("Ranked “yes, if…” options")
        st.caption("Simulated score movement and relative friction assumptions.")
        for index, option in enumerate(counterfactual_options(txn), start=1):
            st.markdown(
                f"""
                <div class="option">
                  <div class="panel-label">Option {index} · score {float(option["score"]):.0%}
                  · ↓ {float(option["reduction"]):.0%}</div>
                  <div class="option-title">{option["action"]}</div>
                  <div class="option-detail">{option["detail"]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown(
            '<div class="notice">Options are explanatory demo scenarios, not causal '
            "guarantees. Only amount and timing sliders change the heuristic score.</div>",
            unsafe_allow_html=True,
        )

elif page == "Checkout plan":
    st.title("From decision to PayPal call.")
    st.markdown(
        '<p class="lede">See which PayPal sandbox calls each decision would trigger. '
        "This page is a dry run: it never contacts PayPal.</p>",
        unsafe_allow_html=True,
    )
    txn = transaction_selector()
    score, _ = score_transaction(txn)
    action = action_for_score(score)
    plan = plan_for_action(action, txn.amount)

    def render_plan(plan):
        for number, step in enumerate(plan.steps, start=1):
            where = "PayPal call" if step.is_paypal_call else "Our app"
            extra = f" · ${step.amount:,.2f}" if step.amount is not None else ""
            extra += " · remainder stays authorized" if step.kind == "capture" and not step.final_capture else ""
            st.markdown(f"{number}. **{step.kind.upper()}** ({where}){extra} — {step.description}")

    left, right = st.columns(2, gap="large")
    with left:
        st.subheader(f"Engine action: {action}")
        st.caption(f"Heuristic score {score:.0%} on ${txn.amount:,.2f}")
        render_plan(plan)
    with right:
        st.subheader("If the buyer takes a “yes, if…” step")
        for option in counterfactual_options(txn):
            st.markdown(f"**{option['action']}**")
            render_plan(plan_for_counterfactual(str(option["action"]), txn.amount))
    st.markdown(
        '<div class="notice">Assumes the order was created with intent AUTHORIZE. '
        "To exercise these calls for real against the sandbox, run "
        "<code>scripts/paypal_smoke_test.py</code>. Step-up and hold are handled by our app, "
        "not by a PayPal feature.</div>",
        unsafe_allow_html=True,
    )

else:
    st.title("When conditions shift, friction matters.")
    st.markdown(
        '<p class="lede">A fixed synthetic experiment comparing cumulative cost under '
        "four illustrative policies.</p>",
        unsafe_allow_html=True,
    )
    curves = policy_cost_series()
    drift_round = 45
    left, right = st.columns([1.3, 0.7], gap="large")
    with left:
        st.subheader("Cumulative simulated cost")
        st.line_chart(curves, height=360, y_label="Relative cost (illustrative units)")
        st.caption(
            f"Assumed environment drift begins at round {drift_round}. "
            "Lower is better within this simulation only."
        )
    with right:
        st.subheader("End-of-run comparison")
        ranked = sorted(
            ((name, values[-1]) for name, values in curves.items()),
            key=lambda pair: pair[1],
        )
        for rank, (name, value) in enumerate(ranked, start=1):
            st.markdown(
                f"""
                <div class="option">
                  <div class="panel-label">Rank {rank}</div>
                  <div class="option-title">{name}</div>
                  <div class="option-detail">Synthetic cumulative cost: {value:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown(
        '<div class="notice">Every curve is generated from fixed demo assumptions. '
        "This is not an experiment on real traffic and does not establish policy effectiveness.</div>",
        unsafe_allow_html=True,
    )

st.divider()
st.caption(
    "yes, if · Prototype for exploration only · No model training, real transaction data, "
    "or production risk decisions"
)
