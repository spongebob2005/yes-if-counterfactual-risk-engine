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
st.set_page_config(
    page_title="yes, if - Payment Risk Engine",
    page_icon=str(PRIMARY_LOGO),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --page: #f3f7f6;
        --surface: #fbfdfc;
        --ink: #172335;
        --muted: #536575;
        --accent: #0b927b;
        --accent-strong: #07594f;
        --accent-soft: #e2f1ed;
        --line: #d7e2df;
        --warning: #73531b;
        --warning-soft: #fff5d9;
        --radius-surface: 14px;
        --radius-control: 8px;
        --radius-pill: 999px;
    }
    .stApp {
        background: var(--page);
        color: var(--ink);
        font-family: "Aptos", "Segoe UI", system-ui, sans-serif;
    }
    [data-testid="stSidebar"] {
        background: var(--surface);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] label {
        color: var(--ink);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: var(--radius-control);
        padding: 8px 10px;
        color: var(--ink);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: var(--accent-soft);
    }
    .block-container { padding: 2.4rem clamp(1.25rem, 4vw, 3.5rem) 3.5rem; max-width: 1440px; }
    h1, h2, h3 { color: var(--ink); letter-spacing: -0.04em; }
    h1 { font-size: clamp(2rem, 3.2vw, 2.7rem); line-height: 1.12; }
    h2, h3 { line-height: 1.2; }
    .eyebrow {
        color: var(--accent-strong); text-transform: uppercase; letter-spacing: .1em;
        font-size: .72rem; font-weight: 800; margin: .5rem 0 .65rem;
    }
    .lede { color: var(--muted); font-size: 1rem; line-height: 1.65; margin-top: -.35rem; max-width: 68ch; }
    .panel {
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-surface);
        padding: 21px 23px; box-shadow: 0 12px 32px rgba(20, 53, 46, .045);
    }
    .panel-label {
        color: var(--muted); text-transform: uppercase; letter-spacing: .1em;
        font-size: .7rem; font-weight: 750;
    }
    .big-value { color: var(--ink); font-size: 2.25rem; font-weight: 750; line-height: 1.05; font-variant-numeric: tabular-nums; }
    .action-pill {
        display: inline-block; background: var(--accent-soft); color: var(--accent-strong);
        border-radius: var(--radius-pill); padding: 6px 11px; font-size: .76rem;
        font-weight: 800; letter-spacing: .04em;
    }
    .notice {
        background: var(--warning-soft); color: var(--warning); border: 1px solid #d9c994;
        border-radius: var(--radius-control); padding: 13px 16px; font-size: .86rem; line-height: 1.55;
    }
    .option {
        border-top: 1px solid var(--line);
        padding: 16px 2px 15px;
    }
    .option-title { font-size: 1rem; font-weight: 750; color: var(--ink); }
    .option-detail { font-size: .84rem; color: var(--muted); margin-top: 5px; line-height: 1.5; }
    .option-metrics { display: flex; flex-wrap: wrap; gap: 8px 18px; margin-top: 9px; color: var(--muted); font-size: .78rem; }
    [data-testid="stMetric"] {
        background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius-control);
        padding: 15px 17px;
    }
    [data-testid="stMetricLabel"], [data-testid="stMetricDelta"] { color: var(--muted); }
    [data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color: var(--ink); font-weight: 650; }
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSlider"] [data-baseweb="slider"] { border-radius: var(--radius-control); }
    [data-testid="stButton"] button {
        min-height: 42px;
        border-radius: var(--radius-control);
        border-color: var(--accent-strong);
        color: var(--accent-strong);
        font-weight: 700;
        transition: background-color .18s ease, transform .18s ease;
    }
    [data-testid="stButton"] button:hover { background: var(--accent-soft); color: var(--accent-strong); }
    [data-testid="stButton"] button:active { transform: scale(.98); }
    hr { border-color: var(--line); }
    .snapshot-grid { display: grid; gap: 12px; margin-top: 16px; }
    .snapshot-item { display: flex; justify-content: space-between; gap: 16px; padding-top: 10px; border-top: 1px solid var(--line); }
    .snapshot-label { color: var(--muted); }
    .snapshot-value { color: var(--ink); font-weight: 650; text-align: right; }
    @media (max-width: 760px) {
        .block-container { padding: 1.5rem 1rem 2.5rem; }
        .panel { padding: 17px; }
        [data-testid="stSidebar"] { border-right: 0; border-bottom: 1px solid var(--line); }
        .snapshot-item { align-items: flex-start; }
    }
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after { scroll-behavior: auto !important; transition-duration: .01ms !important; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
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
    st.image(str(PRIMARY_LOGO), width=190)

st.markdown(
    '<div class="eyebrow">Counterfactual payment risk engine | prototype</div>',
    unsafe_allow_html=True,
)


def transaction_selector() -> Transaction:
    return st.selectbox(
        "Choose a sample transaction",
        SAMPLE_TRANSACTIONS,
        format_func=lambda txn: (
            f"{txn.transaction_id}  |  ${txn.amount:,.2f}"
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
        '<p class="lede">Explore how the demo engine scores a sample payment, and why.</p>',
        unsafe_allow_html=True,
    )
    txn = transaction_selector()
    score, reasons = score_transaction(txn)
    action = action_for_score(score)

    left, right = st.columns([1.15, 0.85], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="panel-label">Illustrative assessment</div>', unsafe_allow_html=True)
            metric, decision = st.columns(2)
            metric.metric("Heuristic risk score", f"{score:.0%}")
            decision.metric("Suggested action", action)
            metric.caption(f"Risk band: {risk_label(score)}")
            st.caption("This hand-written score is not a calibrated probability.")
        st.subheader("Signals behind this result")
        if reasons:
            for name, weight in reasons[:4]:
                st.markdown(f"- **{name}**: illustrative contribution {weight:.0%}")
        else:
            st.markdown("No elevated signals in this sample.")
    with right:
        st.subheader("Transaction snapshot")
        st.markdown(
            f"""
            <div class="panel">
            <div class="panel-label">Sample | {txn.transaction_id}</div>
            <h2 style="margin:.45rem 0 1rem">${txn.amount:,.2f}</h2>
            <div class="snapshot-grid">
              <div class="snapshot-item"><span class="snapshot-label">Time since previous</span><span class="snapshot-value">{txn.minutes_since_last} min</span></div>
              <div class="snapshot-item"><span class="snapshot-label">Distance from usual location</span><span class="snapshot-value">{txn.distance_km:,} km</span></div>
              <div class="snapshot-item"><span class="snapshot-label">Transactions in the last hour</span><span class="snapshot-value">{txn.velocity_1h}</span></div>
              <div class="snapshot-item"><span class="snapshot-label">Device and region</span><span class="snapshot-value">{"New" if txn.new_device else "Known"} | {"International" if txn.international else "Domestic"}</span></div>
            </div>
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
        with st.container(border=True):
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
        before, after = st.columns(2)
        before.metric("Sample baseline", f"{baseline:.0%}")
        after.metric(
            "Adjusted score",
            f"{adjusted:.0%}",
            f"{(adjusted - baseline):+.0%} vs. sample",
            delta_color="inverse",
        )
        st.caption("Both scores are illustrative. The change is not a causal prediction.")
    with result:
        st.subheader("Ranked “yes, if…” options")
        st.caption("Simulated score movement and relative friction assumptions.")
        for index, option in enumerate(counterfactual_options(txn), start=1):
            st.markdown(
                f"""
                <div class="option">
                  <div class="panel-label">Ranked option {index}</div>
                  <div class="option-title">{option["action"]}</div>
                  <div class="option-detail">{option["detail"]}</div>
                  <div class="option-metrics">
                    <span>Simulated score {float(option["score"]):.0%}</span>
                    <span>Reduction {float(option["reduction"]):.0%}</span>
                    <span>Relative friction {float(option["friction"]):.0%}</span>
                  </div>
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
            extra = f" | ${step.amount:,.2f}" if step.amount is not None else ""
            extra += " | remainder stays authorized" if step.kind == "capture" and not step.final_capture else ""
            st.markdown(f"{number}. **{step.kind.upper()}** ({where}){extra}: {step.description}")

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
