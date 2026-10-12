"""Turn a risk decision into a concrete, inspectable PayPal action plan.

A plan is plain data, so the dashboard can show it as a dry run and tests can check it
without any network. `execute_plan` runs only the leading PayPal calls; steps that need
something outside PayPal (a verification, a delay) stop execution and are reported as pending.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

PAYPAL_KINDS = ("capture", "void")


@dataclass(frozen=True)
class Step:
    kind: str  # "capture" | "void" | "verify" | "wait" | "review" | "switch_funding"
    description: str
    amount: float | None = None
    final_capture: bool = True
    minutes: int | None = None

    @property
    def is_paypal_call(self) -> bool:
        return self.kind in PAYPAL_KINDS


@dataclass(frozen=True)
class PaymentPlan:
    label: str
    steps: tuple[Step, ...]

    @property
    def paypal_calls(self) -> tuple[Step, ...]:
        return tuple(step for step in self.steps if step.is_paypal_call)


def _norm(label: str) -> str:
    return " ".join(label.replace("_", " ").upper().split())


def plan_for_action(action: str, amount: float) -> PaymentPlan:
    key = _norm(action)
    if key == "ALLOW":
        return PaymentPlan("ALLOW", (Step("capture", "Capture the full authorization.", amount),))
    if key == "STEP UP":
        return PaymentPlan(
            "STEP UP",
            (
                Step("verify", "Ask the buyer for a one-time verification in our app."),
                Step("capture", "After verification passes, capture in full.", amount),
            ),
        )
    if key == "HOLD":
        return PaymentPlan(
            "HOLD",
            (
                Step("review", "Leave the authorization uncaptured while an analyst reviews the payment."),
                Step("capture", "If the review approves it, capture in full.", amount),
            ),
        )
    if key == "BLOCK":
        return PaymentPlan("BLOCK", (Step("void", "Void the authorization; nothing is charged."),))
    raise ValueError(f"Unknown action: {action!r}")


def partial_capture_plan(amount: float, fraction: float, label: str = "Partial capture") -> PaymentPlan:
    if not 0 < fraction < 1:
        raise ValueError("fraction must be between 0 and 1")
    return PaymentPlan(
        label,
        (
            Step(
                "capture",
                f"Capture {fraction:.0%} now; the remainder stays authorized for a later capture.",
                round(amount * fraction, 2),
                final_capture=False,
            ),
        ),
    )


def plan_for_change(change: Mapping[str, Any], amount: float, card6: str) -> PaymentPlan:
    """Turn a counterfactual from the engine into the PayPal steps that would carry it out.

    `change` is {"amount_mult", "delay_h", "card6"}; the order was created with intent AUTHORIZE.
    """
    steps: list[Step] = []
    if change["card6"] != card6:
        steps.append(Step("switch_funding", f"Ask the buyer to approve again, paying with {change['card6']}."))
    delay = int(change["delay_h"])
    if delay:
        steps.append(Step("wait", f"Keep the authorization and capture in {delay} hour{'s' if delay != 1 else ''}.",
                          minutes=delay * 60))
    mult = float(change["amount_mult"])
    if mult < 1:
        now = round(amount * mult, 2)
        steps += [
            Step("capture", f"Capture {mult:.0%} now; the rest stays authorized.", now, final_capture=False),
            Step("verify", "Verify the buyer before releasing the remainder."),
            Step("capture", "Capture the remainder after verification.", round(amount - now, 2)),
        ]
    else:
        steps.append(Step("capture", "Capture in full.", amount))
    return PaymentPlan("yes, if", tuple(steps))


def execute_plan(
    client: Any, authorization_id: str, plan: PaymentPlan, *, currency: str = "USD"
) -> list[dict[str, Any]]:
    """Run leading PayPal steps; stop at the first step that is not a PayPal call."""
    results: list[dict[str, Any]] = []
    for step in plan.steps:
        if not step.is_paypal_call:
            results.append({"step": step.kind, "pending": step.description})
            break
        if step.kind == "capture":
            response = client.capture(
                authorization_id, step.amount, currency=currency, final_capture=step.final_capture
            )
        else:
            response = client.void(authorization_id)
        results.append({"step": step.kind, "response": response})
    return results
