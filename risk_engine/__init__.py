"""Illustrative payment-risk scoring for the demo dashboard."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Transaction:
    transaction_id: str
    amount: float
    minutes_since_last: int
    distance_km: float
    new_device: bool
    international: bool
    velocity_1h: int


SAMPLE_TRANSACTIONS = (
    Transaction("TXN-2048", 86.40, 240, 12, False, False, 1),
    Transaction("TXN-3192", 1240.00, 3, 4120, True, True, 7),
    Transaction("TXN-5176", 318.75, 18, 840, True, False, 3),
    Transaction("TXN-6081", 42.10, 95, 4, False, False, 2),
)

def score_transaction(
    transaction: Transaction,
    *,
    amount: float | None = None,
    minutes_since_last: int | None = None,
) -> tuple[float, list[tuple[str, float]]]:
    """Return an uncalibrated illustrative score and its heuristic factors."""
    evaluated_amount = transaction.amount if amount is None else amount
    elapsed_minutes = (
        transaction.minutes_since_last
        if minutes_since_last is None
        else minutes_since_last
    )

    factors = [
        ("High transaction amount", min(max(evaluated_amount, 0) / 2000, 1) * 0.24),
        ("New device", 0.20 if transaction.new_device else 0.0),
        ("International transaction", 0.12 if transaction.international else 0.0),
        ("Unusual location distance", min(max(transaction.distance_km, 0) / 5000, 1) * 0.18),
        ("High transaction velocity", min(max(transaction.velocity_1h, 0) / 8, 1) * 0.21),
        (
            "Short time since previous transaction",
            0.15 if elapsed_minutes <= 5 else 0.08 if elapsed_minutes <= 30 else 0.0,
        ),
    ]
    score = min(0.98, 0.05 + sum(value for _, value in factors))
    return score, sorted(
        ((name, value) for name, value in factors if value > 0),
        key=lambda item: item[1],
        reverse=True,
    )

def action_for_score(score: float) -> str:
    if score < 0.24:
        return "ALLOW"
    if score < 0.48:
        return "STEP UP"
    if score < 0.70:
        return "HOLD"
    return "BLOCK"


def counterfactual_options(transaction: Transaction) -> list[dict[str, float | str]]:
    """Rank simple, simulated mitigations by score reduction per friction unit."""
    original_score, _ = score_transaction(transaction)
    candidates = [
        (
            "Verify with a step-up",
            "Add a one-time verification step.",
            original_score * 0.55,
            0.18,
        ),
        (
            "Wait 30 minutes",
            "A short delay reduces the timing signal.",
            score_transaction(transaction, minutes_since_last=30)[0],
            0.08,
        ),
        (
            "Reduce the amount by 25%",
            "Lower the transaction amount before retrying.",
            score_transaction(transaction, amount=transaction.amount * 0.75)[0],
            0.06,
        ),
    ]
    options = [
        {
            "action": label,
            "detail": detail,
            "score": score,
            "reduction": max(0.0, original_score - score),
            "friction": friction,
        }
        for label, detail, score, friction in candidates
    ]
    return sorted(
        options,
        key=lambda option: float(option["reduction"]) / float(option["friction"]),
        reverse=True,
    )


def policy_cost_series(rounds: int = 80, drift_round: int = 45) -> dict[str, list[float]]:
    """Return fixed, synthetic cost curves with an illustrative drift point."""
    policies = {
        "Allow most": (8.8, 3.8),
        "Static rules": (6.7, 5.6),
        "Step-up first": (5.9, 5.2),
        "Adaptive friction": (5.5, 4.1),
    }
    curves: dict[str, list[float]] = {}
    for index, (policy, (before, after)) in enumerate(policies.items()):
        cumulative = 0.0
        values = []
        for round_number in range(rounds):
            base_cost = before if round_number < drift_round else after
            wobble = ((round_number * (index + 3) + index * 7) % 11 - 5) * 0.11
            cumulative += max(0.1, base_cost + wobble)
            values.append(round(cumulative, 2))
        curves[policy] = values
    return curves
