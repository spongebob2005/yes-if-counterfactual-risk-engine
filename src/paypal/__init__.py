"""PayPal sandbox integration: a small REST client and a decision-to-API gateway."""

from .client import PayPalClient, PayPalError, SANDBOX_BASE_URL
from .gateway import (
    PaymentPlan,
    Step,
    execute_plan,
    partial_capture_plan,
    plan_for_action,
    plan_for_change,
)

__all__ = [
    "PayPalClient",
    "PayPalError",
    "SANDBOX_BASE_URL",
    "PaymentPlan",
    "Step",
    "execute_plan",
    "partial_capture_plan",
    "plan_for_action",
    "plan_for_change",
]
