"""Minimal PayPal REST client for the authorize -> capture / void flow.

Sandbox only: the constructor refuses any base URL that is not a PayPal sandbox host.
`requests` is imported lazily so the rest of the project (and its tests) do not need it.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from decimal import Decimal, InvalidOperation
from typing import Any, Callable
from urllib.parse import urlsplit

SANDBOX_BASE_URL = "https://api-m.sandbox.paypal.com"
_SANDBOX_HOSTS = {"api-m.sandbox.paypal.com", "api.sandbox.paypal.com"}
_RESOURCE_ID = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")


class PayPalError(RuntimeError):
    """A PayPal API call failed. Never includes credentials."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        base = super().__str__()
        if self.status_code is None:
            return base
        return f"{base} (HTTP {self.status_code}): {self.body}"


def _money(value: float | str | Decimal) -> str:
    try:
        amount = Decimal(str(value))
        rounded = amount.quantize(Decimal("0.01"))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("Amount must be a positive number with at most two decimals.") from exc
    if not amount.is_finite() or amount <= 0 or amount != rounded:
        raise ValueError("Amount must be a positive number with at most two decimals.")
    return format(rounded, ".2f")


def _validate_resource_id(value: str) -> str:
    if not _RESOURCE_ID.fullmatch(value):
        raise ValueError("Invalid PayPal resource id.")
    return value


class PayPalClient:
    def __init__(
        self,
        client_id: str,
        client_secret: str,
        *,
        base_url: str = SANDBOX_BASE_URL,
        session: Any = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        parsed_base_url = urlsplit(base_url)
        if (
            parsed_base_url.scheme != "https"
            or parsed_base_url.hostname not in _SANDBOX_HOSTS
            or parsed_base_url.path not in ("", "/")
            or parsed_base_url.username
            or parsed_base_url.password
        ):
            raise ValueError("This project only talks to the PayPal sandbox.")
        if not client_id or not client_secret:
            raise ValueError("Missing PayPal client id or secret.")
        if session is None:
            import requests  # lazy: only needed for real network calls

            session = requests.Session()
        self._client_id = client_id
        self._client_secret = client_secret
        self._base_url = base_url.rstrip("/")
        self._session = session
        self._clock = clock
        self._token: str | None = None
        self._token_expiry = 0.0

    @classmethod
    def from_env(cls, **kwargs: Any) -> "PayPalClient":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:
            pass
        return cls(
            os.getenv("PAYPAL_CLIENT_ID", ""),
            os.getenv("PAYPAL_CLIENT_SECRET", ""),
            **kwargs,
        )

    # -- low level ---------------------------------------------------------
    def _send(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        response = self._session.request(
            method, f"{self._base_url}{path}", timeout=30, **kwargs
        )
        if not response.ok:
            raise PayPalError(f"{method} {path} failed", response.status_code, response.text)
        return response.json() if response.text else {}

    def access_token(self) -> str:
        now = self._clock()
        if self._token and now < self._token_expiry - 60:
            return self._token
        payload = self._send(
            "POST",
            "/v1/oauth2/token",
            auth=(self._client_id, self._client_secret),
            data={"grant_type": "client_credentials"},
        )
        self._token = payload["access_token"]
        self._token_expiry = now + float(payload.get("expires_in", 0))
        return self._token

    def _api(
        self, method: str, path: str, body: dict[str, Any] | None = None, request_id: str | None = None
    ) -> dict[str, Any]:
        headers = {
            "Content-Type": "application/json",
        }
        headers["Authorization"] = "Bearer " + self.access_token()
        if method.upper() == "POST":
            headers["PayPal-Request-Id"] = request_id or str(uuid.uuid4())  # idempotency
        kwargs: dict[str, Any] = {"headers": headers}
        if body is not None:
            kwargs["json"] = body
        return self._send(method, path, **kwargs)

    # -- orders ------------------------------------------------------------
    def create_authorize_order(
        self,
        amount: float | str,
        *,
        currency: str = "USD",
        return_url: str = "https://example.com/return",
        cancel_url: str = "https://example.com/cancel",
    ) -> dict[str, Any]:
        """Create an order with intent AUTHORIZE. Returns id, status, approve_url."""
        order = self._api(
            "POST",
            "/v2/checkout/orders",
            {
                "intent": "AUTHORIZE",
                "purchase_units": [
                    {"amount": {"currency_code": currency, "value": _money(amount)}}
                ],
                "payment_source": {
                    "paypal": {
                        "experience_context": {
                            "return_url": return_url,
                            "cancel_url": cancel_url,
                            "user_action": "PAY_NOW",
                        }
                    }
                },
            },
        )
        approve_url = next(
            (link["href"] for link in order.get("links", []) if link.get("rel") in ("payer-action", "approve")),
            None,
        )
        return {"id": order["id"], "status": order.get("status"), "approve_url": approve_url}

    def create_capture_order(
        self,
        amount: float | str | Decimal,
        *,
        currency: str = "USD",
        return_url: str = "http://localhost:8000/",
        cancel_url: str = "http://localhost:8000/",
    ) -> dict[str, Any]:
        """Create a PayPal Checkout order that can be captured after buyer approval."""
        order = self._api(
            "POST",
            "/v2/checkout/orders",
            {
                "intent": "CAPTURE",
                "purchase_units": [
                    {"amount": {"currency_code": currency, "value": _money(amount)}}
                ],
                "payment_source": {
                    "paypal": {
                        "experience_context": {
                            "return_url": return_url,
                            "cancel_url": cancel_url,
                            "user_action": "PAY_NOW",
                        }
                    }
                },
            },
        )
        return {"id": order["id"], "status": order.get("status")}

    def get_order(self, order_id: str) -> dict[str, Any]:
        order_id = _validate_resource_id(order_id)
        return self._api("GET", f"/v2/checkout/orders/{order_id}")

    def capture_order(
        self, order_id: str, *, request_id: str | None = None
    ) -> dict[str, Any]:
        """Capture an approved CAPTURE-intent order."""
        order_id = _validate_resource_id(order_id)
        return self._api(
            "POST",
            f"/v2/checkout/orders/{order_id}/capture",
            request_id=request_id,
        )

    def authorize_order(self, order_id: str) -> dict[str, Any]:
        """Authorize an approved order. Returns authorization id and status."""
        order_id = _validate_resource_id(order_id)
        result = self._api("POST", f"/v2/checkout/orders/{order_id}/authorize")
        try:
            authorization = result["purchase_units"][0]["payments"]["authorizations"][0]
        except (KeyError, IndexError) as exc:
            raise PayPalError("Authorize response had no authorization", body=str(result)) from exc
        return {"id": authorization["id"], "status": authorization.get("status")}

    # -- payments ----------------------------------------------------------
    def capture(
        self,
        authorization_id: str,
        amount: float | str | Decimal | None = None,
        *,
        currency: str = "USD",
        final_capture: bool = True,
    ) -> dict[str, Any]:
        """Capture all (amount=None) or part of an authorization."""
        body: dict[str, Any] = {"final_capture": final_capture}
        if amount is not None:
            body["amount"] = {"currency_code": currency, "value": _money(amount)}
        authorization_id = _validate_resource_id(authorization_id)
        return self._api("POST", f"/v2/payments/authorizations/{authorization_id}/capture", body)

    def void(self, authorization_id: str) -> dict[str, Any]:
        authorization_id = _validate_resource_id(authorization_id)
        return self._api("POST", f"/v2/payments/authorizations/{authorization_id}/void")

    def get_authorization(self, authorization_id: str) -> dict[str, Any]:
        authorization_id = _validate_resource_id(authorization_id)
        return self._api("GET", f"/v2/payments/authorizations/{authorization_id}")
