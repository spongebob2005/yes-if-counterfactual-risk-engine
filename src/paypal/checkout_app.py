"""Sandbox-only PayPal Checkout API for the fixed demo product."""

from __future__ import annotations

import logging
import os
import sqlite3
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from functools import lru_cache
from pathlib import Path
from typing import Callable

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from requests import RequestException

from .client import PayPalClient, PayPalError

logger = logging.getLogger(__name__)
CHECKOUT_PAGE = Path(__file__).resolve().parents[2] / "checkout" / "index.html"


@dataclass(frozen=True)
class CheckoutSettings:
    client_id: str
    client_secret: str
    product_name: str
    amount: Decimal
    currency: str
    database_path: Path
    base_url: str

    @classmethod
    def from_env(cls) -> CheckoutSettings:
        load_dotenv()
        try:
            amount = Decimal(os.getenv("CHECKOUT_DEMO_AMOUNT", "1.00"))
            if not amount.is_finite() or amount <= 0 or amount != amount.quantize(Decimal("0.01")):
                raise ValueError
        except (InvalidOperation, ValueError) as exc:
            raise ValueError("CHECKOUT_DEMO_AMOUNT must be positive with at most two decimals.") from exc
        currency = os.getenv("CHECKOUT_DEMO_CURRENCY", "USD").upper()
        if len(currency) != 3 or not currency.isalpha():
            raise ValueError("CHECKOUT_DEMO_CURRENCY must be a three-letter currency code.")
        return cls(
            client_id=os.getenv("PAYPAL_CLIENT_ID", ""),
            client_secret=os.getenv("PAYPAL_CLIENT_SECRET", ""),
            product_name=os.getenv("CHECKOUT_DEMO_PRODUCT", "yes, if sandbox demo item"),
            amount=amount,
            currency=currency,
            database_path=Path(os.getenv("CHECKOUT_DATABASE_PATH", ".data/checkout.sqlite3")),
            base_url=os.getenv("CHECKOUT_BASE_URL", "http://localhost:8000").rstrip("/"),
        )


def create_app(
    *,
    settings: CheckoutSettings | None = None,
    client_factory: Callable[[], PayPalClient] | None = None,
) -> FastAPI:
    config = settings or CheckoutSettings.from_env()
    config.database_path.parent.mkdir(parents=True, exist_ok=True)

    def connect() -> sqlite3.Connection:
        connection = sqlite3.connect(config.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    with connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS checkout_orders (
                order_id TEXT PRIMARY KEY,
                amount TEXT NOT NULL,
                currency TEXT NOT NULL,
                status TEXT NOT NULL,
                capture_id TEXT
            )
            """
        )

    app = FastAPI(
        title="yes, if sandbox checkout",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @lru_cache(maxsize=1)
    def paypal_client() -> PayPalClient:
        if client_factory is not None:
            return client_factory()
        try:
            return PayPalClient.from_env()
        except ValueError as exc:
            raise HTTPException(
                status_code=503,
                detail="PayPal sandbox credentials are not configured.",
            ) from exc

    @app.get("/", include_in_schema=False)
    def checkout_page() -> FileResponse:
        return FileResponse(CHECKOUT_PAGE, media_type="text/html")

    @app.get("/api/config")
    def public_checkout_config() -> dict[str, str]:
        return {
            "client_id": config.client_id,
            "product_name": config.product_name,
            "amount": format(config.amount, ".2f"),
            "currency": config.currency,
            "environment": "sandbox",
        }

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ok", "environment": "sandbox"}

    @app.get("/readyz")
    def ready() -> dict[str, str]:
        if not config.client_id or not config.client_secret:
            raise HTTPException(
                status_code=503,
                detail="PayPal sandbox credentials are not configured.",
            )
        return {"status": "ready", "environment": "sandbox"}

    @app.post("/api/orders", status_code=201)
    def create_order() -> dict[str, str]:
        try:
            result = paypal_client().create_capture_order(
                config.amount,
                currency=config.currency,
                return_url=f"{config.base_url}/",
                cancel_url=f"{config.base_url}/",
            )
        except HTTPException:
            raise
        except (PayPalError, RequestException) as exc:
            logger.exception("PayPal sandbox order creation failed")
            raise HTTPException(
                status_code=502,
                detail="Could not create a PayPal sandbox order.",
            ) from exc

        order_id = result.get("id")
        if not isinstance(order_id, str) or not order_id:
            logger.error("PayPal sandbox order response did not contain an id")
            raise HTTPException(status_code=502, detail="PayPal returned an invalid order response.")

        try:
            with connect() as connection:
                connection.execute(
                    "INSERT INTO checkout_orders (order_id, amount, currency, status) VALUES (?, ?, ?, ?)",
                    (order_id, format(config.amount, ".2f"), config.currency, "CREATED"),
                )
        except sqlite3.Error as exc:
            logger.exception("Could not persist the PayPal sandbox order")
            raise HTTPException(status_code=503, detail="Could not record the checkout order.") from exc

        return {"id": order_id, "status": result.get("status", "CREATED")}

    @app.post("/api/orders/{order_id}/capture")
    def capture_order(order_id: str) -> dict[str, str | None]:
        try:
            with connect() as connection:
                stored = connection.execute(
                    "SELECT amount, currency, status, capture_id FROM checkout_orders WHERE order_id = ?",
                    (order_id,),
                ).fetchone()
        except sqlite3.Error as exc:
            logger.exception("Could not read the PayPal sandbox order")
            raise HTTPException(status_code=503, detail="Checkout storage is unavailable.") from exc
        if stored is None:
            raise HTTPException(status_code=404, detail="Unknown checkout order.")
        if stored["status"] == "COMPLETED":
            return {
                "id": order_id,
                "status": stored["status"],
                "capture_id": stored["capture_id"],
            }

        client = paypal_client()
        try:
            order = client.get_order(order_id)
            if order.get("status") != "APPROVED":
                raise HTTPException(status_code=409, detail="The PayPal order has not been approved.")
            purchase_units = order.get("purchase_units", [])
            if not purchase_units or not isinstance(purchase_units[0], dict):
                raise HTTPException(status_code=502, detail="PayPal returned an invalid order response.")
            amount = purchase_units[0].get("amount", {})
            if (
                not isinstance(amount, dict)
                or amount.get("currency_code") != stored["currency"]
                or Decimal(str(amount.get("value", "0"))) != Decimal(stored["amount"])
            ):
                raise HTTPException(status_code=409, detail="The approved order does not match this checkout.")
            result = client.capture_order(order_id, request_id=f"capture-{order_id}")
        except HTTPException:
            raise
        except (PayPalError, RequestException, InvalidOperation, ValueError) as exc:
            logger.exception("PayPal sandbox order capture failed")
            raise HTTPException(
                status_code=502,
                detail="Could not capture the PayPal sandbox order.",
            ) from exc

        status = str(result.get("status", "UNKNOWN")).upper()
        capture_id: str | None = None
        try:
            capture_id = result["purchase_units"][0]["payments"]["captures"][0]["id"]
        except (KeyError, IndexError, TypeError):
            if status == "COMPLETED":
                logger.error("Completed PayPal capture response did not contain a capture id")
                raise HTTPException(
                    status_code=502,
                    detail="PayPal returned an incomplete capture response.",
                )

        try:
            with connect() as connection:
                connection.execute(
                    "UPDATE checkout_orders SET status = ?, capture_id = ? WHERE order_id = ?",
                    (status, capture_id, order_id),
                )
        except sqlite3.Error as exc:
            logger.exception("Could not persist the PayPal sandbox capture result")
            raise HTTPException(status_code=503, detail="Could not record the capture result.") from exc

        return {"id": order_id, "status": status, "capture_id": capture_id}

    return app


app = create_app()
