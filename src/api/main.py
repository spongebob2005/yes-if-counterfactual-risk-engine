"""Serving API (M7) and the risk-aware PayPal sandbox checkout.

    uvicorn src.api.main:app --reload        # also serves the website from public/ locally

On Vercel, api/index.py exposes this app and public/ is served as static files.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from collections.abc import Callable
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from risk_engine import RiskEngine, default_engine
from src.counterfactual.search import describe
from src.paypal.client import PayPalClient, PayPalError
from src.paypal.gateway import PaymentPlan, Step, plan_for_action, plan_for_change

from .store import Store, store_from_env

try:  # local development reads .env; Vercel injects environment variables directly
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results.json"
SAMPLES = ROOT / "models" / "samples.json"
ORDER_TTL = 7 * 86400

PRODUCTS = {
    "earbuds": {"name": "Wireless earbuds", "price": "79.00"},
    "watch": {"name": "Smartwatch", "price": "199.00"},
    "headphones": {"name": "Noise-cancelling headphones", "price": "349.00"},
    "laptop": {"name": "14-inch laptop", "price": "1299.00"},
}
# Buyers are built from real test-set transactions; the product sets the amount.
PERSONAS = {
    "clean": "Returning customer on their usual phone",
    "borderline": "Regular customer, slightly out of pattern",
    "large": "Loyal customer shopping late at night",
    "suspicious": "New device, far from home, middle of the night",
    "ring": "Device already used by many other cards",
}


class Transaction(BaseModel):
    TransactionAmt: float = Field(gt=0, le=100_000)
    TransactionDT: int = Field(ge=0, le=10**9)
    ProductCD: str | None = Field(default=None, max_length=8)
    card4: str | None = Field(default=None, max_length=32)
    card6: Literal["debit", "credit"] = "debit"
    dist1: float | None = Field(default=None, ge=0, le=20_000)
    P_emaildomain: str | None = Field(default=None, max_length=64)
    DeviceType: Literal["mobile", "desktop"] | None = None
    prior_cnt: float | None = Field(default=None, ge=0)
    prior_mean_amt: float | None = Field(default=None, gt=0)
    prev_dt: float | None = Field(default=None, ge=0)
    card_device_prior: float | None = Field(default=None, ge=0)
    DeviceInfo_prior_cnt: float | None = Field(default=None, ge=0)
    DeviceInfo_distinct_uids: float | None = Field(default=None, ge=0)
    P_emaildomain_prior_cnt: float | None = Field(default=None, ge=0)
    P_emaildomain_distinct_uids: float | None = Field(default=None, ge=0)
    addr1_prior_cnt: float | None = Field(default=None, ge=0)
    addr1_distinct_uids: float | None = Field(default=None, ge=0)


class QuoteRequest(BaseModel):
    product: str
    persona: str
    offer: int | None = Field(default=None, ge=0, le=5)


class ContinueRequest(BaseModel):
    code: str | None = Field(default=None, max_length=12)
    review: Literal["approve", "reject"] | None = None


def plan_json(plan: PaymentPlan) -> list[dict[str, Any]]:
    return [{**asdict(step), "paypal_call": step.is_paypal_call} for step in plan.steps]


def client_ip(request: Request) -> str:
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def create_app(
    *,
    engine: RiskEngine | None = None,
    store: Store | None = None,
    client_factory: Callable[[], PayPalClient] | None = None,
    analyst_rate_per_min: int = 60,
    serve_static: bool | None = None,
) -> FastAPI:
    app = FastAPI(title="yes, if API", version="2.0", docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    state: dict[str, Any] = {"engine": engine, "store": store}

    def get_engine() -> RiskEngine:
        if state["engine"] is None:
            state["engine"] = default_engine()
        return state["engine"]

    def get_store() -> Store:
        if state["store"] is None:
            state["store"] = store_from_env()
        return state["store"]

    def paypal() -> PayPalClient:
        if client_factory is not None:
            return client_factory()
        if "paypal" not in state:
            try:
                state["paypal"] = PayPalClient.from_env()
            except ValueError as exc:
                raise HTTPException(503, "PayPal sandbox credentials are not configured.") from exc
        return state["paypal"]

    def rate_limit(request: Request, bucket: str, limit: int) -> None:
        key = f"rl:{bucket}:{client_ip(request)}:{int(time.time() // 60)}"
        try:
            count = get_store().hit(key, 60)
        except Exception:  # storage trouble should not take decisions down
            logger.exception("rate limiter unavailable")
            return
        if count > limit:
            raise HTTPException(429, "Too many analyst requests. Try again in a minute.")

    # ---- engine ------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        eng = get_engine()
        return {"ok": True, "model": {k: eng.meta[k] for k in ("trained_at", "data_source", "deployed_model")},
                "metrics": eng.meta["metrics"], "thresholds": eng.thresholds, "policy": eng.policy_mode}

    @app.post("/api/decide")
    def decide(tx: Transaction, request: Request, view: Literal["user", "analyst"] = "user") -> dict[str, Any]:
        """One decision. Analysts get the score, reasons, counterfactuals and PayPal plans; users a coarse message."""
        if view == "analyst":
            rate_limit(request, "analyst", analyst_rate_per_min)
        started = time.perf_counter()
        result = get_engine().decide(tx.model_dump(), view)
        if view == "analyst":
            amount = float(result["transaction"]["TransactionAmt"])
            card6 = result["transaction"]["card6"]
            result["paypal_plan"] = plan_json(plan_for_action(result["action"], amount))
            for option in result["counterfactuals"]:
                option["paypal_plan"] = plan_json(plan_for_change(option["change"], amount, card6))
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 1)
        return result

    @app.get("/api/samples")
    def samples() -> JSONResponse:
        return JSONResponse(json.loads(SAMPLES.read_text()), headers={"Cache-Control": "public, max-age=300"})

    @app.get("/api/results")
    def results() -> JSONResponse:
        return JSONResponse(json.loads(RESULTS.read_text()), headers={"Cache-Control": "public, max-age=300"})

    # ---- checkout -------------------------------------------------------------
    def persona_tx(persona: str, product: str) -> dict[str, Any]:
        if product not in PRODUCTS:
            raise HTTPException(404, "Unknown product.")
        if persona not in PERSONAS:
            raise HTTPException(404, "Unknown buyer.")
        sample = next(s for s in json.loads(SAMPLES.read_text()) if s["key"] == persona)
        return {**sample["transaction"], "TransactionAmt": float(PRODUCTS[product]["price"])}

    def quote(product: str, persona: str) -> dict[str, Any]:
        tx = persona_tx(persona, product)
        decision = get_engine().decide(tx, "analyst")
        # Only offer changes the buyer can make inside PayPal checkout: amount and timing.
        offers = [o for o in decision["counterfactuals"] if o["change"]["card6"] == tx["card6"]]
        return {"tx": tx, "decision": decision, "offers": offers}

    @app.get("/api/checkout/config")
    def checkout_config() -> dict[str, Any]:
        return {
            "client_id": os.getenv("PAYPAL_CLIENT_ID", ""),
            "currency": "USD",
            "environment": "sandbox",
            "products": [{"id": k, **v} for k, v in PRODUCTS.items()],
            "personas": [{"id": k, "label": v} for k, v in PERSONAS.items()],
        }

    @app.post("/api/checkout/quote")
    def checkout_quote(body: QuoteRequest) -> dict[str, Any]:
        q = quote(body.product, body.persona)
        d = q["decision"]
        return {
            "amount": PRODUCTS[body.product]["price"],
            "buyer_sees": {"action": d["action"], "guidance": d["guidance"],
                           "offers": [{"index": i, "label": o["label"]} for i, o in enumerate(q["offers"])]},
            "merchant_sees": {"risk": d["risk"], "action": d["action"], "top_reasons": d["top_reasons"][:3],
                              "thresholds": d["thresholds"],
                              "paypal_plan": plan_json(plan_for_action(d["action"], float(q["tx"]["TransactionAmt"])))},
        }

    @app.post("/api/checkout/orders", status_code=201)
    def create_order(body: QuoteRequest, request: Request) -> dict[str, Any]:
        rate_limit(request, "orders", 20)
        q = quote(body.product, body.persona)
        amount = float(q["tx"]["TransactionAmt"])
        if body.offer is not None:
            if body.offer >= len(q["offers"]):
                raise HTTPException(409, "That option is no longer available.")
            offer = q["offers"][body.offer]
            plan, label = plan_for_change(offer["change"], amount, q["tx"]["card6"]), offer["label"]
        elif q["decision"]["action"] == "BLOCK":
            raise HTTPException(403, "This payment couldn't be approved.")
        else:
            plan, label = plan_for_action(q["decision"]["action"], amount), q["decision"]["action"]
        base = str(request.base_url).rstrip("/")
        try:
            order = paypal().create_authorize_order(PRODUCTS[body.product]["price"], return_url=f"{base}/app#checkout",
                                                    cancel_url=f"{base}/app#checkout")
        except (PayPalError, OSError) as exc:
            logger.exception("PayPal order creation failed")
            raise HTTPException(502, "Could not create a PayPal sandbox order.") from exc
        record = {"order_id": order["id"], "amount": PRODUCTS[body.product]["price"], "currency": "USD",
                  "product": body.product, "persona": body.persona, "label": label, "plan": plan_json(plan),
                  "next_step": 0, "status": "CREATED", "authorization_id": None, "captures": [],
                  "demo_code": f"{secrets.randbelow(10**6):06d}"}
        get_store().put(f"order:{order['id']}", record, ORDER_TTL)
        return {"id": order["id"], "status": order.get("status"), "plan": record["plan"], "label": label}

    def load(order_id: str) -> dict[str, Any]:
        record = get_store().get(f"order:{order_id}")
        if record is None:
            raise HTTPException(404, "Unknown checkout order.")
        return record

    def run_steps(record: dict[str, Any], client: PayPalClient) -> dict[str, Any]:
        """Run PayPal steps until one needs the buyer, a reviewer or time; then pause there."""
        steps = record["plan"]
        while record["next_step"] < len(steps):
            index = record["next_step"]
            step = Step(**{k: v for k, v in steps[index].items() if k != "paypal_call"})
            request_id = f"{record['order_id']}-{index}"
            if step.kind == "capture":
                result = client.capture(record["authorization_id"], step.amount, final_capture=step.final_capture,
                                        request_id=request_id)
                record["captures"].append({"id": result.get("id"), "amount": step.amount, "status": result.get("status")})
            elif step.kind == "void":
                client.void(record["authorization_id"], request_id=request_id)
                record["status"] = "VOIDED"
            else:
                record["status"] = {"verify": "VERIFY_REQUIRED", "review": "IN_REVIEW", "wait": "SCHEDULED"}.get(step.kind, "PENDING")
                return record
            record["next_step"] += 1
        if record["status"] != "VOIDED":
            partial = any(c["amount"] and Decimal(str(c["amount"])) < Decimal(record["amount"]) for c in record["captures"])
            record["status"] = "COMPLETED" if not partial or len(record["captures"]) > 1 else "PARTIALLY_CAPTURED"
        return record

    def public(record: dict[str, Any]) -> dict[str, Any]:
        out = {k: record[k] for k in ("order_id", "status", "amount", "currency", "label", "plan", "next_step", "captures")}
        if record["status"] == "VERIFY_REQUIRED":
            out["demo_code"] = record["demo_code"]  # sandbox stand-in for an SMS or app prompt
        return out

    @app.post("/api/checkout/orders/{order_id}/authorize")
    def authorize(order_id: str) -> dict[str, Any]:
        record = load(order_id)
        if record["authorization_id"]:
            return public(record)
        client = paypal()
        try:
            order = client.get_order(order_id)
            if order.get("status") != "APPROVED":
                raise HTTPException(409, "The PayPal order has not been approved.")
            amount = order["purchase_units"][0]["amount"]
            if amount.get("currency_code") != record["currency"] or Decimal(str(amount.get("value"))) != Decimal(record["amount"]):
                raise HTTPException(409, "The approved order does not match this checkout.")
            record["authorization_id"] = client.authorize_order(order_id, request_id=f"{order_id}-auth")["id"]
            record["status"] = "AUTHORIZED"
            record = run_steps(record, client)
        except HTTPException:
            raise
        except (PayPalError, KeyError, IndexError, TypeError, ValueError, OSError) as exc:
            logger.exception("PayPal sandbox authorization failed")
            raise HTTPException(502, "Could not complete the PayPal sandbox payment.") from exc
        finally:
            get_store().put(f"order:{order_id}", record, ORDER_TTL)
        return public(record)

    @app.post("/api/checkout/orders/{order_id}/continue")
    def continue_order(order_id: str, body: ContinueRequest) -> dict[str, Any]:
        """Resume a paused plan: a verification code, a review decision, or (sandbox only) skipping a wait."""
        record = load(order_id)
        if not record["authorization_id"] or record["next_step"] >= len(record["plan"]):
            return public(record)
        kind = record["plan"][record["next_step"]]["kind"]
        if kind == "verify" and not secrets.compare_digest(body.code or "", record["demo_code"]):
            raise HTTPException(422, "That verification code is not right.")
        if kind == "review":
            if body.review is None:
                raise HTTPException(422, "A review decision is required.")
            if body.review == "reject":
                record["plan"] = record["plan"][: record["next_step"] + 1] + plan_json(
                    PaymentPlan("reject", (Step("void", "Reviewer rejected it: void the authorization."),)))
        record["next_step"] += 1
        try:
            record = run_steps(record, paypal())
        except (PayPalError, OSError) as exc:
            logger.exception("PayPal sandbox capture failed")
            raise HTTPException(502, "Could not complete the PayPal sandbox payment.") from exc
        finally:
            get_store().put(f"order:{order_id}", record, ORDER_TTL)
        return public(record)

    @app.get("/api/checkout/orders/{order_id}")
    def order_status(order_id: str) -> dict[str, Any]:
        return public(load(order_id))

    # ---- local website --------------------------------------------------------
    static = ROOT / "public"
    if serve_static if serve_static is not None else (not os.getenv("VERCEL") and static.is_dir()):
        from fastapi.staticfiles import StaticFiles

        from fastapi.responses import FileResponse

        @app.get("/app", include_in_schema=False)
        def app_page() -> FileResponse:
            return FileResponse(static / "app.html")

        @app.get("/docs", include_in_schema=False)
        def docs_page() -> FileResponse:
            return FileResponse(static / "docs.html")

        app.mount("/", StaticFiles(directory=static, html=True), name="site")

    return app


app = create_app()
