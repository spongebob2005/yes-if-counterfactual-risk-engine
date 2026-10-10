import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from fastapi.testclient import TestClient

from src.paypal.checkout_app import CheckoutSettings, create_app


class FakePayPalClient:
    def __init__(self):
        self.create_calls = []
        self.capture_calls = []
        self.order = {
            "status": "APPROVED",
            "purchase_units": [
                {"amount": {"currency_code": "USD", "value": "1.00"}}
            ],
        }
        self.capture_result = {
            "status": "COMPLETED",
            "purchase_units": [
                {"payments": {"captures": [{"id": "CAPTURE-1"}]}}
            ],
        }

    def create_capture_order(self, amount, **kwargs):
        self.create_calls.append((amount, kwargs))
        return {"id": "ORDER-1", "status": "CREATED"}

    def get_order(self, order_id):
        return self.order

    def capture_order(self, order_id, *, request_id=None):
        self.capture_calls.append((order_id, request_id))
        return self.capture_result


class CheckoutApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.client = FakePayPalClient()
        settings = CheckoutSettings(
            client_id="sandbox-public-id",
            client_secret="sandbox-secret",
            product_name="yes, if sandbox item",
            amount=Decimal("1.00"),
            currency="USD",
            database_path=Path(self.temp_dir.name) / "checkout.sqlite3",
            base_url="http://localhost:8000",
        )
        app = create_app(settings=settings, client_factory=lambda: self.client)
        self.http = TestClient(app)

    def test_config_exposes_public_settings_not_secret(self):
        response = self.http.get("/api/config")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["client_id"], "sandbox-public-id")
        self.assertEqual(response.json()["amount"], "1.00")
        self.assertEqual(response.json()["environment"], "sandbox")
        self.assertNotIn("client_secret", response.json())

    def test_order_amount_is_fixed_on_server(self):
        response = self.http.post("/api/orders", json={"amount": "9999.99"})

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["id"], "ORDER-1")
        amount, kwargs = self.client.create_calls[0]
        self.assertEqual(amount, Decimal("1.00"))
        self.assertEqual(kwargs["currency"], "USD")

    def test_capture_checks_paypal_approval_and_is_idempotent_after_completion(self):
        self.http.post("/api/orders")

        first = self.http.post("/api/orders/ORDER-1/capture")
        second = self.http.post("/api/orders/ORDER-1/capture")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.json()["status"], "COMPLETED")
        self.assertEqual(first.json()["capture_id"], "CAPTURE-1")
        self.assertEqual(second.json(), first.json())
        self.assertEqual(self.client.capture_calls, [("ORDER-1", "capture-ORDER-1")])

    def test_capture_rejects_unapproved_order(self):
        self.http.post("/api/orders")
        self.client.order["status"] = "CREATED"

        response = self.http.post("/api/orders/ORDER-1/capture")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.capture_calls, [])

    def test_capture_rejects_amount_mismatch(self):
        self.http.post("/api/orders")
        self.client.order["purchase_units"][0]["amount"]["value"] = "999.00"

        response = self.http.post("/api/orders/ORDER-1/capture")

        self.assertEqual(response.status_code, 409)
        self.assertEqual(self.client.capture_calls, [])

    def test_capture_rejects_unknown_order(self):
        response = self.http.post("/api/orders/UNKNOWN/capture")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.capture_calls, [])

    def test_capture_does_not_report_success_for_incomplete_paypal_response(self):
        self.http.post("/api/orders")
        self.client.capture_result = {"status": "COMPLETED", "purchase_units": []}

        response = self.http.post("/api/orders/ORDER-1/capture")

        self.assertEqual(response.status_code, 502)
        self.assertEqual(
            response.json()["detail"],
            "PayPal returned an incomplete capture response.",
        )

    def test_readiness_reports_missing_credentials(self):
        settings = CheckoutSettings(
            client_id="",
            client_secret="",
            product_name="demo",
            amount=Decimal("1.00"),
            currency="USD",
            database_path=Path(self.temp_dir.name) / "missing-creds.sqlite3",
            base_url="http://localhost:8000",
        )
        app = create_app(settings=settings, client_factory=lambda: self.client)

        response = TestClient(app).get("/readyz")

        self.assertEqual(response.status_code, 503)


if __name__ == "__main__":
    unittest.main()
