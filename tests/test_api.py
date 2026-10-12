import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from risk_engine import MODELS_DIR, RiskEngine
from src.api.main import PRODUCTS, create_app
from src.api.store import SQLiteStore

SAMPLES = {s["key"]: s for s in json.loads((MODELS_DIR / "samples.json").read_text())}


class FakePayPal:
    def __init__(self):
        self.calls = []
        self.status = "APPROVED"
        self.amount = None

    def create_authorize_order(self, amount, **kwargs):
        self.calls.append(("create", str(amount)))
        self.amount = str(amount)
        return {"id": f"ORDER{len(self.calls)}", "status": "PAYER_ACTION_REQUIRED"}

    def get_order(self, order_id):
        return {"status": self.status, "purchase_units": [{"amount": {"currency_code": "USD", "value": self.amount}}]}

    def authorize_order(self, order_id, request_id=None):
        self.calls.append(("authorize", order_id))
        return {"id": "AUTH1", "status": "CREATED"}

    def capture(self, authorization_id, amount=None, *, final_capture=True, request_id=None):
        self.calls.append(("capture", amount, final_capture))
        return {"id": f"CAP{len(self.calls)}", "status": "COMPLETED"}

    def void(self, authorization_id, *, request_id=None):
        self.calls.append(("void", authorization_id))
        return {}


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RiskEngine.load()

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.paypal = FakePayPal()
        app = create_app(engine=self.engine, store=SQLiteStore(Path(tmp.name) / "kv.sqlite3"),
                         client_factory=lambda: self.paypal, analyst_rate_per_min=3, serve_static=False)
        self.http = TestClient(app)

    def test_health_reports_model_and_thresholds(self):
        body = self.http.get("/api/health").json()
        self.assertTrue(body["ok"])
        self.assertIn("tau_allow", body["thresholds"])

    def test_user_and_analyst_views_differ(self):
        tx = SAMPLES["large"]["transaction"]
        user = self.http.post("/api/decide", json=tx).json()
        analyst = self.http.post("/api/decide?view=analyst", json=tx).json()
        self.assertNotIn("risk", user)
        self.assertIn("risk", analyst)
        self.assertTrue(analyst["counterfactuals"][0]["paypal_plan"])
        self.assertEqual(analyst["paypal_plan"][0]["kind"] if analyst["action"] == "BLOCK" else "void", "void")

    def test_analyst_view_is_rate_limited(self):
        tx = SAMPLES["clean"]["transaction"]
        codes = [self.http.post("/api/decide?view=analyst", json=tx).status_code for _ in range(4)]
        self.assertEqual(codes, [200, 200, 200, 429])
        self.assertEqual(self.http.post("/api/decide", json=tx).status_code, 200)  # user view is not

    def test_invalid_transactions_are_rejected(self):
        self.assertEqual(self.http.post("/api/decide", json={"TransactionAmt": -5, "TransactionDT": 0}).status_code, 422)

    def test_allowed_purchase_is_authorized_and_captured_in_full(self):
        order = self.http.post("/api/checkout/orders", json={"product": "earbuds", "persona": "clean"})
        self.assertEqual(order.status_code, 201)
        self.assertEqual(self.paypal.calls[0], ("create", PRODUCTS["earbuds"]["price"]))
        done = self.http.post(f"/api/checkout/orders/{order.json()['id']}/authorize").json()
        self.assertEqual(done["status"], "COMPLETED")
        self.assertEqual(self.paypal.calls[-1], ("capture", 79.0, True))

    def test_blocked_purchase_never_reaches_paypal(self):
        quote = self.http.post("/api/checkout/quote", json={"product": "laptop", "persona": "ring"}).json()
        self.assertEqual(quote["buyer_sees"]["action"], "BLOCK")
        response = self.http.post("/api/checkout/orders", json={"product": "laptop", "persona": "ring"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.paypal.calls, [])

    def test_yes_if_offer_runs_its_plan_and_pauses_for_the_buyer(self):
        quote = self.http.post("/api/checkout/quote", json={"product": "laptop", "persona": "large"}).json()
        self.assertNotEqual(quote["buyer_sees"]["action"], "ALLOW")
        self.assertTrue(quote["buyer_sees"]["offers"])
        order = self.http.post("/api/checkout/orders", json={"product": "laptop", "persona": "large", "offer": 0}).json()
        state = self.http.post(f"/api/checkout/orders/{order['id']}/authorize").json()
        self.assertIn(state["status"], ("SCHEDULED", "VERIFY_REQUIRED", "COMPLETED", "PARTIALLY_CAPTURED"))
        for _ in range(4):
            if state["status"] in ("COMPLETED", "VOIDED"):
                break
            state = self.http.post(f"/api/checkout/orders/{order['id']}/continue",
                                   json={"code": state.get("demo_code")}).json()
        self.assertEqual(state["status"], "COMPLETED")
        captured = sum(c["amount"] for c in state["captures"])
        self.assertAlmostEqual(captured, 1299.0, places=2)

    def find(self, action):
        for persona in ("borderline", "large", "suspicious", "clean"):
            for product in PRODUCTS:
                q = self.http.post("/api/checkout/quote", json={"product": product, "persona": persona}).json()
                if q["buyer_sees"]["action"] == action:
                    return {"product": product, "persona": persona}
        self.skipTest(f"no persona/product pair gets {action} with the current model")

    def test_wrong_verification_code_is_refused(self):
        pair = self.find("STEP_UP")
        order = self.http.post("/api/checkout/orders", json=pair).json()
        state = self.http.post(f"/api/checkout/orders/{order['id']}/authorize").json()
        self.assertEqual(state["status"], "VERIFY_REQUIRED")
        bad = self.http.post(f"/api/checkout/orders/{order['id']}/continue", json={"code": "000000x"})
        self.assertEqual(bad.status_code, 422)
        good = self.http.post(f"/api/checkout/orders/{order['id']}/continue", json={"code": state["demo_code"]}).json()
        self.assertEqual(good["status"], "COMPLETED")

    def test_unapproved_or_mismatched_orders_are_refused(self):
        order = self.http.post("/api/checkout/orders", json={"product": "earbuds", "persona": "clean"}).json()
        self.paypal.status = "CREATED"
        self.assertEqual(self.http.post(f"/api/checkout/orders/{order['id']}/authorize").status_code, 409)
        self.paypal.status, self.paypal.amount = "APPROVED", "1.00"
        self.assertEqual(self.http.post(f"/api/checkout/orders/{order['id']}/authorize").status_code, 409)
        self.assertNotIn("authorize", [c[0] for c in self.paypal.calls])

    def test_unknown_order_and_product(self):
        self.assertEqual(self.http.post("/api/checkout/orders/NOPE/authorize").status_code, 404)
        self.assertEqual(self.http.post("/api/checkout/quote", json={"product": "boat", "persona": "clean"}).status_code, 404)

    def test_hold_can_be_approved_or_rejected_by_review(self):
        pair = self.find("HOLD")
        for decision, final in (("reject", "VOIDED"), ("approve", "COMPLETED")):
            order = self.http.post("/api/checkout/orders", json=pair).json()
            state = self.http.post(f"/api/checkout/orders/{order['id']}/authorize").json()
            self.assertEqual(state["status"], "IN_REVIEW")
            state = self.http.post(f"/api/checkout/orders/{order['id']}/continue", json={"review": decision}).json()
            self.assertEqual(state["status"], final)
        self.assertIn(("void", "AUTH1"), self.paypal.calls)


if __name__ == "__main__":
    unittest.main()
