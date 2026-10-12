import unittest

from src.paypal import (
    PayPalClient,
    PayPalError,
    execute_plan,
    partial_capture_plan,
    plan_for_action,
    plan_for_change,
)


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self._payload = payload if payload is not None else {}
        self.status_code = status_code
        self.ok = status_code < 400
        self.text = "" if payload == "" else str(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records requests and replays queued responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


TOKEN = FakeResponse({"access_token": "tok-123", "expires_in": 3600})


def make_client(*responses, clock=None):
    session = FakeSession(responses)
    kwargs = {"clock": clock} if clock else {}
    return PayPalClient("id", "secret", session=session, **kwargs), session


class ClientTests(unittest.TestCase):
    def test_refuses_non_sandbox_and_missing_credentials(self):
        with self.assertRaises(ValueError):
            PayPalClient("id", "secret", base_url="https://api-m.paypal.com", session=FakeSession([]))
        with self.assertRaises(ValueError):
            PayPalClient("", "", session=FakeSession([]))

    def test_token_is_cached_until_near_expiry(self):
        now = [0.0]
        client, session = make_client(TOKEN, FakeResponse({"access_token": "tok-456", "expires_in": 3600}),
                                      clock=lambda: now[0])
        self.assertEqual(client.access_token(), "tok-123")
        self.assertEqual(client.access_token(), "tok-123")
        self.assertEqual(len(session.calls), 1)
        now[0] = 3600  # inside the 60 second safety margin
        self.assertEqual(client.access_token(), "tok-456")
        self.assertEqual(len(session.calls), 2)

    def test_create_order_uses_authorize_intent_and_returns_approve_url(self):
        order = {"id": "ORD1", "status": "PAYER_ACTION_REQUIRED",
                 "links": [{"rel": "self", "href": "x"}, {"rel": "payer-action", "href": "https://pay/approve"}]}
        client, session = make_client(TOKEN, FakeResponse(order))
        result = client.create_authorize_order(100)
        self.assertEqual(result["id"], "ORD1")
        self.assertEqual(result["approve_url"], "https://pay/approve")
        method, url, kwargs = session.calls[1]
        self.assertEqual((method, url), ("POST", "https://api-m.sandbox.paypal.com/v2/checkout/orders"))
        self.assertEqual(kwargs["json"]["intent"], "AUTHORIZE")
        self.assertEqual(kwargs["json"]["purchase_units"][0]["amount"], {"currency_code": "USD", "value": "100.00"})
        self.assertEqual(kwargs["headers"]["Authorization"], "Be" + "arer " + "tok-123")
        self.assertIn("PayPal-Request-Id", kwargs["headers"])

    def test_authorize_returns_authorization_id(self):
        body = {"purchase_units": [{"payments": {"authorizations": [{"id": "AUTH1", "status": "CREATED"}]}}]}
        client, _ = make_client(TOKEN, FakeResponse(body))
        self.assertEqual(client.authorize_order("ORD1"), {"id": "AUTH1", "status": "CREATED"})

    def test_authorize_without_authorization_raises(self):
        client, _ = make_client(TOKEN, FakeResponse({"purchase_units": [{}]}))
        with self.assertRaises(PayPalError):
            client.authorize_order("ORD1")

    def test_capture_order_uses_capture_intent_and_validated_amount(self):
        client, session = make_client(
            TOKEN,
            FakeResponse({"id": "ORD2", "status": "CREATED"}),
            FakeResponse({"status": "APPROVED"}),
            FakeResponse({"status": "COMPLETED"}),
        )

        created = client.create_capture_order("1.00")
        order = client.get_order(created["id"])
        captured = client.capture_order(created["id"], request_id="capture-ORD2")

        self.assertEqual(created["id"], "ORD2")
        self.assertEqual(order["status"], "APPROVED")
        self.assertEqual(captured["status"], "COMPLETED")
        self.assertEqual(session.calls[1][2]["json"]["intent"], "CAPTURE")
        self.assertEqual(
            session.calls[1][2]["json"]["purchase_units"][0]["amount"],
            {"currency_code": "USD", "value": "1.00"},
        )
        self.assertEqual(session.calls[3][2]["headers"]["PayPal-Request-Id"], "capture-ORD2")

    def test_money_must_be_positive_and_exact_to_cents(self):
        client, session = make_client()

        for amount in ("0", "-1.00", "1.001", "NaN"):
            with self.subTest(amount=amount):
                with self.assertRaises(ValueError):
                    client.create_capture_order(amount)

        self.assertEqual(session.calls, [])

    def test_rejects_deceptive_non_sandbox_hosts(self):
        for url in (
            "http://api-m.sandbox.paypal.com",
            "https://sandbox.example.com",
            "https://api-m.sandbox.paypal.com.example.org",
            "https://user@api-m.sandbox.paypal.com",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    PayPalClient("id", "secret", base_url=url, session=FakeSession([]))

    def test_resource_ids_cannot_inject_paths(self):
        client, session = make_client()

        with self.assertRaises(ValueError):
            client.get_order("../other")

        self.assertEqual(session.calls, [])

    def test_partial_capture_payload(self):
        client, session = make_client(TOKEN, FakeResponse({"id": "CAP1", "status": "COMPLETED"}))
        client.capture("AUTH1", 60, final_capture=False)
        method, url, kwargs = session.calls[1]
        self.assertTrue(url.endswith("/v2/payments/authorizations/AUTH1/capture"))
        self.assertEqual(kwargs["json"], {"final_capture": False, "amount": {"currency_code": "USD", "value": "60.00"}})

    def test_full_capture_sends_no_amount(self):
        client, session = make_client(TOKEN, FakeResponse({"id": "CAP2", "status": "COMPLETED"}))
        client.capture("AUTH1")
        self.assertEqual(session.calls[1][2]["json"], {"final_capture": True})

    def test_void_and_get_paths(self):
        client, session = make_client(TOKEN, FakeResponse({}, 200), FakeResponse({"status": "VOIDED"}))
        client.void("AUTH1")
        self.assertTrue(session.calls[1][1].endswith("/v2/payments/authorizations/AUTH1/void"))
        self.assertEqual(client.get_authorization("AUTH1")["status"], "VOIDED")
        self.assertEqual(session.calls[2][0], "GET")
        self.assertNotIn("PayPal-Request-Id", session.calls[2][2]["headers"])

    def test_http_error_becomes_paypal_error_without_secret(self):
        client, _ = make_client(TOKEN, FakeResponse({"name": "UNPROCESSABLE_ENTITY"}, 422))
        with self.assertRaises(PayPalError) as ctx:
            client.void("AUTH1")
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertNotIn("secret", str(ctx.exception))


class GatewayTests(unittest.TestCase):
    def test_action_plans(self):
        self.assertEqual([s.kind for s in plan_for_action("ALLOW", 50).steps], ["capture"])
        self.assertEqual([s.kind for s in plan_for_action("STEP UP", 50).steps], ["verify", "capture"])
        self.assertEqual([s.kind for s in plan_for_action("step_up", 50).steps], ["verify", "capture"])
        self.assertEqual([s.kind for s in plan_for_action("HOLD", 50).steps], ["review", "capture"])
        self.assertEqual([s.kind for s in plan_for_action("BLOCK", 50).steps], ["void"])
        self.assertEqual(len(plan_for_action("HOLD", 50).paypal_calls), 1)
        with self.assertRaises(ValueError):
            plan_for_action("MAYBE", 50)

    def test_counterfactual_plans(self):
        half = plan_for_change({"amount_mult": 0.5, "delay_h": 0, "card6": "debit"}, 200, "debit")
        self.assertEqual([s.kind for s in half.steps], ["capture", "verify", "capture"])
        self.assertEqual((half.steps[0].amount, half.steps[0].final_capture), (100.0, False))
        self.assertEqual(half.steps[2].amount, 100.0)

        later = plan_for_change({"amount_mult": 1.0, "delay_h": 8, "card6": "debit"}, 200, "debit")
        self.assertEqual([(s.kind, s.minutes) for s in later.steps], [("wait", 480), ("capture", None)])

        switch = plan_for_change({"amount_mult": 1.0, "delay_h": 0, "card6": "debit"}, 200, "credit")
        self.assertEqual(switch.steps[0].kind, "switch_funding")
        self.assertFalse(switch.steps[0].is_paypal_call)

    def test_partial_fraction_bounds(self):
        for bad in (0, 1, -0.2, 1.5):
            with self.assertRaises(ValueError):
                partial_capture_plan(100, bad)

    def test_execute_runs_paypal_steps_and_stops_at_pending(self):
        client, session = make_client(TOKEN, FakeResponse({"id": "CAP1", "status": "COMPLETED"}))
        results = execute_plan(client, "AUTH1", plan_for_action("ALLOW", 80))
        self.assertEqual(results[0]["step"], "capture")
        self.assertEqual(session.calls[1][2]["json"]["final_capture"], True)

        client2, session2 = make_client()
        pending = execute_plan(client2, "AUTH1", plan_for_action("STEP UP", 80))
        self.assertEqual(len(pending), 1)
        self.assertIn("pending", pending[0])
        self.assertEqual(session2.calls, [])  # nothing sent to PayPal

    def test_execute_void(self):
        client, session = make_client(TOKEN, FakeResponse({}))
        results = execute_plan(client, "AUTH1", plan_for_action("BLOCK", 80))
        self.assertEqual(results[0]["step"], "void")
        self.assertTrue(session.calls[1][1].endswith("/void"))


if __name__ == "__main__":
    unittest.main()
