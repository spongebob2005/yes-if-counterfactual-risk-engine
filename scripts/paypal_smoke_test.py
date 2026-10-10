"""PayPal sandbox smoke test for the 'yes, if' project.

Proves the exact flow the risk engine will control:
  create order (intent AUTHORIZE) -> buyer approves -> authorize
  -> then ONE of: full capture / partial capture / void

Setup:
  pip install requests python-dotenv
  .env  ->  PAYPAL_CLIENT_ID=...   PAYPAL_CLIENT_SECRET=...
Run:
  python paypal_smoke_test.py --action partial     # capture 60% now
  python paypal_smoke_test.py --action full
  python paypal_smoke_test.py --action void
"""
import argparse
import os
import sys
import uuid

import requests
from dotenv import load_dotenv

BASE = "https://api-m.sandbox.paypal.com"   # sandbox only
load_dotenv()
CLIENT_ID = os.getenv("PAYPAL_CLIENT_ID")
CLIENT_SECRET = os.getenv("PAYPAL_CLIENT_SECRET")


def check(resp, what):
    if not resp.ok:
        print(f"\n[FAILED] {what}: HTTP {resp.status_code}\n{resp.text}")
        sys.exit(1)
    return resp.json() if resp.text else {}


def get_token():
    r = requests.post(f"{BASE}/v1/oauth2/token", auth=(CLIENT_ID, CLIENT_SECRET),
                      data={"grant_type": "client_credentials"}, timeout=30)
    return check(r, "get access token")["access_token"]


def api(token, method, path, body=None):
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json",
               "PayPal-Request-Id": str(uuid.uuid4())}   # idempotency key
    r = requests.request(method, f"{BASE}{path}", headers=headers, json=body, timeout=30)
    return check(r, f"{method} {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--action", choices=["full", "partial", "void"], default="partial")
    ap.add_argument("--amount", default="100.00")
    args = ap.parse_args()

    if not CLIENT_ID or not CLIENT_SECRET:
        sys.exit("Missing PAYPAL_CLIENT_ID / PAYPAL_CLIENT_SECRET in .env")

    token = get_token()
    print("1. Got access token.")

    order = api(token, "POST", "/v2/checkout/orders", {
        "intent": "AUTHORIZE",
        "purchase_units": [{"amount": {"currency_code": "USD", "value": args.amount}}],
        "payment_source": {"paypal": {"experience_context": {
            "return_url": "https://example.com/return",
            "cancel_url": "https://example.com/cancel",
            "user_action": "PAY_NOW"}}},
    })
    order_id = order["id"]
    link = next((l["href"] for l in order["links"]
                 if l["rel"] in ("payer-action", "approve")), None)
    print(f"2. Created order {order_id} (status {order['status']}).")
    print(f"\n   Open this in a PRIVATE window and log in with your SANDBOX PERSONAL account:\n   {link}\n")
    print("   After you approve, the browser lands on example.com - that is expected.")
    input("   Press Enter here once you have approved... ")

    auth = api(token, "POST", f"/v2/checkout/orders/{order_id}/authorize")
    authorization = auth["purchase_units"][0]["payments"]["authorizations"][0]
    auth_id = authorization["id"]
    print(f"3. Authorized. authorization id = {auth_id} (status {authorization['status']}).")

    if args.action == "full":
        cap = api(token, "POST", f"/v2/payments/authorizations/{auth_id}/capture",
                  {"final_capture": True})
        print(f"4. Full capture: {cap['status']}")
    elif args.action == "partial":
        part = f"{float(args.amount) * 0.6:.2f}"
        cap = api(token, "POST", f"/v2/payments/authorizations/{auth_id}/capture",
                  {"amount": {"currency_code": "USD", "value": part}, "final_capture": False})
        print(f"4. Partial capture of {part}: {cap['status']} (remainder still authorized)")
    else:
        api(token, "POST", f"/v2/payments/authorizations/{auth_id}/void")
        print("4. Authorization voided.")

    final = api(token, "GET", f"/v2/payments/authorizations/{auth_id}")
    print(f"5. Authorization status now: {final['status']}")
    print("\nSmoke test complete.")


if __name__ == "__main__":
    main()
