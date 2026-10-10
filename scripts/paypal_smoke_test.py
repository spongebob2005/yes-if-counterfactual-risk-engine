"""PayPal sandbox smoke test for the 'yes, if' project.

Runs the flow the risk engine controls:
  create order (intent AUTHORIZE) -> buyer approves -> authorize -> capture / partial / void

Setup:
  pip install requests python-dotenv
  .env  ->  PAYPAL_CLIENT_ID=...   PAYPAL_CLIENT_SECRET=...

Run (from the repo root):
  python scripts/paypal_smoke_test.py --action partial
  python scripts/paypal_smoke_test.py --action full
  python scripts/paypal_smoke_test.py --action void
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.paypal import (  # noqa: E402
    PayPalClient,
    PayPalError,
    execute_plan,
    partial_capture_plan,
    plan_for_action,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["full", "partial", "void"], default="partial")
    parser.add_argument("--amount", type=float, default=100.0)
    parser.add_argument("--fraction", type=float, default=0.6, help="share captured now for 'partial'")
    args = parser.parse_args()

    try:
        client = PayPalClient.from_env()
        client.access_token()
        print("1. Got access token.")

        order = client.create_authorize_order(args.amount)
        print(f"2. Created order {order['id']} (status {order['status']}).")
        print(f"\n   Open this in a PRIVATE window and log in with your SANDBOX PERSONAL account:\n   {order['approve_url']}\n")
        print("   After you approve, the browser lands on example.com - that is expected.")
        input("   Press Enter here once you have approved... ")

        authorization = client.authorize_order(order["id"])
        print(f"3. Authorized: {authorization['id']} (status {authorization['status']}).")

        if args.action == "full":
            plan = plan_for_action("ALLOW", args.amount)
        elif args.action == "void":
            plan = plan_for_action("BLOCK", args.amount)
        else:
            plan = partial_capture_plan(args.amount, args.fraction)
        print(f"4. Plan '{plan.label}': " + "; ".join(step.description for step in plan.steps))
        for result in execute_plan(client, authorization["id"], plan):
            print(f"   -> {result}")

        final = client.get_authorization(authorization["id"])
        print(f"5. Authorization status now: {final['status']}")
        print("\nSmoke test complete.")
    except (PayPalError, ValueError) as error:
        sys.exit(f"\n[FAILED] {error}")


if __name__ == "__main__":
    main()
