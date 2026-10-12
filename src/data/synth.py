"""Reproducible synthetic card transactions in the IEEE-CIS column layout.

Public data has no device graph we can show and IEEE-CIS needs a Kaggle account, so the
live engine is trained on this generator. Every pattern below is an explicit assumption:

- Legitimate cardholders have a home region, one or two personal devices, a usual spend
  level and usual hours. Some legitimately travel, buy something big or use a new device.
- Account takeover: a stolen card used from a fraudster device, often at night, for well
  above the cardholder's usual amount. Credit cards are targeted more often than debit.
- Card testing: a burst of tiny charges seconds apart, then one or two large ones.
- Rings: one or two devices and an email domain shared across several stolen cards.
- "Stealth" fraud runs on the victim's own compromised device; only an unusual amount or
  hour gives it away, so it overlaps heavily with legitimate big or late-night purchases.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAY = 86_400
EMAILS = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com", "comcast.net"]
EMAIL_P = [0.45, 0.15, 0.12, 0.10, 0.10, 0.04, 0.04]
FRAUD_EMAILS = ["protonmail.com", "mail.ru", "anonmail.net", "yandex.ru", "gmail.com"]
PRODUCTS = ["W", "C", "R", "H", "S"]
PRODUCT_P = [0.60, 0.15, 0.10, 0.10, 0.05]
NETWORKS = ["visa", "mastercard", "american express", "discover"]
NETWORK_P = [0.60, 0.30, 0.07, 0.03]


def _device(rng: np.random.Generator) -> str:
    return "dev-" + rng.bytes(5).hex()


def _hour_seconds(rng: np.random.Generator, hours: np.ndarray) -> np.ndarray:
    return (np.mod(hours, 24) * 3600).astype(np.int64) + rng.integers(0, 3600, len(hours))


def generate(
    *,
    cards: int = 6000,
    days: int = 180,
    tx_per_card_per_day: float = 0.15,
    fraud_rate: float = 0.035,
    seed: int = 42,
) -> pd.DataFrame:
    """Return a time-sorted frame with IEEE-CIS style columns plus `isFraud` and `fraud_kind`."""
    rng = np.random.default_rng(seed)
    horizon = days * DAY
    rows: list[dict] = []

    card_ids = rng.choice(np.arange(1000, 1000 + cards * 3), size=cards, replace=False)
    regions = rng.zipf(1.6, cards) % 200 + 100
    networks = rng.choice(NETWORKS, cards, p=NETWORK_P)
    funding = rng.choice(["debit", "credit"], cards, p=[0.62, 0.38])
    emails = rng.choice(EMAILS, cards, p=EMAIL_P)
    device_types = rng.choice(["mobile", "desktop"], cards, p=[0.55, 0.45])
    mean_amt = np.clip(rng.lognormal(np.log(60), 0.8, cards), 3, 3000)
    night_owl = rng.random(cards) < 0.08
    peak_hour = np.where(night_owl, rng.normal(1, 2, cards), rng.normal(14, 3, cards))
    devices = [[_device(rng) for _ in range(rng.integers(1, 3))] for _ in range(cards)]

    def legit(i: int, n: int) -> None:
        times = np.sort(rng.integers(0, horizon, n))
        days_ = times // DAY * DAY
        hours = rng.normal(peak_hour[i], 3, n)
        stamp = days_ + _hour_seconds(rng, hours)
        amounts = np.round(mean_amt[i] * rng.lognormal(0, 0.5, n), 2)
        big = rng.random(n) < 0.04  # occasional large purchase
        amounts[big] *= rng.uniform(3, 8, big.sum())
        travel = rng.random(n) < 0.07
        dist = np.where(travel, rng.exponential(800, n), rng.exponential(15, n))
        new_dev = rng.random(n) < 0.10
        for k in range(n):
            device = _device(rng) if new_dev[k] else devices[i][rng.integers(len(devices[i]))]
            rows.append(_row(i, int(stamp[k]), float(amounts[k]), float(dist[k]), device, emails[i],
                             device_types[i], rng.choice(PRODUCTS, p=PRODUCT_P), 0, "legit"))

    def _row(i, dt, amt, dist, device, email, dtype, product, label, kind):
        return {
            "TransactionDT": dt, "TransactionAmt": round(max(amt, 0.5), 2), "ProductCD": product,
            "card1": int(card_ids[i]), "card4": networks[i], "card6": funding[i], "addr1": int(regions[i]),
            "dist1": round(dist, 1), "P_emaildomain": email, "DeviceType": dtype, "DeviceInfo": device,
            "isFraud": label, "fraud_kind": kind,
        }

    counts = rng.poisson(tx_per_card_per_day * days, cards)
    for i in range(cards):
        legit(i, int(counts[i]))
    n_legit = len(rows)
    target_fraud = int(n_legit * fraud_rate / (1 - fraud_rate))

    # Credit cards are 2.2x as likely to be picked as a victim.
    victim_p = np.where(funding == "credit", 2.2, 1.0)
    victim_p /= victim_p.sum()
    fraud_devices = [_device(rng) for _ in range(150)]

    def victim() -> int:
        return int(rng.choice(cards, p=victim_p))

    made = 0
    while made < target_fraud:
        kind = rng.choice(["takeover", "testing", "ring", "stealth"], p=[0.40, 0.12, 0.10, 0.38])
        start = int(rng.integers(5 * DAY, horizon - DAY))
        if kind == "takeover":
            i = victim()
            pooled = rng.random() < 0.35
            device = fraud_devices[rng.integers(len(fraud_devices))] if pooled else _device(rng)
            email = emails[i] if rng.random() < 0.8 else rng.choice(FRAUD_EMAILS)
            near = rng.random() < 0.5  # proxies close to the billing address
            night = rng.random() < 0.5
            hour = rng.uniform(0, 6) if night else rng.uniform(0, 24)
            t = start // DAY * DAY + int(hour * 3600)
            for _ in range(rng.integers(1, 5)):
                amt = mean_amt[i] * rng.lognormal(0.9, 0.6)
                rows.append(_row(i, t, amt, rng.exponential(15 if near else 600), device, email,
                                 rng.choice(["mobile", "desktop"]), rng.choice(["W", "C"]), 1, kind))
                t += int(rng.integers(60, 1800))
                made += 1
        elif kind == "testing":
            i = victim()
            device = fraud_devices[rng.integers(len(fraud_devices))]
            t = start
            for _ in range(rng.integers(3, 9)):
                rows.append(_row(i, t, rng.uniform(1, 5), rng.exponential(400), device, rng.choice(FRAUD_EMAILS),
                                 "desktop", "C", 1, kind))
                t += int(rng.integers(5, 120))
                made += 1
            for _ in range(rng.integers(1, 3)):
                rows.append(_row(i, t, rng.uniform(150, 900), rng.exponential(400), device, rng.choice(FRAUD_EMAILS),
                                 "desktop", "W", 1, kind))
                t += int(rng.integers(30, 600))
                made += 1
        elif kind == "ring":
            ring_devices = [_device(rng) for _ in range(rng.integers(1, 3))]
            email = rng.choice(FRAUD_EMAILS[:4])
            for _ in range(rng.integers(4, 11)):
                i = victim()
                t = start + int(rng.integers(0, 4 * DAY))
                for _ in range(rng.integers(1, 3)):
                    amt = rng.uniform(80, 600)
                    rows.append(_row(i, min(t, horizon - 1), amt, rng.exponential(300),
                                     ring_devices[rng.integers(len(ring_devices))], email, "mobile", "W", 1, kind))
                    t += int(rng.integers(300, 7200))
                    made += 1
        else:  # stealth: victim's own (compromised) device; only amount and timing stand out
            i = victim()
            hour = rng.uniform(0, 6) if rng.random() < 0.5 else rng.normal(peak_hour[i], 3)
            t = start // DAY * DAY + int(_hour_seconds(rng, np.array([hour]))[0])
            rows.append(_row(i, t, mean_amt[i] * rng.lognormal(0.8, 0.5), rng.exponential(15),
                             devices[i][0], emails[i], device_types[i], rng.choice(PRODUCTS, p=PRODUCT_P), 1, kind))
            made += 1

    frame = pd.DataFrame(rows)
    frame = frame.sort_values(["TransactionDT"], kind="mergesort").reset_index(drop=True)
    frame.insert(0, "TransactionID", np.arange(3_000_000, 3_000_000 + len(frame)))
    return frame
