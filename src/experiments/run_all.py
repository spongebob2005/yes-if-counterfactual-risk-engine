"""Build everything the live engine needs, and every number the site reports.

    python -m src.experiments.run_all            # ~2-4 minutes on a laptop

Writes models/engine.json, models/detector.npz, models/samples.json and reports/results.json.
Seeds, thresholds and simulator costs come from config.yaml.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from risk_engine import RAW_FIELDS, RiskEngine
from src.adversary.attacker import evasion_rate, hill_climb, random_search
from src.counterfactual.search import quality
from src.data.core import CATEGORICAL, FEATURES, GRAPH_FEATURES, feature_matrix
from src.data.features import add_graph_feats, add_history, learn_category_levels, make_features
from src.data.load import SPLIT_NAMES, assert_time_ordered_splits, load_ieee, time_split
from src.data.synth import generate
from src.models.detector import check_parity, choose_thresholds, export, fit_detector, fit_logreg
from src.models.metrics import ece, reliability, report
from src.policy.bandit import LinUCB
from src.policy.bandit import ThompsonFriction
from src.policy.baselines import analytic, replay, run_bandit, run_thompson, single_threshold, static_ladder
from src.policy.simulator import ACTIONS, drifting

ROOT = Path(__file__).resolve().parents[2]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_frame(cfg: dict) -> pd.DataFrame:
    if cfg["data"]["source"] == "ieee":
        return load_ieee(ROOT / "data/raw")
    d = cfg["data"]
    return generate(cards=d["cards"], days=d["days"], tx_per_card_per_day=d["tx_per_card_per_day"],
                    fraud_rate=d["fraud_rate"], seed=cfg["seed"])


def to_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame[list(RAW_FIELDS)].to_dict("records")
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float) and np.isnan(value):
                row[key] = None
            elif isinstance(value, np.generic):
                row[key] = value.item()
    return rows


def downsample(values: np.ndarray, points: int = 200) -> list[float]:
    idx = np.linspace(0, len(values) - 1, points).astype(int)
    return [round(float(values[i]), 2) for i in idx]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    pol = cfg["policy"]
    sim, margin = pol["sim"], pol["margin"]
    results: dict[str, Any] = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "data_source": cfg["data"]["source"], "seed": seed}

    # ---- M1 data and features -------------------------------------------
    log("M1: generating data and causal features")
    frame = load_frame(cfg)
    fractions = tuple(cfg["split"][name] for name in SPLIT_NAMES)
    raw = time_split(frame, fractions)
    assert_time_ordered_splits(raw)
    fallback = float(raw["train"]["TransactionAmt"].median())
    full = make_features(add_graph_feats(add_history(frame, fallback_amount=fallback)))  # once, on the full frame
    splits = time_split(full, fractions)
    assert_time_ordered_splits(splits)
    levels = learn_category_levels(splits["train"], CATEGORICAL)
    X = {k: feature_matrix(v, levels, FEATURES) for k, v in splits.items()}
    y = {k: v["isFraud"].to_numpy() for k, v in splits.items()}
    results["data"] = {k: {"rows": int(len(v)), "fraud_rate": round(float(v["isFraud"].mean()), 4)} for k, v in splits.items()}
    results["data"]["fraud_kinds"] = full.loc[full.isFraud == 1, "fraud_kind"].value_counts().to_dict() if "fraud_kind" in full else {}
    log(f"   {len(full):,} rows; " + ", ".join(f"{k} {v['fraud_rate']:.2%}" for k, v in results["data"].items() if k in SPLIT_NAMES))

    # ---- M2 detector + E1/E2 -------------------------------------------
    log("M2: LightGBM + isotonic calibration")
    det = cfg["detector"]
    # card6 is encoded on sorted levels (credit=0, debit=1): -1 means debit is never riskier.
    direction = {**det.get("monotone", {}), }
    monotone = [int(direction.get(name, 0)) for name in FEATURES]
    base, iso = fit_detector(X["train"], y["train"], X["val"], y["val"], X["calib"], y["calib"], det, seed,
                             monotone=monotone)
    raw_test = base.predict_proba(X["test"])[:, 1]
    cal_test = iso.predict(raw_test)
    logreg = fit_logreg(X["train"], y["train"], seed)
    lr_test = logreg.predict_proba(X["test"])[:, 1]
    results["E1_detector"] = {"lightgbm": report(y["test"], cal_test), "logistic_regression": report(y["test"], lr_test),
                              "trees": int(base.booster_.num_trees())}
    results["E2_calibration"] = {
        "ece_raw": ece(y["test"], raw_test), "ece_isotonic": ece(y["test"], cal_test),
        "reliability_raw": reliability(y["test"], raw_test), "reliability_isotonic": reliability(y["test"], cal_test),
    }
    log(f"   test PR-AUC {results['E1_detector']['lightgbm']['pr_auc']:.3f} "
        f"ROC-AUC {results['E1_detector']['lightgbm']['roc_auc']:.3f} "
        f"(logreg PR-AUC {results['E1_detector']['logistic_regression']['pr_auc']:.3f}); "
        f"ECE {results['E2_calibration']['ece_raw']:.4f} -> {results['E2_calibration']['ece_isotonic']:.4f}")

    # ---- M3 graph ablation (E3) ----------------------------------------
    log("M3: ablation without shared-entity features")
    keep = [i for i, name in enumerate(FEATURES) if name not in GRAPH_FEATURES]
    no_graph, iso_ng = fit_detector(X["train"][:, keep], y["train"], X["val"][:, keep], y["val"],
                                    X["calib"][:, keep], y["calib"], det, seed, monotone=[monotone[i] for i in keep])
    ng = report(y["test"], iso_ng.predict(no_graph.predict_proba(X["test"][:, keep])[:, 1]))
    results["E3_graph"] = {"with_graph": results["E1_detector"]["lightgbm"], "without_graph": ng,
                           "pr_auc_delta": results["E1_detector"]["lightgbm"]["pr_auc"] - ng["pr_auc"]}
    log(f"   PR-AUC with graph {results['E1_detector']['lightgbm']['pr_auc']:.3f} vs without {ng['pr_auc']:.3f}")

    # Thresholds for the base model (needed by the attacker's success condition).
    risk_val = iso.predict(base.predict_proba(X["val"])[:, 1])
    thresholds = choose_thresholds(risk_val, y["val"], splits["val"]["TransactionAmt"].to_numpy(), sim, margin)

    # ---- M6 adversary loop (E6) ----------------------------------------
    log("M6: adversarial retraining")
    adv = cfg["adversary"]
    meta_common = {"features": list(FEATURES), "levels": levels}

    def engine_for(model, iso_, t) -> RiskEngine:
        meta = {**meta_common, "thresholds": t, "policy": {"mode": "thresholds"}}
        return RiskEngine(export(model, iso_), meta)

    # Attacks are built from training-period fraud: validation stays clean for early stopping.
    train_rows, val_rows, test_rows = to_rows(splits["train"]), to_rows(splits["val"]), to_rows(splits["test"])
    tau = thresholds["tau_allow"]
    eng = engine_for(base, iso, thresholds)
    risk_train = eng.score(train_rows)
    train_fraud = [r for r, yy, rr in zip(train_rows, y["train"], risk_train) if yy == 1 and rr >= tau]
    risk_test_base = eng.score(test_rows)
    test_fraud_idx = [i for i in np.where((y["test"] == 1) & (risk_test_base >= tau))[0]]
    test_fraud = [test_rows[i] for i in rng.choice(test_fraud_idx, min(300, len(test_fraud_idx)), replace=False)]
    Xtr_adv, ytr_adv = X["train"], y["train"]
    weights = np.ones(len(ytr_adv))
    model_r, iso_r = base, iso
    rounds = []
    for rnd in range(adv["rounds"] + 1):
        eng_r = engine_for(model_r, iso_r, thresholds)
        evade = evasion_rate(test_fraud, eng_r.score, np.random.default_rng(seed + rnd), adv["query_budget"], tau,
                             random_search, adv["min_amt_frac"])
        clean = report(y["test"], iso_r.predict(model_r.predict_proba(X["test"])[:, 1]))
        rounds.append({"round": rnd, "evasion": round(evade, 4), "pr_auc": round(clean["pr_auc"], 4),
                       "roc_auc": round(clean["roc_auc"], 4)})
        log(f"   round {rnd}: evasion {evade:.1%}, clean PR-AUC {clean['pr_auc']:.3f}")
        if rnd == adv["rounds"]:
            break
        evaded = []
        attack_rng = np.random.default_rng(seed + 100 + rnd)
        for x in train_fraud:
            cand, r, ok = random_search(x, eng_r.score, attack_rng, adv["query_budget"], tau, adv["min_amt_frac"])
            if ok:
                evaded.append(cand)
        if evaded:
            Xtr_adv = np.vstack([Xtr_adv, eng_r.matrix(evaded)])
            ytr_adv = np.concatenate([ytr_adv, np.ones(len(evaded), dtype=ytr_adv.dtype)])
            weights = np.concatenate([weights, np.full(len(evaded), adv["adversarial_weight"])])
        model_r, iso_r = fit_detector(Xtr_adv, ytr_adv, X["val"], y["val"], X["calib"], y["calib"], det, seed, weights,
                                      monotone=monotone)
    results["E6_adversary"] = {"rounds": rounds, "query_budget": adv["query_budget"],
                               "attacked_test_fraud": len(test_fraud)}

    hardened_ok = rounds[-1]["pr_auc"] >= rounds[0]["pr_auc"] - 0.01
    model, iso_final = (model_r, iso_r) if hardened_ok else (base, iso)
    results["deployed_model"] = "adversarially retrained" if hardened_ok else "base"
    log(f"   deploying the {results['deployed_model']} model")

    risk_model = export(model, iso_final)
    results["parity_max_abs_diff"] = check_parity(model, iso_final, risk_model, X["test"])
    risk = {k: risk_model.predict(X[k]) for k in SPLIT_NAMES}
    thresholds = choose_thresholds(risk["val"], y["val"], splits["val"]["TransactionAmt"].to_numpy(), sim, margin)
    results["thresholds"] = thresholds
    results["deployed_metrics"] = report(y["test"], risk["test"])
    log(f"   thresholds {thresholds}")

    # ---- M5 friction policy (E5) ---------------------------------------
    log("M5: policy replay under drift")

    def stream(split: str) -> dict[str, np.ndarray]:
        part = splits[split]
        return {"risk": risk[split], "y": y[split], "amount": part["TransactionAmt"].to_numpy(),
                "hour": part["hour"].to_numpy(), "shared_device": part["DeviceInfo_distinct_uids"].to_numpy()}

    sv, st = stream("val"), stream("test")
    sim_val = drifting(sim, int(len(sv["risk"]) * pol["drift_at"]), pol["step_up_catch_after"])
    tuning = {}
    for alpha in (0.1, 0.3, 1.0):
        for gamma in (1.0, 0.9995, 0.999):
            res, _ = run_bandit(sv, sim_val, margin, alpha, pol["reward_clip"], seed, gamma=gamma,
                                tau_allow=thresholds["tau_allow"])
            tuning[f"alpha={alpha}, gamma={gamma}"] = (float(res["cumulative"][-1]), alpha, gamma)
    best_alpha, best_gamma = min(tuning.values())[1:]
    alpha_scores = {k: v[0] for k, v in tuning.items()}
    drift_i = int(len(st["risk"]) * pol["drift_at"])
    sim_test = drifting(sim, drift_i, pol["step_up_catch_after"])
    stale = analytic(sim, margin)
    after = {**sim, "STEP_UP": {**sim["STEP_UP"], "catch": pol["step_up_catch_after"]}}
    oracle_after = analytic(after, margin)
    policies = {
        "B1 single threshold": lambda i: single_threshold(st["risk"][i], thresholds),
        "B2 static ladder": lambda i: static_ladder(st["risk"][i], thresholds),
        "B3 analytic (stale)": lambda i: stale(st["risk"][i], st["amount"][i]),
        "Oracle (knows the drift)": lambda i: (stale if i < drift_i else oracle_after)(st["risk"][i], st["amount"][i]),
    }
    e5: dict[str, Any] = {"drift_index": drift_i, "stream_length": len(st["risk"]), "tuning": alpha_scores,
                          "alpha": best_alpha, "gamma": best_gamma, "policies": {}}
    def summarize(res) -> dict[str, Any]:
        return {"curve": downsample(res["cumulative"]), "total": round(float(res["cumulative"][-1]), 1),
                "breakdown": dict(zip(("fraud_loss", "friction", "legit_loss"), map(float, res["parts"]))),
                "actions": dict(zip(ACTIONS, np.bincount(res["actions"], minlength=4).tolist()))}

    for name, choose in policies.items():
        e5["policies"][name] = summarize(replay(st, choose, sim_test, margin, seed))
    # Warm-start on validation (earlier in time, so the pre-drift world), then learn online through test.
    _, warm = run_bandit(sv, lambda i: sim, margin, best_alpha, pol["reward_clip"], seed, gamma=best_gamma,
                         tau_allow=thresholds["tau_allow"])
    res, bandit = run_bandit(st, sim_test, margin, best_alpha, pol["reward_clip"], seed, bandit=warm,
                             tau_allow=thresholds["tau_allow"])
    e5["policies"]["LinUCB bandit (plan)"] = summarize(res)
    learned = {}
    for gamma in (1.0, 0.999, 0.995):
        r_val, _ = run_thompson(sv, sim_val, margin, seed, ThompsonFriction(sim, margin, ACTIONS, gamma=gamma),
                                thresholds["tau_allow"])
        learned[gamma] = float(r_val["cumulative"][-1])
    best_ts_gamma = min(learned, key=learned.get)
    e5["thompson_tuning"] = {str(k): v for k, v in learned.items()}
    e5["thompson_gamma"] = best_ts_gamma
    _, warm_ts = run_thompson(sv, lambda i: sim, margin, seed,
                              ThompsonFriction(sim, margin, ACTIONS, gamma=best_ts_gamma), thresholds["tau_allow"])
    res_ts, thompson = run_thompson(st, sim_test, margin, seed, warm_ts, thresholds["tau_allow"])
    e5["policies"]["Learned friction (ours)"] = summarize(res_ts)
    e5["learned_rates_end"] = thompson.rates()
    # Late-window comparison: who adapts after the drift?
    for name, data in e5["policies"].items():
        curve = data["curve"]
        k = int(len(curve) * pol["drift_at"])
        data["after_drift"] = round(curve[-1] - curve[k], 1)
    results["E5_policy"] = e5
    for name, data in e5["policies"].items():
        log(f"   {name:28s} total {data['total']:>10,.0f}  after drift {data['after_drift']:>10,.0f}")

    # ---- deployable engine ------------------------------------------------
    numeric = [f for f in RAW_FIELDS if f not in CATEGORICAL]
    defaults = {f: float(splits["train"][f].median()) for f in numeric if f != "prev_dt"}
    defaults.update({c: str(splits["train"][c].mode().iloc[0]) for c in CATEGORICAL})
    last = full.groupby("P_emaildomain").tail(1)
    email_stats = {str(r.P_emaildomain): {"P_emaildomain_prior_cnt": float(r.P_emaildomain_prior_cnt),
                                          "P_emaildomain_distinct_uids": float(r.P_emaildomain_distinct_uids)}
                   for r in last.itertuples()}
    email_stats["__unknown__"] = {"P_emaildomain_prior_cnt": 0.0, "P_emaildomain_distinct_uids": 1.0}
    cf_cfg = {k: cfg["counterfactual"][k] for k in ("amt_mults", "delays_h", "card6_opts", "weights", "top_k")}
    bandit.alpha = best_alpha
    meta = {
        "version": 2,
        "trained_at": results["generated_at"],
        "data_source": cfg["data"]["source"],
        "deployed_model": results["deployed_model"],
        "features": list(FEATURES),
        "levels": levels,
        "thresholds": thresholds,
        "defaults": defaults,
        "email_stats": email_stats,
        "counterfactual": cf_cfg,
        "policy": {"mode": "thompson", "thompson": thompson.state(), "bandit": bandit.state(), "sim": sim, "margin": margin},
        "metrics": {k: round(v, 4) for k, v in results["deployed_metrics"].items()},
    }
    models_dir = ROOT / "models"
    models_dir.mkdir(exist_ok=True)
    risk_model.save(models_dir / "detector.npz")
    (models_dir / "engine.json").write_text(json.dumps(meta, indent=1) + "\n")
    engine = RiskEngine.load(models_dir)
    assert np.allclose(engine.score(test_rows[:500]), risk["test"][:500]), "engine/model mismatch"

    # ---- M4 counterfactual quality (E4) ---------------------------------
    log("M4: counterfactual quality")
    flagged = np.where(risk["test"] >= thresholds["tau_allow"])[0]
    sample = rng.choice(flagged, min(400, len(flagged)), replace=False)
    by_label: dict[str, list] = {"all": [], "legit": [], "fraud": []}
    for i in sample:
        cfs = engine.counterfactuals(test_rows[i])
        by_label["all"].append(cfs)
        by_label["fraud" if y["test"][i] else "legit"].append(cfs)
    results["E4_counterfactual"] = {k: quality(v) for k, v in by_label.items()}
    log(f"   validity {results['E4_counterfactual']['all']['validity']:.1%}, "
        f"sparsity {results['E4_counterfactual']['all']['sparsity']:.2f}")

    # ---- E7 defense ------------------------------------------------------
    log("E7: evasion vs query budget")
    budgets = adv["budgets"]
    e7 = {"budgets": budgets, "open_scores": [], "coarse_output": [], "rate_limited": []}
    flagged_fraud = [test_rows[i] for i in np.where((y["test"] == 1) & (risk["test"] >= tau))[0]]
    targets = [flagged_fraud[i] for i in rng.choice(len(flagged_fraud), min(200, len(flagged_fraud)), replace=False)]
    tau = thresholds["tau_allow"]
    for q in budgets:
        e7["open_scores"].append(round(evasion_rate(targets, engine.score, np.random.default_rng(seed), q, tau, hill_climb), 4))
        e7["coarse_output"].append(round(evasion_rate(targets, engine.score, np.random.default_rng(seed), q, tau, random_search), 4))
        e7["rate_limited"].append(round(evasion_rate(targets, engine.score, np.random.default_rng(seed),
                                                     min(q, adv["rate_limit"]), tau, random_search), 4))
    e7["rate_limit"] = adv["rate_limit"]
    results["E7_defense"] = e7

    # ---- fairness check ----------------------------------------------------
    test = splits["test"].assign(risk=risk["test"])
    legit = test[test.isFraud == 0]
    results["fairness"] = {
        seg: {str(k): {"false_positive_rate": round(float((g.risk >= thresholds["tau_allow"]).mean()), 4), "n": int(len(g))}
              for k, g in legit.groupby(seg)}
        for seg in ("DeviceType", "ProductCD", "card6")
    }

    # ---- sample transactions for the API, site and dashboard -------------
    log("Samples and ring example")
    test["action"] = [engine.choose_action(r, row) for r, row in zip(risk["test"], test_rows)]

    def pick(mask, by, ascending=True, simple_yes=False) -> int:
        """First row by `by`; with simple_yes, the first whose best "yes, if" changes one thing."""
        cand = test[mask].sort_values(by, ascending=ascending)
        if simple_yes:
            for idx in cand.index[:60]:
                cfs = engine.counterfactuals(test_rows[test.index.get_loc(idx)])
                if cfs and len(cfs[0]["kind"]) == 1:
                    return int(idx)
        return int(cand.index[0])

    t = thresholds
    kinds = test.get("fraud_kind", pd.Series("", index=test.index))
    chosen = [
        ("clean", "A regular purchase on a known device",
         pick((test.isFraud == 0) & (test.prior_cnt > 20) & test.TransactionAmt.between(40, 120) & (test.card_device_prior > 10), "risk")),
        ("large", "A loyal customer's unusually large purchase",
         pick((test.isFraud == 0) & (test.amt_ratio > 4) & (test.risk >= t["tau_allow"]) & (test.risk < t["tau_block"]),
              "risk", False, simple_yes=True)),
        ("borderline", "Just over the approval line",
         pick((test.risk >= t["tau_allow"]) & (test.risk < t["tau_allow"] * 2) & (test.isFraud == 0), "risk",
              simple_yes=True)),
        ("suspicious", "Stolen card, new device, middle of the night",
         pick((kinds == "takeover") & (test.risk >= t["tau_allow"]), "risk", False)),
        ("ring", "Device shared by many cards", pick((kinds == "ring") & (test.risk >= t["tau_allow"]), "DeviceInfo_distinct_uids", False)),
        ("testing", "Card testing: tiny charge seconds after the last",
         pick((kinds == "testing") & (test.TransactionAmt < 6) & (test.DeviceInfo_distinct_uids <= 2), "since_last")),
    ]
    samples = []
    for key, title, idx in chosen:
        row = to_rows(test.loc[[idx]])[0]
        samples.append({"id": f"TXN-{int(test.loc[idx, 'TransactionID'])}", "key": key, "title": title,
                        "is_fraud": bool(test.loc[idx, "isFraud"]), "transaction": row})
    (models_dir / "samples.json").write_text(json.dumps(samples, indent=1) + "\n")

    ring_idx = chosen[4][2]
    device = test.loc[ring_idx, "DeviceInfo"]
    ring = full[(full.DeviceInfo == device) & (full.TransactionDT <= test.loc[ring_idx, "TransactionDT"])]
    results["ring_example"] = {
        "device": "device " + str(device)[-4:],
        "edges": [{"card": f"card {int(r.card1)}", "amount": float(r.TransactionAmt), "fraud": bool(r.isFraud),
                   "email": str(r.P_emaildomain)} for r in ring.itertuples()],
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "results.json").write_text(json.dumps(results, indent=1, default=float) + "\n")
    log("Wrote models/ and reports/results.json")


if __name__ == "__main__":
    main()
