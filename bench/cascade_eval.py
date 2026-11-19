#!/usr/bin/env python3
"""Replicate the JEV-as-a-Judge cascade on local models, following the paper's protocol.

Source: "JEV-as-a-Judge: Accept When Confident, Escalate When Unsure" (arXiv:2609.26550).
  Section 8: two-order gate (eq. 1), confidence q = max probability, threshold rule.
  Section 7.2: temperature scaling fitted on the selection set, checked on held-out data.
  Section 9: freeze the policy on selection pairs, then run the cascade live on held-out pairs.

Steps:
1. Judge every selection pair with the first stage (both orders) and the fallback (base order).
2. Fit a temperature on the selection pairs (minimum negative log-likelihood).
3. Choose tau on the selection pairs with the paper's lower-bound rule. Freeze tau and T in policy.json.
4. Judge the held-out pairs offline, and run the frozen cascade live on them.
5. Report accuracy, escalation, paired difference with a bootstrap CI, ECE, Brier, NLL, and error AUROC.

The paper's own held-out results are not reproduced here: this build uses local models and a small,
hand-labeled set. Needs Ollama with both models pulled. Standard library only.
"""
from __future__ import annotations

import json
import math
import os
import pathlib
import random
import statistics
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import server  # noqa: E402
from pairs import PAIRS  # noqa: E402  (legacy held-out and the first 75 selection pairs)
from pairs_extra import EXTRA_SELECT, HELDOUT  # noqa: E402

FIRST = os.environ.get("JEV_MODEL", "qwen3:4b-instruct")
FALLBACK = os.environ.get("JEV_FALLBACK_MODEL", "qwen3:8b")
INSTRUCTIONS = "Which response correctly and fully answers the question?"
GRID = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]  # the paper's threshold grid
TEMPERATURES = [0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 10.0]
MARGIN = -0.02  # the paper's tolerated loss versus the fallback
Z_ONE_SIDED_95 = 1.645
RESULTS = ROOT / "bench" / "results"


def orient(pairs: List[tuple], seed: int = 7) -> List[Dict[str, Any]]:
    """Put the correct answer in position a or b at random, so position bias cannot leak the label."""
    rng = random.Random(seed)
    items = []
    for index, (split, question, correct, wrong) in enumerate(pairs):
        flip = rng.random() < 0.5
        a, b = (wrong, correct) if flip else (correct, wrong)
        items.append({"id": f"{split}-{index:03d}", "split": split, "question": question,
                      "a": a, "b": b, "truth": "b" if flip else "a"})
    return items


def credit(choice: Optional[str], truth: str) -> float:
    """Exact ties get half credit, as in the paper. A missing or invalid output gets none."""
    if choice is None:
        return 0.0
    if choice == "tie":
        return 0.5
    return 1.0 if choice == truth else 0.0


def pair_p_a(p_ab: float, p_ba: float, temperature: float) -> float:
    """The paper's eq. (1), with temperature applied to each order before averaging."""
    return 0.5 * (server.sharpen({"A": p_ab, "B": 1 - p_ab}, temperature)["A"]
                  + 1 - server.sharpen({"A": p_ba, "B": 1 - p_ba}, temperature)["A"])


def materialize(raw: List[Dict[str, Any]], temperature: float) -> List[Dict[str, Any]]:
    """Turn raw first-stage orders into gate inputs (q, choice, credit) at a given temperature."""
    out = []
    for r in raw:
        if r["p_ab"] is None:
            out.append({"id": r["id"], "valid": False, "q": 0.0, "choice": None, "c1": 0.0, "c2": r["c2"]})
            continue
        p_a = pair_p_a(r["p_ab"], r["p_ba"], temperature)
        choice = "a" if p_a > 0.5 else ("b" if p_a < 0.5 else "tie")
        out.append({"id": r["id"], "valid": True, "q": max(p_a, 1 - p_a), "choice": choice,
                    "p_a": p_a, "c1": credit(choice, r["truth"]), "c2": r["c2"], "truth": r["truth"]})
    return out


def nll(raw: List[Dict[str, Any]], temperature: float) -> float:
    total = 0.0
    valid = [r for r in raw if r["p_ab"] is not None]
    for r in valid:
        p_a = pair_p_a(r["p_ab"], r["p_ba"], temperature)
        p_truth = p_a if r["truth"] == "a" else 1 - p_a
        total += -math.log(min(max(p_truth, 1e-6), 1 - 1e-6))  # clipped, as in the paper
    return total / len(valid)


def fit_temperature(raw: List[Dict[str, Any]]) -> float:
    """Pick the temperature with the lowest selection-set NLL. Ties go to the one nearest 1."""
    return min(TEMPERATURES, key=lambda t: (round(nll(raw, t), 10), abs(math.log(t))))


def choose_threshold(records: List[Dict[str, Any]], rule: str) -> Optional[float]:
    """The paper's rule: the most permissive tau whose loss versus the fallback is within the margin.

    `rule` is "point" (mean difference) or "lower_bound" (one-sided 95% lower confidence bound).
    Ties go to the stricter threshold. If nothing qualifies, every item is escalated (returns None).
    """
    best = None  # (accepted, tau)
    for tau in GRID:
        diffs, accepted = [], 0
        for r in records:
            accept = r["valid"] and r["q"] >= tau and r["choice"] != "tie"
            cascade_credit = r["c1"] if accept else r["c2"]
            diffs.append(cascade_credit - r["c2"])
            accepted += accept
        n = len(diffs)
        mean = sum(diffs) / n
        spread = statistics.stdev(diffs) if n > 1 else 0.0
        bound = mean - Z_ONE_SIDED_95 * spread / math.sqrt(n) if rule == "lower_bound" else mean
        if bound >= MARGIN and (best is None or accepted > best[0] or (accepted == best[0] and tau > best[1])):
            best = (accepted, tau)
    return best[1] if best else None


def simulate(records: List[Dict[str, Any]], tau: Optional[float]) -> List[Dict[str, Any]]:
    """Offline cascade: keep the first stage when q >= tau, otherwise take the fallback's decision."""
    rows = []
    for r in records:
        accept = tau is not None and r["valid"] and r["q"] >= tau and r["choice"] != "tie"
        rows.append({"accepted": bool(accept), "cascade": r["c1"] if accept else r["c2"],
                     "first": r["c1"] if r["valid"] else 0.0, "fallback": r["c2"], "q": r["q"] if r["valid"] else 0.0,
                     "valid": r["valid"]})
    return rows


def ece(confidences: List[float], correct: List[float], bins: int = 10) -> float:
    total, out = len(confidences), 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(confidences) if min(int(c * bins), bins - 1) == b]
        if idx:
            out += len(idx) / total * abs(statistics.mean(confidences[i] for i in idx) - statistics.mean(correct[i] for i in idx))
    return out


def auroc(scores: List[float], positive: List[bool]) -> Optional[float]:
    pos = [s for s, p in zip(scores, positive) if p]
    neg = [s for s, p in zip(scores, positive) if not p]
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def bootstrap_mean(values: List[float], resamples: int = 2000, seed: int = 11) -> List[float]:
    rng = random.Random(seed)
    n = len(values)
    means = sorted(statistics.mean(rng.choice(values) for _ in range(n)) for _ in range(resamples))
    return [round(means[int(0.025 * resamples)], 4), round(means[int(0.975 * resamples) - 1], 4)]


def summarize(records: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Metrics on held-out pairs, following Sections 3.4 and 7 of the paper."""
    n = len(rows)
    diffs = [r["cascade"] - r["fallback"] for r in rows]
    valid = [i for i in range(n) if rows[i]["valid"]]
    q_valid = [rows[i]["q"] for i in valid]
    correct_valid = [1.0 if rows[i]["first"] == 1.0 else 0.0 for i in valid]
    p_truth = [(records[i]["p_a"] if records[i]["truth"] == "a" else 1 - records[i]["p_a"]) for i in valid]
    errors_first = [rows[i]["first"] < 1.0 for i in range(n)]
    auc = auroc([1 - q for q in q_valid], [not c for c in correct_valid])
    ranked = sorted(valid, key=lambda i: -rows[i]["q"])

    def selective(coverage: float) -> float:
        keep = ranked[: max(1, math.ceil(coverage * n))]
        return round(statistics.mean(rows[i]["first"] for i in keep), 4)

    return {
        "n": n,
        "valid_first_stage": len(valid),
        "accuracy": {"cascade": round(statistics.mean(r["cascade"] for r in rows), 4),
                     "first_stage_alone": round(statistics.mean(r["first"] for r in rows), 4),
                     "fallback_alone": round(statistics.mean(r["fallback"] for r in rows), 4)},
        "cascade_minus_fallback_pp": {"mean": round(100 * statistics.mean(diffs), 2),
                                      "bootstrap_95": [round(100 * x, 2) for x in bootstrap_mean(diffs)]},
        "escalated_pct": round(100 * sum(1 for r in rows if not r["accepted"]) / n, 1),
        "first_stage_errors_caught_pct": round(100 * sum(1 for e, r in zip(errors_first, rows) if e and not r["accepted"]) / max(1, sum(errors_first)), 1),
        "first_stage_quality": {
            "ece_10_bins": round(ece(q_valid, correct_valid), 4),
            "brier_mean": round(statistics.mean(2 * (1 - p) ** 2 for p in p_truth), 4),
            "nll_mean": round(statistics.mean(-math.log(min(max(p, 1e-6), 1 - 1e-6)) for p in p_truth), 4),
            "error_auroc_of_1_minus_q": round(auc, 4) if auc is not None else None,
            "selective_accuracy_at_80_pct_coverage": selective(0.8),
            "selective_accuracy_at_50_pct_coverage": selective(0.5),
        },
    }


def timed(fn, *args):
    started = time.perf_counter()
    try:
        return fn(*args), (time.perf_counter() - started) * 1000
    except server.InvalidAnswer:
        return None, (time.perf_counter() - started) * 1000


def judge_raw(first: server.JevLite, fallback: server.JevLite, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Both models judge every item, with no temperature and no gate, so later steps can reuse the raw orders."""
    out = []
    for item in items:
        pair, first_ms = timed(first.judge_pair, item["question"], INSTRUCTIONS, item["a"], item["b"])
        base, fb_ms = timed(fallback.judge_base, item["question"], INSTRUCTIONS, item["a"], item["b"])
        out.append({
            "id": item["id"], "truth": item["truth"],
            "p_ab": pair["orders"]["a_first"] if pair else None,
            "p_ba": pair["orders"]["b_first"] if pair else None,
            "c2": credit(base["choice"], item["truth"]) if base else 0.0,
            "first_ms": round(first_ms, 1), "fallback_ms": round(fb_ms, 1),
        })
    return out


def run_live(live: server.JevLite, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The frozen cascade, run through the real method on each held-out pair, with timings."""
    rows = []
    for item in items:
        started = time.perf_counter()
        try:
            result = live.cascade(item["question"], INSTRUCTIONS, item["a"], item["b"])
            decision, path = result["decision"], result["path"]
        except server.InvalidAnswer:
            decision, path = None, "escalated"  # the fallback gave no valid answer: counts as wrong
        rows.append({"id": item["id"], "decision": decision, "path": path, "truth": item["truth"],
                     "ms": round((time.perf_counter() - started) * 1000, 1),
                     "cascade": credit(decision, item["truth"])})
    return rows


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    items = orient(PAIRS + EXTRA_SELECT + HELDOUT)
    select = [i for i in items if i["split"] == "select"]
    test = [i for i in items if i["split"] == "heldout"]  # fresh: scored once, after the policy is frozen
    legacy = [i for i in items if i["split"] == "legacy"]  # previously inspected: a sanity check only
    first = server.JevLite(server.OllamaStepModel(model=FIRST))  # temperature 1.0 on purpose: raw orders
    fallback = server.JevLite(server.OllamaStepModel(model=FALLBACK))
    print(f"select: {len(select)}, heldout (fresh): {len(test)}, legacy: {len(legacy)}; first={FIRST} fallback={FALLBACK}")

    # 1-2. Selection set: raw judgments, then a temperature.
    raw_select = judge_raw(first, fallback, select)
    for r, item in zip(raw_select, select):
        r["question"] = item["question"]
    T = fit_temperature(raw_select)
    (RESULTS / "select_raw.json").write_text(json.dumps(raw_select, indent=2))

    # 3. Threshold on the selection set, with and without temperature (the paper's rule).
    rec_t1 = materialize(raw_select, 1.0)
    rec_T = materialize(raw_select, T)
    tau_uncal = choose_threshold(rec_t1, "lower_bound")
    tau_cal = choose_threshold(rec_T, "lower_bound")
    point_cal = choose_threshold(rec_T, "point")
    policy = {"tau": tau_cal, "temperature": T, "rule": "lower_bound", "margin": MARGIN, "grid": GRID,
              "first_stage_model": FIRST, "fallback_model": FALLBACK, "selection_pairs": len(select),
              "selection_nll_at_T1": round(nll(raw_select, 1.0), 4), "selection_nll_at_T": round(nll(raw_select, T), 4),
              "point_rule_tau_for_comparison": point_cal}
    (ROOT / "policy.json").write_text(json.dumps(policy, indent=2))
    print(f"fitted T={T}; frozen tau={tau_cal} (uncalibrated would have been {tau_uncal})")

    # 4. Held-out set: offline simulation of both policies, plus the live frozen cascade.
    raw_test = judge_raw(first, fallback, test)
    for r, item in zip(raw_test, test):
        r["question"] = item["question"]
    (RESULTS / "test_raw.json").write_text(json.dumps(raw_test, indent=2))
    test_t1 = materialize(raw_test, 1.0)
    test_T = materialize(raw_test, T)
    offline_uncal = summarize(test_t1, simulate(test_t1, tau_uncal))
    offline_cal = summarize(test_T, simulate(test_T, tau_cal))

    live = server.JevLite(server.OllamaStepModel(model=FIRST), fallback=server.JevLite(server.OllamaStepModel(model=FALLBACK)),
                          policy=policy, temperature=T)
    live_rows = run_live(live, test)
    sim_rows = simulate(test_T, tau_cal)
    agree = sum(1 for lr, sr in zip(live_rows, sim_rows) if lr["path"] == ("first_stage" if sr["accepted"] else "escalated")) / len(test)
    live_summary = {
        "accuracy": round(statistics.mean(r["cascade"] for r in live_rows), 4),
        "escalated_pct": round(100 * sum(1 for r in live_rows if r["path"] == "escalated") / len(test), 1),
        "gate_agreement_with_offline_pct": round(100 * agree, 1),
        "latency_ms": {
            "cascade_p50": round(statistics.median(r["ms"] for r in live_rows), 1),
            "cascade_p95": round(sorted(r["ms"] for r in live_rows)[int(0.95 * (len(test) - 1))], 1),
            "first_path_p50": round(statistics.median(r["ms"] for r in live_rows if r["path"] == "first_stage"), 1)
            if any(r["path"] == "first_stage" for r in live_rows) else None,
            "escalated_p50": round(statistics.median(r["ms"] for r in live_rows if r["path"] == "escalated"), 1)
            if any(r["path"] == "escalated" for r in live_rows) else None,
            "fallback_alone_p50": round(statistics.median(r["fallback_ms"] for r in raw_test), 1),
        },
    }
    # Legacy set (inspected earlier): reported only as a sanity check on the same frozen policy.
    raw_legacy = judge_raw(first, fallback, legacy)
    legacy_T = materialize(raw_legacy, T)
    legacy_summary = summarize(legacy_T, simulate(legacy_T, tau_cal))

    # Post hoc sweep on the fresh held-out set. Printed for reading only; the policy is not changed by it.
    sweep = []
    for tau in GRID:
        rows = simulate(test_T, tau)
        sweep.append({"tau": tau,
                      "escalated_pct": round(100 * sum(1 for r in rows if not r["accepted"]) / len(rows), 1),
                      "cascade_minus_fallback_pp": round(100 * statistics.mean(r["cascade"] - r["fallback"] for r in rows), 2)})

    report = {
        "policy": policy,
        "selection_counts": {"select": len(select), "heldout_fresh": len(test), "legacy_previously_inspected": len(legacy)},
        "heldout_fresh_offline_uncalibrated_T1": offline_uncal,
        "heldout_fresh_offline_calibrated": offline_cal,
        "heldout_fresh_live_frozen_cascade": live_summary,
        "legacy_previously_inspected_calibrated": legacy_summary,
        "post_hoc_sweep_on_heldout_fresh_NOT_used_for_policy": sweep,
    }
    (RESULTS / "summary.json").write_text(json.dumps(report, indent=2))
    (RESULTS / "live_rows.json").write_text(json.dumps(live_rows, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
