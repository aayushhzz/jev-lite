#!/usr/bin/env python3
"""Measure Jev Lite on a small labeled set: accuracy, calibration, and latency.

Needs a running server (python3 server.py) and Ollama. Standard library only.
The 40 labels were written by hand before any model run; they are a small sanity check,
not a broad benchmark.
"""
from __future__ import annotations

import json
import os
import statistics
import time
import urllib.request

URL = os.environ.get("JEV_URL", "http://127.0.0.1:8876") + "/api/system-one"

CRITERIA = {
    "billing": "payments, invoices, charges, refunds, or subscriptions",
    "bug": "something broken or behaving incorrectly",
    "feature": "a request for new functionality",
    "account": "login, profile, or account management",
}

CASES = [
    ("I was charged twice for my subscription this month.", "billing"),
    ("The app crashes every time I open the settings page on Android.", "bug"),
    ("Could you add dark mode to the dashboard?", "feature"),
    ("I can't log in, it says my password is wrong but I just reset it.", "account"),
    ("Please refund the annual plan I bought by mistake yesterday.", "billing"),
    ("Exporting a CSV with more than 10k rows returns an empty file.", "bug"),
    ("It would be great to have a Slack integration for alerts.", "feature"),
    ("How do I change the email address on my account?", "account"),
    ("My invoice shows the wrong VAT rate for Germany.", "billing"),
    ("Clicking save on the profile page does nothing.", "bug"),
    ("Please support SSO with Okta for enterprise teams.", "feature"),
    ("I want to delete my account and all my data.", "account"),
    ("The credit card on file expired, how do I update it?", "billing"),
    ("Search results show deleted items after reindexing.", "bug"),
    ("Can you add keyboard shortcuts for navigation?", "feature"),
    ("Two-factor codes are not arriving by SMS anymore.", "account"),
    ("I see a charge from your company that I don't recognize.", "billing"),
    ("The mobile app freezes when I upload a photo.", "bug"),
    ("Please allow exporting reports as PDF.", "feature"),
    ("I forgot which email I registered with.", "account"),
    ("Our team needs seat-based pricing with monthly invoices.", "billing"),
    ("The API returns 500 for any request with unicode in the path.", "bug"),
    ("Would be nice to have a dark theme for the mobile app.", "feature"),
    ("Transfer ownership of my workspace to my colleague.", "account"),
    ("Why was I charged a late fee when I paid on time?", "billing"),
    ("Notifications are sent twice for every comment.", "bug"),
    ("Add a bulk-edit option for tags.", "feature"),
    ("My session logs me out every five minutes.", "bug"),
    ("Please update my name on the account profile.", "account"),
    ("There's a discount code I can't apply at checkout, it says invalid.", "billing"),
    ("Add support for markdown in comments.", "feature"),
    ("Can I get a copy of my data in JSON?", "account"),
    ("The build page shows a blank log for failed jobs.", "bug"),
    ("Please add an option to pause my subscription for a month.", "billing"),
    ("I need a way to invite external guests without a seat.", "feature"),
    ("Password reset emails go to spam for all our users.", "bug"),
    ("Can you add a calendar view?", "feature"),
    ("I was double-billed for the add-on, please fix it.", "billing"),
    ("How do I merge two accounts that I created by accident?", "account"),
    ("The page throws a null pointer error when the cart is empty.", "bug"),
]


def post(body: dict) -> dict:
    request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def percentile(values, p):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(p / 100 * (len(ordered) - 1))))]


def main() -> None:
    labels = list(CRITERIA)
    correct, brier, bins = 0, [], [[] for _ in range(10)]
    single_ms = []
    for text, truth in CASES:
        started = time.perf_counter()
        answer = post({"state": text, "questions": {"q": {"type": "choice", "instructions": "Which category fits this support ticket?", "criteria": CRITERIA}}})["answers"]["q"]
        single_ms.append((time.perf_counter() - started) * 1000)
        probs = answer["probabilities"]
        correct += answer["choice"] == truth
        brier.append(sum((probs[k] - (1.0 if k == truth else 0.0)) ** 2 for k in labels))
        confidence = max(probs.values())  # the top probability, which is what a caller thresholds on
        bins[min(9, int(confidence * 10))].append((confidence, answer["choice"] == truth))

    n = len(CASES)
    ece = sum(len(b) / n * abs(statistics.mean(c for c, _ in b) - statistics.mean(1.0 if ok else 0.0 for _, ok in b)) for b in bins if b)

    # Latency only: 4 identical questions about one ticket per request, to time the parallel path.
    batch_ms = []
    for start in range(0, n, 4):
        text, _ = CASES[start]
        questions = {f"t{i}": {"type": "choice", "instructions": "Which category fits this support ticket?", "criteria": CRITERIA} for i in range(4)}
        started = time.perf_counter()
        post({"state": text, "questions": questions})
        batch_ms.append((time.perf_counter() - started) * 1000)

    print(json.dumps({
        "cases": n,
        "accuracy": round(correct / n, 3),
        "brier_mean": round(statistics.mean(brier), 4),
        "ece_top_probability_10_bins": round(ece, 4),
        "single_call_ms": {"p50": round(percentile(single_ms, 50), 1), "p95": round(percentile(single_ms, 95), 1)},
        "four_question_batch_ms": {"p50": round(percentile(batch_ms, 50), 1), "p95": round(percentile(batch_ms, 95), 1)},
    }, indent=2))


if __name__ == "__main__":
    main()
