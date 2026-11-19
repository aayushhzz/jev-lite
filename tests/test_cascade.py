"""Offline tests for the two-order gate, the cascade, and the threshold rule. No model needed."""
from __future__ import annotations

import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "bench"))

import server  # noqa: E402
import cascade_eval  # noqa: E402
from pairs import PAIRS  # noqa: E402


class OrderAware:
    """Gives 0.9 to the option whose text contains 'right'. With bias_first set, ignores content
    and always returns that probability for the first position, the way a position-biased judge would."""

    def __init__(self, bias_first=None):
        self.bias_first = bias_first

    def label_probs(self, system, user, labels):
        options = {}
        for line in user.splitlines():
            if len(line) > 3 and line[0] in labels and line[1:3] == ". ":
                options[line[0]] = line[3:]
        if self.bias_first is not None:
            return {label: (self.bias_first if label == "A" else 1 - self.bias_first) for label in labels}
        return {label: (0.9 if "right" in options.get(label, "") else 0.1) for label in labels}


class FixedStep:
    def __init__(self, dist):
        self.dist = dist
        self.calls = 0

    def label_probs(self, system, user, labels):
        self.calls += 1
        total = sum(self.dist.get(label, 0.0) for label in labels)
        return {label: (self.dist.get(label, 0.0) / total if total else 0.0) for label in labels}


class PairGateTests(unittest.TestCase):
    def test_order_averaging_recovers_the_right_answer(self):
        result = server.JevLite(OrderAware()).judge_pair("Q?", "Which is right?", "right answer", "wrong answer")
        self.assertEqual(result["choice"], "a")
        self.assertAlmostEqual(result["p_a"], 0.9)
        self.assertAlmostEqual(result["confidence_q"], 0.9)

    def test_order_averaging_cancels_position_bias(self):
        # A judge that always favours the first position gives 0.8 in both orders: the average is a tie.
        result = server.JevLite(OrderAware(bias_first=0.8)).judge_pair("Q?", "Which is right?", "x", "y")
        self.assertEqual(result["choice"], "tie")
        self.assertAlmostEqual(result["p_a"], 0.5)

    def test_second_candidate_can_win(self):
        result = server.JevLite(OrderAware()).judge_pair("Q?", "Which is right?", "wrong answer", "right answer")
        self.assertEqual(result["choice"], "b")
        self.assertAlmostEqual(result["confidence_q"], 0.9)

    def test_candidates_are_flattened_and_bounded(self):
        a, b = server.pair_candidates({"a": "line one\nline two", "b": "other"})
        self.assertEqual(a, "line one line two")
        with self.assertRaises(server.BadRequest):
            server.pair_candidates({"a": "x" * (server.MAX_CANDIDATE_CHARS + 1), "b": "y"})


class CascadeTests(unittest.TestCase):
    def build(self, first_model, fallback_model, tau):
        fallback = server.JevLite(fallback_model)
        return server.JevLite(first_model, fallback=fallback, policy={"tau": tau})

    def test_confident_first_stage_is_accepted_without_the_fallback(self):
        fallback_model = FixedStep({"B": 1.0})  # would decide b if called
        svc = self.build(OrderAware(), fallback_model, tau=0.85)
        result = svc.cascade("Q?", "Which is right?", "right answer", "wrong answer")
        self.assertEqual(result["path"], "first_stage")
        self.assertEqual(result["decision"], "a")
        self.assertEqual(fallback_model.calls, 0)

    def test_unconfident_first_stage_escalates_to_the_fallback(self):
        svc = self.build(OrderAware(), FixedStep({"B": 1.0}), tau=0.95)
        result = svc.cascade("Q?", "Which is right?", "right answer", "wrong answer")
        self.assertEqual(result["path"], "escalated")
        self.assertEqual(result["decision"], "b")  # the fallback's decision replaces the first stage's

    def test_no_threshold_means_every_item_escalates(self):
        svc = self.build(OrderAware(), FixedStep({"A": 1.0}), tau=None)
        self.assertEqual(svc.cascade("Q?", "Which is right?", "right answer", "wrong answer")["path"], "escalated")

    def test_invalid_first_stage_always_escalates(self):
        svc = self.build(FixedStep({"Z": 1.0}), FixedStep({"A": 1.0}), tau=0.5)
        result = svc.cascade("Q?", "Which is right?", "x", "y")
        self.assertEqual(result["path"], "escalated")
        self.assertIsNone(result["first_stage"])
        self.assertEqual(result["decision"], "a")

    def test_cascade_route_needs_a_frozen_policy(self):
        svc = server.JevLite(OrderAware(), fallback=server.JevLite(FixedStep({"A": 1.0})), policy=None)
        with self.assertRaises(server.ModelUnavailable):
            server.route_cascade(svc, {"instructions": "Which?", "a": "x", "b": "y"})


class ThresholdRuleTests(unittest.TestCase):
    def records(self, high, low):
        # high: confident and right for both stages. low: first stage wrong at q=0.6, fallback right.
        return [{"valid": True, "q": 0.95, "choice": "a", "c1": 1.0, "c2": 1.0}] * high + \
               [{"valid": True, "q": 0.6, "choice": "a", "c1": 0.0, "c2": 1.0}] * low

    def test_picks_the_most_permissive_safe_threshold_and_ties_to_stricter(self):
        # Accepting the 0.6 items would lose to the fallback, so tau must be above 0.6.
        # Above 0.6 the gate accepts only the 0.95 items, and 0.95 is the stricter of the tied thresholds.
        for rule in ("point", "lower_bound"):
            self.assertEqual(cascade_eval.choose_threshold(self.records(20, 20), rule), 0.95)

    def test_returns_none_when_nothing_qualifies(self):
        recs = [{"valid": True, "q": 0.99, "choice": "a", "c1": 0.0, "c2": 1.0}] * 30
        self.assertIsNone(cascade_eval.choose_threshold(recs, "lower_bound"))

    def test_orientation_keeps_the_label_truthful(self):
        items = cascade_eval.orient(PAIRS)
        for item, (_, _, correct, _) in zip(items, PAIRS):
            self.assertEqual({"a": item["a"], "b": item["b"]}[item["truth"]], correct)


class StatsTests(unittest.TestCase):
    def test_auroc_ranks_errors_above_correct(self):
        self.assertAlmostEqual(cascade_eval.auroc([0.9, 0.1], [True, False]), 1.0)
        self.assertAlmostEqual(cascade_eval.auroc([0.1, 0.9], [True, False]), 0.0)

    def test_ece_is_zero_when_confidence_matches_accuracy(self):
        self.assertAlmostEqual(cascade_eval.ece([1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]), 0.0)


class CascadeHttpTests(unittest.TestCase):
    def test_cascade_route_answers_with_the_path(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        httpd.jev = server.JevLite(OrderAware(), fallback=server.JevLite(FixedStep({"B": 1.0})), policy={"tau": 0.85})
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            body = json.dumps({"state": "Q?", "instructions": "Which is right?", "a": "right answer", "b": "wrong answer"}).encode()
            request = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}/api/cascade", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=5) as response:
                data = json.loads(response.read())
            self.assertEqual(data["path"], "first_stage")
            self.assertEqual(data["decision"], "a")
        finally:
            httpd.shutdown()
            httpd.server_close()


if __name__ == "__main__":
    unittest.main()
