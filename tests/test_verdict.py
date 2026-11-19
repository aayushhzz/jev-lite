"""Offline tests for the thinking fallback: verdict parsing and how judge_base uses it. No model needed."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import server  # noqa: E402


class FakeVerdict:
    """Stands in for OllamaVerdictModel: returns a scripted final answer, or raises InvalidAnswer."""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = 0

    def verdict(self, system, user, labels):
        self.calls += 1
        answer = self.answers.pop(0)
        if answer is None:
            raise server.InvalidAnswer("no option letter in the final answer")
        return answer


class ParseVerdictTests(unittest.TestCase):
    def test_plain_letter(self):
        self.assertEqual(server.parse_verdict("B", ["A", "B"]), "B")

    def test_markdown_bold(self):
        self.assertEqual(server.parse_verdict("**A**", ["A", "B"]), "A")

    def test_letter_inside_a_sentence(self):
        self.assertEqual(server.parse_verdict("The answer is B.", ["A", "B"]), "B")

    def test_last_standalone_letter_wins(self):
        self.assertEqual(server.parse_verdict("Option A is wrong, so B", ["A", "B"]), "B")

    def test_lowercase_article_is_not_a_letter(self):
        self.assertIsNone(server.parse_verdict("a reasonable answer", ["A", "B"]))

    def test_letter_outside_the_options_is_ignored(self):
        self.assertIsNone(server.parse_verdict("C", ["A", "B"]))
        self.assertIsNone(server.parse_verdict("", ["A", "B"]))


class ThinkingFallbackTests(unittest.TestCase):
    def test_verdict_model_sets_the_choice_without_a_probability(self):
        fallback = server.JevLite(FakeVerdict("B"))
        result = fallback.judge_base("Q?", "Which is right?", "wrong", "right")
        self.assertEqual(result["choice"], "b")
        self.assertTrue(result["verdict_only"])
        self.assertNotIn("p_a", result)

    def test_first_position_verdict_maps_to_a(self):
        result = server.JevLite(FakeVerdict("A")).judge_base("Q?", "Which is right?", "right", "wrong")
        self.assertEqual(result["choice"], "a")

    def test_invalid_verdict_propagates_so_the_caller_can_count_it_wrong(self):
        with self.assertRaises(server.InvalidAnswer):
            server.JevLite(FakeVerdict(None)).judge_base("Q?", "Which is right?", "x", "y")

    def test_cascade_escalates_to_the_thinking_fallback(self):
        class Unsure:
            def label_probs(self, system, user, labels):
                return {"A": 0.5, "B": 0.5}  # q = 0.5 is below any tau above 0.5

        fallback_model = FakeVerdict("A")
        svc = server.JevLite(Unsure(), fallback=server.JevLite(fallback_model), policy={"tau": 0.9}, temperature=1.0)
        result = svc.cascade("Q?", "Which is right?", "x", "y")
        self.assertEqual(result["path"], "escalated")
        self.assertEqual(result["decision"], "a")
        self.assertEqual(fallback_model.calls, 1)


if __name__ == "__main__":
    unittest.main()
