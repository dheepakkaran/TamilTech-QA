"""Tests for the automatic + Tanglish-specific evaluation metrics."""

from __future__ import annotations

import unittest

from src.evaluation.metrics import (
    answer_length_ratio,
    bleu_scores,
    code_switch_preservation_score,
    exact_match,
    rouge_scores,
    tamil_connector_fluency,
    technical_term_retention,
    token_f1,
)


class AutomaticMetricsTests(unittest.TestCase):
    """Smoke + sanity checks on the standard automatic metrics."""

    def test_exact_match_basic(self) -> None:
        self.assertEqual(exact_match(["Hello!"], ["hello"]), 1.0)
        self.assertEqual(exact_match(["foo"], ["bar"]), 0.0)

    def test_token_f1_full_overlap(self) -> None:
        self.assertEqual(token_f1(["a b c"], ["a b c"]), 1.0)

    def test_token_f1_partial(self) -> None:
        score = token_f1(["a b c d"], ["a b x y"])
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_bleu_runs_and_in_range(self) -> None:
        scores = bleu_scores(
            ["the gradient descent algorithm minimizes loss"],
            ["gradient descent minimizes the loss function"],
        )
        for k in ("bleu_1", "bleu_2", "bleu_4"):
            self.assertIn(k, scores)
            self.assertGreaterEqual(scores[k], 0.0)
            self.assertLessEqual(scores[k], 1.0)

    def test_rouge_runs_and_in_range(self) -> None:
        scores = rouge_scores(
            ["the gradient descent algorithm minimizes loss"],
            ["gradient descent minimizes the loss function"],
        )
        for k in ("rouge_1", "rouge_2", "rouge_l"):
            self.assertIn(k, scores)
            self.assertGreaterEqual(scores[k], 0.0)
            self.assertLessEqual(scores[k], 1.0)

    def test_answer_length_ratio(self) -> None:
        # pred half as long as ref → ratio 0.5
        self.assertAlmostEqual(answer_length_ratio(["a b"], ["a b c d"]), 0.5, places=4)


class TanglishMetricsTests(unittest.TestCase):
    """Sanity checks on CSPS / TTR / TCF."""

    def test_csps_perfect_when_same_ratio(self) -> None:
        # Identical strings → identical tanglish ratios → CSPS == 1.0
        pred = "indha function-a sollunga konjam"
        ref = "indha function-a sollunga konjam"
        self.assertAlmostEqual(
            code_switch_preservation_score([pred], [ref]), 1.0, places=4
        )

    def test_csps_drops_for_register_shift(self) -> None:
        ref = "indha gradient descent-a konjam explain pannunga"
        pred = "Gradient descent is an optimization method."
        score = code_switch_preservation_score([pred], [ref])
        self.assertLess(score, 1.0)

    def test_ttr_keeps_keyword(self) -> None:
        kws = ["gradient", "function"]
        score = technical_term_retention(
            ["gradient and function are explained"],
            ["the gradient of the function"],
            technical_keywords=kws,
        )
        self.assertEqual(score, 1.0)

    def test_ttr_drops_for_missing_keyword(self) -> None:
        kws = ["gradient", "function"]
        score = technical_term_retention(
            ["nothing here"],
            ["the gradient of the function"],
            technical_keywords=kws,
        )
        self.assertEqual(score, 0.0)

    def test_tcf_neutral_when_no_connectors_in_ref(self) -> None:
        score = tamil_connector_fluency(
            ["it works fine"],
            ["it works fine"],
            connectors=["enna na", "sollunga"],
        )
        # ref has 0 connectors, pred has 0 → neutral 0.5
        self.assertAlmostEqual(score, 0.5, places=4)

    def test_tcf_high_when_pred_matches_ref_connectors(self) -> None:
        score = tamil_connector_fluency(
            ["sollunga apdi patha"],
            ["sollunga apdi patha"],
            connectors=["sollunga", "apdi patha"],
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
