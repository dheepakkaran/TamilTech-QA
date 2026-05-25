"""Tests for the Alpaca/ChatML formatter and the stratified splitter."""

from __future__ import annotations

import json
import unittest
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

from src.preprocessing.qa_formatter import (
    compute_stats,
    stratified_split,
    to_alpaca,
    to_chatml,
)


class FormattingTests(unittest.TestCase):
    """Schema-level checks for the Alpaca and ChatML renderers."""

    def test_alpaca_schema(self) -> None:
        rec = {"question": "Q?", "answer": "A."}
        out = to_alpaca(rec, "Be helpful.")
        self.assertEqual(set(out), {"instruction", "input", "output"})
        self.assertEqual(out["input"], "Q?")
        self.assertEqual(out["output"], "A.")

    def test_chatml_contains_markers(self) -> None:
        rec = {"question": "Q?", "answer": "A."}
        s = to_chatml(rec, "system msg")
        self.assertIn("<|im_start|>system", s)
        self.assertIn("<|im_start|>user", s)
        self.assertIn("<|im_start|>assistant", s)
        self.assertIn("<|im_end|>", s)


class StratifiedSplitTests(unittest.TestCase):
    """The split should preserve topic/difficulty proportions within rounding."""

    def _records(self, n_per_bucket: int = 20):
        recs = []
        for topic in ("python", "ml", "dsa"):
            for diff in ("easy", "medium", "hard"):
                for i in range(n_per_bucket):
                    recs.append(
                        {
                            "question": f"{topic} {diff} {i}",
                            "answer": "x" * 10,
                            "topic": topic,
                            "difficulty": diff,
                        }
                    )
        return recs

    def test_fractions_sum_check(self) -> None:
        with self.assertRaises(ValueError):
            stratified_split([], train=0.5, val=0.2, test=0.2)

    def test_split_sizes(self) -> None:
        recs = self._records(20)
        tr, va, te = stratified_split(recs, train=0.8, val=0.1, test=0.1, seed=1)
        self.assertEqual(len(tr) + len(va) + len(te), len(recs))
        # Each stratum should put roughly the right number into each split.
        for topic in ("python", "ml", "dsa"):
            n_topic = sum(1 for r in recs if r["topic"] == topic)
            n_topic_tr = sum(1 for r in tr if r["topic"] == topic)
            self.assertAlmostEqual(n_topic_tr / n_topic, 0.8, delta=0.1)

    def test_stats_payload(self) -> None:
        recs = self._records(5)
        tr, va, te = stratified_split(recs, seed=2)
        stats = compute_stats({"train": tr, "val": va, "test": te})
        self.assertEqual(stats["overall"]["n_samples"], len(recs))
        self.assertIn("per_topic", stats["train"])
        self.assertIn("vocab_size", stats["overall"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
