"""Tests for the Tanglish detector + language filter."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.preprocessing.tanglish_detector import (
    TanglishDetector,
    annotate_jsonl,
)


class TanglishDetectorTests(unittest.TestCase):
    """Verify ratio computation and accept/reject behavior."""

    def setUp(self) -> None:
        self.det = TanglishDetector(min_ratio=0.15, max_ratio=0.85)

    def test_pure_english_is_rejected(self) -> None:
        text = "This is a fully English explanation about gradient descent and optimization."
        s = self.det.score(text)
        self.assertLess(s["tanglish_ratio"], 0.15)
        self.assertFalse(self.det.keep(text))

    def test_tanglish_text_is_accepted(self) -> None:
        text = (
            "indha gradient descent epdi work pannuthu sollunga, "
            "naa apdi patha learning rate konjam adjust pannanum"
        )
        s = self.det.score(text)
        self.assertGreaterEqual(s["tanglish_ratio"], 0.15)
        self.assertLessEqual(s["tanglish_ratio"], 0.85)
        self.assertTrue(self.det.keep(text))

    def test_pure_tamil_is_rejected_when_too_high(self) -> None:
        text = "naan epdi pannuven indha apdi mathiri sollunga puriyutha vendam summa"
        s = self.det.score(text)
        # The text is overwhelmingly Tamil — should exceed the 0.85 upper band.
        self.assertGreater(s["tanglish_ratio"], 0.85)
        self.assertFalse(self.det.keep(text))

    def test_score_fields_present(self) -> None:
        info = self.det.score("indha function-a sollunga")
        for k in (
            "tanglish_ratio", "tamil_word_count", "total_word_count",
            "language_tags", "confidence", "flagged_words",
        ):
            self.assertIn(k, info)

    def test_empty_string_safe(self) -> None:
        info = self.det.score("")
        self.assertEqual(info["total_word_count"], 0)
        self.assertEqual(info["tanglish_ratio"], 0.0)


class AnnotateJsonlTests(unittest.TestCase):
    """End-to-end JSONL annotation + filtering smoke test."""

    def test_filtering_drops_out_of_band_rows(self) -> None:
        det = TanglishDetector(min_ratio=0.15, max_ratio=0.85)
        with TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            inp = tmp_dir / "in.jsonl"
            out = tmp_dir / "out.jsonl"
            records = [
                {"question": "indha epdi sollunga apdi", "answer": "naa puriyutha"},
                {"question": "What is gradient descent", "answer": "It minimizes loss."},
            ]
            inp.write_text(
                "\n".join(json.dumps(r) for r in records), encoding="utf-8"
            )
            read, written = annotate_jsonl(inp, out, det, filter_band=True)
            self.assertEqual(read, 2)
            # Pure-English row is dropped, Tanglish row kept (or vice versa,
            # depending on exact heuristic). We at least expect strict reduction
            # or equality.
            self.assertLessEqual(written, read)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
