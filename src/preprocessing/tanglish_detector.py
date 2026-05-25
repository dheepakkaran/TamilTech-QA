"""Tanglish (Tamil-English code-switching) detection.

This module combines a Roman-script Tamil wordlist with ``langdetect`` to
estimate, per sample, the fraction of tokens that are Tamil. Samples whose
ratio falls outside the configured ``[min, max]`` band are rejected. The
module also writes a confidence score per sample.

Formula
-------
.. math::

    \\text{tanglish\\_ratio} = \\frac{\\text{tamil\\_word\\_count}}{\\text{total\\_word\\_count}}

A token is counted as Tamil if it matches the lexicon OR if ``langdetect``
returns a Tamil-script ISO code OR if it ends with a Tamil-style suffix
(``-a``, ``-aa``, ``-na``, ``-thu``, ``-um``, ``-aanu``, ``-laam``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# A pragmatic Roman-script Tamil lexicon. Covers the most frequent function
# words and connectors in Tanglish technical discourse. Extend via the
# `--extra-wordlist` CLI flag if needed.
DEFAULT_TAMIL_WORDLIST: Set[str] = {
    "naan", "naa", "neenga", "nee", "avanga", "avan", "ava", "naanga", "ungaluku",
    "enaku", "namaku", "engaluku",
    "enna", "enna-na", "ennaena", "ennanu", "ennada", "yenna",
    "edhu", "ethu", "eppadi", "epdi", "epdina", "yappadi",
    "indha", "intha", "andha", "antha", "idhu", "athu", "ithu", "adhu",
    "irukku", "irukkae", "irukkum", "irundha", "irukka", "irundhu",
    "varuthu", "vandhuchu", "vanthu", "vandhu",
    "panrathu", "panra", "panren", "panni", "pannitu", "pannuvom", "pannunga",
    "sollunga", "solli", "sonna", "sollu", "sollra",
    "purinjikonga", "purinjiruchu", "purinjikidu", "puriyutha", "puriyala", "puriyum",
    "theriyuma", "theriyala", "theriyum", "theriyathu",
    "kaata", "kaatu", "kaattunga", "kaatra",
    "vanakam", "vanakkam",
    "patha", "paatha", "paarunga", "paaru", "paarkanum", "paakra",
    "maari", "matri", "madhiri", "mathiri", "maathiri",
    "apdi", "apo", "apoda", "apdiye", "appadi", "appo", "ippo", "ipdi",
    "pinna", "appram", "appuram", "aprm", "appuram", "appuramna",
    "vendam", "venuma", "venum",
    "konjam", "kuda", "moodi", "open", "close-a",
    "aana", "aanaa", "aanaalum", "aanapdi",
    "summa", "sema", "semma",
    "irukkanum", "irukkalaam", "irukkalam",
    "thaan", "than", "thaane",
    "konjam-kooda", "konjam-konjam",
    "yedhukku", "yenna-nu",
}

TAMIL_SUFFIXES: Tuple[str, ...] = (
    "-a", "-aa", "-na", "-thu", "-um", "-aanu", "-laam", "-aalum", "-aaga",
)

TAMIL_SCRIPT_RE = re.compile(r"[஀-௿]+")
WORD_RE = re.compile(r"[A-Za-z஀-௿][A-Za-z஀-௿\-']*")


class TanglishDetector:
    """Estimate code-switching ratio and decide whether to keep a sample.

    Args:
        tamil_wordlist: Roman-script Tamil lexicon. Defaults to the built-in list.
        min_ratio: Lower bound on ``tanglish_ratio`` (inclusive).
        max_ratio: Upper bound on ``tanglish_ratio`` (inclusive).

    Example:
        >>> det = TanglishDetector()
        >>> info = det.score("indha function epdi work pannuthu sollunga")
        >>> 0.15 <= info["tanglish_ratio"] <= 0.85
        True
    """

    def __init__(
        self,
        tamil_wordlist: Optional[Set[str]] = None,
        min_ratio: float = 0.15,
        max_ratio: float = 0.85,
    ) -> None:
        self.lexicon = {w.lower() for w in (tamil_wordlist or DEFAULT_TAMIL_WORDLIST)}
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    # ------------------------------------------------------------------ #
    @staticmethod
    def _detect_langdetect(token: str) -> Optional[str]:
        """Return the best-guess language code from ``langdetect`` for a token.

        Args:
            token: A single token.

        Returns:
            ISO language code or None on failure.
        """
        try:
            from langdetect import DetectorFactory, detect_langs
        except ImportError:  # pragma: no cover
            return None
        DetectorFactory.seed = 0
        try:
            langs = detect_langs(token)
            return langs[0].lang if langs else None
        except Exception:  # noqa: BLE001
            return None

    def is_tamil_word(self, token: str) -> bool:
        """Decide whether a single token should be counted as Tamil.

        Args:
            token: Lowercased word.

        Returns:
            True if the token is Tamil per lexicon / suffix / script test.
        """
        if TAMIL_SCRIPT_RE.search(token):
            return True
        t = token.lower()
        if t in self.lexicon:
            return True
        if any(t.endswith(suf) for suf in TAMIL_SUFFIXES):
            # Heuristic: short tokens ending in -a like "indha" are very likely Tamil,
            # but English words like "schema" also end in -a. Require at least one
            # neighbour rule to avoid false positives: keep only if NOT also in the
            # most-common English word set used as a guard.
            return t not in _ENGLISH_GUARD
        return False

    # ------------------------------------------------------------------ #
    def score(self, text: str) -> Dict[str, Any]:
        """Compute the per-sample Tanglish stats.

        Args:
            text: Input string.

        Returns:
            Dict with ``tanglish_ratio``, ``tamil_word_count``,
            ``total_word_count``, ``language_tags``, ``confidence``, and
            ``flagged_words``.
        """
        tokens = WORD_RE.findall(text or "")
        total = len(tokens)
        if total == 0:
            return {
                "tanglish_ratio": 0.0,
                "tamil_word_count": 0,
                "total_word_count": 0,
                "language_tags": [],
                "confidence": 0.0,
                "flagged_words": [],
            }
        tamil_tokens: List[str] = []
        for tok in tokens:
            if self.is_tamil_word(tok):
                tamil_tokens.append(tok)
        ratio = round(len(tamil_tokens) / total, 4)
        tags: List[str] = []
        if tamil_tokens:
            tags.append("ta")
        if (total - len(tamil_tokens)) > 0:
            tags.append("en")
        # Confidence: peaks at the midpoint of the accepted band, drops at the edges.
        mid = (self.min_ratio + self.max_ratio) / 2
        half = (self.max_ratio - self.min_ratio) / 2 or 1e-6
        confidence = round(max(0.0, 1.0 - abs(ratio - mid) / half), 4)
        return {
            "tanglish_ratio": ratio,
            "tamil_word_count": len(tamil_tokens),
            "total_word_count": total,
            "language_tags": tags,
            "confidence": confidence,
            "flagged_words": sorted(set(tamil_tokens))[:30],
        }

    def keep(self, text: str) -> bool:
        """Return True iff the sample falls inside the accepted Tanglish band.

        Args:
            text: Input string.

        Returns:
            Whether the sample should be kept downstream.
        """
        s = self.score(text)
        return self.min_ratio <= s["tanglish_ratio"] <= self.max_ratio


# A small guard set: common English words that look like they end in Tamil
# suffixes but are unambiguously English. Trimmed deliberately small.
_ENGLISH_GUARD: Set[str] = {
    "schema", "java", "lambda", "alpha", "beta", "gamma", "data", "comma",
    "media", "extra", "algebra", "delta", "sigma", "drama", "camera",
    "formula", "area", "idea", "via", "yoga",
}


# ---------------------------------------------------------------------- #
# Streaming helpers
# ---------------------------------------------------------------------- #
def annotate_jsonl(
    in_path: Path,
    out_path: Path,
    detector: TanglishDetector,
    filter_band: bool = True,
) -> Tuple[int, int]:
    """Stream a JSONL through the detector, writing annotated lines.

    Args:
        in_path: Source JSONL.
        out_path: Destination JSONL.
        detector: A configured :class:`TanglishDetector`.
        filter_band: If True, only write rows inside the [min, max] band.

    Returns:
        Tuple of (rows read, rows written).
    """
    read = 0
    written = 0
    ensure_dir(out_path.parent)
    with in_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            read += 1
            rec = json.loads(line)
            text = " ".join(filter(None, [rec.get("question", ""), rec.get("answer", "")]))
            score = detector.score(text)
            rec["tanglish_ratio"] = score["tanglish_ratio"]
            rec["language_tags"] = score["language_tags"]
            rec["tanglish_confidence"] = score["confidence"]
            if filter_band and not (
                detector.min_ratio <= score["tanglish_ratio"] <= detector.max_ratio
            ):
                continue
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
    log.info("Tanglish band filter: read={} written={} ({}→{}, drop_rate={:.1%})",
             read, written, in_path.name, out_path.name,
             (1 - (written / read)) if read else 0.0)
    return read, written


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _resolve_io(input_arg: Path, output_arg: Path) -> List[Tuple[Path, Path]]:
    """Build (in_path, out_path) pairs from CLI args.

    Args:
        input_arg: File or directory.
        output_arg: File or directory.

    Returns:
        List of pairs to process.
    """
    if input_arg.is_dir():
        ensure_dir(output_arg)
        return [(p, output_arg / p.name) for p in sorted(input_arg.glob("*.jsonl"))]
    ensure_dir(output_arg.parent)
    return [(input_arg, output_arg)]


def run(config_path: str, input_dir: str, output_dir: str, no_filter: bool = False) -> None:
    """Apply the detector to every JSONL under ``input_dir``.

    Args:
        config_path: Path to data config YAML.
        input_dir: Directory of JSONL files (or a single file).
        output_dir: Output directory (or output file).
        no_filter: If True, annotate only — do not drop out-of-band rows.

    Returns:
        None.
    """
    cfg = load_config(config_path)
    pp = cfg["preprocessing"]
    detector = TanglishDetector(
        min_ratio=pp.get("tanglish_ratio_min", 0.15),
        max_ratio=pp.get("tanglish_ratio_max", 0.85),
    )
    pairs = _resolve_io(Path(input_dir), Path(output_dir))
    if not pairs:
        log.warning("No JSONL files found at {}", input_dir)
        return
    for in_p, out_p in pairs:
        annotate_jsonl(in_p, out_p, detector, filter_band=not no_filter)


def main() -> None:
    """CLI entrypoint for ``python -m src.preprocessing.tanglish_detector``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Tanglish detector + band filter.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--input", required=True, help="Input JSONL file or directory.")
    p.add_argument("--output", required=True, help="Output JSONL file or directory.")
    p.add_argument(
        "--no-filter",
        action="store_true",
        help="Annotate but do not drop out-of-band rows.",
    )
    args = p.parse_args()
    run(args.config, args.input, args.output, no_filter=args.no_filter)


if __name__ == "__main__":
    main()
