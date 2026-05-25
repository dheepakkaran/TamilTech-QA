"""Text cleaning pipeline with MinHash-LSH deduplication.

Stages, in order:

1. URL / HTML / emoji-ish junk removal.
2. Unicode NFC normalization and whitespace collapse.
3. Technical-keyword gate (drop if no keyword present).
4. Length gate (drop if < min_words or > max_tokens tokens).
5. Near-duplicate removal via MinHash + LSH (``datasketch``).

Counts at each stage are logged.
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
PUNCT_REPEAT_RE = re.compile(r"([!?.,])\1{2,}")
MULTI_WS_RE = re.compile(r"\s+")
NON_PRINTABLE_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class TextCleaner:
    """Apply normalization + filtering + dedup to a stream of records.

    Args:
        technical_keywords: Lower-cased keyword list. A record must contain at
            least one keyword (in question, answer, or raw_text) to survive
            the technical-content gate.
        min_words: Reject records with fewer than this many words.
        max_tokens: Reject records longer than this many whitespace tokens.
        minhash_threshold: Jaccard threshold for the LSH near-dup search.
        minhash_num_perm: Number of MinHash permutations.

    Example:
        >>> cleaner = TextCleaner(["function"], min_words=2, max_tokens=20)
        >>> cleaner.normalize("Hello!!!  <b>world</b> http://x.com")
        'Hello! world'
    """

    def __init__(
        self,
        technical_keywords: Iterable[str],
        min_words: int = 20,
        max_tokens: int = 512,
        minhash_threshold: float = 0.85,
        minhash_num_perm: int = 128,
    ) -> None:
        self.tech_kws = [k.lower() for k in technical_keywords]
        self.min_words = min_words
        self.max_tokens = max_tokens
        self.minhash_threshold = minhash_threshold
        self.minhash_num_perm = minhash_num_perm

    # ------------------------------------------------------------------ #
    # Stage helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def normalize(text: str) -> str:
        """Apply URL/HTML strip, NFC normalize, collapse whitespace, fix punct.

        Args:
            text: Raw input string.

        Returns:
            Normalized string.
        """
        if not text:
            return ""
        text = URL_RE.sub("", text)
        text = HTML_RE.sub("", text)
        text = NON_PRINTABLE_RE.sub(" ", text)
        text = unicodedata.normalize("NFC", text)
        text = PUNCT_REPEAT_RE.sub(r"\1", text)
        text = MULTI_WS_RE.sub(" ", text).strip()
        return text

    def has_technical_content(self, *fields: str) -> bool:
        """Return True if any technical keyword is present across the fields.

        Args:
            *fields: Strings to check.

        Returns:
            Whether at least one keyword matches.
        """
        blob = " ".join(f or "" for f in fields).lower()
        return any(kw in blob for kw in self.tech_kws)

    def length_ok(self, text: str) -> bool:
        """Check the length gate.

        Args:
            text: Combined text.

        Returns:
            True if ``min_words <= words <= max_tokens``.
        """
        n = len(text.split())
        return self.min_words <= n <= self.max_tokens

    # ------------------------------------------------------------------ #
    # Core pipeline
    # ------------------------------------------------------------------ #
    def clean_stream(self, records: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """Run the full cleaning pipeline over an iterable of records.

        Args:
            records: Iterable of input dicts.

        Returns:
            Tuple of (kept records, stage counters).
        """
        counts = {
            "read": 0,
            "after_normalize": 0,
            "after_tech_gate": 0,
            "after_length_gate": 0,
            "after_dedup": 0,
        }

        # ---- Stages 1-4: per-record streaming -------------------------
        staged: List[Dict[str, Any]] = []
        for rec in records:
            counts["read"] += 1
            rec = dict(rec)
            q = self.normalize(rec.get("question", ""))
            a = self.normalize(rec.get("answer", ""))
            rt = self.normalize(rec.get("raw_text", ""))
            if not (q or a or rt):
                continue
            counts["after_normalize"] += 1

            if not self.has_technical_content(q, a, rt):
                continue
            counts["after_tech_gate"] += 1

            combined = (q + " " + a).strip() or rt
            if not self.length_ok(combined):
                continue
            counts["after_length_gate"] += 1

            rec["question"] = q
            rec["answer"] = a
            rec["raw_text"] = rt or combined
            staged.append(rec)

        # ---- Stage 5: MinHash LSH dedup ------------------------------
        kept = self._dedup_minhash(staged)
        counts["after_dedup"] = len(kept)

        log.info(
            "Cleaning counts: read={read} norm={after_normalize} "
            "tech={after_tech_gate} len={after_length_gate} dedup={after_dedup}",
            **counts,
        )
        return kept, counts

    # ------------------------------------------------------------------ #
    def _dedup_minhash(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop near-duplicates with MinHash + LSH.

        Args:
            records: Records to deduplicate.

        Returns:
            Deduplicated list (preserves first occurrence).
        """
        if not records:
            return []
        try:
            from datasketch import MinHash, MinHashLSH
        except ImportError:  # pragma: no cover
            log.warning("datasketch not installed; skipping near-duplicate removal.")
            return records

        lsh = MinHashLSH(threshold=self.minhash_threshold, num_perm=self.minhash_num_perm)
        kept: List[Dict[str, Any]] = []
        for idx, rec in enumerate(records):
            text = (rec.get("question", "") + " " + rec.get("answer", "")).lower()
            shingles = _shingles(text, n=5)
            if not shingles:
                continue
            mh = MinHash(num_perm=self.minhash_num_perm)
            for s in shingles:
                mh.update(s.encode("utf-8"))
            if lsh.query(mh):
                continue
            lsh.insert(f"d{idx}", mh)
            kept.append(rec)
        return kept


def _shingles(text: str, n: int = 5) -> List[str]:
    """Return word-level n-gram shingles for a string.

    Args:
        text: Input lowercased text.
        n: Shingle size.

    Returns:
        List of shingle strings.
    """
    toks = re.findall(r"[A-Za-z஀-௿][A-Za-z஀-௿\-']*", text)
    if len(toks) < n:
        return [" ".join(toks)] if toks else []
    return [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]


# ---------------------------------------------------------------------- #
# Streaming helpers + CLI
# ---------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield JSON records from a JSONL file.

    Args:
        path: File path.

    Yields:
        Decoded dictionaries.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write records to a JSONL file.

    Args:
        records: Iterable of dicts.
        path: Destination path (parent is created).

    Returns:
        None.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(config_path: str, input_path: str, output_path: str) -> None:
    """Clean every JSONL under ``input_path`` and write merged output.

    If ``input_path`` is a directory, all ``*.jsonl`` files are concatenated.
    The output is always a single JSONL.

    Args:
        config_path: Path to data config YAML.
        input_path: Input file or directory.
        output_path: Output JSONL path.

    Returns:
        None.
    """
    cfg = load_config(config_path)
    pp = cfg["preprocessing"]
    cleaner = TextCleaner(
        technical_keywords=pp["technical_keywords"],
        min_words=pp.get("min_words", 20),
        max_tokens=pp.get("max_tokens", 512),
        minhash_threshold=pp.get("minhash", {}).get("threshold", 0.85),
        minhash_num_perm=pp.get("minhash", {}).get("num_perm", 128),
    )

    in_path = Path(input_path)
    out_path = Path(output_path)
    if in_path.is_dir():
        sources: List[Path] = sorted(in_path.glob("*.jsonl"))
    else:
        sources = [in_path]
    if not sources:
        log.warning("No input JSONL files found at {}", input_path)
        return

    def merged() -> Iterator[Dict[str, Any]]:
        for src in sources:
            log.info("Cleaning source: {}", src)
            yield from _read_jsonl(src)

    cleaned, counts = cleaner.clean_stream(merged())
    if out_path.is_dir() or output_path.endswith("/"):
        out_path = out_path / "cleaned.jsonl"
    _write_jsonl(cleaned, out_path)

    stats_path = out_path.parent / "clean_stats.json"
    stats_path.write_text(json.dumps(counts, indent=2), encoding="utf-8")
    log.info("Wrote {} cleaned records to {} (stats → {})", len(cleaned), out_path, stats_path)


def main() -> None:
    """CLI entrypoint for ``python -m src.preprocessing.text_cleaner``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Clean and deduplicate the merged corpus.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--input", required=True, help="JSONL file or directory.")
    p.add_argument("--output", required=True, help="Output JSONL file.")
    args = p.parse_args()
    run(args.config, args.input, args.output)


if __name__ == "__main__":
    main()
