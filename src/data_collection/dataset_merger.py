"""Merge YouTube, synthetic, and public HuggingFace datasets into one corpus.

Outputs a single unified JSONL at ``data/raw/merged_corpus.jsonl`` with the
canonical schema described in the prompt:

.. code-block:: json

    {
      "id": "tamiltech_001",
      "source": "youtube|synthetic|existing",
      "question": "...",
      "answer": "...",
      "topic": "python|ml|ece|dsa|web|os|general",
      "difficulty": "easy|medium|hard|unknown",
      "tanglish_ratio": 0.0,
      "language_tags": ["ta", "en"],
      "raw_text": "..."
    }

The ``tanglish_ratio`` filled here is a quick lexical estimate; the
preprocessing stage refines it.
"""

from __future__ import annotations

import argparse
import json
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

# Coarse topic detection from text — kept simple and fully data-driven.
TOPIC_KEYWORDS: Dict[str, List[str]] = {
    "python": ["python", "decorator", "list comprehension", "lambda", "tuple", "django", "flask"],
    "ml": ["machine learning", "neural network", "gradient", "cnn", "rnn", "transformer",
           "overfit", "regression", "classification", "tensor", "model training"],
    "ece": ["transistor", "op-amp", "opamp", "circuit", "voltage", "current", "kirchhoff",
            "signal processing", "fourier", "vlsi", "digital logic"],
    "dsa": ["array", "linked list", "binary tree", "graph", "hash map", "stack",
            "queue", "heap", "sorting", "searching", "dynamic programming", "recursion"],
    "web": ["rest", "api", "http", "frontend", "backend", "react", "node", "javascript",
            "html", "css", "fetch", "ajax", "database", "sql"],
    "os": ["operating system", "process", "thread", "deadlock", "scheduling", "memory",
           "page table", "kernel", "context switch", "semaphore"],
    "networking": ["tcp", "udp", "osi", "subnet", "router", "dns", "http", "https"],
    "databases": ["sql", "join", "index", "schema", "transaction", "nosql", "mongodb"],
    "algorithms": ["complexity", "big-o", "greedy", "divide and conquer", "graph algorithm"],
    "debugging": ["debug", "error", "exception", "stack trace", "breakpoint", "log"],
}

TAMIL_HINT_WORDS = {
    "enna", "na", "naa", "apdi", "patha", "paatha", "indha", "intha", "maari",
    "sollunga", "puriyutha", "puriyala", "puriyum", "theriyuma", "theriyala",
    "vanakam", "vanakkam", "paaru", "panrathu", "panra", "panren", "irukku",
    "irukum", "epdi", "yenna", "ennoda", "unga", "unakku", "vendam", "kaata",
    "kelvi", "answer-a", "concept-a", "code-a", "neenga",
}

URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")


def _quick_tanglish_ratio(text: str) -> float:
    """Quick heuristic Tanglish ratio used at merge time.

    The preprocessing stage replaces this with the rigorous detector. We need a
    cheap estimate so we can skim obvious non-Tanglish content out.

    Args:
        text: Input string.

    Returns:
        Estimated fraction of tokens that look Tamil (0.0 - 1.0).

    Example:
        >>> _quick_tanglish_ratio("indha function epdi work pannuthu")
        0.6
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())
    if not tokens:
        return 0.0
    tamil = sum(1 for t in tokens if t in TAMIL_HINT_WORDS)
    return round(tamil / len(tokens), 3)


def _detect_topic(text: str) -> str:
    """Pick the most likely topic by keyword count.

    Args:
        text: Input string.

    Returns:
        One of the canonical topic labels or ``"general"``.
    """
    text_l = text.lower()
    best_topic, best_score = "general", 0
    for topic, kws in TOPIC_KEYWORDS.items():
        score = sum(1 for kw in kws if kw in text_l)
        if score > best_score:
            best_topic, best_score = topic, score
    return best_topic


def _strip_noise(text: str) -> str:
    """Strip URLs and HTML; collapse whitespace.

    Args:
        text: Input string.

    Returns:
        Cleaned string.
    """
    text = URL_RE.sub("", text or "")
    text = HTML_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------- #
# Loaders for each source
# ---------------------------------------------------------------------- #
def iter_youtube_records(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield unified records from all YouTube JSONL files in ``raw_dir``.

    Args:
        raw_dir: Directory containing ``youtube_comments_*.jsonl`` files.

    Yields:
        Canonical record dicts (with ``source="youtube"``).
    """
    for jsonl in sorted(raw_dir.glob("youtube_comments_*.jsonl")):
        log.info("Reading {}", jsonl)
        with jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                vid = json.loads(line)
                vid_title = _strip_noise(vid.get("title", ""))
                vid_desc = _strip_noise(vid.get("description", ""))
                for c in vid.get("comments", []) or []:
                    q_text = _strip_noise(c.get("comment_text", ""))
                    if not q_text:
                        continue
                    replies = c.get("reply_text") or []
                    a_text = _strip_noise(" ".join(replies)) if replies else ""
                    if not a_text:
                        # Comment with no reply: treat as raw_text only.
                        a_text = vid_title
                    raw_blob = " | ".join(filter(None, [vid_title, q_text, a_text, vid_desc]))
                    yield {
                        "question": q_text,
                        "answer": a_text,
                        "topic": _detect_topic(raw_blob),
                        "difficulty": "unknown",
                        "tanglish_ratio": _quick_tanglish_ratio(raw_blob),
                        "language_tags": ["ta", "en"],
                        "raw_text": raw_blob,
                        "source": "youtube",
                    }


def iter_synthetic_records(raw_dir: Path) -> Iterator[Dict[str, Any]]:
    """Yield unified records from ``synthetic_qa.jsonl``.

    Args:
        raw_dir: Directory containing ``synthetic_qa.jsonl``.

    Yields:
        Canonical record dicts (with ``source="synthetic"``).
    """
    path = raw_dir / "synthetic_qa.jsonl"
    if not path.exists():
        log.warning("No synthetic file at {}", path)
        return
    log.info("Reading {}", path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            it = json.loads(line)
            q = _strip_noise(it.get("question", ""))
            a = _strip_noise(it.get("answer", ""))
            blob = f"{q} {a}"
            yield {
                "question": q,
                "answer": a,
                "topic": it.get("topic") or _detect_topic(blob),
                "difficulty": it.get("difficulty", "medium"),
                "tanglish_ratio": _quick_tanglish_ratio(blob),
                "language_tags": ["ta", "en"],
                "raw_text": blob,
                "source": "synthetic",
            }


def iter_existing_records(cfg: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Yield unified records from configured HuggingFace datasets.

    Only records with at least one technical keyword are kept.

    Args:
        cfg: The full data config dict.

    Yields:
        Canonical record dicts (with ``source="existing"``).
    """
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        log.warning("datasets not installed; skipping existing datasets.")
        return

    tech_kws = [k.lower() for k in cfg["preprocessing"]["technical_keywords"]]

    for entry in cfg.get("existing_datasets", []) or []:
        if not entry.get("enable", True):
            continue
        hf_id = entry["hf_id"]
        split = entry.get("split", "train")
        field = entry.get("text_field", "text")
        try:
            ds = load_dataset(hf_id, split=split)
        except Exception as e:  # noqa: BLE001
            log.warning("Could not load {} ({}): {}", hf_id, split, e)
            continue

        log.info("Loaded {} with {} rows; filtering for technical content.", hf_id, len(ds))
        kept = 0
        for row in ds:
            text = ""
            if isinstance(row, dict):
                text = row.get(field) or row.get("text") or ""
                if isinstance(text, list):
                    text = " ".join(str(x) for x in text)
            if not isinstance(text, str) or not text.strip():
                continue
            tl = text.lower()
            if not any(kw in tl for kw in tech_kws):
                continue
            text = _strip_noise(text)
            yield {
                "question": text,
                "answer": "",
                "topic": _detect_topic(text),
                "difficulty": "unknown",
                "tanglish_ratio": _quick_tanglish_ratio(text),
                "language_tags": ["ta", "en"],
                "raw_text": text,
                "source": "existing",
            }
            kept += 1
        log.info("From {}: kept {} technical rows.", hf_id, kept)


# ---------------------------------------------------------------------- #
# Merge
# ---------------------------------------------------------------------- #
def merge_all(config_path: str, out_path: Optional[Path] = None) -> Tuple[Path, Dict[str, int]]:
    """Merge all three sources to one JSONL and return per-source counts.

    Args:
        config_path: Path to ``config/data_config.yaml``.
        out_path: Optional explicit output path.

    Returns:
        Tuple of (output path, per-source count dict).
    """
    cfg = load_config(config_path)
    raw_dir = project_root() / cfg["paths"]["raw_dir"]
    ensure_dir(raw_dir)
    out_path = out_path or (raw_dir / "merged_corpus.jsonl")

    counts: Dict[str, int] = {"youtube": 0, "synthetic": 0, "existing": 0}
    next_idx = 1

    iters: Iterable[Iterator[Dict[str, Any]]] = (
        iter_youtube_records(raw_dir),
        iter_synthetic_records(raw_dir),
        iter_existing_records(cfg),
    )

    with out_path.open("w", encoding="utf-8") as f:
        for it in iters:
            for rec in it:
                rec["id"] = f"tamiltech_{next_idx:06d}"
                # tag with a per-record uuid for traceability if same `id` is regenerated
                rec["uuid"] = uuid.uuid4().hex
                counts[rec["source"]] += 1
                next_idx += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    log.info("Merged corpus written to {} (counts={})", out_path, counts)
    return out_path, counts


def main() -> None:
    """CLI entrypoint for ``python -m src.data_collection.dataset_merger``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Merge YouTube / synthetic / public datasets.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--out", default=None, help="Optional output JSONL path.")
    args = p.parse_args()
    merge_all(args.config, out_path=Path(args.out) if args.out else None)


if __name__ == "__main__":
    main()
