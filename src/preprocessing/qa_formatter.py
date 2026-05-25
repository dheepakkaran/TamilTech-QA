"""Convert the cleaned corpus into Alpaca + ChatML formats and split it.

Outputs (under ``data/final/``):

- ``train.jsonl``, ``val.jsonl``, ``test.jsonl``  — Alpaca-style records
  (``instruction``, ``input``, ``output``) plus the ChatML-rendered string
  in a ``chatml`` field, plus the original metadata.
- HuggingFace ``DatasetDict`` saved with ``save_to_disk``.
- ``dataset_stats.json`` with the publication-ready statistics table.

The split is stratified by ``(topic, difficulty)`` so each subset's topic and
difficulty distribution closely matches the full corpus.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def to_alpaca(record: Dict[str, Any], instruction: str) -> Dict[str, Any]:
    """Convert a cleaned record into the Alpaca instruction-tuning schema.

    Args:
        record: Cleaned record with ``question`` and ``answer`` fields.
        instruction: System-level instruction string to set on every sample.

    Returns:
        Dict with ``instruction``, ``input``, ``output``.

    Example:
        >>> r = {"question": "Q?", "answer": "A."}
        >>> out = to_alpaca(r, "Be helpful.")
        >>> sorted(out)
        ['input', 'instruction', 'output']
    """
    return {
        "instruction": instruction,
        "input": record.get("question", ""),
        "output": record.get("answer", ""),
    }


def to_chatml(record: Dict[str, Any], system_msg: str) -> str:
    """Render a record as a ChatML conversation string.

    Args:
        record: Cleaned record.
        system_msg: System message text.

    Returns:
        ChatML-formatted multi-turn string.

    Example:
        >>> s = to_chatml({"question": "Q", "answer": "A"}, "sys")
        >>> "<|im_start|>system" in s and "<|im_end|>" in s
        True
    """
    q = record.get("question", "")
    a = record.get("answer", "")
    return (
        "<|im_start|>system\n"
        f"{system_msg}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{q}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{a}\n"
        "<|im_end|>"
    )


# ---------------------------------------------------------------------- #
# Stratified split
# ---------------------------------------------------------------------- #
def stratified_split(
    records: List[Dict[str, Any]],
    train: float = 0.8,
    val: float = 0.1,
    test: float = 0.1,
    stratify_by: Tuple[str, ...] = ("topic", "difficulty"),
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split records into train/val/test while preserving strata proportions.

    Args:
        records: Input records.
        train: Train fraction.
        val: Validation fraction.
        test: Test fraction.
        stratify_by: Tuple of field names that define each stratum.
        seed: RNG seed.

    Returns:
        Tuple of (train, val, test).

    Raises:
        ValueError: If the fractions do not approximately sum to 1.
    """
    total = train + val + test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")

    rng = random.Random(seed)
    buckets: Dict[Tuple[str, ...], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        key = tuple(str(r.get(k, "unknown")) for k in stratify_by)
        buckets[key].append(r)

    tr: List[Dict[str, Any]] = []
    va: List[Dict[str, Any]] = []
    te: List[Dict[str, Any]] = []
    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        # ensure no overflow when n is tiny
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        tr.extend(items[:n_tr])
        va.extend(items[n_tr : n_tr + n_va])
        te.extend(items[n_tr + n_va :])

    rng.shuffle(tr)
    rng.shuffle(va)
    rng.shuffle(te)
    return tr, va, te


# ---------------------------------------------------------------------- #
# Stats
# ---------------------------------------------------------------------- #
def compute_stats(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Compute publication-grade dataset statistics across splits.

    Args:
        splits: Dict with keys ``train``, ``val``, ``test``.

    Returns:
        Stats dict suitable for ``json.dump``.
    """
    stats: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []
    for name, recs in splits.items():
        all_records.extend(recs)
        topic_dist = Counter(r.get("topic", "unknown") for r in recs)
        diff_dist = Counter(r.get("difficulty", "unknown") for r in recs)
        q_lens = [len(str(r.get("question", "")).split()) for r in recs]
        a_lens = [len(str(r.get("output", r.get("answer", ""))).split()) for r in recs]
        tr_ratios = [float(r.get("tanglish_ratio", 0.0)) for r in recs]
        stats[name] = {
            "n_samples": len(recs),
            "per_topic": dict(topic_dist),
            "per_difficulty": dict(diff_dist),
            "avg_question_tokens": round(_mean(q_lens), 2),
            "avg_answer_tokens": round(_mean(a_lens), 2),
            "avg_tanglish_ratio": round(_mean(tr_ratios), 4),
        }

    vocab: set[str] = set()
    for r in all_records:
        for field in ("question", "answer", "output"):
            v = r.get(field, "")
            if isinstance(v, str):
                vocab.update(v.lower().split())
    stats["overall"] = {
        "n_samples": len(all_records),
        "vocab_size": len(vocab),
        "sources": dict(Counter(r.get("source", "unknown") for r in all_records)),
    }
    return stats


def _mean(xs: Iterable[float]) -> float:
    """Return the arithmetic mean of an iterable of numbers (0.0 if empty).

    Args:
        xs: Numbers.

    Returns:
        Mean as a float.
    """
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        path: Path to JSONL.

    Returns:
        List of records.
    """
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    """Write a sequence of dicts to a JSONL file.

    Args:
        records: Records.
        path: Output path.

    Returns:
        None.
    """
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_and_split(
    input_path: str,
    output_dir: str,
    config_path: str,
) -> Dict[str, Any]:
    """Format the cleaned corpus into Alpaca + ChatML and split it.

    Args:
        input_path: Path to the cleaned JSONL.
        output_dir: Directory for ``train/val/test.jsonl`` and stats.
        config_path: Path to data config YAML.

    Returns:
        The stats dict that was written to ``dataset_stats.json``.
    """
    cfg = load_config(config_path)
    fmt = cfg["formatting"]
    sp = cfg["split"]

    out_dir = ensure_dir(Path(output_dir))
    raw_records = _read_jsonl(Path(input_path))
    log.info("Loaded {} cleaned records.", len(raw_records))

    # Build formatted records: keep metadata, add Alpaca fields + chatml.
    formatted: List[Dict[str, Any]] = []
    for r in raw_records:
        alpaca = to_alpaca(r, fmt["alpaca_instruction"])
        chatml = to_chatml(r, fmt["chatml_system"])
        formatted.append(
            {
                **r,
                **alpaca,
                "chatml": chatml,
            }
        )

    tr, va, te = stratified_split(
        formatted,
        train=sp.get("train_ratio", 0.8),
        val=sp.get("val_ratio", 0.1),
        test=sp.get("test_ratio", 0.1),
        stratify_by=tuple(sp.get("stratify_by", ["topic", "difficulty"])),
        seed=cfg.get("seed", 42),
    )

    _write_jsonl(tr, out_dir / "train.jsonl")
    _write_jsonl(va, out_dir / "val.jsonl")
    _write_jsonl(te, out_dir / "test.jsonl")
    log.info("Wrote train={} val={} test={} to {}", len(tr), len(va), len(te), out_dir)

    # Optionally also save as a HuggingFace DatasetDict
    try:
        from datasets import Dataset, DatasetDict  # noqa: WPS433

        dsd = DatasetDict(
            {
                "train": Dataset.from_list(tr),
                "validation": Dataset.from_list(va),
                "test": Dataset.from_list(te),
            }
        )
        hf_dir = out_dir / "hf_dataset"
        dsd.save_to_disk(str(hf_dir))
        log.info("HuggingFace DatasetDict saved to {}", hf_dir)
    except ImportError:  # pragma: no cover
        log.warning("`datasets` not installed; skipping HF DatasetDict export.")
    except Exception as e:  # noqa: BLE001
        log.warning("Could not save HF DatasetDict: {}", e)

    stats = compute_stats({"train": tr, "val": va, "test": te})
    stats_path = out_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Stats written to {}", stats_path)
    return stats


def main() -> None:
    """CLI entrypoint for ``python -m src.preprocessing.qa_formatter``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Format + split the cleaned corpus.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--input", required=True, help="Cleaned JSONL.")
    p.add_argument("--output", required=True, help="Directory to write splits + stats.")
    args = p.parse_args()
    format_and_split(args.input, args.output, args.config)


if __name__ == "__main__":
    main()
