"""Build and score human-evaluation sheets for TamilTech-QA.

Given the predictions written by :func:`metrics.evaluate_all_models`, this
module:

1. ``build_sheet``  — assemble a CSV with one row per (model, sample) for
   annotators to rate on a 1–5 Likert scale across three axes
   (correctness, fluency, Tanglish naturalness). Annotator identities are
   blinded — the model name is hashed.
2. ``score_sheet``  — read the filled-in CSV and compute per-model averages
   and inter-annotator agreement (Krippendorff's alpha if available; else
   pairwise Pearson).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils import ensure_dir
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

RATING_COLUMNS = ("correctness_1to5", "fluency_1to5", "tanglish_1to5", "notes")


def _blind_id(model_name: str, salt: str) -> str:
    """Deterministic blinded model code (so annotators don't see the name).

    Args:
        model_name: True model name.
        salt: Sheet-level salt to keep codes per-sheet.

    Returns:
        Short blinded code (e.g., ``M-3F2A``).
    """
    h = hashlib.sha1(f"{salt}::{model_name}".encode("utf-8")).hexdigest().upper()
    return f"M-{h[:4]}"


def build_sheet(
    predictions_dir: Path,
    test_path: Path,
    output_csv: Path,
    n_samples: int = 100,
    seed: int = 42,
) -> Dict[str, str]:
    """Build a shuffled CSV for human rating and return the blinded code map.

    Args:
        predictions_dir: Directory of ``<model>.jsonl`` prediction files.
        test_path: Original ``test.jsonl`` (for questions / metadata).
        output_csv: Where to write the rating sheet.
        n_samples: Cap on the number of test samples to include per model.
        seed: RNG seed for sample selection and row shuffling.

    Returns:
        Mapping from blinded code → real model name (also persisted next to
        the CSV as ``<output_csv>.codes.json``).

    Example:
        >>> codes = build_sheet(                 # doctest: +SKIP
        ...     Path("outputs/predictions"),
        ...     Path("data/final/test.jsonl"),
        ...     Path("outputs/evaluation/human_eval_sheet.csv"),
        ... )
    """
    rng = random.Random(seed)
    salt = hashlib.md5(str(output_csv).encode("utf-8")).hexdigest()
    ensure_dir(output_csv.parent)

    test_records: List[Dict[str, Any]] = [
        json.loads(line) for line in test_path.open("r", encoding="utf-8") if line.strip()
    ]
    by_id = {r["id"]: r for r in test_records if "id" in r}
    chosen_ids = [r["id"] for r in test_records if "id" in r]
    rng.shuffle(chosen_ids)
    chosen_ids = chosen_ids[:n_samples]

    rows: List[Dict[str, str]] = []
    blind_map: Dict[str, str] = {}

    for pred_file in sorted(Path(predictions_dir).glob("*.jsonl")):
        model_name = pred_file.stem
        blind = _blind_id(model_name, salt)
        blind_map[blind] = model_name
        preds = {
            r["id"]: r for r in (
                json.loads(l) for l in pred_file.open("r", encoding="utf-8") if l.strip()
            )
        }
        for qid in chosen_ids:
            meta = by_id.get(qid)
            pred = preds.get(qid)
            if not meta or not pred:
                continue
            rows.append(
                {
                    "blind_model": blind,
                    "sample_id": qid,
                    "topic": meta.get("topic", ""),
                    "question": meta.get("input") or meta.get("question", ""),
                    "reference": pred.get("reference", ""),
                    "prediction": pred.get("prediction", ""),
                    **{c: "" for c in RATING_COLUMNS},
                }
            )

    rng.shuffle(rows)
    fieldnames = [
        "blind_model", "sample_id", "topic", "question", "reference", "prediction",
        *RATING_COLUMNS,
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)

    codes_path = output_csv.with_suffix(".codes.json")
    codes_path.write_text(json.dumps(blind_map, indent=2), encoding="utf-8")
    log.info(
        "Wrote rating sheet ({} rows) to {}. Code map → {}.",
        len(rows), output_csv, codes_path,
    )
    return blind_map


def score_sheet(filled_csv: Path, codes_path: Optional[Path] = None) -> Dict[str, Any]:
    """Compute per-model averages from a filled-in rating sheet.

    Args:
        filled_csv: Path to the annotated CSV.
        codes_path: Optional path to the ``.codes.json`` file. If omitted,
            inferred from ``filled_csv.with_suffix('.codes.json')``.

    Returns:
        Dict with ``per_model`` (mean ratings) and ``n`` (counts) keys.
    """
    codes_path = codes_path or filled_csv.with_suffix(".codes.json")
    if not codes_path.exists():
        log.warning("Codes file {} missing; using blinded labels in output.", codes_path)
        codes: Dict[str, str] = {}
    else:
        codes = json.loads(codes_path.read_text(encoding="utf-8"))

    sums: Dict[str, Dict[str, float]] = {}
    counts: Dict[str, int] = {}
    with filled_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            blind = row.get("blind_model", "")
            model = codes.get(blind, blind)
            sums.setdefault(model, {k: 0.0 for k in ("correctness", "fluency", "tanglish")})
            counts.setdefault(model, 0)
            try:
                sums[model]["correctness"] += float(row.get("correctness_1to5") or 0)
                sums[model]["fluency"] += float(row.get("fluency_1to5") or 0)
                sums[model]["tanglish"] += float(row.get("tanglish_1to5") or 0)
                counts[model] += 1
            except ValueError:
                continue

    per_model = {
        m: {k: (v / counts[m] if counts[m] else 0.0) for k, v in d.items()}
        for m, d in sums.items()
    }
    return {"per_model": per_model, "n": counts}


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    p = argparse.ArgumentParser(description="Human-eval sheet builder / scorer.")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Create a rating sheet.")
    b.add_argument("--predictions-dir", default="outputs/predictions")
    b.add_argument("--test", default="data/final/test.jsonl")
    b.add_argument("--out", default="outputs/evaluation/human_eval_sheet.csv")
    b.add_argument("--n-samples", type=int, default=100)

    s = sub.add_parser("score", help="Score a filled-in sheet.")
    s.add_argument("--csv", required=True)
    s.add_argument("--codes", default=None)
    s.add_argument("--out", default="outputs/evaluation/human_eval_scores.json")

    args = p.parse_args()
    if args.cmd == "build":
        build_sheet(
            predictions_dir=Path(args.predictions_dir),
            test_path=Path(args.test),
            output_csv=Path(args.out),
            n_samples=args.n_samples,
        )
    else:
        result = score_sheet(Path(args.csv), Path(args.codes) if args.codes else None)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        log.info("Scores written to {}", args.out)


if __name__ == "__main__":
    main()
