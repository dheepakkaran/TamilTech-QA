"""Thin CLI wrapper around :mod:`tanglish_detector` for explicit filtering.

This module exists so the pipeline can be read top-to-bottom as
``language_filter → text_cleaner → qa_formatter``. Internally it just calls
the detector with ``filter_band=True``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing.tanglish_detector import TanglishDetector, annotate_jsonl
from src.utils import ensure_dir, load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def filter_jsonl(
    in_path: Path,
    out_path: Path,
    min_ratio: float = 0.15,
    max_ratio: float = 0.85,
) -> None:
    """Drop rows whose ``tanglish_ratio`` is outside [min, max].

    Args:
        in_path: Source JSONL.
        out_path: Destination JSONL.
        min_ratio: Inclusive lower bound.
        max_ratio: Inclusive upper bound.

    Returns:
        None.
    """
    detector = TanglishDetector(min_ratio=min_ratio, max_ratio=max_ratio)
    ensure_dir(out_path.parent)
    annotate_jsonl(in_path, out_path, detector, filter_band=True)


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    p = argparse.ArgumentParser(description="Filter samples by Tanglish ratio.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    pp = cfg["preprocessing"]
    filter_jsonl(
        Path(args.input),
        Path(args.output),
        min_ratio=pp.get("tanglish_ratio_min", 0.15),
        max_ratio=pp.get("tanglish_ratio_max", 0.85),
    )


if __name__ == "__main__":
    main()
