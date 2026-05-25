"""Ablation studies for TamilTech-QA.

Studies
-------
1. ``lora_rank``    — compare lora_r8 / lora_r16 / lora_r64.
2. ``data_size``    — re-train lora_r16 on 25/50/75/100 % of train and re-evaluate.
3. ``data_source``  — train on YouTube-only / synthetic-only / combined.
4. ``per_topic``    — break down test metrics by topic.

Each study writes a JSON + CSV under ``outputs/evaluation/ablation/`` and
delegates the plotting to :mod:`src.evaluation.visualizer`.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.evaluation.metrics import (
    code_switch_preservation_score,
    compute_all_metrics,
    perplexity,
)
from src.preprocessing.tanglish_detector import TanglishDetector
from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Common helpers
# ---------------------------------------------------------------------- #
def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts.

    Args:
        p: Path to JSONL.

    Returns:
        Records.
    """
    return [json.loads(line) for line in p.open("r", encoding="utf-8") if line.strip()]


def _load_predictions(name: str, preds_dir: Path) -> List[Dict[str, str]]:
    """Load predictions written by :func:`metrics.evaluate_all_models`.

    Args:
        name: Model name (used as the filename stem).
        preds_dir: Directory holding ``<name>.jsonl``.

    Returns:
        List of ``{id, prediction, reference}`` records.
    """
    path = preds_dir / f"{name}.jsonl"
    if not path.exists():
        log.warning("No predictions file for {} at {}", name, path)
        return []
    return _read_jsonl(path)


def _ensure_dir(p: Path) -> Path:
    """Wrapper kept for symmetry with other modules."""
    return ensure_dir(p)


# ---------------------------------------------------------------------- #
# Study 1 — LoRA rank
# ---------------------------------------------------------------------- #
def study_lora_rank(
    eval_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    out_dir: Path,
    rank_models: Iterable[str] = ("lora_r8", "lora_r16", "lora_r64"),
) -> Dict[str, Any]:
    """Compare lora_r{8,16,64} on every metric and report rank-vs-metric data.

    Reads predictions already produced by :func:`metrics.evaluate_all_models`.

    Args:
        eval_cfg: Loaded eval config dict.
        data_cfg: Loaded data config dict.
        out_dir: Directory to write the study output.
        rank_models: Model names to include.

    Returns:
        Study payload dict (also written to disk).
    """
    preds_dir = Path(eval_cfg["paths"]["predictions_dir"])
    detector = TanglishDetector(
        min_ratio=data_cfg["preprocessing"].get("tanglish_ratio_min", 0.15),
        max_ratio=data_cfg["preprocessing"].get("tanglish_ratio_max", 0.85),
    )
    tech_kws = data_cfg["preprocessing"]["technical_keywords"]
    connectors = data_cfg["preprocessing"]["tamil_connectors"]

    rank_map = {"lora_r8": 8, "lora_r16": 16, "lora_r64": 64, "lora_r16_full": 16}
    rows: List[Dict[str, Any]] = []
    for name in rank_models:
        records = _load_predictions(name, preds_dir)
        if not records:
            continue
        preds = [r["prediction"] for r in records]
        refs = [r["reference"] for r in records]
        metrics = compute_all_metrics(
            preds, refs,
            technical_keywords=tech_kws,
            connectors=connectors,
            detector=detector,
            compute_bertscore=False,  # already computed in metrics.py; keep cheap
        )
        rows.append({"model": name, "rank": rank_map.get(name), **metrics})

    out = {"study": "lora_rank", "rows": rows}
    _write_study(out, out_dir, "study_lora_rank")
    return out


# ---------------------------------------------------------------------- #
# Study 2 — Data size impact
# ---------------------------------------------------------------------- #
def study_data_size(
    eval_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    dataset_dir: Path,
    out_dir: Path,
    fractions: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
    config_name: str = "lora_r16",
    model_config_path: str = "config/model_config.yaml",
    train_fn: Optional[Any] = None,
    eval_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Train ``config_name`` on shrinking train fractions and re-evaluate.

    For each fraction in ``fractions`` the train.jsonl is downsampled
    (deterministic seed), the model is fine-tuned, and metrics are recorded.

    Args:
        eval_cfg: Eval config dict.
        data_cfg: Data config dict.
        dataset_dir: Directory of train/val/test splits.
        out_dir: Output directory for the study payload.
        fractions: Training-data fractions to evaluate.
        config_name: Which LoRA config to use (default lora_r16).
        model_config_path: Path to the model config YAML.
        train_fn: Optional injected trainer function (for testing).
        eval_fn: Optional injected evaluator function (for testing).

    Returns:
        Study payload dict.
    """
    from src.evaluation.metrics import evaluate_all_models as default_eval_fn
    from src.training.trainer import train_one as default_train_fn

    train_fn = train_fn or default_train_fn
    eval_fn = eval_fn or default_eval_fn

    base_train = dataset_dir / "train.jsonl"
    if not base_train.exists():
        log.warning("No train.jsonl at {}; aborting data-size study.", base_train)
        return {"study": "data_size", "rows": []}
    all_train = _read_jsonl(base_train)

    rows: List[Dict[str, Any]] = []
    tmp_root = ensure_dir(out_dir / "_tmp_data_size")

    for frac in fractions:
        rng = random.Random(int(frac * 1000))
        n = max(1, int(len(all_train) * frac))
        sample = rng.sample(all_train, n)
        subset_dir = ensure_dir(tmp_root / f"frac_{frac:.2f}")
        (subset_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in sample),
            encoding="utf-8",
        )
        # Re-use val / test as-is
        for src in ("val.jsonl", "test.jsonl"):
            sp = dataset_dir / src
            if sp.exists():
                shutil.copy(sp, subset_dir / src)

        run_dir = train_fn(
            config_name=config_name,
            dataset_dir=subset_dir,
            model_config_path=model_config_path,
            timestamp=f"frac_{int(frac * 100):03d}",
            merge_after=False,
        )

        # Patch eval_config in-memory: point lora_r16's adapter_path at this run.
        patched_eval = json.loads(json.dumps(eval_cfg))  # deep copy
        for m in patched_eval.get("models_to_eval", []):
            if m.get("name") == config_name:
                m["adapter_path"] = str(run_dir)
        patched_path = subset_dir / "_patched_eval_config.yaml"
        import yaml as _yaml  # local import to keep top-level imports clean

        patched_path.write_text(_yaml.safe_dump(patched_eval), encoding="utf-8")
        metrics_map = eval_fn(
            eval_config_path=str(patched_path),
            model_config_path=model_config_path,
            data_config_path="config/data_config.yaml",
            models_filter=[config_name],
        )
        m = metrics_map.get(config_name, {})
        rows.append({"fraction": frac, "n_train": n, **m})

    out = {"study": "data_size", "rows": rows}
    _write_study(out, out_dir, "study_data_size")
    return out


# ---------------------------------------------------------------------- #
# Study 3 — Data source impact
# ---------------------------------------------------------------------- #
def study_data_source(
    eval_cfg: Dict[str, Any],
    dataset_dir: Path,
    out_dir: Path,
    config_name: str = "lora_r16",
    model_config_path: str = "config/model_config.yaml",
    train_fn: Optional[Any] = None,
    eval_fn: Optional[Any] = None,
) -> Dict[str, Any]:
    """Re-train on YouTube-only, synthetic-only, and combined data.

    Args:
        eval_cfg: Eval config dict.
        dataset_dir: Directory containing the standard splits.
        out_dir: Where to drop the study payload.
        config_name: Which LoRA config to use.
        model_config_path: Path to model config YAML.
        train_fn: Optional injected trainer function.
        eval_fn: Optional injected evaluator function.

    Returns:
        Study payload dict.
    """
    from src.evaluation.metrics import evaluate_all_models as default_eval_fn
    from src.training.trainer import train_one as default_train_fn

    train_fn = train_fn or default_train_fn
    eval_fn = eval_fn or default_eval_fn

    base_train = dataset_dir / "train.jsonl"
    if not base_train.exists():
        log.warning("No train.jsonl at {}; aborting data-source study.", base_train)
        return {"study": "data_source", "rows": []}

    all_train = _read_jsonl(base_train)
    rows: List[Dict[str, Any]] = []
    tmp_root = ensure_dir(out_dir / "_tmp_data_source")

    splits = {
        "youtube_only": [r for r in all_train if r.get("source") == "youtube"],
        "synthetic_only": [r for r in all_train if r.get("source") == "synthetic"],
        "combined": all_train,
    }

    for label, recs in splits.items():
        if not recs:
            log.warning("Source '{}' has 0 records; skipping.", label)
            continue
        subset_dir = ensure_dir(tmp_root / label)
        (subset_dir / "train.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in recs),
            encoding="utf-8",
        )
        for src in ("val.jsonl", "test.jsonl"):
            sp = dataset_dir / src
            if sp.exists():
                shutil.copy(sp, subset_dir / src)

        run_dir = train_fn(
            config_name=config_name,
            dataset_dir=subset_dir,
            model_config_path=model_config_path,
            timestamp=f"src_{label}",
            merge_after=False,
        )

        patched_eval = json.loads(json.dumps(eval_cfg))
        for m in patched_eval.get("models_to_eval", []):
            if m.get("name") == config_name:
                m["adapter_path"] = str(run_dir)
        import yaml as _yaml

        patched_path = subset_dir / "_patched_eval_config.yaml"
        patched_path.write_text(_yaml.safe_dump(patched_eval), encoding="utf-8")
        metrics_map = eval_fn(
            eval_config_path=str(patched_path),
            model_config_path=model_config_path,
            data_config_path="config/data_config.yaml",
            models_filter=[config_name],
        )
        m = metrics_map.get(config_name, {})
        rows.append({"source": label, "n_train": len(recs), **m})

    out = {"study": "data_source", "rows": rows}
    _write_study(out, out_dir, "study_data_source")
    return out


# ---------------------------------------------------------------------- #
# Study 4 — Per-topic breakdown
# ---------------------------------------------------------------------- #
def study_per_topic(
    eval_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    out_dir: Path,
    topics: Iterable[str] = ("python", "ml", "ece", "dsa", "web", "os"),
) -> Dict[str, Any]:
    """Slice each model's predictions by topic and compute metrics per-topic.

    Args:
        eval_cfg: Eval config dict.
        data_cfg: Data config dict.
        out_dir: Output directory.
        topics: Topics to break down by.

    Returns:
        Study payload dict.
    """
    preds_dir = Path(eval_cfg["paths"]["predictions_dir"])
    test_path = Path(eval_cfg["paths"]["test_set"])
    if not test_path.exists():
        log.warning("test set not found at {}; aborting per-topic study.", test_path)
        return {"study": "per_topic", "rows": []}

    test_records = _read_jsonl(test_path)
    by_id = {r["id"]: r for r in test_records if "id" in r}
    detector = TanglishDetector()
    tech_kws = data_cfg["preprocessing"]["technical_keywords"]
    connectors = data_cfg["preprocessing"]["tamil_connectors"]

    rows: List[Dict[str, Any]] = []
    for spec in eval_cfg.get("models_to_eval", []):
        name = spec["name"]
        records = _load_predictions(name, preds_dir)
        if not records:
            continue
        per_topic: Dict[str, Tuple[List[str], List[str]]] = {t: ([], []) for t in topics}
        for rec in records:
            meta = by_id.get(rec.get("id"))
            if not meta:
                continue
            topic = meta.get("topic", "general")
            if topic in per_topic:
                per_topic[topic][0].append(rec["prediction"])
                per_topic[topic][1].append(rec["reference"])
        for topic, (preds, refs) in per_topic.items():
            if not preds:
                continue
            m = compute_all_metrics(
                preds, refs,
                technical_keywords=tech_kws,
                connectors=connectors,
                detector=detector,
                compute_bertscore=False,
            )
            rows.append({"model": name, "topic": topic, "n": len(preds), **m})

    out = {"study": "per_topic", "rows": rows}
    _write_study(out, out_dir, "study_per_topic")
    return out


# ---------------------------------------------------------------------- #
# Persistence
# ---------------------------------------------------------------------- #
def _write_study(payload: Dict[str, Any], out_dir: Path, stem: str) -> None:
    """Write a study payload as JSON + CSV.

    Args:
        payload: Dict with ``study`` and ``rows`` keys.
        out_dir: Output directory.
        stem: Filename stem.

    Returns:
        None.
    """
    ensure_dir(out_dir)
    json_path = out_dir / f"{stem}.json"
    csv_path = out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = payload.get("rows", [])
    if not rows:
        csv_path.write_text("", encoding="utf-8")
        return
    cols: List[str] = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(_csv_value(r.get(c, "")) for c in cols) + "\n")
    log.info("Study written: {} (json + csv)", stem)


def _csv_value(v: Any) -> str:
    """Render a value for CSV (floats with 6 decimals, others as ``str``)."""
    if isinstance(v, float):
        return f"{v:.6f}"
    return str(v)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    """CLI entrypoint for ``python -m src.evaluation.ablation``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Run TamilTech-QA ablation studies.")
    p.add_argument("--study", default="all",
                   choices=["all", "lora_rank", "data_size", "data_source", "per_topic"])
    p.add_argument("--eval-config", default="config/eval_config.yaml")
    p.add_argument("--data-config", default="config/data_config.yaml")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--dataset", default="data/final/")
    p.add_argument("--out-dir", default="outputs/evaluation/ablation")
    args = p.parse_args()

    seed_everything(42)
    eval_cfg = load_config(args.eval_config)
    data_cfg = load_config(args.data_config)
    out_dir = ensure_dir(Path(args.out_dir))
    ds_dir = Path(args.dataset)

    if args.study in ("all", "lora_rank"):
        study_lora_rank(eval_cfg, data_cfg, out_dir)
    if args.study in ("all", "data_size"):
        study_data_size(eval_cfg, data_cfg, ds_dir, out_dir,
                        fractions=tuple(eval_cfg.get("ablation", {}).get("data_size_fractions", [0.25, 0.5, 0.75, 1.0])),
                        model_config_path=args.model_config)
    if args.study in ("all", "data_source"):
        study_data_source(eval_cfg, ds_dir, out_dir, model_config_path=args.model_config)
    if args.study in ("all", "per_topic"):
        study_per_topic(eval_cfg, data_cfg, out_dir,
                        topics=tuple(eval_cfg.get("ablation", {}).get("topics",
                                                                       ["python", "ml", "ece", "dsa", "web", "os"])))


if __name__ == "__main__":
    main()
