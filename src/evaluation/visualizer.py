"""Publication-quality plots for TamilTech-QA.

All plots are saved as 300 DPI PNG under ``outputs/figures/``.

Charts produced
---------------
- ``bar_models_metrics.png``  — bar chart of every model × every metric.
- ``heatmap_topic_metric.png`` — topic × metric heatmap for the best model.
- ``loss_curves.png``         — per-config training loss vs. step.
- ``radar_base_vs_best.png``  — radar chart, base vs best LoRA, 6 metrics.
- ``tanglish_ratio_hist.png`` — dataset-wide Tanglish ratio distribution.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


def _apply_style(eval_cfg_path: str) -> Tuple[int, str]:
    """Apply the configured matplotlib style.

    Args:
        eval_cfg_path: Path to ``eval_config.yaml``.

    Returns:
        Tuple of ``(dpi, palette_name)``.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  (registers style)
    import seaborn as sns

    cfg = load_config(eval_cfg_path).get("visualization", {})
    dpi = int(cfg.get("dpi", 300))
    palette = cfg.get("palette", "deep")
    try:
        import matplotlib.pyplot as plt2

        plt2.style.use(cfg.get("style", "seaborn-v0_8-whitegrid"))
    except Exception:  # noqa: BLE001
        pass
    sns.set_palette(palette)
    return dpi, palette


# ---------------------------------------------------------------------- #
# Loaders
# ---------------------------------------------------------------------- #
def _load_metrics(results_path: Path) -> Dict[str, Dict[str, float]]:
    """Load ``metrics_comparison.json`` if present, else empty.

    Args:
        results_path: Path to the JSON.

    Returns:
        Mapping ``model -> metrics dict``.
    """
    if not results_path.exists():
        log.warning("No metrics JSON at {}", results_path)
        return {}
    return json.loads(results_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------- #
# Plots
# ---------------------------------------------------------------------- #
def plot_bar_models_metrics(metrics: Dict[str, Dict[str, float]], out_path: Path, dpi: int) -> None:
    """Grouped bar chart of every model across every metric.

    Args:
        metrics: ``model -> metric -> float``.
        out_path: Destination PNG.
        dpi: Figure DPI.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not metrics:
        log.warning("No metrics provided; skipping bar plot.")
        return

    models = list(metrics.keys())
    metric_names = sorted({m for d in metrics.values() for m in d})
    metric_names = [m for m in metric_names if m != "perplexity"]
    x = np.arange(len(metric_names))
    w = 0.8 / max(1, len(models))

    fig, ax = plt.subplots(figsize=(max(8, len(metric_names) * 1.0), 5))
    for i, model in enumerate(models):
        vals = [metrics[model].get(m, 0.0) for m in metric_names]
        ax.bar(x + i * w, vals, width=w, label=model)
    ax.set_xticks(x + w * (len(models) - 1) / 2)
    ax.set_xticklabels(metric_names, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("TamilTech-QA: model × metric")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved {}", out_path)


def plot_heatmap_topic_metric(per_topic_payload: Dict[str, Any], out_path: Path, dpi: int,
                              metrics_to_show: Sequence[str] = ("bleu_4", "rouge_l", "csps", "ttr")) -> None:
    """Heatmap of topic × metric for each model row.

    Args:
        per_topic_payload: Output dict from :func:`ablation.study_per_topic`.
        out_path: Destination PNG.
        dpi: Figure DPI.
        metrics_to_show: Metrics to include as separate panels.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns

    rows = per_topic_payload.get("rows", [])
    if not rows:
        log.warning("No per-topic rows; skipping heatmap.")
        return

    models = sorted({r["model"] for r in rows})
    topics = sorted({r["topic"] for r in rows})
    n_metrics = len(metrics_to_show)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.2 * n_metrics, max(3, 0.45 * len(models))))
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_show):
        mat = np.zeros((len(models), len(topics)))
        for r in rows:
            i = models.index(r["model"])
            j = topics.index(r["topic"])
            mat[i, j] = r.get(metric, 0.0)
        sns.heatmap(
            mat, annot=True, fmt=".2f",
            xticklabels=topics, yticklabels=models, cmap="YlGnBu",
            cbar_kws={"label": metric}, ax=ax,
        )
        ax.set_title(metric)
        ax.set_xlabel("topic")
        if ax is axes[0]:
            ax.set_ylabel("model")
        else:
            ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved {}", out_path)


def plot_loss_curves(outputs_root: Path, out_path: Path, dpi: int) -> None:
    """Overlay training-loss curves from every run's ``loss_history.json``.

    Args:
        outputs_root: ``outputs/`` directory containing per-run subdirs.
        out_path: Destination PNG.
        dpi: Figure DPI.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    runs = []
    for hist_file in outputs_root.glob("*/loss_history.json"):
        try:
            entries = json.loads(hist_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        steps = [e["step"] for e in entries if "loss" in e]
        losses = [e["loss"] for e in entries if "loss" in e]
        if steps:
            runs.append((hist_file.parent.name, steps, losses))

    if not runs:
        log.warning("No loss_history.json files found under {}", outputs_root)
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, steps, losses in runs:
        ax.plot(steps, losses, label=name, linewidth=1.2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training loss")
    ax.set_title("QLoRA training loss")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved {}", out_path)


def plot_radar(
    metrics: Dict[str, Dict[str, float]],
    out_path: Path,
    dpi: int,
    metric_set: Sequence[str] = ("bleu_4", "rouge_l", "bertscore_f1", "csps", "ttr", "tcf"),
    base_name_hint: str = "base_zero_shot",
) -> None:
    """Radar chart of base model vs the best fine-tuned model across 6 metrics.

    "Best" is chosen by mean of the radar metrics.

    Args:
        metrics: ``model -> metric -> float``.
        out_path: Destination PNG.
        dpi: Figure DPI.
        metric_set: 6 metrics to plot.
        base_name_hint: Exact name of the baseline model (fallback: any with ``base``).

    Returns:
        None.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    if not metrics:
        log.warning("No metrics; skipping radar.")
        return

    base = metrics.get(base_name_hint)
    if base is None:
        for n in metrics:
            if "base" in n:
                base = metrics[n]
                base_name_hint = n
                break
    if base is None:
        base = next(iter(metrics.values()))
        base_name_hint = next(iter(metrics))

    candidates = [(n, m) for n, m in metrics.items() if n != base_name_hint]
    if not candidates:
        log.warning("Need at least 2 models for radar.")
        return
    best_name, best = max(
        candidates,
        key=lambda kv: sum(kv[1].get(x, 0.0) for x in metric_set) / len(metric_set),
    )

    angles = np.linspace(0, 2 * math.pi, len(metric_set), endpoint=False).tolist()
    angles += angles[:1]

    def loop(d: Dict[str, float]) -> List[float]:
        vals = [float(d.get(x, 0.0)) for x in metric_set]
        return vals + vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "polar"})
    ax.plot(angles, loop(base), label=base_name_hint, linewidth=2)
    ax.fill(angles, loop(base), alpha=0.15)
    ax.plot(angles, loop(best), label=best_name, linewidth=2)
    ax.fill(angles, loop(best), alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(list(metric_set))
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(0, 1)
    ax.set_title("Base vs best fine-tuned (6 metrics)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved {}", out_path)


def plot_tanglish_ratio_hist(dataset_dir: Path, out_path: Path, dpi: int) -> None:
    """Histogram of ``tanglish_ratio`` across the final dataset.

    Args:
        dataset_dir: Directory containing the final train/val/test JSONL.
        out_path: Destination PNG.
        dpi: Figure DPI.

    Returns:
        None.
    """
    import matplotlib.pyplot as plt

    ratios: List[float] = []
    for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
        p = dataset_dir / name
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "tanglish_ratio" in rec:
                        ratios.append(float(rec["tanglish_ratio"]))

    if not ratios:
        log.warning("No tanglish_ratio values found under {}", dataset_dir)
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(ratios, bins=30, edgecolor="black", alpha=0.85)
    ax.axvline(0.15, color="red", linestyle="--", label="band lower (0.15)")
    ax.axvline(0.85, color="red", linestyle="--", label="band upper (0.85)")
    ax.set_xlabel("Tanglish ratio")
    ax.set_ylabel("Count")
    ax.set_title(f"Tanglish ratio distribution (n={len(ratios)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved {}", out_path)


# ---------------------------------------------------------------------- #
# Driver
# ---------------------------------------------------------------------- #
def generate_all(
    eval_cfg_path: str = "config/eval_config.yaml",
    results_path: Optional[str] = None,
    per_topic_path: Optional[str] = None,
    dataset_dir: Optional[str] = None,
    outputs_root: Optional[str] = None,
    figures_dir: Optional[str] = None,
) -> None:
    """Render every plot listed in the module docstring.

    Args:
        eval_cfg_path: Path to ``eval_config.yaml``.
        results_path: Override for ``metrics_comparison.json``.
        per_topic_path: Override for ``study_per_topic.json``.
        dataset_dir: Override for ``data/final/``.
        outputs_root: Override for the runs root (``outputs/``).
        figures_dir: Override for ``outputs/figures/``.

    Returns:
        None.
    """
    dpi, _ = _apply_style(eval_cfg_path)
    cfg = load_config(eval_cfg_path)
    figures = ensure_dir(Path(figures_dir or cfg["paths"]["figures_dir"]))
    results = Path(results_path or Path(cfg["paths"]["results_dir"]) / "metrics_comparison.json")
    per_topic = Path(per_topic_path or Path(cfg["paths"]["results_dir"]) / "ablation" / "study_per_topic.json")
    runs_root = Path(outputs_root or "outputs")
    final_dir = Path(dataset_dir or "data/final")

    metrics = _load_metrics(results)
    plot_bar_models_metrics(metrics, figures / "bar_models_metrics.png", dpi)
    plot_radar(metrics, figures / "radar_base_vs_best.png", dpi)
    plot_loss_curves(runs_root, figures / "loss_curves.png", dpi)
    plot_tanglish_ratio_hist(final_dir, figures / "tanglish_ratio_hist.png", dpi)

    if per_topic.exists():
        payload = json.loads(per_topic.read_text(encoding="utf-8"))
        plot_heatmap_topic_metric(payload, figures / "heatmap_topic_metric.png", dpi)
    else:
        log.warning("Per-topic study not found ({}); skipping heatmap.", per_topic)


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    p = argparse.ArgumentParser(description="Render publication-quality plots.")
    p.add_argument("--eval-config", default="config/eval_config.yaml")
    p.add_argument("--results", default=None,
                   help="Override metrics_comparison.json path.")
    p.add_argument("--per-topic", default=None,
                   help="Override study_per_topic.json path.")
    p.add_argument("--dataset", default=None, help="Override data/final/ path.")
    p.add_argument("--outputs-root", default=None, help="Override outputs/ path.")
    p.add_argument("--figures-dir", default=None, help="Override figures dir.")
    args = p.parse_args()
    generate_all(
        eval_cfg_path=args.eval_config,
        results_path=args.results,
        per_topic_path=args.per_topic,
        dataset_dir=args.dataset,
        outputs_root=args.outputs_root,
        figures_dir=args.figures_dir,
    )


if __name__ == "__main__":
    main()
