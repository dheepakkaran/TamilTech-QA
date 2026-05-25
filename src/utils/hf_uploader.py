"""Push the TamilTech-QA dataset and model to the HuggingFace Hub.

Usage
-----

.. code-block:: bash

    # Dataset
    python -m src.utils.hf_uploader --dataset \
        --dataset-dir data/final --repo your-username/TamilTech-QA

    # Model (merged adapter + base)
    python -m src.utils.hf_uploader --model outputs/lora_r16 \
        --repo your-username/TamilTech-QA-Llama3.1-8B-QLoRA
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Dataset card
# ---------------------------------------------------------------------- #
DATASET_CARD_TEMPLATE = """---
language:
  - ta
  - en
multilinguality: multilingual
tags:
  - tanglish
  - code-switching
  - tamil-english
  - technical-qa
  - instruction-tuning
task_categories:
  - question-answering
  - text-generation
license: mit
---

# TamilTech-QA

The first **Tanglish (Tamil-English code-switched) technical question-answering**
dataset. ~{n_samples} samples across Python, ML, ECE, DSA, web, OS — collected
from YouTube comments on Tamil tech channels, synthesised with GPT-4o,
and curated from public Tanglish corpora.

## Motivation

Tamil engineering students consume technical content in a code-switched register
("indha function-a epdi optimise pannuvanga?", "this gradient descent slow-a varuthu"),
yet no open dataset captures this register for technical QA. TamilTech-QA closes
that gap so that fine-tuned LLMs can produce answers in the register learners
actually use.

## Language

Tanglish — Roman-script Tamil scaffolding with English technical vocabulary kept
in English (e.g., ``function``, ``gradient``, ``pointer``, ``transistor``).

## Task

Open-ended question answering / instruction following.

## Source breakdown

| Source | Description |
|---|---|
| ``youtube``   | Comments and replies from Tamil tech channels (Brototype Tamil, CS in Tamil, …) |
| ``synthetic`` | GPT-4o generated QA pairs with the prompt in the paper / repository |
| ``existing``  | Technical-keyword filtered slices of DravidianCodeMix and Aya_Tamil |

## Statistics

```json
{stats_block}
```

## Example records

{examples_block}

## Splits

| Split | n |
|---|---|
| train | {n_train} |
| validation | {n_val} |
| test | {n_test} |

## Limitations and biases

- The YouTube portion inherits biases of the underlying channels (CS / EE focus).
- The synthetic portion inherits GPT-4o's idea of what "fluent Tanglish" looks
  like, which can be conservative.
- Topics are skewed toward Python / ML / DSA; ECE coverage is thinner.
- No personally identifying information is included by design, but the
  scraping pipeline does not run named-entity redaction — users redistributing
  derivative work should re-check.

## Citation

```bibtex
@misc{{tamiltech-qa-2026,
  title  = {{TamilTech-QA: A Tanglish Technical Question-Answering Dataset and Fine-Tuned LLM}},
  author = {{TamilTech-QA Authors}},
  year   = {{2026}},
  url    = {{https://huggingface.co/datasets/{repo}}},
}}
```
"""


def _build_examples_block(dataset_dir: Path, topics: List[str], per_topic: int = 1) -> str:
    """Render a markdown block of example samples per topic.

    Args:
        dataset_dir: Directory containing ``train.jsonl``.
        topics: Topics to draw examples from.
        per_topic: How many examples per topic.

    Returns:
        Markdown-formatted string of examples (may be empty).
    """
    train = dataset_dir / "train.jsonl"
    if not train.exists():
        return "_(No train.jsonl found — examples skipped.)_"
    examples: Dict[str, List[Dict[str, Any]]] = {t: [] for t in topics}
    with train.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            t = rec.get("topic", "general")
            if t in examples and len(examples[t]) < per_topic:
                examples[t].append(rec)
            if all(len(v) >= per_topic for v in examples.values()):
                break
    parts: List[str] = []
    for topic, recs in examples.items():
        for r in recs:
            parts.append(
                f"### Topic: `{topic}`\n\n"
                f"**Q:** {r.get('input') or r.get('question', '')}\n\n"
                f"**A:** {r.get('output') or r.get('answer', '')}\n"
            )
    return "\n".join(parts) if parts else "_(No examples available.)_"


def write_dataset_card(dataset_dir: Path, repo: str) -> Path:
    """Write a Markdown dataset card based on ``dataset_stats.json``.

    Args:
        dataset_dir: Directory containing ``dataset_stats.json``.
        repo: HF repo ID (``username/name``) for the citation.

    Returns:
        Path to the written README.md.
    """
    stats_path = dataset_dir / "dataset_stats.json"
    stats: Dict[str, Any] = {}
    if stats_path.exists():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
    n_train = stats.get("train", {}).get("n_samples", 0)
    n_val = stats.get("val", {}).get("n_samples", 0)
    n_test = stats.get("test", {}).get("n_samples", 0)
    n_total = stats.get("overall", {}).get("n_samples", n_train + n_val + n_test)

    topics = sorted(stats.get("train", {}).get("per_topic", {}).keys()) or [
        "python", "ml", "ece", "dsa", "web", "os"
    ]
    body = DATASET_CARD_TEMPLATE.format(
        n_samples=n_total,
        stats_block=json.dumps(stats, indent=2)[:2000],
        examples_block=_build_examples_block(dataset_dir, topics),
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        repo=repo,
    )
    out = dataset_dir / "README.md"
    out.write_text(body, encoding="utf-8")
    log.info("Wrote dataset card → {}", out)
    return out


# ---------------------------------------------------------------------- #
# Model card
# ---------------------------------------------------------------------- #
MODEL_CARD_TEMPLATE = """---
language:
  - ta
  - en
tags:
  - tanglish
  - tamil
  - english
  - code-switching
  - qlora
  - peft
  - text-generation
license: mit
base_model: {base_model}
datasets:
  - {dataset_repo}
---

# {repo}

QLoRA fine-tuned **{base_model}** for **Tanglish technical question answering**.

## Method

- **Base**: ``{base_model}`` loaded in 4-bit with bitsandbytes NF4 + double-quant.
- **Adapters**: LoRA (config ``{lora_name}``).
- **Trainer**: ``trl.SFTTrainer``.
- **Data**: [TamilTech-QA dataset]({dataset_link}).

## Results

| Metric | Base (zero-shot) | Fine-tuned (ours) |
|---|---|---|
{metrics_table}

Full per-metric, per-topic numbers in the repository's
``outputs/evaluation/metrics_comparison.csv`` and ablation studies.

## Quick inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

repo = "{repo}"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, torch_dtype=torch.float16, device_map="auto")

prompt = (
    "<|im_start|>system\\nYou are a helpful Tanglish technical assistant.<|im_end|>\\n"
    "<|im_start|>user\\nindha gradient descent enna na konjam explain pannunga<|im_end|>\\n"
    "<|im_start|>assistant\\n"
)
inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
print(tok.decode(out[0], skip_special_tokens=True))
```

## Hardware

Trained on a single NVIDIA A100 (40 GB) — typical run ≈ 2–4 GPU-hours per
LoRA configuration on ~10K samples.

## Limitations

- Outputs Tanglish only; for pure English answers, prompt explicitly.
- Strongest on Python / ML / DSA; weaker on ECE due to less training data.
- May hallucinate API signatures — verify any code it produces.
- Safety: not red-teamed; do not deploy in user-facing safety-critical settings.

## Citation

```bibtex
@misc{{tamiltech-qa-2026,
  title  = {{TamilTech-QA: A Tanglish Technical Question-Answering Dataset and Fine-Tuned LLM}},
  author = {{TamilTech-QA Authors}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{repo}}},
}}
```
"""


def _format_metrics_table(metrics_json: Optional[Path]) -> str:
    """Render a Markdown table of base vs fine-tuned metrics.

    Args:
        metrics_json: Path to ``metrics_comparison.json`` (may be None).

    Returns:
        Multi-line Markdown table body (rows only).
    """
    rows = []
    rows_keys = ("bleu_4", "rouge_l", "bertscore_f1", "csps", "ttr", "tcf")
    if metrics_json and metrics_json.exists():
        data = json.loads(metrics_json.read_text(encoding="utf-8"))
        base = data.get("base_zero_shot") or next(
            (v for k, v in data.items() if "base" in k), {}
        )
        ft = data.get("lora_r16") or next(
            (v for k, v in data.items() if k.startswith("lora")), {}
        )
        for k in rows_keys:
            rows.append(
                f"| {k} | {base.get(k, 0.0):.4f} | {ft.get(k, 0.0):.4f} |"
            )
    else:
        for k in rows_keys:
            rows.append(f"| {k} | — | — |")
    return "\n".join(rows)


def write_model_card(
    model_dir: Path,
    repo: str,
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_name: str = "lora_r16",
    dataset_repo: str = "your-username/TamilTech-QA",
    metrics_json: Optional[Path] = None,
) -> Path:
    """Write a Markdown model card.

    Args:
        model_dir: Directory holding the saved model artifacts.
        repo: HF repo ID for this model.
        base_model: Base model HF ID.
        lora_name: Name of the LoRA config used.
        dataset_repo: HF dataset repo ID.
        metrics_json: Optional path to metrics JSON.

    Returns:
        Path to the written ``README.md``.
    """
    body = MODEL_CARD_TEMPLATE.format(
        repo=repo,
        base_model=base_model,
        lora_name=lora_name,
        dataset_repo=dataset_repo,
        dataset_link=f"https://huggingface.co/datasets/{dataset_repo}",
        metrics_table=_format_metrics_table(metrics_json),
    )
    out = model_dir / "README.md"
    out.write_text(body, encoding="utf-8")
    log.info("Wrote model card → {}", out)
    return out


# ---------------------------------------------------------------------- #
# HF push
# ---------------------------------------------------------------------- #
def push_dataset(dataset_dir: Path, repo: str, private: bool = False) -> None:
    """Push the dataset to the Hub via ``datasets.push_to_hub``.

    Args:
        dataset_dir: Directory containing ``hf_dataset/`` (or JSONL splits).
        repo: HF repo ID.
        private: Make the repo private.

    Returns:
        None.
    """
    try:
        from datasets import load_from_disk, Dataset, DatasetDict
    except ImportError as e:  # pragma: no cover
        raise ImportError("`datasets` not installed.") from e

    write_dataset_card(dataset_dir, repo)
    hf_dir = dataset_dir / "hf_dataset"
    if hf_dir.exists():
        dsd = load_from_disk(str(hf_dir))
    else:
        # Build a DatasetDict from raw JSONL.
        parts: Dict[str, Any] = {}
        for name, split in [("train", "train.jsonl"), ("validation", "val.jsonl"), ("test", "test.jsonl")]:
            p = dataset_dir / split
            if p.exists():
                parts[name] = Dataset.from_json(str(p))
        dsd = DatasetDict(parts)
    log.info("Pushing dataset → {} (private={})", repo, private)
    dsd.push_to_hub(repo, private=private)


def push_model(
    model_dir: Path,
    repo: str,
    private: bool = False,
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct",
    lora_name: str = "lora_r16",
    dataset_repo: str = "your-username/TamilTech-QA",
    metrics_json: Optional[Path] = None,
) -> None:
    """Push a fine-tuned model + tokenizer + model card to the Hub.

    If the run directory contains a ``merged/`` subdirectory (produced by the
    trainer's merge step), that is uploaded. Otherwise the adapter directory
    itself is uploaded so the model can be loaded with ``PeftModel``.

    Args:
        model_dir: Path to the run directory.
        repo: HF repo ID for the model.
        private: Make the repo private.
        base_model: Base model HF ID (for the card).
        lora_name: LoRA config name (for the card).
        dataset_repo: Dataset repo ID (for the card).
        metrics_json: Optional path to ``metrics_comparison.json``.

    Returns:
        None.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise ImportError("`transformers` not installed.") from e

    merged = model_dir / "merged"
    upload_dir = merged if merged.exists() else model_dir
    log.info("Uploading from {} → {}", upload_dir, repo)
    write_model_card(
        upload_dir, repo, base_model=base_model, lora_name=lora_name,
        dataset_repo=dataset_repo, metrics_json=metrics_json,
    )
    tok = AutoTokenizer.from_pretrained(str(upload_dir))
    model = AutoModelForCausalLM.from_pretrained(str(upload_dir))
    model.push_to_hub(repo, private=private)
    tok.push_to_hub(repo, private=private)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    p = argparse.ArgumentParser(description="Push TamilTech-QA dataset / model to the Hub.")
    p.add_argument("--dataset", action="store_true", help="Push the dataset.")
    p.add_argument("--dataset-dir", default="data/final")
    p.add_argument(
        "--model",
        default=None,
        help="Path to a run directory to upload (e.g. outputs/lora_r16_TIMESTAMP).",
    )
    p.add_argument("--repo", required=True, help="HF repo ID (username/name).")
    p.add_argument("--dataset-repo", default=None,
                   help="Companion dataset repo ID for model-card cross-link.")
    p.add_argument("--lora-name", default="lora_r16")
    p.add_argument("--base-model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument(
        "--metrics-json",
        default="outputs/evaluation/metrics_comparison.json",
    )
    p.add_argument("--private", action="store_true")
    args = p.parse_args()

    if args.dataset:
        push_dataset(Path(args.dataset_dir), args.repo, private=args.private)
    if args.model:
        push_model(
            Path(args.model),
            args.repo,
            private=args.private,
            base_model=args.base_model,
            lora_name=args.lora_name,
            dataset_repo=args.dataset_repo or args.repo,
            metrics_json=Path(args.metrics_json) if Path(args.metrics_json).exists() else None,
        )


if __name__ == "__main__":
    main()
