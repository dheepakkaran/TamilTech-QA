"""QLoRA fine-tuning via ``trl.SFTTrainer`` with the prompt's hyper-parameters.

Usage
-----

.. code-block:: bash

    # Train a single config
    python -m src.training.trainer --config lora_r16 --dataset data/final/

    # Run the full 4-config ablation
    python -m src.training.trainer --all-configs --dataset data/final/

    # Smoke-test (no GPU required) — show the CLI without crashing
    python -m src.training.trainer --help
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.training.callbacks import (
    LossHistoryCallback,
    MemoryCallback,
    SampleGenerationCallback,
    make_early_stopping_callback,
)
from src.training.lora_config import (
    LoRASpec,
    build_all,
    count_trainable_parameters,
    get_lora_config,
)
from src.training.model_loader import LoadedModel, load_quantized_model
from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

log = get_logger(__name__)


# ---------------------------------------------------------------------- #
# Dataset loading
# ---------------------------------------------------------------------- #
def load_splits(dataset_dir: Path):
    """Load train + val splits from a ``data/final`` directory.

    Prefers a HuggingFace ``hf_dataset`` directory, falls back to JSONL files.

    Args:
        dataset_dir: Directory containing ``train.jsonl`` / ``val.jsonl`` (and
            optionally ``hf_dataset/``).

    Returns:
        Tuple of (train ``Dataset``, eval ``Dataset``).

    Raises:
        FileNotFoundError: If no train data is found.
    """
    from datasets import Dataset, load_from_disk

    hf_dir = dataset_dir / "hf_dataset"
    if hf_dir.exists():
        log.info("Loading HuggingFace DatasetDict from {}", hf_dir)
        dsd = load_from_disk(str(hf_dir))
        return dsd["train"], dsd.get("validation") or dsd.get("test")

    train_p = dataset_dir / "train.jsonl"
    val_p = dataset_dir / "val.jsonl"
    if not train_p.exists():
        raise FileNotFoundError(f"No train.jsonl or hf_dataset in {dataset_dir}")
    log.info("Loading JSONL splits from {}", dataset_dir)
    train_ds = Dataset.from_json(str(train_p))
    eval_ds = Dataset.from_json(str(val_p)) if val_p.exists() else None
    return train_ds, eval_ds


def _formatting_func(example: Dict[str, Any]) -> str:
    """Render a record as a single prompt+answer string for SFTTrainer.

    Prefers the pre-rendered ``chatml`` field (written by the formatter);
    falls back to a plain Alpaca-style stub.

    Args:
        example: One dataset record.

    Returns:
        Formatted training string.
    """
    if isinstance(example.get("chatml"), str) and example["chatml"]:
        return example["chatml"]
    inst = example.get("instruction", "")
    q = example.get("input") or example.get("question", "")
    a = example.get("output") or example.get("answer", "")
    return f"### Instruction:\n{inst}\n\n### Input:\n{q}\n\n### Response:\n{a}"


# ---------------------------------------------------------------------- #
# Training driver
# ---------------------------------------------------------------------- #
def _build_training_args(
    cfg: Dict[str, Any],
    output_dir: Path,
    run_name: str,
):
    """Construct ``TrainingArguments`` from the model config dict.

    Args:
        cfg: Resolved model-config dict.
        output_dir: Per-run output directory.
        run_name: ``wandb`` / log label.

    Returns:
        ``transformers.TrainingArguments`` instance.
    """
    from transformers import TrainingArguments

    tr = cfg["training"]
    kwargs: Dict[str, Any] = dict(
        output_dir=str(output_dir),
        num_train_epochs=float(tr.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(tr.get("per_device_train_batch_size", 4)),
        per_device_eval_batch_size=int(tr.get("per_device_eval_batch_size", 4)),
        gradient_accumulation_steps=int(tr.get("gradient_accumulation_steps", 4)),
        warmup_ratio=float(tr.get("warmup_ratio", 0.03)),
        learning_rate=float(tr.get("learning_rate", 2e-4)),
        weight_decay=float(tr.get("weight_decay", 0.0)),
        fp16=bool(tr.get("fp16", True)),
        bf16=bool(tr.get("bf16", False)),
        logging_steps=int(tr.get("logging_steps", 10)),
        eval_steps=int(tr.get("eval_steps", 100)),
        save_steps=int(tr.get("save_steps", 200)),
        save_total_limit=int(tr.get("save_total_limit", 3)),
        load_best_model_at_end=bool(tr.get("load_best_model_at_end", True)),
        metric_for_best_model=tr.get("metric_for_best_model", "eval_loss"),
        greater_is_better=bool(tr.get("greater_is_better", False)),
        lr_scheduler_type=tr.get("lr_scheduler_type", "cosine"),
        optim=tr.get("optim", "paged_adamw_8bit"),
        report_to=tr.get("report_to", "wandb"),
        run_name=run_name,
        gradient_checkpointing=bool(tr.get("gradient_checkpointing", True)),
        save_safetensors=True,
    )
    # transformers renamed evaluation_strategy -> eval_strategy in 4.46+
    eval_strategy = tr.get("evaluation_strategy", "steps")
    try:
        kwargs["evaluation_strategy"] = eval_strategy
        return TrainingArguments(**kwargs)
    except TypeError:
        kwargs.pop("evaluation_strategy", None)
        kwargs["eval_strategy"] = eval_strategy
        return TrainingArguments(**kwargs)


def train_one(
    config_name: str,
    dataset_dir: Path,
    model_config_path: str,
    base_model_override: Optional[str] = None,
    timestamp: Optional[str] = None,
    sample_prompts: Optional[List[str]] = None,
    merge_after: bool = True,
) -> Path:
    """Fine-tune one LoRA config and return the output directory.

    Args:
        config_name: Name of the LoRA config to use (e.g. ``lora_r16``).
        dataset_dir: Directory containing ``train.jsonl`` / ``val.jsonl``.
        model_config_path: Path to ``config/model_config.yaml``.
        base_model_override: Optional alternative base model HF ID.
        timestamp: Optional timestamp suffix for the run name.
        sample_prompts: Prompts used by :class:`SampleGenerationCallback`.
        merge_after: If True, save a merged (adapter+base) model for inference.

    Returns:
        Path to the per-run output directory.
    """
    cfg = load_config(model_config_path)
    seed_everything(cfg.get("seed", 42))

    ts = timestamp or time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{config_name}_{ts}"
    output_root = project_root() / cfg["training"].get("output_root", "outputs")
    output_dir = ensure_dir(output_root / run_name)

    # ----- Model + tokenizer ------------------------------------------
    bundle: LoadedModel = load_quantized_model(
        model_name=base_model_override,
        model_config_path=model_config_path,
        use_4bit=True,
    )
    model, tokenizer = bundle.model, bundle.tokenizer

    # ----- PEFT prep ---------------------------------------------------
    from peft import get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=cfg["training"].get("gradient_checkpointing", True)
    )
    spec: LoRASpec = get_lora_config(config_name, model_config_path)
    log.info("Using LoRA spec: {}", spec)
    peft_model = get_peft_model(model, spec.to_peft())
    count_trainable_parameters(peft_model)

    # ----- Data --------------------------------------------------------
    train_ds, eval_ds = load_splits(dataset_dir)
    log.info("Train rows={} Val rows={}", len(train_ds), 0 if eval_ds is None else len(eval_ds))

    # ----- Training args + trainer ------------------------------------
    args = _build_training_args(cfg, output_dir, run_name)

    cb_cfg = cfg.get("callbacks", {})
    callbacks = [
        MemoryCallback(every_steps=int(cb_cfg.get("memory_log_every_steps", 100))),
        LossHistoryCallback(),
        make_early_stopping_callback(
            patience=int(cb_cfg.get("early_stopping_patience", 3)),
            threshold=float(cb_cfg.get("early_stopping_threshold", 0.0)),
        ),
    ]
    if sample_prompts:
        callbacks.append(
            SampleGenerationCallback(
                prompts=sample_prompts,
                tokenizer=tokenizer,
                every_steps=int(cb_cfg.get("sample_generation_every_steps", 200)),
                num_prompts=int(cb_cfg.get("num_sample_prompts", 5)),
            )
        )

    from trl import SFTTrainer

    sft_kwargs: Dict[str, Any] = dict(
        model=peft_model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=_formatting_func,
        callbacks=callbacks,
    )
    # trl's API changed across versions: try the modern signature first.
    try:
        trainer = SFTTrainer(
            **sft_kwargs,
            processing_class=tokenizer,
            max_seq_length=cfg["tokenizer"].get("max_seq_length", 1024),
            packing=cfg["training"].get("packing", False),
        )
    except TypeError:
        try:
            trainer = SFTTrainer(
                **sft_kwargs,
                tokenizer=tokenizer,
                max_seq_length=cfg["tokenizer"].get("max_seq_length", 1024),
                packing=cfg["training"].get("packing", False),
            )
        except TypeError:
            trainer = SFTTrainer(**sft_kwargs, tokenizer=tokenizer)

    log.info("Starting training run: {}", run_name)
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # ----- Persist metadata -------------------------------------------
    meta = {
        "config_name": config_name,
        "lora_spec": asdict(spec),
        "base_model": bundle.model_name,
        "quantization": bundle.quant_config,
        "run_name": run_name,
        "output_dir": str(output_dir),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )

    # ----- Merge adapter into base for downstream inference ----------
    if merge_after:
        try:
            log.info("Merging LoRA adapter into base model for inference...")
            merged = peft_model.merge_and_unload()
            merged.save_pretrained(str(output_dir / "merged"))
            tokenizer.save_pretrained(str(output_dir / "merged"))
        except Exception as e:  # noqa: BLE001
            log.warning("Merge step skipped due to: {}", e)

    log.info("Run complete. Artifacts in: {}", output_dir)
    return output_dir


def train_all(
    dataset_dir: Path,
    model_config_path: str,
    sample_prompts: Optional[List[str]] = None,
) -> List[Path]:
    """Run all configured LoRA specs sequentially.

    Args:
        dataset_dir: Path to the prepared dataset directory.
        model_config_path: Path to the model config.
        sample_prompts: Optional list of prompts for the sample callback.

    Returns:
        List of per-run output directories.
    """
    specs = build_all(model_config_path)
    ts = time.strftime("%Y%m%d_%H%M%S")
    outs: List[Path] = []
    for name in specs:
        outs.append(
            train_one(
                config_name=name,
                dataset_dir=dataset_dir,
                model_config_path=model_config_path,
                timestamp=ts,
                sample_prompts=sample_prompts,
            )
        )
    return outs


def _read_sample_prompts(dataset_dir: Path, k: int = 5) -> List[str]:
    """Pull a few user prompts from ``val.jsonl`` for periodic generation logging.

    Args:
        dataset_dir: Dataset directory.
        k: Max number of prompts.

    Returns:
        List of prompt strings.
    """
    val = dataset_dir / "val.jsonl"
    if not val.exists():
        return []
    prompts: List[str] = []
    with val.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            q = rec.get("input") or rec.get("question", "")
            if q:
                prompts.append(q)
            if len(prompts) >= k:
                break
    return prompts


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    """CLI entrypoint for ``python -m src.training.trainer``."""
    setup_logging()
    p = argparse.ArgumentParser(description="QLoRA fine-tune a base model on TamilTech-QA.")
    p.add_argument(
        "--config",
        default=None,
        help="LoRA config name (e.g. lora_r16). Required unless --all-configs.",
    )
    p.add_argument(
        "--all-configs",
        action="store_true",
        help="Train every configured LoRA spec sequentially.",
    )
    p.add_argument("--dataset", required=True, help="Directory of train/val splits.")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--base-model", default=None, help="Override base model HF ID.")
    p.add_argument(
        "--no-merge",
        action="store_true",
        help="Skip the post-training adapter merge.",
    )
    args = p.parse_args()

    ds_dir = Path(args.dataset)
    sample_prompts = _read_sample_prompts(ds_dir)

    if args.all_configs:
        train_all(ds_dir, args.model_config, sample_prompts=sample_prompts)
    else:
        if not args.config:
            p.error("Either --config <name> or --all-configs is required.")
        train_one(
            config_name=args.config,
            dataset_dir=ds_dir,
            model_config_path=args.model_config,
            base_model_override=args.base_model,
            sample_prompts=sample_prompts,
            merge_after=not args.no_merge,
        )


if __name__ == "__main__":
    main()
