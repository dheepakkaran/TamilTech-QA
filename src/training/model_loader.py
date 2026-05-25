"""4-bit quantized model + tokenizer loader for QLoRA fine-tuning.

Logs VRAM usage before and after loading so it is trivial to confirm the
A100 budget is respected.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from src.utils import load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


@dataclass
class LoadedModel:
    """Bundle of model + tokenizer + the resolved config dict."""

    model: Any
    tokenizer: Any
    model_name: str
    quant_config: Dict[str, Any]


def _gpu_memory_summary(prefix: str) -> None:
    """Log allocated and reserved CUDA memory for the current device.

    Args:
        prefix: Free-form label written at the start of the line.

    Returns:
        None.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            log.info("[{}] CUDA not available — running on CPU.", prefix)
            return
        dev = torch.cuda.current_device()
        alloc = torch.cuda.memory_allocated(dev) / 1024**3
        reserv = torch.cuda.memory_reserved(dev) / 1024**3
        log.info(
            "[{}] GPU{} memory: allocated={:.2f} GiB reserved={:.2f} GiB",
            prefix, dev, alloc, reserv,
        )
    except ImportError:  # pragma: no cover
        pass


def _resolve_dtype(name: str) -> Any:
    """Resolve a torch dtype string to the actual dtype.

    Args:
        name: One of ``"float16"``, ``"bfloat16"``, ``"float32"``.

    Returns:
        The corresponding torch dtype.

    Raises:
        ValueError: If ``name`` is unknown.
    """
    import torch

    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }.get(name.lower()) or _raise_dtype(name)


def _raise_dtype(name: str):
    raise ValueError(f"Unknown dtype string: {name!r}")


def load_quantized_model(
    model_name: Optional[str] = None,
    model_config_path: str = "config/model_config.yaml",
    use_4bit: bool = True,
) -> LoadedModel:
    """Load a base causal-LM in 4-bit precision with the prompt's BNB settings.

    Args:
        model_name: Override for the base model HF ID. If ``None`` the config's
            ``base_model.default`` is used.
        model_config_path: Path to ``config/model_config.yaml``.
        use_4bit: If False, load in fp16 instead of 4-bit (debug only).

    Returns:
        :class:`LoadedModel` bundle.

    Example:
        >>> bundle = load_quantized_model()             # doctest: +SKIP
        >>> bundle.model.config.model_type              # doctest: +SKIP
        'llama'
    """
    cfg = load_config(model_config_path)
    bm = cfg["base_model"]
    qc = cfg["quantization"]
    tk_cfg = cfg["tokenizer"]
    name = model_name or bm["default"]

    log.info("Loading tokenizer for {}", name)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        name,
        trust_remote_code=bm.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        if tk_cfg.get("pad_token_strategy", "eos") == "eos" and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    tokenizer.padding_side = tk_cfg.get("padding_side", "right")
    tokenizer.truncation_side = tk_cfg.get("truncation_side", "right")

    _gpu_memory_summary("before model load")

    log.info("Loading base model {} (4bit={})", name, use_4bit)
    from transformers import AutoModelForCausalLM

    quant_kwargs: Dict[str, Any] = {}
    quant_config_dict: Dict[str, Any] = {}
    if use_4bit:
        from transformers import BitsAndBytesConfig
        import torch

        bnb_dtype = _resolve_dtype(qc.get("bnb_4bit_compute_dtype", "float16"))
        bnb = BitsAndBytesConfig(
            load_in_4bit=bool(qc.get("load_in_4bit", True)),
            bnb_4bit_compute_dtype=bnb_dtype,
            bnb_4bit_use_double_quant=bool(qc.get("bnb_4bit_use_double_quant", True)),
            bnb_4bit_quant_type=qc.get("bnb_4bit_quant_type", "nf4"),
        )
        quant_kwargs["quantization_config"] = bnb
        quant_kwargs["torch_dtype"] = bnb_dtype
        quant_config_dict = {
            "load_in_4bit": True,
            "bnb_4bit_compute_dtype": str(bnb_dtype),
            "bnb_4bit_use_double_quant": bnb.bnb_4bit_use_double_quant,
            "bnb_4bit_quant_type": bnb.bnb_4bit_quant_type,
        }
    else:
        import torch

        quant_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",
        trust_remote_code=bm.get("trust_remote_code", False),
        **quant_kwargs,
    )
    # Resize if we added a pad token
    if tokenizer.pad_token == "<|pad|>" and getattr(model.get_input_embeddings(), "num_embeddings", 0) != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    model.config.use_cache = False
    if hasattr(model, "config"):
        model.config.pad_token_id = tokenizer.pad_token_id

    _gpu_memory_summary("after model load")
    return LoadedModel(model=model, tokenizer=tokenizer, model_name=name, quant_config=quant_config_dict)


def main() -> None:
    """CLI: load a model and print its memory footprint (smoke test)."""
    setup_logging()
    p = argparse.ArgumentParser(description="Smoke-test the 4-bit model loader.")
    p.add_argument("--model", default=None)
    p.add_argument("--config", default="config/model_config.yaml")
    p.add_argument("--no-4bit", action="store_true")
    args = p.parse_args()
    load_quantized_model(
        model_name=args.model,
        model_config_path=args.config,
        use_4bit=not args.no_4bit,
    )


if __name__ == "__main__":
    main()
