"""Pre-defined LoRA configurations for the ablation study.

Configs (matching the spec):

- ``lora_r8``      — r=8,  alpha=16,  attention modules only.
- ``lora_r16``     — r=16, alpha=32,  attention modules only.
- ``lora_r64``     — r=64, alpha=128, attention modules only.
- ``lora_r16_full``— r=16, alpha=32,  attention + MLP gate/up/down projections.

Use :func:`get_lora_config` to fetch a config, or :func:`build_all` to
materialise the full dict from the YAML.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List

from src.utils import load_config
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)


@dataclass(frozen=True)
class LoRASpec:
    """Plain-Python mirror of a ``peft.LoraConfig`` for dependency-free use."""

    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float
    bias: str
    task_type: str

    def to_peft(self):
        """Return an actual ``peft.LoraConfig`` (imports peft lazily).

        Returns:
            A ``peft.LoraConfig`` instance.
        """
        from peft import LoraConfig

        return LoraConfig(
            r=self.r,
            lora_alpha=self.lora_alpha,
            target_modules=list(self.target_modules),
            lora_dropout=self.lora_dropout,
            bias=self.bias,
            task_type=self.task_type,
        )


def build_all(model_config_path: str = "config/model_config.yaml") -> Dict[str, LoRASpec]:
    """Materialise all configured LoRA specs from the YAML.

    Args:
        model_config_path: Path to ``config/model_config.yaml``.

    Returns:
        Mapping ``config_name -> LoRASpec``.

    Example:
        >>> specs = build_all()                          # doctest: +SKIP
        >>> sorted(specs)[:2]                            # doctest: +SKIP
        ['lora_r16', 'lora_r16_full']
    """
    cfg = load_config(model_config_path)
    out: Dict[str, LoRASpec] = {}
    for name, params in (cfg.get("lora_configs") or {}).items():
        out[name] = LoRASpec(
            r=int(params["r"]),
            lora_alpha=int(params["lora_alpha"]),
            target_modules=list(params["target_modules"]),
            lora_dropout=float(params.get("lora_dropout", 0.05)),
            bias=str(params.get("bias", "none")),
            task_type=str(params.get("task_type", "CAUSAL_LM")),
        )
    return out


def get_lora_config(
    config_name: str,
    model_config_path: str = "config/model_config.yaml",
) -> LoRASpec:
    """Return the named LoRA spec, raising if not configured.

    Args:
        config_name: One of the keys under ``lora_configs`` in the YAML.
        model_config_path: Path to the model config YAML.

    Returns:
        :class:`LoRASpec` for the requested config.

    Raises:
        KeyError: If ``config_name`` is not present in the YAML.
    """
    specs = build_all(model_config_path)
    if config_name not in specs:
        raise KeyError(
            f"Unknown LoRA config: {config_name!r}. Available: {sorted(specs)}"
        )
    return specs[config_name]


def count_trainable_parameters(model: Any) -> Dict[str, int]:
    """Count and log trainable vs total parameters of a (possibly PEFT) model.

    Args:
        model: A PyTorch ``nn.Module`` (e.g., a PEFT-wrapped model).

    Returns:
        Dict with ``trainable``, ``total``, and ``ratio`` keys.

    Example:
        >>> info = count_trainable_parameters(model)     # doctest: +SKIP
        >>> info["ratio"] < 0.05                          # doctest: +SKIP
        True
    """
    trainable = 0
    total = 0
    for _, p in model.named_parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    ratio = trainable / total if total else 0.0
    log.info(
        "Trainable parameters: {:,} / {:,} ({:.2%})",
        trainable, total, ratio,
    )
    return {"trainable": trainable, "total": total, "ratio": ratio}


def main() -> None:
    """CLI: print all configured LoRA specs."""
    setup_logging()
    p = argparse.ArgumentParser(description="Inspect available LoRA configs.")
    p.add_argument("--config", default="config/model_config.yaml")
    args = p.parse_args()
    specs = build_all(args.config)
    for name, spec in specs.items():
        log.info("{}: {}", name, spec)


if __name__ == "__main__":
    main()
