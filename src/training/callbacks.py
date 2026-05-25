"""Custom HuggingFace Trainer callbacks for TamilTech-QA.

- :class:`MemoryCallback`           — log GPU memory every N steps.
- :class:`SampleGenerationCallback` — generate sample outputs every N steps.
- :class:`LossHistoryCallback`      — accumulate the loss history for plotting.

We use ``transformers.TrainerCallback`` (and the built-in
``EarlyStoppingCallback``) so the spec's early-stopping requirement is met
without re-implementation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

log = get_logger(__name__)

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState
    from transformers.training_args import TrainingArguments
    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    TrainerCallback = object  # type: ignore[misc,assignment]
    TrainerControl = object  # type: ignore[misc,assignment]
    TrainerState = object  # type: ignore[misc,assignment]
    TrainingArguments = object  # type: ignore[misc,assignment]
    _HAS_TRANSFORMERS = False


class MemoryCallback(TrainerCallback):  # type: ignore[misc]
    """Log GPU memory usage every ``every_steps`` training steps.

    Args:
        every_steps: Logging interval (in optimizer steps).

    Example:
        >>> cb = MemoryCallback(every_steps=100)  # doctest: +SKIP
    """

    def __init__(self, every_steps: int = 100) -> None:
        self.every_steps = max(1, int(every_steps))

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Log VRAM after every ``every_steps`` steps.

        Args:
            args: Trainer arguments.
            state: Trainer state.
            control: Trainer control flow object.
            **kwargs: Additional context from the trainer.

        Returns:
            The unchanged ``control``.
        """
        if state.global_step and state.global_step % self.every_steps == 0:
            try:
                import torch

                if torch.cuda.is_available():
                    dev = torch.cuda.current_device()
                    alloc = torch.cuda.memory_allocated(dev) / 1024**3
                    reserv = torch.cuda.memory_reserved(dev) / 1024**3
                    log.info(
                        "[step {}] GPU{} mem: allocated={:.2f} GiB, reserved={:.2f} GiB",
                        state.global_step, dev, alloc, reserv,
                    )
            except ImportError:  # pragma: no cover
                pass
        return control


class SampleGenerationCallback(TrainerCallback):  # type: ignore[misc]
    """Generate ``num_prompts`` sample outputs every ``every_steps``.

    The callback expects ``trainer.tokenizer`` to be set (SFTTrainer does this).
    Generations are appended to ``output_dir/sample_outputs.jsonl``.

    Args:
        prompts: List of prompt strings.
        tokenizer: HF tokenizer.
        every_steps: How often to sample (in optimizer steps).
        max_new_tokens: Generation cap.
        num_prompts: Cap on number of prompts to evaluate (subset of `prompts`).
    """

    def __init__(
        self,
        prompts: List[str],
        tokenizer: Any,
        every_steps: int = 200,
        max_new_tokens: int = 128,
        num_prompts: int = 5,
    ) -> None:
        self.prompts = prompts[:num_prompts]
        self.tokenizer = tokenizer
        self.every_steps = max(1, int(every_steps))
        self.max_new_tokens = max_new_tokens

    def _generate(self, model: Any, prompt: str) -> str:
        """Generate a single response (used internally).

        Args:
            model: The current model.
            prompt: User prompt.

        Returns:
            Decoded model continuation.
        """
        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> TrainerControl:
        """Trigger sample generation periodically.

        Args:
            args: Trainer args.
            state: Trainer state.
            control: Trainer control.
            **kwargs: Includes ``model``.

        Returns:
            Unchanged ``control``.
        """
        if not (state.global_step and state.global_step % self.every_steps == 0):
            return control
        model = kwargs.get("model")
        if model is None or not self.prompts:
            return control
        out_path = Path(args.output_dir) / "sample_outputs.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as f:
            for prompt in self.prompts:
                try:
                    completion = self._generate(model, prompt)
                except Exception as e:  # noqa: BLE001
                    completion = f"<<generation error: {e}>>"
                f.write(
                    json.dumps(
                        {
                            "step": state.global_step,
                            "prompt": prompt,
                            "completion": completion,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        log.info("[step {}] wrote {} sample generations to {}",
                 state.global_step, len(self.prompts), out_path)
        return control


class LossHistoryCallback(TrainerCallback):  # type: ignore[misc]
    """Accumulate the training/eval loss history and persist it as JSON.

    Output file: ``{output_dir}/loss_history.json``.

    Args:
        out_path: Optional explicit path. Defaults to ``output_dir/loss_history.json``.
    """

    def __init__(self, out_path: Optional[str] = None) -> None:
        self.out_path = out_path
        self.history: List[Dict[str, Any]] = []

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> TrainerControl:
        """Record any logged metrics with the step number.

        Args:
            args: Trainer args.
            state: Trainer state.
            control: Trainer control.
            logs: Dict of metric values being logged.
            **kwargs: Additional context.

        Returns:
            Unchanged ``control``.
        """
        if not logs:
            return control
        entry = {"step": state.global_step, "epoch": state.epoch}
        entry.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        self.history.append(entry)
        target = self.out_path or str(Path(args.output_dir) / "loss_history.json")
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text(json.dumps(self.history, indent=2), encoding="utf-8")
        return control


def make_early_stopping_callback(patience: int = 3, threshold: float = 0.0):
    """Construct the built-in early-stopping callback.

    Args:
        patience: Number of eval steps without improvement before stopping.
        threshold: Minimum improvement to count as progress.

    Returns:
        ``transformers.EarlyStoppingCallback`` instance.

    Example:
        >>> cb = make_early_stopping_callback(patience=3)  # doctest: +SKIP
    """
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=threshold)
