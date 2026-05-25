"""Phase 3 — QLoRA fine-tuning modules.

Submodules:
- ``model_loader``: 4-bit quantized base-model loading.
- ``lora_config``: pre-defined LoRA configs for the ablation study.
- ``trainer``: SFTTrainer orchestration with logging and early stopping.
- ``callbacks``: memory, sample-generation, early-stopping callbacks.
"""
