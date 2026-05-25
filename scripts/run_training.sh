#!/usr/bin/env bash
# Phase 3 — QLoRA fine-tuning.
#
# Usage:
#   bash scripts/run_training.sh                  # train lora_r16 (default)
#   bash scripts/run_training.sh all              # train all 4 configs
#   bash scripts/run_training.sh lora_r64         # train a specific config

set -euo pipefail

cd "$(dirname "$0")/.."

CFG="${1:-lora_r16}"

if [[ "$CFG" == "all" ]]; then
    echo "Training every LoRA config (this is the ablation sweep)..."
    python -m src.training.trainer --all-configs --dataset data/final/
else
    echo "Training config: $CFG"
    python -m src.training.trainer --config "$CFG" --dataset data/final/
fi

echo "Phase 3 done. Artifacts under outputs/<run_name>/"
