#!/usr/bin/env bash
# Phase 4 — full evaluation, ablation, and visualization.
#
# Usage:
#   bash scripts/run_evaluation.sh

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[4A] Computing core + Tanglish-specific metrics..."
python -m src.evaluation.metrics \
    --eval-config config/eval_config.yaml \
    --model-config config/model_config.yaml \
    --data-config config/data_config.yaml

echo "[4B] Running ablation studies..."
python -m src.evaluation.ablation --study all

echo "[4C] Rendering plots..."
python -m src.evaluation.visualizer --eval-config config/eval_config.yaml

echo "Phase 4 done. Results under outputs/evaluation/ and outputs/figures/"
