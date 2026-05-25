#!/usr/bin/env bash
# Phase 2 — clean, filter, format, and split the merged corpus.
#
# Usage:
#   bash scripts/run_preprocessing.sh

set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/processed data/final

echo "[2A] Cleaning + technical-keyword gate + MinHash dedup..."
python -m src.preprocessing.text_cleaner \
    --config config/data_config.yaml \
    --input data/raw/merged_corpus.jsonl \
    --output data/processed/cleaned.jsonl

echo "[2B] Tanglish detection + band filter..."
python -m src.preprocessing.tanglish_detector \
    --config config/data_config.yaml \
    --input data/processed/cleaned.jsonl \
    --output data/processed/tanglish.jsonl

echo "[2C] Alpaca + ChatML formatting + stratified split..."
python -m src.preprocessing.qa_formatter \
    --config config/data_config.yaml \
    --input data/processed/tanglish.jsonl \
    --output data/final/

echo "Phase 2 done. Output: data/final/{train,val,test}.jsonl + dataset_stats.json"
