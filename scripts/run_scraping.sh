#!/usr/bin/env bash
# Phase 1 — collect raw data from YouTube + GPT-4o synthesis + public corpora.
#
# Usage:
#   bash scripts/run_scraping.sh
#
# Requires .env to define YOUTUBE_API_KEY and OPENAI_API_KEY.

set -euo pipefail

cd "$(dirname "$0")/.."

echo "[1A] YouTube scraping..."
python -m src.data_collection.youtube_scraper --config config/data_config.yaml

echo "[1B] Synthetic QA generation..."
python -m src.data_collection.synthetic_generator --config config/data_config.yaml

echo "[1C] Dataset merge..."
python -m src.data_collection.dataset_merger --config config/data_config.yaml

echo "Phase 1 done. Output: data/raw/merged_corpus.jsonl"
