# 🏗️ Document 3 — Local Setup From Scratch (Tanglish Edition)

> **Intha doc-ku purpose:** Empty laptop-le start panni, TamilTech-QA-a fully **own-aa build pannardhu**. Each line, each function, each import — meaning + purpose explain panren.
>
> **Reading time:** ~3 hours (with hands-on practice).
> **Prereq:** Python 3.10+ installed, GitHub account, HuggingFace account, OpenAI account, YouTube Data API key.

---

## Table of Contents

1. [Pre-requisites — Software install](#1-pre-requisites--software-install)
2. [Project skeleton — Folder structure create](#2-project-skeleton--folder-structure-create)
3. [Python environment — venv + requirements.txt](#3-python-environment--venv--requirementstxt)
4. [Secrets — `.env` file setup](#4-secrets--env-file-setup)
5. [Configuration YAMLs explained](#5-configuration-yamls-explained)
6. [`src/utils/` walkthrough](#6-srcutils-walkthrough)
7. [`src/data_collection/` walkthrough](#7-srcdata_collection-walkthrough)
8. [`src/preprocessing/` walkthrough](#8-srcpreprocessing-walkthrough)
9. [`src/training/` walkthrough](#9-srctraining-walkthrough)
10. [`src/evaluation/` walkthrough](#10-srcevaluation-walkthrough)
11. [Tests — `tests/` folder](#11-tests--tests-folder)
12. [Scripts — `scripts/` folder](#12-scripts--scripts-folder)
13. [Local dry-run end-to-end](#13-local-dry-run-end-to-end)
14. [Common pitfalls (and fixes)](#14-common-pitfalls-and-fixes)

---

## 1. Pre-requisites — Software install

### 🛠️ Required software (Windows)

| Software | Version | Install command |
|---|---|---|
| Python | 3.10 or 3.11 (3.12 has package conflicts) | https://python.org/downloads |
| Git | latest | https://git-scm.com/download/win |
| VS Code | latest | https://code.visualstudio.com |
| Notepad++ or VS Code | for YAML editing | (with VS Code) |

### 🔑 API keys to obtain

| Service | Where to get | Purpose | Free tier? |
|---|---|---|---|
| **YouTube Data API v3** | https://console.cloud.google.com/apis/library/youtube.googleapis.com | Scrape comments | Yes — 10K units/day |
| **OpenAI** | https://platform.openai.com/api-keys | Synthetic data via gpt-4o-mini | Pay-as-you-go (~$0.10 for project) |
| **HuggingFace** | https://huggingface.co/settings/tokens | Upload model/dataset | Free |

### Verify installs

```powershell
python --version          # should show 3.10.x or 3.11.x
git --version             # any 2.x is fine
```

Issue-na: Python `PATH` la add aagala-na, installer-le **"Add Python to PATH"** checkbox check pannitu reinstall pannu.

---

## 2. Project skeleton — Folder structure create

### Step-by-step

```powershell
# 1. Navigate to where you keep projects
cd C:\Users\LENOVO\Downloads

# 2. Create root folder
mkdir tamiltech-qa
cd tamiltech-qa
```

### Target folder structure

```
tamiltech-qa/
├── .env                    # secrets (DO NOT commit)
├── .gitignore              # protect .env + data + outputs
├── README.md               # project documentation
├── LICENSE                 # MIT
├── requirements.txt        # Python dependencies
├── setup.py                # package metadata
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── eval_config.yaml
├── src/
│   ├── __init__.py
│   ├── data_collection/
│   │   ├── __init__.py
│   │   ├── youtube_scraper.py
│   │   ├── synthetic_generator.py
│   │   └── dataset_merger.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── text_cleaner.py
│   │   ├── tanglish_detector.py
│   │   ├── language_filter.py
│   │   └── qa_formatter.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── model_loader.py
│   │   ├── lora_config.py
│   │   ├── callbacks.py
│   │   └── trainer.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   ├── ablation.py
│   │   ├── human_eval.py
│   │   └── visualizer.py
│   └── utils/
│       ├── __init__.py
│       ├── logger.py
│       ├── seed.py
│       └── hf_uploader.py
├── data/
│   ├── raw/                # scraped data (gitignored)
│   ├── processed/          # cleaned (gitignored)
│   └── final/              # train/val/test (gitignored)
├── outputs/                # training artifacts (gitignored)
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_analysis.ipynb
│   ├── 03_training_monitoring.ipynb
│   ├── 04_results_visualization.ipynb
│   └── train_on_kaggle.ipynb
├── scripts/
│   ├── run_scraping.sh
│   ├── run_preprocessing.sh
│   ├── run_training.sh
│   └── run_evaluation.sh
└── tests/
    ├── __init__.py
    ├── test_language_filter.py
    ├── test_metrics.py
    └── test_qa_formatter.py
```

### Quick create script

```powershell
# All folders
New-Item -ItemType Directory -Force -Path config, src, src\data_collection, src\preprocessing, src\training, src\evaluation, src\utils, data\raw, data\processed, data\final, outputs, notebooks, scripts, tests

# All __init__.py files (empty for now)
$pkgs = "src", "src\data_collection", "src\preprocessing", "src\training", "src\evaluation", "src\utils", "tests"
foreach ($p in $pkgs) { New-Item -ItemType File -Force -Path "$p\__init__.py" }
```

---

## 3. Python environment — venv + requirements.txt

### Step 3.1: Create virtual env

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Why venv?** Each project gets its **own dependencies**, no global pollution. Different projects can use different `transformers` versions without conflict.

**Activation verify:**
```powershell
where python
# Should show: C:\Users\LENOVO\Downloads\tamiltech-qa\.venv\Scripts\python.exe
```

### Step 3.2: Create `requirements.txt`

```text
# Core ML
torch>=2.1.0
transformers>=4.40.0
peft>=0.10.0
bitsandbytes>=0.43.0
accelerate>=0.30.0
trl>=0.8.0
datasets>=2.18.0
sentencepiece>=0.2.0

# Data collection
google-api-python-client>=2.100.0
openai>=1.30.0
tenacity>=8.2.0

# Data processing
pandas>=2.0.0
numpy>=1.24.0
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.66.0
loguru>=0.7.0
langdetect>=1.0.9

# Evaluation
sacrebleu>=2.4.0
rouge-score>=0.1.2
nltk>=3.8.0
bert-score>=0.3.13

# Visualization
matplotlib>=3.7.0
seaborn>=0.13.0

# Deployment
huggingface_hub>=0.23.0
gradio>=4.30.0

# Dev / testing
pytest>=8.0.0
pytest-cov>=5.0.0
```

#### Version pinning philosophy

- **Lower bound only (`>=`)** for libraries that frequently get fixes
- **Use latest** for reproducibility — but in production, freeze with `pip freeze`
- **`torch` and `bitsandbytes`** are CUDA-sensitive — pin if you hit ABI issues

### Step 3.3: Install

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

⏱️ Takes ~5-10 minutes. `torch` is the big download (~2 GB).

### Step 3.4: Verify torch + CUDA (if NVIDIA GPU)

```python
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

CPU-only-na **False, no GPU** varum. Training Kaggle-le pannardhu by design, so CPU local is fine.

---

## 4. Secrets — `.env` file setup

### Step 4.1: Create `.env`

```powershell
notepad .env
```

Paste:

```
# YouTube Data API v3
YOUTUBE_API_KEY=AIzaSy...your_key_here

# OpenAI (gpt-4o-mini for synthetic data)
OPENAI_API_KEY=sk-proj-...your_key_here

# HuggingFace Hub
HF_TOKEN=hf_...your_token_here

# Optional: Weights & Biases
# WANDB_API_KEY=...
```

**Critical rules:**
- ✅ `.env` already in `.gitignore` — git won't commit it
- ✅ No quotes around values
- ✅ No spaces around `=`
- ❌ **Never** paste keys in code, configs, or chat messages
- ❌ **Never** check `.env` into git (even private repos — habits stick)

### Step 4.2: Create `.gitignore`

```gitignore
# Secrets (NEVER commit)
.env
.env.*
*.key
*.token
secrets.yaml
credentials.json

# Generated data
data/raw/
data/processed/
data/final/
outputs/

# Python
__pycache__/
*.pyc
*.egg-info/
build/
dist/
.pytest_cache/
.coverage

# Virtual env
.venv/
venv/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# Logs
*.log
```

### Step 4.3: Load `.env` in Python

We use `python-dotenv`:

```python
# At top of any script that needs env vars
from dotenv import load_dotenv
load_dotenv()  # reads .env into os.environ

import os
api_key = os.environ["YOUTUBE_API_KEY"]
```

**Important:** `load_dotenv()` is **idempotent** — safe to call multiple times. Always call **before** reading env vars.

---

## 5. Configuration YAMLs explained

### 5.1 `config/data_config.yaml`

Full file (copy-paste-able):

```yaml
# Data collection and preprocessing configuration
seed: 42

paths:
  raw_dir: data/raw
  processed_dir: data/processed
  final_dir: data/final

youtube:
  api_key: "${YOUTUBE_API_KEY}"        # expanded from .env
  target_channels:
    - UCIFQgj1Rhx-FFgyo0zzPSfw         # Brototype Tamil
    - UCKOob5-7sMljgW3f4pO_Dyg         # Tamil Coding Wizard
    # ... 10 more
  max_comments_per_video: 200
  max_videos_per_channel: 80
  retry:
    max_attempts: 5
    base_delay_seconds: 2.0
    max_delay_seconds: 60.0

synthetic:
  openai_api_key: "${OPENAI_API_KEY}"
  model: "gpt-4o-mini"
  temperature: 0.8
  max_tokens: 3000
  n_pairs_per_topic: 50
  topics: [python, dsa, ml, ece, web, os, networking, databases, algorithms, debugging]

preprocessing:
  tanglish_ratio_min: 0.05             # loose lower bound
  tanglish_ratio_max: 0.95             # loose upper bound
  min_words: 10
  max_tokens: 512
  technical_keywords: [function, array, class, error, ...]
  tamil_connectors: [enna na, apdi patha, ...]

split:
  train_ratio: 0.80
  val_ratio: 0.10
  test_ratio: 0.10
  stratify_by: [topic, difficulty]
```

#### Line-by-line meaning

- `seed: 42` — reproducibility seed for any randomization
- `paths.raw_dir` — relative to project root
- `youtube.api_key: "${YOUTUBE_API_KEY}"` — placeholder expanded at config load time
- `target_channels` — list of YouTube channel IDs (find via channel page → URL or third-party tool)
- `max_videos_per_channel: 80` — quota-budget knob
- `retry` block — exponential backoff config
- `tanglish_ratio_min/max` — band filter bounds
- `min_words: 10` — drop comments shorter than 10 words (too short to be useful QA)
- `stratify_by` — split keys (preserve topic+difficulty proportions across train/val/test)

### 5.2 `config/model_config.yaml`

```yaml
seed: 42

base_model:
  default: "meta-llama/Llama-3.1-8B-Instruct"
  trust_remote_code: false

quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: "float16"
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: "nf4"

tokenizer:
  pad_token_strategy: "eos"
  padding_side: "right"
  truncation_side: "right"
  max_seq_length: 512

lora_configs:
  lora_r8:                                    # the v0.1 default
    r: 8
    lora_alpha: 16
    target_modules: [q_proj, v_proj, k_proj, o_proj]
    lora_dropout: 0.05
    bias: "none"
    task_type: "CAUSAL_LM"

training:
  output_root: "outputs"
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8              # effective batch = 16
  warmup_ratio: 0.03
  learning_rate: 2.0e-4
  bf16: true                                   # ← critical for Llama-3.1
  fp16: false                                  # ← MUST be false
  logging_steps: 5
  eval_strategy: "steps"
  eval_steps: 30
  save_steps: 60
  save_total_limit: 2
  load_best_model_at_end: true
  lr_scheduler_type: "cosine"
  optim: "paged_adamw_8bit"
  gradient_checkpointing: true
```

#### Why each setting

| Setting | Why this value |
|---|---|
| `Llama-3.1-8B-Instruct` | Best open model that fits T4 in 4-bit |
| `load_in_4bit: true` | Memory: 16 GB → 4 GB |
| `bnb_4bit_quant_type: "nf4"` | NormalFloat-4, better than uniform |
| `pad_token_strategy: "eos"` | Llama-3.1 has no native pad token |
| `lora_r8` | Sweet spot: enough capacity, fast train |
| `target_modules: [q,k,v,o]` | Attention weights — standard LoRA practice |
| `per_device_train_batch_size: 2` | Largest that fits 16 GB T4 with seq=512 |
| `gradient_accumulation_steps: 8` | Effective batch 16 → stable gradients |
| `learning_rate: 2.0e-4` | QLoRA standard (Dettmers paper) |
| `bf16: true` | Llama-3.1 native dtype |
| `optim: "paged_adamw_8bit"` | Saves optimizer state memory |

---

## 6. `src/utils/` walkthrough

### 6.1 `src/utils/__init__.py`

```python
"""Project-wide utilities and config loader."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict

import yaml
from dotenv import load_dotenv

# Load .env once at import time
load_dotenv()


def project_root() -> Path:
    """Return absolute path to the project root."""
    return Path(__file__).resolve().parents[2]


def ensure_dir(path: Path | str) -> Path:
    """Create directory if missing. Returns the Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


_ENV_VAR_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _expand_env_vars(text: str) -> str:
    """Replace ${VAR} with os.environ[VAR]. Empty if missing."""
    return _ENV_VAR_RE.sub(lambda m: os.environ.get(m.group(1), ""), text)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load YAML config with ${VAR} substitution."""
    raw = Path(path).read_text(encoding="utf-8")
    expanded = _expand_env_vars(raw)
    return yaml.safe_load(expanded)
```

#### Line-by-line breakdown

| Line(s) | Meaning |
|---|---|
| `from __future__ import annotations` | Enable forward references in type hints (Python 3.10 compatibility) |
| `load_dotenv()` | Read `.env` into `os.environ` — runs once on first import |
| `project_root()` | `Path(__file__)` = `.../src/utils/__init__.py`; `.parents[2]` = `.../tamiltech-qa/` |
| `ensure_dir(path)` | Recursively create folder; idempotent (no error if exists) |
| `_ENV_VAR_RE` | Regex matching `${VAR_NAME}` style placeholders |
| `_expand_env_vars(text)` | For each `${VAR}` match, look up `os.environ[VAR]`; return `""` if missing |
| `load_config(path)` | Read file → expand vars → parse YAML |

#### Why this pattern?

- ❌ **Bad:** secrets directly in YAML → committed by accident
- ✅ **Good:** YAML has `${VAR}` placeholders → resolved from `.env` at runtime

### 6.2 `src/utils/logger.py`

```python
"""Structured logging via loguru."""
import sys
from loguru import logger as _logger

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure loguru once. Idempotent."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    _logger.remove()  # remove default handler
    _logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level:<8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    _CONFIGURED = True


def get_logger(name: str):
    """Get a logger bound to the module name."""
    setup_logging()
    return _logger.bind(name=name)
```

#### Why loguru over stdlib `logging`?

- **Zero config** to look pretty
- **Built-in colors** in terminal
- **Better tracebacks** (full local vars on exception)
- **f-string style** formatting: `log.info("Got {} items", n)` instead of `%s` % n

### 6.3 `src/utils/seed.py`

```python
"""Deterministic seeding for reproducibility."""
import os
import random

import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Seed all RNGs across Python, NumPy, PyTorch."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
```

#### What each line does

| Line | Effect |
|---|---|
| `os.environ["PYTHONHASHSEED"]` | Stabilizes dict ordering (Python hash randomization) |
| `random.seed` | Python `random` module |
| `np.random.seed` | NumPy global RNG |
| `torch.manual_seed` | CPU PyTorch RNG |
| `torch.cuda.manual_seed_all` | GPU RNG (all devices) |
| `cudnn.deterministic = True` | Disable nondeterministic CUDNN kernels (10-20% slower but reproducible) |

---

## 7. `src/data_collection/` walkthrough

### 7.1 `youtube_scraper.py` — Key functions

```python
"""YouTube Data API v3 client for Tanglish tech channel scraping."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, List, Dict, Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from tenacity import retry, stop_after_attempt, wait_exponential

from src.utils import load_config, ensure_dir, project_root
from src.utils.logger import get_logger

log = get_logger(__name__)


class YouTubeScraper:
    def __init__(self, config_path: str = "config/data_config.yaml"):
        self.cfg = load_config(config_path)
        self.api_key = self.cfg["youtube"]["api_key"]
        if not self.api_key:
            raise RuntimeError("YOUTUBE_API_KEY not set in .env")
        self.service = build("youtube", "v3", developerKey=self.api_key)
        self.state_file = project_root() / "data" / "raw" / ".scraped_videos.json"
        self.scraped_video_ids: set = self._load_state()
```

#### Walkthrough

- `build("youtube", "v3", developerKey=...)` — creates an authorized API client
- `state_file` — JSON tracking which videos we've already processed (resume support)
- `scraped_video_ids` — set, in-memory for fast lookup

```python
    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=lambda r: isinstance(r.outcome.exception(), HttpError),
    )
    def fetch_video_comments(self, video_id: str) -> List[Dict[str, Any]]:
        """Fetch top-level comments for a video. Returns list of comment dicts."""
        comments = []
        next_page = None
        while True:
            req = self.service.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page,
                textFormat="plainText",
            )
            resp = req.execute()
            for item in resp.get("items", []):
                top = item["snippet"]["topLevelComment"]["snippet"]
                comments.append({
                    "video_id": video_id,
                    "comment_id": item["id"],
                    "text": top["textDisplay"],
                    "author": top["authorDisplayName"],
                    "likes": top["likeCount"],
                    "published_at": top["publishedAt"],
                })
            next_page = resp.get("nextPageToken")
            if not next_page or len(comments) >= self.cfg["youtube"]["max_comments_per_video"]:
                break
        return comments
```

#### Key concepts

| Decorator/Code | Purpose |
|---|---|
| `@retry(stop=stop_after_attempt(5), wait=wait_exponential(...))` | Tenacity decorator — retry 5x with exponential backoff (2s, 4s, 8s, 16s, 32s) |
| `pageToken=next_page` | YouTube paginates results, this fetches next page |
| `maxResults=100` | API per-page limit |
| `if not next_page or len >= max` | Stop when out of pages OR limit reached |

```python
    def run(self) -> None:
        """Main entry: scrape all configured channels."""
        out_dir = ensure_dir(project_root() / self.cfg["paths"]["raw_dir"])
        out_path = out_dir / "youtube_comments.jsonl"

        with out_path.open("a", encoding="utf-8") as f:  # append for resume
            for ch_id in self.cfg["youtube"]["target_channels"]:
                videos = self.fetch_channel_videos(ch_id)
                for video in videos:
                    if video["id"] in self.scraped_video_ids:
                        log.info("Skipping already-scraped {}", video["id"])
                        continue
                    try:
                        comments = self.fetch_video_comments(video["id"])
                        for c in comments:
                            f.write(json.dumps(c, ensure_ascii=False) + "\n")
                        self._mark_scraped(video["id"])
                    except HttpError as e:
                        log.warning("Failed video {}: {}", video["id"], e)
                        continue
```

#### Walkthrough

- `out_path.open("a", ...)` — **append mode** — important for resume safety
- `if ... in self.scraped_video_ids` — skip if already processed
- `ensure_ascii=False` — preserve Unicode (Tamil characters)
- `try/except HttpError` — keep going if one video fails

### 7.2 `synthetic_generator.py`

```python
"""Synthetic Tanglish QA generation via OpenAI gpt-4o-mini."""
import json
from openai import OpenAI

from src.utils import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


PROMPT_TEMPLATE = """You are an expert Tanglish (Tamil-English code-switched) 
technical content creator. Generate {n} question-answer pairs about {topic}.

Rules:
1. Use Tanglish: Tamil scaffolding + English technical terms
2. Roman-script Tamil (no Tamil script)
3. Casual, natural tone — how real Tamil techies talk
4. Preserve English technical vocabulary verbatim
5. Each answer 30-150 words

Output JSON with exactly this format:
{{
  "pairs": [
    {{"question": "...", "answer": "..."}},
    ...
  ]
}}
"""


def generate_topic(topic: str, n_pairs: int, cfg: dict) -> list:
    client = OpenAI(api_key=cfg["synthetic"]["openai_api_key"])

    response = client.chat.completions.create(
        model=cfg["synthetic"]["model"],
        temperature=cfg["synthetic"]["temperature"],
        max_tokens=cfg["synthetic"]["max_tokens"],
        response_format={"type": "json_object"},      # ← CRITICAL
        messages=[
            {"role": "system", "content": "You output only valid JSON."},
            {"role": "user", "content": PROMPT_TEMPLATE.format(n=n_pairs, topic=topic)},
        ],
    )
    content = response.choices[0].message.content
    data = json.loads(content)
    return data.get("pairs", [])
```

#### Why every line matters

| Line | Reason |
|---|---|
| `response_format={"type": "json_object"}` | Forces gpt-4o-mini into JSON mode — **without this, 30% of responses are malformed** |
| `"You output only valid JSON."` | System message reinforces the JSON discipline |
| `temperature=0.8` | High enough for variety, low enough for coherence |
| `max_tokens=3000` | Enough room for 50 pairs × ~50 tokens each |
| `data.get("pairs", [])` | Safe default if key missing |

### 7.3 `dataset_merger.py`

```python
"""Merge YouTube + synthetic + HF datasets into single JSONL."""
import json
from pathlib import Path
from typing import Iterator

from src.utils import load_config, ensure_dir, project_root


def stream_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def merge_sources(cfg: dict) -> None:
    raw_dir = project_root() / cfg["paths"]["raw_dir"]
    out_path = ensure_dir(raw_dir) / "merged.jsonl"

    with out_path.open("w", encoding="utf-8") as out:
        # Tag each source for downstream stratification
        for source_file, source_name in [
            (raw_dir / "youtube_comments.jsonl", "youtube"),
            (raw_dir / "synthetic_pairs.jsonl", "synthetic"),
        ]:
            if not source_file.exists():
                continue
            for rec in stream_jsonl(source_file):
                rec["source"] = source_name
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
```

#### Key insight: streaming I/O

Don't load all data into memory. Use generator (`yield`) — lazy, scales to millions of lines.

---

## 8. `src/preprocessing/` walkthrough

### 8.1 `text_cleaner.py`

```python
"""Strip emoji, normalize Unicode, fix encoding."""
import re
import unicodedata


EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"     # emoticons
    "\U0001F300-\U0001F5FF"     # symbols & pictographs
    "\U0001F680-\U0001F6FF"     # transport
    "\U0001F1E0-\U0001F1FF"     # flags
    "]+",
    flags=re.UNICODE,
)


def clean_text(text: str) -> str:
    """Apply standard cleanup."""
    if not text:
        return ""
    # 1. Unicode normalize (handle decomposed Tamil characters)
    text = unicodedata.normalize("NFKC", text)
    # 2. Strip emoji
    text = EMOJI_RE.sub("", text)
    # 3. Collapse whitespace
    text = re.sub(r"\s+", " ", text)
    # 4. Strip leading/trailing
    return text.strip()
```

#### Why each step

| Step | Reason |
|---|---|
| `NFKC normalize` | Same Tamil character can have multiple Unicode representations; canonicalize |
| `EMOJI_RE.sub("", text)` | Emoji-only "comments" pollute training (e.g., "👍👍👍") |
| `\s+ → " "` | Multiple newlines/tabs → single space |
| `.strip()` | Remove leading/trailing whitespace |

### 8.2 `tanglish_detector.py` — Core logic

```python
DEFAULT_TAMIL_WORDLIST = {
    "naan", "neenga", "avan", "ava", "naanga",
    "indha", "intha", "athu", "ithu",
    "irukku", "irundha", "irukka",
    "panrathu", "panna", "panren", "pannu",
    "sollunga", "solli", "sonna",
    "athukku", "ippo", "apdi", "appdi",
    # ... ~60 total
}

TAMIL_SUFFIXES = ("-a", "-aa", "-na", "-thu", "-um", "-aanu", "-laam")

TAMIL_SCRIPT_RE = re.compile(r"[஀-௿]+")     # Tamil Unicode block
WORD_RE = re.compile(r"[A-Za-z஀-௿][A-Za-z஀-௿\-']*")


class TanglishDetector:
    def __init__(self, min_ratio=0.05, max_ratio=0.95, wordlist=None):
        self.lexicon = wordlist or DEFAULT_TAMIL_WORDLIST
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio

    def is_tamil_word(self, token: str) -> bool:
        """Three-criteria check: script, lexicon, suffix."""
        if TAMIL_SCRIPT_RE.search(token):           # 1. Tamil script
            return True
        t = token.lower()
        if t in self.lexicon:                       # 2. Lexicon match
            return True
        if any(t.endswith(s) for s in TAMIL_SUFFIXES):
            return t not in _ENGLISH_GUARD          # 3. Suffix + not in guard
        return False

    def score(self, text: str) -> dict:
        tokens = WORD_RE.findall(text or "")
        total = len(tokens)
        if total == 0:
            return {"tanglish_ratio": 0.0, "total_word_count": 0}
        tamil_count = sum(1 for t in tokens if self.is_tamil_word(t))
        return {
            "tanglish_ratio": round(tamil_count / total, 4),
            "tamil_word_count": tamil_count,
            "total_word_count": total,
        }

    def keep(self, text: str) -> bool:
        s = self.score(text)
        return self.min_ratio <= s["tanglish_ratio"] <= self.max_ratio


_ENGLISH_GUARD = {"schema", "java", "lambda", "alpha", "beta", "data", "extra"}
```

#### Walkthrough

| Concept | Explanation |
|---|---|
| `r"[஀-௿]+"` | Tamil Unicode block (U+0B80 to U+0BFF) — catches any character in Tamil script |
| `WORD_RE` | Match a "word" = letters + possibly hyphens/apostrophes |
| `is_tamil_word` 3 checks | **Script**: `வணக்கம்` matches. **Lexicon**: `naan` matches. **Suffix**: `irukkuna` ends in `-na` |
| `_ENGLISH_GUARD` | Prevents false positives like `schema` (ends in `-a` but is English) |
| `score` returns dict | Easier to extend than returning float |

### 8.3 `qa_formatter.py`

```python
"""Format raw rows as instruction/response or ChatML."""
import json
from typing import List, Dict


ALPACA_TEMPLATE = (
    "### Instruction:\n"
    "{instruction}\n\n"
    "### Response:\n"
    "{response}"
)


def to_alpaca(sample: dict) -> str:
    """Format as Alpaca instruction-tuning template."""
    return ALPACA_TEMPLATE.format(
        instruction=sample["question"],
        response=sample["answer"],
    )


def to_chatml(sample: dict, system_prompt: str) -> List[Dict[str, str]]:
    """Format as ChatML (Llama-3.1-Instruct native format)."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]


def apply_chat_template(tokenizer, sample: dict, system_prompt: str) -> str:
    """Use tokenizer's chat template to produce model-ready text."""
    messages = to_chatml(sample, system_prompt)
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,  # we're training, not generating
    )
```

#### Why ChatML for Llama-3.1?

Llama-3.1-Instruct pretraining used this exact format:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a helpful Tanglish assistant.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Anna React-a explain pannunga
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
React na frontend library...
<|eot_id|>
```

`tokenizer.apply_chat_template()` automatically wraps your messages with the correct special tokens.

---

## 9. `src/training/` walkthrough

### 9.1 `model_loader.py` — Detailed

```python
from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.utils import load_config
from src.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class LoadedModel:
    """Bundle of model + tokenizer + the resolved config dict."""
    model: Any
    tokenizer: Any
    model_name: str
    quant_config: Dict[str, Any]


def load_quantized_model(
    model_name: Optional[str] = None,
    model_config_path: str = "config/model_config.yaml",
    use_4bit: bool = True,
) -> LoadedModel:
    cfg = load_config(model_config_path)
    bm = cfg["base_model"]
    qc = cfg["quantization"]
    tk_cfg = cfg["tokenizer"]
    name = model_name or bm["default"]

    # Step 1: Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)

    # Step 2: Set pad token (Llama-3.1 doesn't have one)
    if tokenizer.pad_token is None:
        if tk_cfg.get("pad_token_strategy", "eos") == "eos":
            tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = tk_cfg.get("padding_side", "right")

    # Step 3: Build quantization config
    if use_4bit:
        from transformers import BitsAndBytesConfig
        import torch

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        quant_kwargs = {
            "quantization_config": bnb,
            "torch_dtype": torch.float16,
        }
    else:
        quant_kwargs = {"torch_dtype": torch.float16}

    # Step 4: Load model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        name,
        device_map="auto",         # auto-distribute across GPUs if multi
        **quant_kwargs,
    )

    # Step 5: Configure for training
    model.config.use_cache = False    # incompatible with gradient checkpointing
    model.config.pad_token_id = tokenizer.pad_token_id

    return LoadedModel(model=model, tokenizer=tokenizer, model_name=name, quant_config={})
```

#### Critical lines explained

| Line | Why |
|---|---|
| `use_fast=True` | Use Rust-backed tokenizer (10-100× faster than Python) |
| `tokenizer.pad_token = tokenizer.eos_token` | Llama-3.1 has no pad token; reuse EOS |
| `bnb_4bit_use_double_quant=True` | Quantize the quantization scales too — saves 0.4 bits/param |
| `bnb_4bit_quant_type="nf4"` | NormalFloat-4 — better than uniform `fp4` |
| `device_map="auto"` | `accelerate` decides GPU placement; on 1 GPU = put it all there |
| `model.config.use_cache = False` | KV cache conflicts with gradient checkpointing |

### 9.2 `lora_config.py`

```python
"""Build PEFT LoRA config from YAML profile."""
from typing import Dict, Any

from src.utils import load_config


def build_lora_config(profile: str = "lora_r8", config_path: str = "config/model_config.yaml"):
    from peft import LoraConfig

    cfg = load_config(config_path)
    p = cfg["lora_configs"][profile]
    return LoraConfig(
        r=p["r"],
        lora_alpha=p["lora_alpha"],
        target_modules=p["target_modules"],
        lora_dropout=p["lora_dropout"],
        bias=p["bias"],
        task_type=p["task_type"],
    )
```

#### Why so simple?

YAML carries all the config knobs. This function just hydrates them into a PEFT-native object. Adding `lora_r16` profile = just edit YAML, no code change.

### 9.3 `callbacks.py`

```python
"""Custom Trainer callbacks for logging + sample generation."""
from transformers import TrainerCallback


class MemoryLoggingCallback(TrainerCallback):
    """Log GPU memory usage every N steps."""

    def __init__(self, every_n_steps: int = 30):
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            try:
                import torch
                if torch.cuda.is_available():
                    alloc = torch.cuda.memory_allocated() / 1024**3
                    reserv = torch.cuda.memory_reserved() / 1024**3
                    print(f"[step {state.global_step}] GPU mem: alloc={alloc:.2f} GiB, reserv={reserv:.2f} GiB")
            except ImportError:
                pass


class SampleGenerationCallback(TrainerCallback):
    """Generate sample outputs every N steps to monitor qualitative drift."""

    def __init__(self, tokenizer, prompts: list, every_n_steps: int = 60):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps != 0 or model is None:
            return
        for p in self.prompts:
            inputs = self.tokenizer(p, return_tensors="pt").to(model.device)
            with __import__("torch").no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            print(f"[step {state.global_step}] sample: {text[:200]}")
```

### 9.4 `trainer.py`

```python
"""Main training entrypoint."""
from datasets import load_dataset
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTConfig, SFTTrainer

from src.training.model_loader import load_quantized_model
from src.training.lora_config import build_lora_config
from src.training.callbacks import MemoryLoggingCallback, SampleGenerationCallback
from src.utils import load_config, project_root
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

log = get_logger(__name__)


def main(data_config_path: str = "config/data_config.yaml",
         model_config_path: str = "config/model_config.yaml",
         lora_profile: str = "lora_r8"):
    seed_everything(42)

    # 1. Load model + tokenizer
    bundle = load_quantized_model(model_config_path=model_config_path)
    model, tokenizer = bundle.model, bundle.tokenizer

    # 2. Prepare for k-bit training (cast norms, enable grad ckpt)
    model = prepare_model_for_kbit_training(model)

    # 3. Attach LoRA adapters
    lora_cfg = build_lora_config(profile=lora_profile, config_path=model_config_path)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    # Expected: ~3.4M trainable / 8B total = 0.04%

    # 4. Load datasets
    final_dir = project_root() / "data" / "final"
    train_ds = load_dataset("json", data_files=str(final_dir / "train.jsonl"), split="train")
    val_ds = load_dataset("json", data_files=str(final_dir / "val.jsonl"), split="train")

    # 5. Build training config
    model_cfg = load_config(model_config_path)
    tr_cfg = model_cfg["training"]

    args = SFTConfig(
        output_dir=tr_cfg["output_root"],
        num_train_epochs=tr_cfg["num_train_epochs"],
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        learning_rate=tr_cfg["learning_rate"],
        bf16=True, fp16=False,                       # ← critical for Llama-3.1
        logging_steps=tr_cfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=tr_cfg["eval_steps"],
        save_steps=tr_cfg["save_steps"],
        save_total_limit=tr_cfg["save_total_limit"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        lr_scheduler_type=tr_cfg["lr_scheduler_type"],
        optim=tr_cfg["optim"],
        gradient_checkpointing=True,
        packing=False,
        report_to="none",
    )

    # Workaround: max_seq_length API changed across trl versions
    try:
        args.max_seq_length = 512
    except Exception:
        pass
    tokenizer.model_max_length = 512

    # 6. Build trainer (handle API shift: tokenizer vs processing_class)
    trainer_kwargs = dict(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[
            MemoryLoggingCallback(every_n_steps=30),
            SampleGenerationCallback(
                tokenizer=tokenizer,
                prompts=["What is a Python list comprehension?"],
                every_n_steps=60,
            ),
        ],
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

    # 7. Train
    trainer.train()

    # 8. Save final adapter
    trainer.save_model(str(project_root() / "outputs" / "best"))
    log.info("Training complete. Adapter saved to outputs/best/")


if __name__ == "__main__":
    main()
```

#### Critical workarounds

| Workaround | Reason |
|---|---|
| `try/except` around `args.max_seq_length` | TRL ≥0.9 dropped this from `SFTConfig` |
| `try/except` around `processing_class` vs `tokenizer` | TRL changed API in 2024 |
| `bf16=True, fp16=False` | Llama-3.1 uses bf16 natively; fp16 grad scaler fails |
| `model.config.use_cache = False` (in loader) | Conflicts with gradient checkpointing |
| `prepare_model_for_kbit_training` before LoRA | Casts norms to fp32 for stability |

---

## 10. `src/evaluation/` walkthrough

### 10.1 `metrics.py` — Novel metrics

```python
"""TamilTech-QA evaluation metrics.

Standard: BLEU, ROUGE, BERTScore, exact-match, token-F1, perplexity, length ratio.
Novel:    CSPS, TTR, TCF.
"""
import math
from typing import List, Sequence

from src.preprocessing.tanglish_detector import TanglishDetector


def code_switch_preservation_score(
    predictions: Sequence[str],
    references: Sequence[str],
    detector: TanglishDetector,
) -> float:
    """CSPS — how close pred's Tanglish ratio is to ref's.
    
    CSPS_i = 1 - |tanglish_ratio(pred) - tanglish_ratio(ref)|
    """
    scores = []
    for p, r in zip(predictions, references):
        p_ratio = detector.score(p)["tanglish_ratio"]
        r_ratio = detector.score(r)["tanglish_ratio"]
        scores.append(1.0 - abs(p_ratio - r_ratio))
    return sum(scores) / max(len(scores), 1)


def technical_term_retention(
    predictions: Sequence[str],
    references: Sequence[str],
    technical_keywords: Sequence[str],
) -> float:
    """TTR — fraction of ref's tech terms preserved in pred."""
    kw_set = {k.lower() for k in technical_keywords}
    scores = []
    for pred, ref in zip(predictions, references):
        ref_terms = {t for t in ref.lower().split() if t in kw_set}
        if not ref_terms:
            continue
        pred_terms = {t for t in pred.lower().split() if t in kw_set}
        overlap = ref_terms & pred_terms
        scores.append(len(overlap) / len(ref_terms))
    return sum(scores) / max(len(scores), 1) if scores else 0.0


def tamil_connector_fluency(
    predictions: Sequence[str],
    references: Sequence[str],
    connectors: Sequence[str],
) -> float:
    """TCF — Tamil discourse connector usage alignment."""
    conn_set = [c.lower() for c in connectors]

    def find_connectors(text: str) -> set:
        t = text.lower()
        return {c for c in conn_set if c in t}

    scores = []
    for pred, ref in zip(predictions, references):
        ref_c = find_connectors(ref)
        pred_c = find_connectors(pred)
        if ref_c:
            scores.append(len(ref_c & pred_c) / len(ref_c))
        else:
            # Reference has no connectors — neutral baseline
            scores.append(1.0 if pred_c else 0.5)
    return sum(scores) / max(len(scores), 1)


def compute_perplexity(model, tokenizer, texts: List[str], device: str = "cuda") -> float:
    """Average perplexity across test set."""
    import torch
    model.eval()
    nlls = []
    with torch.no_grad():
        for text in texts:
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            out = model(**enc, labels=enc.input_ids)
            nll = out.loss.item() * enc.input_ids.size(1)
            nlls.append((nll, enc.input_ids.size(1)))
    total_nll = sum(n for n, _ in nlls)
    total_tokens = sum(t for _, t in nlls)
    avg_nll = total_nll / max(total_tokens, 1)
    return math.exp(avg_nll)
```

### 10.2 `ablation.py`

```python
"""Compare base model vs fine-tuned across all metrics."""
import json
from pathlib import Path

from src.evaluation.metrics import (
    code_switch_preservation_score,
    technical_term_retention,
    tamil_connector_fluency,
    compute_perplexity,
)
from src.preprocessing.tanglish_detector import TanglishDetector
from src.utils import load_config, project_root


def run_ablation(test_jsonl: str, base_model_id: str, adapter_path: str):
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config("config/data_config.yaml")
    detector = TanglishDetector()

    # Load test set
    test_data = [json.loads(line) for line in open(test_jsonl)]
    questions = [d["question"] for d in test_data]
    references = [d["answer"] for d in test_data]

    results = {"base": {}, "finetuned": {}}

    # Load base model (no adapter)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base = AutoModelForCausalLM.from_pretrained(base_model_id, load_in_4bit=True, device_map="auto")
    base_preds = generate_all(base, tokenizer, questions)

    # Compute base metrics
    results["base"]["csps"] = code_switch_preservation_score(base_preds, references, detector)
    results["base"]["ttr"] = technical_term_retention(base_preds, references, cfg["preprocessing"]["technical_keywords"])
    results["base"]["tcf"] = tamil_connector_fluency(base_preds, references, cfg["preprocessing"]["tamil_connectors"])
    results["base"]["ppl"] = compute_perplexity(base, tokenizer, references)

    # Load adapter
    finetuned = PeftModel.from_pretrained(base, adapter_path)
    ft_preds = generate_all(finetuned, tokenizer, questions)
    results["finetuned"]["csps"] = code_switch_preservation_score(ft_preds, references, detector)
    # ... same for ttr, tcf, ppl

    out = project_root() / "outputs" / "eval_report.json"
    out.write_text(json.dumps(results, indent=2))
```

---

## 11. Tests — `tests/` folder

### `tests/test_language_filter.py`

```python
import pytest
from src.preprocessing.tanglish_detector import TanglishDetector


def test_pure_english_rejected():
    det = TanglishDetector(min_ratio=0.15, max_ratio=0.85)
    assert not det.keep("This is a pure English sentence without any Tamil words at all.")


def test_pure_tamil_rejected():
    det = TanglishDetector(min_ratio=0.15, max_ratio=0.85)
    assert not det.keep("வணக்கம் நான் தமிழ் மட்டும்")  # 100% Tamil script


def test_natural_tanglish_kept():
    det = TanglishDetector(min_ratio=0.15, max_ratio=0.85)
    assert det.keep("Anna intha function epdi work pannuthu sollunga please")
```

### `tests/test_metrics.py`

```python
from src.evaluation.metrics import code_switch_preservation_score, tamil_connector_fluency
from src.preprocessing.tanglish_detector import TanglishDetector


def test_csps_perfect_match():
    det = TanglishDetector()
    preds = ["naan code-a paaren"]
    refs = ["naan code-a paaren"]
    score = code_switch_preservation_score(preds, refs, det)
    assert score == 1.0


def test_csps_mismatch_lower():
    det = TanglishDetector()
    preds = ["I see the code"]               # pure English
    refs = ["naan code-a paaren"]            # Tanglish
    score = code_switch_preservation_score(preds, refs, det)
    assert score < 0.5  # significant drift
```

### Running tests

```powershell
pytest tests/ -v
```

---

## 12. Scripts — `scripts/` folder

Bash scripts for orchestration. Even on Windows, can run via Git Bash or convert to `.ps1`.

### `scripts/run_scraping.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.data_collection.youtube_scraper --config config/data_config.yaml
python -m src.data_collection.synthetic_generator --config config/data_config.yaml
python -m src.data_collection.dataset_merger --config config/data_config.yaml
```

### `scripts/run_preprocessing.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.preprocessing.text_cleaner --input data/raw/merged.jsonl --output data/processed/cleaned.jsonl
python -m src.preprocessing.tanglish_detector --input data/processed/cleaned.jsonl --output data/processed/tanglish.jsonl
python -m src.preprocessing.language_filter --input data/processed/tanglish.jsonl --output data/processed/filtered.jsonl
python -m src.preprocessing.qa_formatter --input data/processed/filtered.jsonl --output-dir data/final
```

### `scripts/run_training.sh`

```bash
#!/usr/bin/env bash
set -e
python -m src.training.trainer \
    --data-config config/data_config.yaml \
    --model-config config/model_config.yaml \
    --lora-profile lora_r8
```

---

## 13. Local dry-run end-to-end

Without GPU, you can verify the **pipeline plumbing** works:

```powershell
# 1. Scrape (uses real APIs)
.\.venv\Scripts\Activate.ps1
python -m src.data_collection.youtube_scraper

# 2. Preprocess (CPU-only — fast)
python -m src.preprocessing.tanglish_detector --input data/raw/merged.jsonl --output data/processed/tanglish.jsonl

# 3. Format
python -m src.preprocessing.qa_formatter --input data/processed/tanglish.jsonl --output-dir data/final

# 4. Sanity check sizes
Get-ChildItem data\final\
# Expected: train.jsonl, val.jsonl, test.jsonl

# 5. Inspect a sample
Get-Content data\final\train.jsonl -TotalCount 1 | ConvertFrom-Json | Format-List

# 6. Load model on CPU (smoke test)
python -c "from src.training.model_loader import load_quantized_model; m = load_quantized_model(use_4bit=False); print(type(m.model))"
```

⚠️ Step 6 will be **very slow** on CPU (10+ min for 8B model). Don't actually train locally — only verify code paths.

---

## 14. Common pitfalls (and fixes)

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'YOUTUBE_API_KEY'` | `.env` not loaded OR key missing | Verify `.env` in project root; call `load_dotenv()` before reading |
| `ImportError: cannot import name 'load_in_4bit'` | Old `transformers` | `pip install -U transformers` |
| `TypeError: unsupported operand for /: 'PosixPath' and 'str'` | Path joining issue | Use `path / "subdir"` (forward slash) |
| `JSON decode error` (synthetic gen) | gpt-4o-mini malformed | Add `response_format={"type": "json_object"}` |
| `OSError: Couldn't reach hf.co` | Network / HF down | Set `HF_HUB_OFFLINE=1` if cached, else retry |
| `Tokenizer pad_token is None` | Llama-3.1 no native pad | `tokenizer.pad_token = tokenizer.eos_token` |
| `Tanglish ratio always 0.0` | WORD_RE not matching | Verify Unicode block `஀-௿` |
| `RuntimeError: PassThroughHasher` (MinHash) | datasketch issue with strings | Use exact dedup by `(question, answer)` tuple |
| `git tracking .env` | Forgot `.gitignore` first | `git rm --cached .env` + commit |
| `Permission denied: .venv\Scripts\activate` | PowerShell exec policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## ✅ End of Doc 3

**Next up:** `04_kaggle_training_walkthrough.md` — Kaggle session, cell-by-cell, errors we hit, lessons learned.

**Padichu paaru:**
- Function explanations sufficient-a, or need more depth?
- Pitfalls section ungalukku useful-a?
- Tanglish tone OK or want adjustment?

Confirm pannina apparam Doc 4 start panren. 🚀
