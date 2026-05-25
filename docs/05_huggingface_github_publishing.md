# 📤 Document 5 — HuggingFace + GitHub Publishing (Tanglish Edition)

> **Intha doc-ku purpose:** Project ready aana apparam, **public-aa publish pannardhu epdi**? HuggingFace Hub (Dataset + Model + Space), GitHub repo creation, READMEs, metadata, social setup — ellame step-by-step + gotchas.
>
> **Reading time:** ~90 min.
> **Prereq:** Docs 1-4. HF + GitHub accounts ready.

---

## Table of Contents

1. [Publishing strategy — Three platforms, one project](#1-publishing-strategy--three-platforms-one-project)
2. [HuggingFace account + token setup](#2-huggingface-account--token-setup)
3. [HF Dataset push workflow](#3-hf-dataset-push-workflow)
4. [HF Model push workflow](#4-hf-model-push-workflow)
5. [HF Space — Gradio demo deployment](#5-hf-space--gradio-demo-deployment)
6. [The README override gotcha](#6-the-readme-override-gotcha)
7. [Model + Dataset cards — Anatomy](#7-model--dataset-cards--anatomy)
8. [GitHub repo creation](#8-github-repo-creation)
9. [`.gitignore` strategy + secret safety](#9-gitignore-strategy--secret-safety)
10. [git push flow — Initial commit](#10-git-push-flow--initial-commit)
11. [GitHub repo polish — Topics, pin, homepage](#11-github-repo-polish--topics-pin-homepage)
12. [Co-author attribution etiquette](#12-co-author-attribution-etiquette)
13. [Cross-linking the three platforms](#13-cross-linking-the-three-platforms)
14. [Common publishing errors + fixes](#14-common-publishing-errors--fixes)

---

## 1. Publishing strategy — Three platforms, one project

### 🎭 The three personas

For a research artifact, you publish to **three places**, each serving a different audience:

| Platform | What lives there | Audience |
|---|---|---|
| **HuggingFace Hub** | Dataset + Model + Space | ML researchers, practitioners |
| **GitHub** | Source code + notebooks + docs | Engineers reviewing your work |
| **HF Space (Gradio)** | Live demo / showcase | Recruiters, hiring managers, casual visitors |

### 🪜 Publishing order

```
1. Local code stable + tested
       ↓
2. Push DATASET to HF Hub first
       ↓ (so model README can link to dataset)
3. Push MODEL adapter to HF Hub
       ↓ (so Space README can link to both)
4. Build SPACE (Gradio app pointing to model)
       ↓
5. Push CODE to GitHub
       ↓ (so all README cards can cross-reference)
6. Update each platform's README with cross-links
       ↓
7. Pin GitHub repo, add topics, set homepage
```

### 🥥 Wedding analogy

Publishing = **kalyana arrangements**:
- **Dataset** = venue (everything happens here, must be ready first)
- **Model** = groom/bride (the headline)
- **Space** = invitation card (catches attention, shows the highlight)
- **GitHub repo** = guest list + program book (formal records)
- **Cross-links** = directions on the invite

Mistake order-le = chaos. Right order-le = smooth.

---

## 2. HuggingFace account + token setup

### Step 2.1: Create account

https://huggingface.co/join — free, email confirm.

**Important:** Choose your **username carefully** — appears in every URL forever:
- `huggingface.co/<username>/repo` (models)
- `huggingface.co/datasets/<username>/repo` (datasets)
- `huggingface.co/spaces/<username>/repo` (spaces)

For TamilTech-QA: `dheepakkaran` username chosen.

### Step 2.2: Accept Llama-3.1 license

Llama-3.1 is a **gated model** — needs license acceptance.

1. Go to https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
2. Click "Agree and access repository"
3. Fill Meta's form (name, org, country, intended use)
4. Wait 5-10 minutes for approval (usually instant)

**Sign of approval:** Page shows "You have been granted access". If pending → wait.

### Step 2.3: Generate access token

https://huggingface.co/settings/tokens

**Token types:**

| Type | Scope | Use case |
|---|---|---|
| **Read** | Pull models/datasets | Inference only |
| **Write** | Push + manage repos | Required for upload ✅ |
| **Fine-grained** | Per-repo permissions | Production safer; v0.1 used Write |

**Settings for write token:**
- Name: `tamiltech-qa-write` (descriptive!)
- Type: **Write**
- Click "Generate token"
- **Copy immediately** — only shown once
- Paste into `.env` as `HF_TOKEN=hf_...`

### Step 2.4: Login locally

```bash
huggingface-cli login
# Paste token when prompted
```

Or programmatically:

```python
from huggingface_hub import login
import os
login(token=os.environ["HF_TOKEN"])
```

⚠️ **Token rotation:** If you accidentally share a token (chat, screenshot, commit), **revoke it immediately** at https://huggingface.co/settings/tokens. Generate a new one.

---

## 3. HF Dataset push workflow

### 🎯 What we're publishing

`train.jsonl`, `val.jsonl`, `test.jsonl` — 4,415 Tanglish QA pairs.

Final URL: `https://huggingface.co/datasets/dheepakkaran/TamilTech-QA`

### Step 3.1: Programmatic push (recommended)

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
DATASET_ID = "dheepakkaran/TamilTech-QA"

# 1. Create dataset repo
create_repo(
    repo_id=DATASET_ID,
    repo_type="dataset",
    private=False,
    exist_ok=True,
)
print(f"Repo created/exists: https://huggingface.co/datasets/{DATASET_ID}")

# 2. Upload jsonl files
for file_name in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    api.upload_file(
        path_or_fileobj=f"data/final/{file_name}",
        path_in_repo=file_name,
        repo_id=DATASET_ID,
        repo_type="dataset",
        commit_message=f"Add {file_name}",
    )
    print(f"Uploaded {file_name}")
```

### Step 3.2: Alternative — `datasets` library

```python
from datasets import load_dataset, DatasetDict

ds = DatasetDict({
    "train": load_dataset("json", data_files="data/final/train.jsonl", split="train"),
    "validation": load_dataset("json", data_files="data/final/val.jsonl", split="train"),
    "test": load_dataset("json", data_files="data/final/test.jsonl", split="train"),
})

ds.push_to_hub("dheepakkaran/TamilTech-QA")
```

**Pro of this approach:** HF auto-generates a `dataset_infos.json` with feature schema + sample counts.
**Con:** Will overwrite README with default boilerplate (gotcha — see Section 6).

### Step 3.3: Upload README separately

```python
api.upload_file(
    path_or_fileobj="dataset_card.md",
    path_in_repo="README.md",
    repo_id=DATASET_ID,
    repo_type="dataset",
    commit_message="Add custom dataset card",
)
```

### Step 3.4: Verify

Open `https://huggingface.co/datasets/dheepakkaran/TamilTech-QA`:
- ✅ Files & versions tab shows train/val/test.jsonl
- ✅ Dataset Card (README) renders correctly
- ✅ Preview tab shows first rows
- ✅ "Use in dataset library" code snippet works

### Quick test of published dataset

```python
from datasets import load_dataset
ds = load_dataset("dheepakkaran/TamilTech-QA")
print(ds)
print(ds["train"][0])
```

---

## 4. HF Model push workflow

### 🎯 What we're publishing

LoRA adapter (~50 MB) — **not** the full Llama base. Users load Llama themselves and apply adapter.

Final URL: `https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA`

### Step 4.1: Repo naming

**Convention:** `<base-model-shortname>-<task>-<method>` or `<dataset>-<base>-<method>`.

Choices we considered:
- ❌ `llama-tanglish` (too generic)
- ❌ `dheepakkaran-llama-finetune` (no info)
- ✅ `TamilTech-QA-Llama3.1-8B-QLoRA` — **dataset** + **base** + **method** all visible

### Step 4.2: Push adapter

```python
from huggingface_hub import HfApi, create_repo

api = HfApi()
MODEL_ID = "dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA"

create_repo(repo_id=MODEL_ID, repo_type="model", private=False, exist_ok=True)

api.upload_folder(
    folder_path="outputs/best",        # contains adapter_model.safetensors + adapter_config.json
    repo_id=MODEL_ID,
    repo_type="model",
    commit_message="Upload TamilTech-QA QLoRA adapter v0.1",
)
print(f"Model uploaded: https://huggingface.co/{MODEL_ID}")
```

### Step 4.3: What ends up there

```
outputs/best/
├── adapter_model.safetensors      # the learned LoRA weights
├── adapter_config.json            # rank, alpha, target_modules
├── tokenizer.json                 # tokenizer (optional — base tokenizer works)
├── tokenizer_config.json
└── special_tokens_map.json
```

**Important:** Don't upload the **full Llama base model** — license forbids redistribution. Adapter only.

### Step 4.4: Upload custom model card

```python
api.upload_file(
    path_or_fileobj="model_card.md",
    path_in_repo="README.md",
    repo_id=MODEL_ID,
    repo_type="model",
    commit_message="Add comprehensive model card",
)
```

### Step 4.5: Verify usage works

In a new Python session:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "meta-llama/Llama-3.1-8B-Instruct"
adapter_id = "dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA"

tokenizer = AutoTokenizer.from_pretrained(base_id)
base = AutoModelForCausalLM.from_pretrained(
    base_id, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, adapter_id)
print("Loaded adapter from HF Hub successfully")
```

---

## 5. HF Space — Gradio demo deployment

### 🎬 What is a Space?

A **HuggingFace Space** = a free hosted web app, typically Gradio or Streamlit. Lives at `huggingface.co/spaces/<user>/<name>`.

For TamilTech-QA: a **comparison demo** showing base vs fine-tuned outputs side-by-side.

Final URL: `https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo`

### 5.1: Hardware tiers (free)

| Tier | Hardware | Cost | OK for our 8B model? |
|---|---|---|---|
| Free CPU | 16 GB RAM | $0 | ❌ OOMs loading Llama-3.1-8B |
| ZeroGPU | T4 (shared, lazy) | $9/mo HF PRO | ✅ Would work |
| A10G Small | dedicated | $0.60/hr | ✅ But expensive |
| **Static + Colab live** ✅ | CPU only for landing | Free | **Compromise: showcase + Colab notebook** |

We picked **the compromise**: Free CPU Space showing static demo + Colab notebook for live inference.

### 5.2: Create a Space

1. https://huggingface.co/new-space
2. Name: `TamilTech-QA-Demo`
3. SDK: **Gradio**
4. Hardware: **CPU basic (free)**
5. Visibility: Public

Auto-creates a git repo at `https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo`.

### 5.3: Clone the Space repo

```bash
git clone https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo
cd TamilTech-QA-Demo
```

### 5.4: Write `app.py`

```python
"""TamilTech-QA Demo — static showcase + interactive Tanglish analyzer."""
import gradio as gr
import re

# Pre-cached qualitative comparisons (computed offline)
SAMPLES = [
    {
        "question": "React hooks-a explain pannunga",
        "base_output": "ரியாக்ட் ஹூக்ஸ் என்பது...",  # Tamil-script Tamil
        "ft_output": "React hooks na functional component-le state manage panrathuku oru way...",
    },
    # ... 5 more samples
]


def analyze_tanglish(text):
    """Compute live Tanglish ratio for user input."""
    from collections import Counter
    tamil_words = {"naan", "neenga", "indha", "antha", "iruku", "panrathu", "sollu",
                   "athu", "ithu", "ippo", "apdi", "ena", "epdi"}
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return "No words detected.", 0
    tamil_count = sum(1 for t in tokens if t in tamil_words)
    ratio = tamil_count / len(tokens)
    label = "Pure English" if ratio < 0.10 else \
            "Natural Tanglish ✓" if ratio < 0.85 else "Pure Tamil-Romanized"
    return label, round(ratio * 100, 1)


with gr.Blocks(title="TamilTech-QA Demo") as demo:
    gr.Markdown("""
    # 🪔 TamilTech-QA — Tanglish Technical QA
    First open Tanglish (Tamil-English code-switched) technical QA dataset and fine-tuned Llama-3.1-8B.

    🤖 [Model](https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA)
    📊 [Dataset](https://huggingface.co/datasets/dheepakkaran/TamilTech-QA)
    💻 [GitHub](https://github.com/dheepakkaran/TamilTech-QA)
    🎮 [Live Colab Demo](https://colab.research.google.com/...) (free GPU)
    """)

    with gr.Tab("📊 Comparison Showcase"):
        for sample in SAMPLES:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"**Q:** {sample['question']}")
                    gr.Markdown(f"**Base Llama-3.1:** {sample['base_output']}")
                    gr.Markdown(f"**Fine-tuned (Ours):** {sample['ft_output']}")
            gr.Markdown("---")

    with gr.Tab("🔍 Tanglish Analyzer"):
        gr.Markdown("Type any text — see how Tanglish it is.")
        inp = gr.Textbox(lines=3, placeholder="Anna intha React hooks-a explain pannunga")
        out_label = gr.Textbox(label="Classification")
        out_ratio = gr.Number(label="Tanglish %")
        btn = gr.Button("Analyze")
        btn.click(analyze_tanglish, inputs=inp, outputs=[out_label, out_ratio])


demo.launch()
```

### 5.5: Write `requirements.txt`

```
gradio>=4.30.0
```

(Light deps because we're not loading the model on CPU Space.)

### 5.6: Push to Space

```bash
git add app.py requirements.txt README.md
git commit -m "Initial Gradio demo"
git push origin main
```

HF auto-builds the Space (~2-3 min) → app live.

### 5.7: Space README (front-matter)

```markdown
---
title: TamilTech-QA Demo
emoji: 🪔
colorFrom: orange
colorTo: red
sdk: gradio
sdk_version: 4.30.0
app_file: app.py
pinned: false
license: mit
---

# TamilTech-QA Demo
[full description below]
```

The YAML frontmatter is parsed by HF — controls thumbnail color, SDK version, app file.

---

## 6. The README override gotcha

### 😡 The bug

When you call:

```python
model.push_to_hub("user/repo")    # OR
trainer.push_to_hub()
```

**HuggingFace silently auto-generates a default README and uploads it**, overwriting any custom README you uploaded earlier.

This bit us **twice** during v0.1 publishing.

### 🩹 Three fixes

#### Fix A: Use `upload_folder` (doesn't generate README)

```python
api.upload_folder(folder_path="outputs/best", repo_id=MODEL_ID, repo_type="model")
```

`upload_folder` is a passive sync — uploads exactly what's there, no auto-generation.

#### Fix B: Upload README LAST

```python
# 1. Push model (which generates default README)
trainer.push_to_hub()

# 2. Then overwrite with custom README
api.upload_file(
    path_or_fileobj="model_card.md",
    path_in_repo="README.md",
    repo_id=MODEL_ID,
    repo_type="model",
    commit_message="Restore custom model card",
)
```

#### Fix C: Edit on website

If you forgot at push time:
1. Open `https://huggingface.co/your-repo`
2. Click "Files and versions" → README.md → ✏️ Edit
3. Paste your custom card → Commit

We used Fix C for the initial v0.1 push, then automated with Fix A in subsequent runs.

---

## 7. Model + Dataset cards — Anatomy

### 📋 Dataset card structure

````markdown
---
license: mit
task_categories:
  - text-generation
  - question-answering
language:
  - ta
  - en
tags:
  - tamil
  - tanglish
  - code-switching
  - technical-qa
size_categories:
  - 1K<n<10K
---

# TamilTech-QA Dataset

## Summary
[1-2 paragraph elevator pitch]

## Statistics
| Split | Count |
|---|---|
| Train | 3,536 |
| Validation | 431 |
| Test | 447 |
| **Total** | **4,415** |

## Data Format
```json
{
  "id": "...",
  "question": "Anna React hooks-a explain pannunga",
  "answer": "Hook na functional component-le state manage...",
  "source": "youtube",
  "tanglish_ratio": 0.42
}
```

## Collection Methodology
[YouTube + synthetic + filtering details]

## Citation
```bibtex
@misc{tamiltech-qa-2026,
  ...
}
```

## License
MIT
````

### 📋 Model card structure

````markdown
---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
tags:
  - qlora
  - tanglish
  - tamil
  - code-switching
language:
  - ta
  - en
datasets:
  - dheepakkaran/TamilTech-QA
---

# TamilTech-QA Llama-3.1-8B QLoRA

QLoRA fine-tuned Llama-3.1-8B for **Tanglish (Tamil-English code-switched) technical QA**.

## Headline Result
**Perplexity reduction: 78%** (57.05 → 12.40) on Tanglish test set.

## How to Use
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

base = "meta-llama/Llama-3.1-8B-Instruct"
adapter = "dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base, load_in_4bit=True, device_map="auto", torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(model, adapter)

inputs = tokenizer(
    "What is a Python decorator?", return_tensors="pt"
).to(model.device)
out = model.generate(**inputs, max_new_tokens=120)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

## Training Details
- Hardware: Free Kaggle T4 (16 GB VRAM)
- Time: ~6 hours / 1 epoch
- Effective batch: 16 (batch=2, grad_accum=8)
- LoRA: r=8, alpha=16, on q/k/v/o projections

## Metrics
| Metric | Base | Fine-tuned | Δ |
|---|---|---|---|
| Perplexity ↓ | 57.05 | 12.40 | -78% |
| CSPS ↑ | 0.62 | 0.66 | +3.7% |
| TTR ↑ | 0.81 | 0.89 | +8% |

## Limitations
- 1 epoch only — under-converged on some patterns
- Roman-script Tanglish only (no Tamil-script outputs)
- 14× length ratio (verbose vs reference)

## Citation
```bibtex
@misc{...}
```
````

---

## 8. GitHub repo creation

### Step 8.1: Create empty repo

https://github.com/new

**Settings:**
- Owner: your username
- Repository name: `TamilTech-QA` (match HF for consistency)
- Description: *[research-grade single sentence]*

> "TamilTech-QA: A Tamil-English Code-Switched Benchmark for Technical Question Answering with QLoRA Fine-Tuning and Novel Code-Switching Evaluation Metrics (CSPS / TTR / TCF). Published on HuggingFace Hub."

- Public ✓
- **Do NOT init with README/LICENSE/.gitignore** (we already have ours)

Click **Create repository**.

### Step 8.2: What to put in the repo

```
TamilTech-QA/
├── .gitignore                 # MUST commit first
├── LICENSE                    # MIT
├── README.md                  # extensive — see Doc 1 §13
├── requirements.txt
├── setup.py
├── config/
│   ├── data_config.yaml
│   ├── model_config.yaml
│   └── eval_config.yaml
├── src/                       # full pipeline code
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── train_on_kaggle_v2.ipynb     # the working notebook
│   └── eval_recovery.ipynb
├── tests/
│   ├── test_language_filter.py
│   ├── test_metrics.py
│   └── test_qa_formatter.py
├── scripts/                   # bash orchestration
└── live_demo.ipynb            # Colab demo
```

### What NOT to put

- `.env` ❌ (secrets)
- `data/` ❌ (regenerable + huge)
- `outputs/` ❌ (training artifacts + huge)
- `.venv/` ❌ (huge, machine-specific)
- `.claude/memory/` ❌ (personal context, not for sharing)

---

## 9. `.gitignore` strategy + secret safety

### 🛡️ The full `.gitignore`

```gitignore
# Secrets (NEVER commit these)
.env
.env.*
*.key
*.token
secrets.yaml
credentials.json

# Generated data (regenerated by pipeline)
data/raw/
data/processed/
data/final/
outputs/

# Browser automation / debug artifacts
.playwright-mcp/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
build/
dist/
.pytest_cache/
.coverage
.tox/

# Virtual environments
.venv/
venv/
env/

# IDE / editor
.vscode/
.idea/
.DS_Store
*.swp

# Claude Code memory (personal context)
.claude/memory/

# Temporary / debug files
_paste_*.txt
_prep_*.py
*.zip
*.tar.gz

# Jupyter checkpoints
.ipynb_checkpoints/

# Logs
*.log
logs/

# OS
Thumbs.db
desktop.ini
```

### 🚨 Secret-leak triage (if it happens)

If you accidentally commit a token/key:

1. **Revoke immediately** at the provider (HF/OpenAI/etc.)
2. **Generate a fresh token**
3. **Don't just delete the line** — git history still has it. You must either:
   - Force-push a rewritten history (`git filter-repo` or BFG Repo-Cleaner), OR
   - Accept the token is permanently public; revocation is the real protection
4. **Re-rotate downstream** anywhere the old token was used

**Best prevention:** Pre-commit hook scanning for `sk-`, `hf_`, `AIzaSy`, etc.

```bash
# .git/hooks/pre-commit
#!/bin/bash
if git diff --cached | grep -E "(sk-proj-|hf_[a-zA-Z]{30}|AIzaSy)"; then
    echo "❌ Potential secret detected. Aborting commit."
    exit 1
fi
```

---

## 10. git push flow — Initial commit

### Step 10.1: Initialize git locally

```bash
cd C:\Users\LENOVO\Downloads\tamiltech-qa\TamilTech-QA
git init -b main             # init with default branch "main"
```

### Step 10.2: Verify no secrets staged

```bash
git status
git add .gitignore           # commit .gitignore FIRST
git commit -m "Add .gitignore"

git add .
git status --short           # should list ~46 files, NO .env or data/
```

### Step 10.3: First commit

```bash
git commit -m "Initial commit: TamilTech-QA v0.1

First publicly released Tanglish (Tamil-English code-switched) technical
QA dataset and QLoRA fine-tuned Llama-3.1-8B model, with three novel
code-switching evaluation metrics (CSPS, TTR, TCF).

Headline result: 78% perplexity reduction on Tanglish test set.

Published artifacts:
- Dataset: https://huggingface.co/datasets/dheepakkaran/TamilTech-QA
- Model:   https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA
- Space:   https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo"
```

### Step 10.4: Add remote + push

```bash
git remote add origin https://github.com/dheepakkaran/TamilTech-QA.git
git push -u origin main
```

**First push prompts for auth.** Two options:

#### Option A: Personal Access Token (HTTPS)

GitHub → Settings → Developer settings → Personal access tokens → generate → use as password.

Password prompt → paste token (NOT GitHub password — passwords for git are deprecated).

#### Option B: SSH key

```bash
ssh-keygen -t ed25519 -C "your@email.com"
cat ~/.ssh/id_ed25519.pub      # copy this
# Paste at: GitHub → Settings → SSH and GPG keys → New SSH key

# Change remote to SSH
git remote set-url origin git@github.com:dheepakkaran/TamilTech-QA.git
git push -u origin main
```

### Step 10.5: Verify

Open `https://github.com/dheepakkaran/TamilTech-QA`:
- ✅ All files visible
- ✅ README renders
- ✅ No `.env`, `data/`, `outputs/`
- ✅ Contributors: only you (see Section 12 for Claude attribution)

---

## 11. GitHub repo polish — Topics, pin, homepage

### 11.1: Add topics (for discoverability)

Click ⚙️ next to "About" → fill **Topics** field:

```
nlp, llm, qlora, tamil, tanglish, code-switching, llama, huggingface, fine-tuning, dataset
```

**Why topics matter:** GitHub search by topic returns trending repos. Topic = community.

### 11.2: Set Website (Homepage) link

Same panel — paste the **most representative HF link**:

For a research project: HF Model page wins (combines dataset + adapter + usage code).

```
https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA
```

### 11.3: Pin the repo to profile

Your GitHub profile → "Customize your pins" → tick `TamilTech-QA` → Save.

**Why:** Recruiters visiting your profile see pinned repos first. TamilTech-QA = your headline portfolio piece.

### 11.4: Update About section text

The description box (same gear icon):

```
Tanglish (Tamil-English code-switched) technical QA dataset + QLoRA fine-tuned Llama-3.1-8B with novel code-switching evaluation metrics (CSPS / TTR / TCF). v0.1 released 2026-05-25.
```

Keep under ~250 chars (GitHub UI cuts off).

### 11.5: Programmatic version (via API)

If you want to script the polish steps:

```bash
gh repo edit dheepakkaran/TamilTech-QA \
    --description "TamilTech-QA: ..." \
    --homepage "https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA" \
    --add-topic nlp \
    --add-topic llm \
    --add-topic qlora \
    --add-topic tamil \
    --add-topic tanglish \
    --add-topic code-switching \
    --add-topic llama \
    --add-topic huggingface
```

(Requires `gh` CLI installed + authenticated.)

---

## 12. Co-author attribution etiquette

### 🤔 The `Co-Authored-By` trailer

Git supports adding co-authors to commits:

```
Initial commit: TamilTech-QA v0.1

Description here...

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
```

GitHub **parses this trailer** and shows the co-author as a contributor on the repo's Contributors graph.

### When to keep it

- ✅ Pair programming with a real human teammate
- ✅ AI-assistance is acknowledged elsewhere AND the project is internal
- ❌ Public research/portfolio repo where you want **sole authorship attribution**

### How to remove it (after the fact)

```bash
# Amend the last commit
git commit --amend
# Editor opens — delete the Co-Authored-By: line — save

# Force push (with safety check)
git push --force-with-lease origin main
```

⚠️ `--force-with-lease` is safer than `--force` — it refuses to push if someone else committed since your last fetch.

### Permanent prevention

For future commits — when AI helps:
1. Use the AI's suggestions, but write commit messages yourself
2. Don't paste auto-generated `Co-Authored-By` lines

For Dheepak's repos: **no `Co-Authored-By: Claude` on public repos** is the agreed convention.

---

## 13. Cross-linking the three platforms

### 🔗 The linking matrix

| From | Link to | Reason |
|---|---|---|
| GitHub README | HF Dataset, Model, Space | Discoverability |
| HF Model card | GitHub, HF Dataset | Code provenance |
| HF Dataset card | GitHub, HF Model | Usage examples |
| HF Space README | All three | Footer credits |
| GitHub "About" / Website | HF Model | Single primary link |

### 📝 Standard footer (paste in every README)

```markdown
## 🔗 Project Links
- 💻 **Code (GitHub):** https://github.com/dheepakkaran/TamilTech-QA
- 📊 **Dataset:** https://huggingface.co/datasets/dheepakkaran/TamilTech-QA
- 🤖 **Model:** https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA
- 🎮 **Live Demo:** https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo
- 📓 **Colab Demo:** [open in Colab](https://colab.research.google.com/...)

## 👤 Author
**Dheepak Karan E S** — graduate student, Northeastern University.
- [GitHub](https://github.com/dheepakkaran)
- [HuggingFace](https://huggingface.co/dheepakkaran)
- [LinkedIn](https://linkedin.com/in/...)
```

---

## 14. Common publishing errors + fixes

### 🐛 HF push errors

| Symptom | Cause | Fix |
|---|---|---|
| `401 Unauthorized` | Token expired or wrong scope | Regenerate write-scope token |
| `403 Forbidden: gated repo` | Llama-3.1 license not accepted | Accept on HF model page |
| `Custom README missing after push` | `push_to_hub` overwrites | Upload README separately AFTER |
| `Filename contains backslash` | PowerShell zip | Use Python `zipfile` |
| `Disk quota exceeded` | Free tier 50 GB | Delete old commits or use LFS |
| Empty repo after push | Pushed but no commit made | Check `git log` — files must be committed before push |

### 🐛 GitHub errors

| Symptom | Cause | Fix |
|---|---|---|
| `support for password auth removed` | Using password instead of PAT | Generate Personal Access Token |
| `non-fast-forward` rejected | Remote has commits you don't | `git pull --rebase` then push |
| Files >100 MB rejected | Git LFS not used | Use HF Hub for large files, not Git |
| `.env` already committed | Forgot `.gitignore` first | `git rm --cached .env` + re-commit + rotate token |
| Contributor list shows AI | `Co-Authored-By` trailer | `git commit --amend` + force-push |

### 🐛 Space errors

| Symptom | Cause | Fix |
|---|---|---|
| Build fails: "module not found" | Missing in `requirements.txt` | Add the dep |
| Space stuck "Building" | Memory OOM during install | Pin smaller deps; remove heavy ones |
| Space loads Llama but OOMs | Free CPU = 16 GB, model = 16+ GB | Use static demo + Colab live (compromise) |
| `app.py` not detected | Wrong filename | Must be exactly `app.py` (or set `app_file:` in frontmatter) |

---

## ✅ End of Doc 5

**Next up:** `06_interview_100_questions.md` — **the big one**. 100 senior technical interviewer questions on TamilTech-QA, with detailed Tanglish answers + "interviewer ena expect pandraru" hints. Estimated ~60-80 pages.

**Padichu paaru:**
- HF + GitHub workflows clear-a?
- README anatomy useful-a (for replicating on future projects)?
- Co-author section relevant-a?
- Anything to add before the big interview prep doc?

Confirm pannina apparam **Doc 6** start panren (the final doc — and the most important for your job hunt). 🚀
