# 🚂 Document 4 — Kaggle Training Walkthrough (Tanglish Edition)

> **Intha doc-ku purpose:** Kaggle phase-le ena nadanthu, **cell-by-cell** notebook-a explain pannardhu, namba **ena thappu pani kathukutom**-na, athula irundhu **ena lessons** learn pannom — full timeline.
>
> **Reading time:** ~2 hours.
> **Prereq:** Docs 1, 2, 3 padichirukanum. Especially Doc 3's training section.

---

## Table of Contents

1. [Why Kaggle? Platform choice rationale](#1-why-kaggle-platform-choice-rationale)
2. [Kaggle environment basics](#2-kaggle-environment-basics)
3. [v1 vs v2 — Why we rewrote the notebook](#3-v1-vs-v2--why-we-rewrote-the-notebook)
4. [Pre-flight checklist — Before starting](#4-pre-flight-checklist--before-starting)
5. [Notebook v2 — Cell-by-cell walkthrough](#5-notebook-v2--cell-by-cell-walkthrough)
6. [The error log — Every bug we hit](#6-the-error-log--every-bug-we-hit)
7. [Session timeout recovery saga](#7-session-timeout-recovery-saga)
8. [Memory optimization — How we squeezed into 16 GB](#8-memory-optimization--how-we-squeezed-into-16-gb)
9. [Saving + uploading from Kaggle](#9-saving--uploading-from-kaggle)
10. [Lessons learned — Production wisdom](#10-lessons-learned--production-wisdom)
11. [v2 → v3 roadmap (what we'd change)](#11-v2--v3-roadmap-what-wed-change)

---

## 1. Why Kaggle? Platform choice rationale

### 🏟️ The contenders

| Platform | GPU | VRAM | Session limit | Cost | Verdict |
|---|---|---|---|---|---|
| **Kaggle Free** ✅ | T4 (dedicated) | 16 GB | 12 hr | Free, 30 hr/wk | **WINNER** |
| Colab Free | T4 (shared) | 16 GB | 3-4 hr (interruptible) | Free | Too unstable |
| Colab Pro | T4/A100 | 16-40 GB | Longer | $10/mo | Costs $$$ |
| Paperspace Free | M4000 | 8 GB | 6 hr | Free | Won't fit Llama-3.1-8B |
| AWS SageMaker Studio Lab | T4 | 16 GB | 4 hr | Free | Limited storage |
| RunPod / Lambda | A100 / H100 | 40-80 GB | Unlimited | $1-2/hr | Costs $$$ |

### Why Kaggle won

1. **12-hour sessions** — Long enough for 1-epoch QLoRA on ~4,400 samples
2. **30 hours/week budget** — Multiple training runs + eval + experimentation
3. **Dedicated T4** (not shared) — Reliable, no random kicks
4. **Persistent storage** via "Save Version" — recovery from timeouts
5. **Free pre-installed** — `torch`, `transformers`, basic libs ready
6. **GPU + CPU + storage all free** — No surprises

### 🥥 Tamil-style analogy

Kaggle = **government library la free reading room** — limited hours per week, but free, has chairs, has electricity, has wifi.

Colab = **friend-oda house** — comfortable but interruptions varum.

RunPod/AWS = **paid coworking space** — best facilities, but daily bill.

Kaggle ku gondhu varathu epdi? **Time budget management.**

### 💰 Cost comparison for this project

| Resource | Kaggle approach | Alternative |
|---|---|---|
| Training (1 epoch, ~6 hrs) | **$0** | Lambda A100: ~$10 |
| Storage of training artifacts | **$0** (Save Version) | S3: $0.50/mo |
| Recovery on crash | **$0** (reload Saved Version) | RunPod re-launch fees |
| **TamilTech-QA v0.1 total compute cost** | **$0** ✅ | $50-100 anywhere else |

---

## 2. Kaggle environment basics

### 🗂️ Filesystem layout

```
/kaggle/
├── input/                       # READ-ONLY — uploaded datasets
│   ├── tamiltech-qa-repo/       # our source code zip-uploaded
│   ├── tamiltech-qa-data/       # preprocessed jsonl files
│   └── saved-version-N/         # previous run's outputs (after timeout)
│
├── working/                     # READ-WRITE — your scratch space
│   ├── outputs/                 # training checkpoints
│   └── ...                      # ⚠️ WIPED on session end unless "Save Version"
│
└── tmp/                         # ephemeral
```

**Critical mental model:**

- `/kaggle/input/` = **READ-ONLY**. Once a dataset is attached, can't modify.
- `/kaggle/working/` = **READ-WRITE**. Your working space. Persists during session.
- **Session ends** → `/kaggle/working/` wiped UNLESS you clicked "Save Version" first.

### 🔧 Session lifecycle

```
[Click "Edit"]
   ↓
[Notebook opens, session starts]
   ↓
[Run cells, do work, save to /kaggle/working/]
   ↓
[Either: stay active 12 hours]
   OR
[Click "Save Version" — snapshots /kaggle/working/ to a permanent "version"]
   ↓
[Session ends — /kaggle/working/ wiped]
   ↓
[Later: open notebook, see "Output" tab — previous Save Version's files accessible]
   OR
[Attach Saved Version's output as input to a NEW notebook]
```

### 🔑 Secrets in Kaggle

Add-ons → Secrets:

```
HF_TOKEN     = hf_...
OPENAI_API_KEY = sk-proj-...
WANDB_API_KEY = ...      # optional
```

In notebook:

```python
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()
hf_token = secrets.get_secret("HF_TOKEN")
```

⚠️ Secrets are **per-notebook**. Toggle them ON in the right panel before running.

### 🌐 Internet on/off

- **Off by default** in Kaggle notebooks (anti-cheating in competitions)
- For our use: **MUST turn ON** (right panel → Settings → Internet: ON)
- Needed for: HuggingFace model download, HF Hub push

---

## 3. v1 vs v2 — Why we rewrote the notebook

### 📛 v1 — `train_on_kaggle.ipynb` (deprecated)

The first attempt. Had multiple show-stopper bugs:

| Issue | Symptom |
|---|---|
| **OOM crash** | `CUDA out of memory` on step 1 — batch=4 too big for T4 |
| **bf16/fp16 conflict** | `NotImplementedError: BFloat16 not supported in fp16 grad scaler` |
| **SFTConfig API drift** | `TypeError: __init__() got unexpected keyword 'max_seq_length'` |
| **save_safetensors error** | `TypeError: unexpected keyword 'save_safetensors'` |
| **tokenizer kwarg removed** | `SFTTrainer` removed `tokenizer=` in newer TRL |
| **Path resolution** | Hardcoded paths broke when dataset attached at different mount points |
| **Recovery impossible** | One bug → restart from scratch (lost 6 hours) |

### ✅ v2 — `train_on_kaggle_v2.ipynb` (working pipeline)

All patches baked in. Key differences:

| Aspect | v1 | v2 |
|---|---|---|
| Batch size | 4 (OOM) | **2** with grad_accum=8 → eff 16 |
| Mixed precision | `fp16=True` (failed) | **`bf16=True, fp16=False`** ✅ |
| Sequence length | 1024 (too big) | **512** |
| `max_seq_length` arg | Direct to SFTConfig (failed) | **`setattr` fallback + tokenizer.model_max_length** |
| Tokenizer passing | `tokenizer=` (removed) | **`processing_class=` with try/except fallback** |
| Path resolution | Hardcoded | **Recursive `find_repo_root()`** walks tree |
| Zip uploads | PowerShell Compress-Archive (broken) | **Python `zipfile` module** |
| Save strategy | Manual | **Auto-save every 60 steps + "Save Version" before timeout** |
| Recovery | None | **Eval-only notebook for adapter-from-checkpoint** |

---

## 4. Pre-flight checklist — Before starting

### ✈️ Checklist (do this once)

**Local side:**
- [ ] Generate `data/final/{train,val,test}.jsonl` locally first
- [ ] Zip with **Python `zipfile`** (not PowerShell):
  ```python
  import zipfile, os
  with zipfile.ZipFile('tamiltech-qa-data.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
      for f in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
          zf.write(f'data/final/{f}', arcname=f)
  ```
- [ ] Zip the `src/` folder + config:
  ```python
  with zipfile.ZipFile('tamiltech-qa-repo.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
      for root, _, files in os.walk('.'):
          if any(x in root for x in ['.git', '.venv', 'data', 'outputs']):
              continue
          for f in files:
              path = os.path.join(root, f)
              zf.write(path, arcname=os.path.relpath(path))
  ```

**Kaggle side:**
- [ ] Create Kaggle account at kaggle.com
- [ ] Verify phone for GPU access (free)
- [ ] **Datasets → New Dataset → upload** `tamiltech-qa-repo.zip` (set as "tamiltech-qa-repo")
- [ ] **Datasets → New Dataset → upload** `tamiltech-qa-data.zip` (set as "tamiltech-qa-data")
- [ ] **Code → New Notebook** → name it `train-tamiltech-qa-v2`
- [ ] **Settings panel (right):**
  - Accelerator: **GPU T4 × 1**
  - Internet: **ON**
  - Persistence: **Files only**
- [ ] **Add-ons → Secrets:**
  - `HF_TOKEN` = your HF write-scope token
- [ ] **Input data:**
  - Add Dataset → `tamiltech-qa-repo` (latest version)
  - Add Dataset → `tamiltech-qa-data` (latest version)

### Pre-run sanity check

```python
# Cell 0 — verify environment
import os, sys
print("Python:", sys.version)
print("CWD:", os.getcwd())
print("/kaggle/input contents:", os.listdir("/kaggle/input"))
import torch
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
```

Expected:
```
Python: 3.10.x
CWD: /kaggle/working
/kaggle/input contents: ['tamiltech-qa-repo', 'tamiltech-qa-data']
CUDA: True Tesla T4
```

❌ If you see no CUDA → Settings panel left Accelerator at "None". Switch and restart.

---

## 5. Notebook v2 — Cell-by-cell walkthrough

The notebook has **33 cells** divided into 8 logical sections.

### Section A: Setup (cells 1-5)

#### 📦 Cell 1 — Markdown intro

```markdown
# TamilTech-QA — Kaggle Training v2

QLoRA fine-tuning of Llama-3.1-8B on Tanglish technical QA pairs.

**Hardware:** Free T4 (16 GB VRAM)
**Time budget:** ~6 hours for 1 epoch on 3,536 train samples
**Adapter output:** ~50 MB
```

#### 📦 Cell 2 — Install missing packages

```python
!pip install -q -U \
    transformers==4.44.0 \
    peft==0.12.0 \
    bitsandbytes==0.43.3 \
    trl==0.10.1 \
    accelerate==0.33.0 \
    datasets==2.21.0 \
    huggingface_hub
```

**Why pin versions?** Kaggle preinstalls older versions. TRL/PEFT APIs changed 2024 → pin to known-good combo.

**`-q -U`:** quiet + upgrade. Don't spam logs.

#### 📦 Cell 3 — Recursive repo finder (critical patch)

```python
import os
from pathlib import Path


def find_repo_root() -> Path:
    """Walk /kaggle/input tree to find the dir containing src/ + config/."""
    base = Path("/kaggle/input")
    for root in base.rglob("*"):
        if root.is_dir() and (root / "src").exists() and (root / "config").exists():
            return root
    raise FileNotFoundError("Could not locate repo root under /kaggle/input")


REPO_ROOT = find_repo_root()
print("Repo root:", REPO_ROOT)
# Expected: /kaggle/input/tamiltech-qa-repo/tamiltech-qa-repo/  (nested!)
```

**Why this is necessary:**
Kaggle dataset uploads sometimes mount at `/kaggle/input/<dataset-name>/` and sometimes nest deeper if zip had a wrapping folder. Hard-coding paths breaks. `rglob` walks tree, finds the actual location.

**The "thappu" we learned from:**
v1 used `REPO = '/kaggle/input/tamiltech-qa-repo'` directly → `FileNotFoundError: src/ not found`. v2 walks the tree. Robust.

#### 📦 Cell 4 — Add repo to PYTHONPATH

```python
import sys
sys.path.insert(0, str(REPO_ROOT))

# Verify imports work
from src.utils import load_config, project_root
from src.utils.logger import get_logger
from src.utils.seed import seed_everything

print("Imports OK")
seed_everything(42)
```

**Note:** `project_root()` from utils returns wrong path on Kaggle (uses `__file__`). We override below.

#### 📦 Cell 5 — Find data + define output paths

```python
DATA_ROOT = None
for root in Path("/kaggle/input").rglob("train.jsonl"):
    DATA_ROOT = root.parent
    break

if DATA_ROOT is None:
    raise FileNotFoundError("Could not find train.jsonl in /kaggle/input")

print("Data root:", DATA_ROOT)

OUTPUT_ROOT = Path("/kaggle/working/outputs")
OUTPUT_ROOT.mkdir(exist_ok=True)
ADAPTER_OUT = OUTPUT_ROOT / "best"
ADAPTER_OUT.mkdir(exist_ok=True)
```

---

### Section B: Auth + Model Loading (cells 6-10)

#### 📦 Cell 6 — HuggingFace authentication

```python
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login

secrets = UserSecretsClient()
HF_TOKEN = secrets.get_secret("HF_TOKEN")
login(token=HF_TOKEN, add_to_git_credential=False)
print("HF login OK")
```

**Why `add_to_git_credential=False`?** Kaggle has no git setup → would error.

**Required:** HF token must have **WRITE** scope to push later.

#### 📦 Cell 7 — Load tokenizer

```python
from transformers import AutoTokenizer

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
tokenizer.model_max_length = 512

print("Tokenizer loaded. Vocab size:", tokenizer.vocab_size)
print("Pad token:", tokenizer.pad_token)
```

**Llama-3.1 gate access:** First time on a new HF account, you must accept Meta's license at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct. Otherwise: `401 access denied`.

#### 📦 Cell 8 — Load base model in 4-bit

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model (this takes ~3 minutes)...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.config.use_cache = False
model.config.pad_token_id = tokenizer.pad_token_id

print("Model loaded.")
print(f"GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GiB")
# Expected: ~4 GB
```

#### 📦 Cell 9 — Prepare for k-bit training

```python
from peft import prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)
print("Model prepared for QLoRA training.")
```

**What this does:**
- Casts LayerNorm to fp32 (stability — norms hate low precision)
- Enables gradient checkpointing
- Freezes base model weights
- Sets `model.enable_input_require_grads()`

#### 📦 Cell 10 — Attach LoRA adapters

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected: trainable params: 3,407,872 || all params: 8,033,002,496 || trainable%: 0.0424
```

**The 0.04%:** Only 3.4M parameters out of 8B are trainable. This is what makes QLoRA fit.

---

### Section C: Data Loading (cells 11-14)

#### 📦 Cell 11 — Load datasets

```python
from datasets import load_dataset

train_ds = load_dataset("json", data_files=str(DATA_ROOT / "train.jsonl"), split="train")
val_ds = load_dataset("json", data_files=str(DATA_ROOT / "val.jsonl"), split="train")
test_ds = load_dataset("json", data_files=str(DATA_ROOT / "test.jsonl"), split="train")

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
# Expected: Train: 3,536, Val: 431, Test: 447
```

#### 📦 Cell 12 — Inspect one sample

```python
print(train_ds[0])
# {'id': '...', 'question': '...', 'answer': '...', 'source': 'youtube', ...}
```

**Why inspect?** Prove the JSONL parsed correctly — field names matter for next step.

#### 📦 Cell 13 — Define chat template formatter

```python
SYSTEM_PROMPT = (
    "You are a helpful Tanglish (Tamil-English mixed) technical assistant. "
    "Answer questions in casual Tanglish using Roman-script Tamil scaffolding "
    "and English technical terms."
)


def format_sample(sample):
    """Render one row as Llama-3.1 chat template."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
    sample["text"] = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False,
    )
    return sample


train_ds = train_ds.map(format_sample)
val_ds = val_ds.map(format_sample)
print(train_ds[0]["text"][:500])
```

**Output snippet:**
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a helpful Tanglish...<|eot_id|>
<|start_header_id|>user<|end_header_id|>
React hooks-a explain pannunga<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Hook na functional component-le...<|eot_id|>
```

**Crucial:** Use the **tokenizer's own** `apply_chat_template` — it knows the exact special tokens. Don't hand-write the template.

#### 📦 Cell 14 — Check token length distribution

```python
import numpy as np

lengths = [len(tokenizer(t["text"]).input_ids) for t in train_ds.select(range(100))]
print(f"Median: {int(np.median(lengths))}, p95: {int(np.percentile(lengths, 95))}, max: {max(lengths)}")
# Expected: Median 180, p95 380, max ~480
```

**Why check?** If p95 > max_seq_length (512), we'd truncate too aggressively. 380 << 512, good.

---

### Section D: Training Setup (cells 15-20)

#### 📦 Cell 15 — SFTConfig with all critical settings

```python
from trl import SFTConfig

args = SFTConfig(
    output_dir=str(OUTPUT_ROOT),
    num_train_epochs=1,
    per_device_train_batch_size=2,          # ← critical, OOM if higher
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,          # ← effective batch 16
    warmup_ratio=0.03,
    learning_rate=2e-4,
    bf16=True,                              # ← MUST be True for Llama-3.1
    fp16=False,                             # ← MUST be False
    logging_steps=5,
    eval_strategy="steps",
    eval_steps=30,
    save_steps=60,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    lr_scheduler_type="cosine",
    optim="paged_adamw_8bit",
    report_to="none",
    gradient_checkpointing=True,
    packing=False,
    dataset_text_field="text",              # ← tells SFTTrainer which field to train on
)

# Workaround: max_seq_length API changed across TRL versions
try:
    args.max_seq_length = 512
except Exception:
    pass

print("SFTConfig built. Effective batch:", args.per_device_train_batch_size * args.gradient_accumulation_steps)
# Effective batch: 16
```

#### 📦 Cell 16 — Build trainer with API fallback

```python
from trl import SFTTrainer

trainer_kwargs = dict(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

try:
    # New API (TRL ≥ 0.10)
    trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)
except TypeError:
    # Old API
    trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)

print("Trainer built")
```

**Why try/except?** TRL's `tokenizer=` → `processing_class=` rename happened mid-2024. Notebook should work on multiple TRL versions.

#### 📦 Cell 17 — Sanity-check trainer (one step)

```python
# Optional smoke test — comment out for real run
# trainer.train(resume_from_checkpoint=None)
# (just run cell 18 directly)
```

#### 📦 Cell 18 — TRAIN

```python
print("Starting training. This will take ~6 hours.")
print("If session ends before completion: click Save Version BEFORE the 12-hour limit.")

train_result = trainer.train()
print("Training complete.")
print(f"Final loss: {train_result.training_loss:.4f}")
```

**Expected output (sampled):**
```
{'loss': 2.3145, 'learning_rate': 1.94e-05, 'epoch': 0.05}
{'loss': 1.8920, 'learning_rate': 1.98e-04, 'epoch': 0.10}
{'eval_loss': 1.5523, 'epoch': 0.13}
{'loss': 1.6438, 'epoch': 0.15}
...
{'eval_loss': 0.9712, 'epoch': 0.95}
{'loss': 0.9531, 'epoch': 1.00}
```

**Watch for:**
- ✅ Loss decreasing monotonically
- ✅ Eval loss following train loss (no overfitting yet)
- ✅ No NaN spikes
- ✅ GPU memory stable around ~10-12 GB

---

### Section E: Evaluation (cells 19-24)

#### 📦 Cell 19 — Save adapter

```python
trainer.save_model(str(ADAPTER_OUT))
print(f"Adapter saved to {ADAPTER_OUT}")
print(f"Size: {sum(f.stat().st_size for f in ADAPTER_OUT.rglob('*'))/1024**2:.2f} MB")
# Expected: ~50 MB
```

#### 📦 Cell 20 — Perplexity on test set

```python
import math, torch
from datasets import Dataset

model.eval()

def compute_ppl(texts):
    total_nll, total_tokens = 0.0, 0
    with torch.no_grad():
        for t in texts:
            enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            out = model(**enc, labels=enc.input_ids)
            total_nll += out.loss.item() * enc.input_ids.size(1)
            total_tokens += enc.input_ids.size(1)
    return math.exp(total_nll / total_tokens)


test_texts = [
    f"Question: {s['question']}\nAnswer: {s['answer']}"
    for s in test_ds
]
ppl = compute_ppl(test_texts[:100])  # subset for speed
print(f"Perplexity on test: {ppl:.2f}")
# Expected: ~12.40
```

#### 📦 Cell 21 — Sample generations (qualitative)

```python
sample_questions = [
    "Anna React hooks-a epdi work pannuthu sollunga",
    "Python list comprehension-a explain panna mudiyuma?",
    "TCP ku UDP ku ena difference",
]

model.eval()
for q in sample_questions:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": q},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    answer = tokenizer.decode(out[0][inputs.input_ids.size(1):], skip_special_tokens=True)
    print("Q:", q)
    print("A:", answer)
    print("-" * 60)
```

**Look for:**
- Roman-script Tamil scaffolding ("na", "athula", "ippadi")
- English tech terms preserved (function, array, useState)
- Coherent flow

#### 📦 Cells 22-24 — CSPS / TTR / TCF computation

```python
sys.path.insert(0, str(REPO_ROOT))
from src.evaluation.metrics import (
    code_switch_preservation_score,
    technical_term_retention,
    tamil_connector_fluency,
)
from src.preprocessing.tanglish_detector import TanglishDetector

detector = TanglishDetector(min_ratio=0.05, max_ratio=0.95)

# Generate predictions for first 50 test samples
preds = []
for s in test_ds.select(range(50)):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": s["question"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120, do_sample=False)
    preds.append(tokenizer.decode(out[0][inputs.input_ids.size(1):], skip_special_tokens=True))

refs = [s["answer"] for s in test_ds.select(range(50))]

# Compute novel metrics
data_cfg = load_config(REPO_ROOT / "config" / "data_config.yaml")
csps = code_switch_preservation_score(preds, refs, detector)
ttr = technical_term_retention(preds, refs, data_cfg["preprocessing"]["technical_keywords"])
tcf = tamil_connector_fluency(preds, refs, data_cfg["preprocessing"]["tamil_connectors"])

print(f"CSPS: {csps:.4f}")
print(f"TTR:  {ttr:.4f}")
print(f"TCF:  {tcf:.4f}")
```

---

### Section F: Save outputs (cells 25-28)

#### 📦 Cell 25 — Write eval report

```python
import json

report = {
    "model": BASE_MODEL,
    "adapter": "lora_r8",
    "perplexity": ppl,
    "csps": csps,
    "ttr": ttr,
    "tcf": tcf,
    "training_loss_final": train_result.training_loss,
}

report_path = OUTPUT_ROOT / "eval_report.json"
report_path.write_text(json.dumps(report, indent=2))
print(f"Report saved: {report_path}")
```

#### 📦 Cell 26 — Zip adapter for download

```python
import zipfile

zip_path = OUTPUT_ROOT / "adapter.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ADAPTER_OUT.rglob('*'):
        if f.is_file():
            zf.write(f, arcname=f.relative_to(ADAPTER_OUT))
print(f"Zip saved: {zip_path}")
```

---

### Section G: Push to HuggingFace (cells 29-32)

#### 📦 Cell 29 — Push adapter

```python
from huggingface_hub import HfApi

api = HfApi()
REPO_ID = "dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA"

# Create repo if doesn't exist
api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True, private=False)

# Upload adapter folder
api.upload_folder(
    folder_path=str(ADAPTER_OUT),
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Upload TamilTech-QA LoRA adapter v0.1",
)
print(f"Pushed to: https://huggingface.co/{REPO_ID}")
```

#### 📦 Cell 30 — Push dataset (separate from model)

```python
DATASET_ID = "dheepakkaran/TamilTech-QA"

api.create_repo(repo_id=DATASET_ID, repo_type="dataset", exist_ok=True, private=False)

for f in ["train.jsonl", "val.jsonl", "test.jsonl"]:
    api.upload_file(
        path_or_fileobj=str(DATA_ROOT / f),
        path_in_repo=f,
        repo_id=DATASET_ID,
        repo_type="dataset",
    )
print(f"Dataset uploaded: https://huggingface.co/datasets/{DATASET_ID}")
```

#### 📦 Cell 31 — Push README (CRITICAL — don't let push_to_hub overwrite!)

```python
README_CONTENT = """---
license: llama3.1
tags: [tamil, tanglish, code-switching, qlora, llama-3.1]
base_model: meta-llama/Llama-3.1-8B-Instruct
---

# TamilTech-QA Llama-3.1-8B QLoRA Adapter

(... your full model card ...)
"""

readme_path = OUTPUT_ROOT / "README.md"
readme_path.write_text(README_CONTENT)

api.upload_file(
    path_or_fileobj=str(readme_path),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="model",
    commit_message="Add custom model card",
)
```

**⚠️ The lesson:** `trainer.push_to_hub()` **auto-generates a default README and overwrites yours**. Either:
1. Use `upload_folder` (doesn't write README), then upload README separately, OR
2. Edit the README on the HF website after push

#### 📦 Cell 32 — Final cleanup

```python
print("All done!")
print(f"Adapter:  https://huggingface.co/{REPO_ID}")
print(f"Dataset:  https://huggingface.co/datasets/{DATASET_ID}")
print()
print("⚠️ NOW click 'Save Version' in the top-right to preserve /kaggle/working/")
```

---

## 6. The error log — Every bug we hit

### 🐛 Error #1: CUDA OOM at step 1

**Symptom:**
```
RuntimeError: CUDA out of memory. Tried to allocate 1.20 GiB.
GPU 0: Tesla T4 (UUID: GPU-...)
```

**Diagnosis:** `batch_size=4`, `max_seq_length=1024` → 4 × 1024 × hidden_dim × layers × bytes too big.

**Fix:**
- `per_device_train_batch_size: 2` (was 4)
- `max_seq_length: 512` (was 1024)
- `gradient_accumulation_steps: 8` (was 4) → keep effective batch=16

**Lesson:** Memory math BEFORE training, not after crash:
```
activations = batch × seq_len × hidden × layers × 4 bytes (fp32 baseline)
             = 4 × 1024 × 4096 × 32 × 4 = ~2 GB per sample!
With bf16: ~1 GB. With gradient checkpointing: ~30% of that.
```

---

### 🐛 Error #2: BFloat16 in fp16 grad scaler

**Symptom:**
```
NotImplementedError: BFloat16 is not supported by torch.cuda.amp.GradScaler
```

**Diagnosis:** Llama-3.1 stores weights in **bf16**. We set `fp16=True` → trainer tries to use fp16 grad scaler → fails on bf16 gradients.

**Fix:**
```python
bf16=True,
fp16=False,    # both must be set
```

**Lesson:** Check **base model's native dtype** before choosing mixed precision:
```python
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
print(model.config.torch_dtype)   # bfloat16 → use bf16
```

---

### 🐛 Error #3: `max_seq_length` API removed

**Symptom:**
```
TypeError: SFTConfig.__init__() got an unexpected keyword 'max_seq_length'
```

**Diagnosis:** TRL ≥ 0.9 removed it from `SFTConfig`. Set via tokenizer or runtime attribute.

**Fix:**
```python
args = SFTConfig(...)  # no max_seq_length kwarg
try:
    args.max_seq_length = 512
except Exception:
    pass
tokenizer.model_max_length = 512
```

**Lesson:** Pin library versions OR write **version-agnostic try/except**.

---

### 🐛 Error #4: `tokenizer` kwarg removed from SFTTrainer

**Symptom:**
```
DeprecationWarning: 'tokenizer' arg is deprecated, use 'processing_class' instead
```

(or hard error in TRL 0.11+)

**Fix:**
```python
try:
    trainer = SFTTrainer(processing_class=tokenizer, **kw)
except TypeError:
    trainer = SFTTrainer(tokenizer=tokenizer, **kw)
```

---

### 🐛 Error #5: `save_safetensors` rejected

**Symptom:**
```
TypeError: __init__() got unexpected keyword 'save_safetensors'
```

**Fix:** Just remove it from `SFTConfig` kwargs. New TRL saves safetensors by default.

---

### 🐛 Error #6: gpt-4o-mini malformed JSON

**Symptom:**
```
json.JSONDecodeError: Expecting value: line 1 column 5 (char 4)
```

Happens ~30% of the time without JSON mode.

**Fix:**
```python
response = client.chat.completions.create(
    ...,
    response_format={"type": "json_object"},   # ← magic
    messages=[
        {"role": "system", "content": "You output only valid JSON."},
        ...
    ],
)
```

**Also crucial:** prompt MUST request a JSON **object** with a `pairs` array, not bare array. JSON mode requires top-level `{}`.

---

### 🐛 Error #7: MinHash dedup too slow

**Symptom:** Dedup hung for >30 min on 7,466 records.

**Diagnosis:** `datasketch.MinHashLSH` is O(n) per insert; LSH index init was slow on Kaggle's CPU.

**Fix:** Replaced with **exact dedup by tuple key**:
```python
seen = set()
deduped = []
for r in records:
    key = (r["question"].strip().lower(), r["answer"].strip().lower())
    if key not in seen:
        seen.add(key)
        deduped.append(r)
```

**Lesson:** Use the **simplest tool** that works. MinHash is for cross-document near-duplicates at million-scale. We had 7K records and most dupes were exact.

---

### 🐛 Error #8: Repo path nested deeper

**Symptom:**
```
ModuleNotFoundError: No module named 'src'
```

After `sys.path.insert(0, '/kaggle/input/tamiltech-qa-repo')`.

**Diagnosis:** Kaggle dataset upload of a zip with a wrapper folder mounts as `/kaggle/input/tamiltech-qa-repo/tamiltech-qa-repo/`.

**Fix:** `find_repo_root()` walks tree (Cell 3).

---

### 🐛 Error #9: PowerShell zip rejected by HuggingFace

**Symptom:** HF dataset viewer fails to parse uploaded files. Or path errors when extracting.

**Diagnosis:** PowerShell `Compress-Archive` writes paths with `\` (Windows separator). HF expects POSIX `/`.

**Fix:** Use Python `zipfile`:
```python
import zipfile, os
with zipfile.ZipFile('out.zip', 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk('src'):
        for f in files:
            full = os.path.join(root, f)
            zf.write(full, arcname=full.replace('\\', '/'))   # force /
```

---

### 🐛 Error #10: Regex bad character range

**Symptom:**
```
re.error: bad character range \-x at position N
```

**Diagnosis:** Regex `r"[A-Za-z\\\\-]"` interpreted backslash-backslash as escape → bad range.

**Fix:** Use single backslash `r"[A-Za-z\\-]"` or place `-` at end: `r"[A-Za-z-]"`.

---

## 7. Session timeout recovery saga

### 😱 The story

During v0.1 training, **session timeout hit at hour 6.5**, but training had completed at hour 6 — adapter was in `/kaggle/working/outputs/best/`. Did NOT click "Save Version". Session ended → outputs **wiped**.

**6 hours of training lost.** ⚠️

### 🛠️ Recovery flow

After loss → realized Kaggle keeps the trained checkpoint **if you click "Save Version"** at any point. Since I'd forgotten, I had to retrain.

**But on the second run**, I built recovery infrastructure:

#### Layer 1: Auto-save during training

```python
SFTConfig(
    save_steps=60,             # save every 60 steps
    save_total_limit=2,        # keep latest 2
    load_best_model_at_end=True,  # restore best after training
    ...
)
```

#### Layer 2: Click "Save Version" mid-run

Click "Save Version" → "Quick Save" — snapshots `/kaggle/working/` **WITHOUT killing your live session**. You can run again after, and final state is preserved as another version.

#### Layer 3: Recovery notebook (`eval_recovery.ipynb`)

If session dies and you have a saved version with `/kaggle/working/outputs/best/`:

```python
# Recovery flow in new notebook
# 1. Settings → Add Data → search "tamiltech-qa-trained" (your saved version)
# 2. Mount appears at /kaggle/input/tamiltech-qa-trained/

ADAPTER_PATH = "/kaggle/input/tamiltech-qa-trained/outputs/best"

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    load_in_4bit=True,
    device_map="auto",
    torch_dtype=torch.float16,
)
model = PeftModel.from_pretrained(base, ADAPTER_PATH)
print("Recovered!")
```

**Now you can:** evaluate, generate samples, push to HF — without retraining.

### 📋 Recovery checklist (do this NEXT time)

- [ ] Click "Save Version" at **hour 4** (mid-train), not at hour 11.5
- [ ] Verify `/kaggle/working/outputs/` has files before saving
- [ ] After save: note the version number (e.g., "Version 3")
- [ ] If timeout happens: open new notebook, attach Version 3's output as dataset
- [ ] Skip training cells, jump straight to eval/push cells

---

## 8. Memory optimization — How we squeezed into 16 GB

### 📊 Memory budget on T4

```
Total T4 VRAM:              16 GB
─────────────────────────────────
Llama-3.1-8B in 4-bit:      4 GB
LoRA adapters (fp16):       <100 MB
Optimizer states (AdamW8):  ~300 MB
Activations (bs=2,seq=512): ~5 GB
Gradients (LoRA only):      <100 MB
CUDA reserved overhead:     ~1 GB
─────────────────────────────────
Total in use:               ~10-11 GB
Buffer:                     ~5 GB ✅
```

### 🪛 Each optimization's contribution

| Technique | Memory saved | Speed cost |
|---|---|---|
| 4-bit base model | ~12 GB | minimal |
| Gradient checkpointing | ~3 GB | +25% time |
| `paged_adamw_8bit` | ~1.5 GB | minimal |
| LoRA (vs full FT) | ~15 GB | none |
| bf16 (vs fp32) | ~50% | none |
| seq_len=512 (vs 1024) | ~50% activations | none for our data |

### What we tried but didn't keep

- **DeepSpeed ZeRO-3:** overkill for single GPU; added complexity
- **CPU offload (`accelerate`):** worked but 3× slower
- **8-bit base model:** worked but used more VRAM than 4-bit

---

## 9. Saving + uploading from Kaggle

### Strategy 1: Direct push to HF Hub (preferred)

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path=str(ADAPTER_OUT), repo_id=REPO_ID, repo_type="model")
```

**Pros:** Adapter on HF immediately, no manual download.
**Cons:** Needs internet on, HF token in secrets.

### Strategy 2: Download zip locally

```python
import zipfile
zip_path = "/kaggle/working/adapter.zip"
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in ADAPTER_OUT.rglob('*'):
        if f.is_file():
            zf.write(f, arcname=f.relative_to(ADAPTER_OUT))
```

Then in Kaggle UI: **Output tab → adapter.zip → Download**.

### Strategy 3: Save Version (Kaggle-internal)

Click **Save Version → Quick Save** → outputs persist as a **Kaggle Dataset** under your account → attachable to any future notebook.

---

## 10. Lessons learned — Production wisdom

### 🧠 What I'd tell past-me

1. **Click "Save Version" at hour 4, not hour 11.5.** Trust nothing about session lifetime.

2. **Pin library versions BEFORE training**, not after first crash. `pip install transformers==4.44.0 peft==0.12.0 trl==0.10.1` saves you 2 hours of debugging.

3. **Memory math FIRST.** Don't OOM, calculate. `batch × seq × hidden × layers × bytes_per_param ≈ activations`.

4. **Match precision to model.** Check `model.config.torch_dtype` → set training precision to match. Llama-3.1=bf16, Mistral=bf16, older Llama=fp16.

5. **Sanity check 1 batch before full training.** `trainer.train()` for 5 steps with `--max_steps=5` proves no API errors.

6. **Use the tokenizer's chat template.** Don't hand-write `<|start_header_id|>` etc. — too easy to break.

7. **Save eval predictions, not just metrics.** Numbers fade; samples teach.

8. **Push README separately.** `push_to_hub()` overwrites README with default. Upload custom README **after** model push.

9. **`response_format={"type": "json_object"}` is non-negotiable** for structured LLM outputs.

10. **Recursive `rglob` for path discovery on shared platforms.** Hard-coded paths are bombs.

### 🚦 Pre-commit-style checklist for any LLM training

Before clicking `trainer.train()`:
- [ ] `model.config.torch_dtype` matches `bf16`/`fp16` config
- [ ] Tokenizer pad token set
- [ ] `model.config.use_cache = False`
- [ ] `prepare_model_for_kbit_training` called BEFORE `get_peft_model`
- [ ] Effective batch size = `bs × accum` ≥ 16 (stable gradients)
- [ ] `eval_steps` reasonable (~5-10% of total steps)
- [ ] `save_steps` ≥ `eval_steps` (don't save more often than eval)
- [ ] Internet ON if pulling base model
- [ ] HF login successful (if pulling gated model)
- [ ] `output_dir` writable (`/kaggle/working/` ON Kaggle, not `/kaggle/input/`)
- [ ] `gradient_checkpointing=True` if VRAM tight

---

## 11. v2 → v3 roadmap (what we'd change)

For **TamilTech-QA v0.2 / v2**:

| v0.1 (current) | v2 plan | Reason |
|---|---|---|
| 1 epoch | **3-5 epochs** with early stopping | More learning; current model under-converged |
| `lora_r8` | **`lora_r16`** + add FFN targets (gate/up/down) | More capacity |
| 4,415 samples | **15K+ samples** | Smaller dataset = more epochs OK; larger = better generalization |
| Single training notebook | **Multi-stage training** (warmup → main → DPO) | Quality stages |
| Fixed seq=512 | **Dynamic seq up to 1024** with sample sorting | Use full context capacity |
| Push to HF manually at end | **Push every 30 min during training** | Crash safety |
| BLEU/ROUGE flat (insensitive) | **Add GPT-4 judge + MAUVE + BERTScore** | Better signal |
| Tamil-script samples dropped | **Two models: Roman-only + script** | Cover both populations |

### Tooling upgrades

- **W&B integration** for live loss plots (currently `report_to="none"`)
- **PR for `find_repo_root`** to upstream as Kaggle utility
- **GitHub Action** to run notebook tests
- **GGUF export** for llama.cpp inference (smaller, faster CPU inference)

---

## ✅ End of Doc 4

**Next up:** `05_huggingface_github_publishing.md` — HF dataset/model/Space + GitHub push process.

**Padichu paaru:**
- Cell walkthrough depth correct-a?
- Error log helpful for interview prep?
- Recovery section clear?
- Lessons section useful, or want more?

Confirm pannina apparam **Doc 5** start panren. 🚀
