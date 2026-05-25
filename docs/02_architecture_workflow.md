# 🏗️ Document 2 — Architecture & Workflow (Tanglish Edition)

> **Intha doc-ku purpose:** TamilTech-QA-oda **full system architecture**-a end-to-end visualize pannardhu. Ovvoru module ena pannum, modules-edam epdi talk panniku, decisions enna basis-le edukapatuthu, failure-na ena pannardhu — ellame onna.
>
> **Reading time:** ~90 minutes.
> **Prereq:** Doc 1 (basics) padichirukanum.

---

## Table of Contents

1. [10,000-foot view — System overview](#1-10000-foot-view--system-overview)
2. [Data flow architecture](#2-data-flow-architecture)
3. [Layered architecture — Components view](#3-layered-architecture--components-view)
4. [Module-by-module breakdown](#4-module-by-module-breakdown)
5. [Sequence diagrams — Major flows](#5-sequence-diagrams--major-flows)
6. [Decision tree — Why this design](#6-decision-tree--why-this-design)
7. [Configuration architecture](#7-configuration-architecture)
8. [Storage & artifact hierarchy](#8-storage--artifact-hierarchy)
9. [Failure modes & recovery paths](#9-failure-modes--recovery-paths)
10. [Workflow walkthrough — Cold start to publish](#10-workflow-walkthrough--cold-start-to-publish)
11. [Architecture trade-offs & alternatives](#11-architecture-tradeoffs--alternatives)

---

## 1. 10,000-foot view — System overview

### 🌍 Big picture

```
┌────────────────────────────────────────────────────────────────────┐
│                    TAMILTECH-QA SYSTEM                              │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐      │
│  │  INPUT   │ → │ COLLECT  │ → │ PROCESS  │ → │   TRAIN    │       │
│  │ Sources  │   │  Phase   │   │  Phase   │   │   Phase    │       │
│  └──────────┘   └──────────┘   └──────────┘   └────────────┘      │
│                                                       ↓             │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                        │
│  │ PUBLISH  │ ← │  DEPLOY  │ ← │ EVALUATE │                        │
│  │ Artifacts│   │  Demos   │   │  Phase   │                        │
│  └──────────┘   └──────────┘   └──────────┘                        │
└────────────────────────────────────────────────────────────────────┘

INPUT      → YouTube comments (12 channels) + OpenAI gpt-4o-mini synthetic
COLLECT    → API scraping, rate limiting, retry logic
PROCESS    → Language filter, Tanglish detection, dedup, split
TRAIN      → QLoRA Llama-3.1-8B on Kaggle T4 (free)
EVALUATE   → PPL, BLEU, ROUGE, CSPS, TTR, TCF
DEPLOY     → HF Space (static + Colab live), Gradio UI
PUBLISH    → HF Dataset + HF Model + GitHub repo
```

### 🎬 Movie analogy

TamilTech-QA project = oru **movie production**:

| Movie phase | TamilTech-QA equivalent |
|---|---|
| **Story idea** | Tanglish QA needed-nu identify pannardhu |
| **Script writing** | Data collection (YouTube + synthetic) |
| **Casting** | Choose Llama-3.1-8B as base model |
| **Editing** | Preprocessing pipeline |
| **Shooting** | Training on Kaggle |
| **Post-production** | Evaluation + metrics |
| **Trailer** | HF Space demo + Colab live |
| **Theatre release** | Publish to GitHub + HF Hub |
| **Reviews** | Community feedback (after release) |

---

## 2. Data flow architecture

### 📊 Detailed flow diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                          PHASE 1: DATA COLLECTION                       │
└────────────────────────────────────────────────────────────────────────┘

   ┌──────────────────┐                ┌──────────────────┐
   │  YouTube API v3  │                │  OpenAI API      │
   │  (12 Tamil tech  │                │  (gpt-4o-mini)   │
   │   channels)      │                │                  │
   └────────┬─────────┘                └────────┬─────────┘
            │                                   │
            │ 80 videos × 200 comments/video    │ 10 topics ×
            │ ≈ 19,200 raw comments max         │ 50 pairs/topic
            ↓                                   ↓
   ┌──────────────────┐                ┌──────────────────┐
   │  raw_youtube.    │                │  raw_synthetic.  │
   │     jsonl        │                │     jsonl        │
   │  ~7,500 entries  │                │  ~500 entries    │
   └────────┬─────────┘                └────────┬─────────┘
            │                                   │
            └─────────────┬─────────────────────┘
                          ↓
                ┌──────────────────────┐
                │  dataset_merger.py   │
                │  combine + tag       │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  data/raw/merged.    │
                │      jsonl           │
                │  ~8,000 raw entries  │
                └──────────┬───────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                       PHASE 2: PREPROCESSING                            │
└────────────────────────────────────────────────────────────────────────┘
                           ↓
                ┌──────────────────────┐
                │  text_cleaner.py     │ → strip emoji, fix Unicode,
                │                      │   normalize whitespace
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ tanglish_detector.py │ → keep only 5%-95% Tanglish band
                │                      │   (drops pure-English & pure-Tamil)
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ language_filter.py   │ → require technical keywords,
                │                      │   min_words ≥ 10
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  qa_formatter.py     │ → format as instruction/input/output
                │                      │   triples (Alpaca / ChatML)
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  exact dedup         │ → dedupe by (question, answer) tuple
                │  (MinHash optional)  │
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │ stratified split     │ → 80/10/10 by topic+difficulty
                └──────────┬───────────┘
                           ↓
                ┌──────────────────────┐
                │  data/final/         │
                │   train.jsonl  3,536 │
                │   val.jsonl     431  │
                │   test.jsonl    447  │
                └──────────┬───────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE 3: TRAINING                                │
└────────────────────────────────────────────────────────────────────────┘
                           ↓
                ┌──────────────────────┐
                │  Kaggle T4 notebook  │
                │  (train_on_kaggle_   │
                │     v2.ipynb)        │
                └──────────┬───────────┘
                           ↓
   ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
   │ model_loader  │ │ lora_config  │ │  trainer.py    │
   │   .py         │ │   .py        │ │ (SFTTrainer)   │
   │ Load Llama-   │ │ Build LoRA   │ │ Run training   │
   │ 3.1-8B in 4bit│ │   adapter    │ │   loop         │
   └───────┬───────┘ └──────┬───────┘ └────────┬───────┘
           └────────────────┼─────────────────┘
                            ↓
                ┌──────────────────────┐
                │  outputs/checkpoint- │
                │   N/                 │
                │  (LoRA adapter,      │
                │   ~50 MB)            │
                └──────────┬───────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                       PHASE 4: EVALUATION                               │
└────────────────────────────────────────────────────────────────────────┘
                           ↓
   ┌───────────────┐ ┌──────────────┐ ┌────────────────┐
   │  metrics.py   │ │  ablation.py │ │ human_eval.py  │
   │ PPL/BLEU/     │ │  vs base     │ │  side-by-side  │
   │ ROUGE/CSPS/   │ │  comparison  │ │  inspection    │
   │ TTR/TCF       │ │              │ │                │
   └───────┬───────┘ └──────┬───────┘ └────────┬───────┘
           └────────────────┼─────────────────┘
                            ↓
                ┌──────────────────────┐
                │  outputs/eval/       │
                │   eval_report.json   │
                │   eval_report.md     │
                │   plots/*.png        │
                └──────────┬───────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                  PHASE 5: PUBLISH + DEPLOY                              │
└────────────────────────────────────────────────────────────────────────┘
                           ↓
   ┌──────────────────────┐ ┌──────────────────────┐
   │  HF Hub upload       │ │  GitHub push          │
   │  (hf_uploader.py)    │ │                       │
   │                      │ │                       │
   │  - dataset           │ │  - code + notebooks   │
   │  - model adapter     │ │  - README.md          │
   │  - Space (Gradio)    │ │  - LICENSE            │
   └──────────────────────┘ └──────────────────────┘
                           ↓
                ┌──────────────────────┐
                │   Live demo (Colab)  │
                │   side-by-side: base │
                │   vs fine-tuned      │
                └──────────────────────┘
```

### 🚂 Train station analogy

Idhu oru **rail journey** mathiri:

| Station | Action |
|---|---|
| **Egmore** (Start) | Raw data collected |
| **Trichy** | Preprocessing — cleaning + filtering |
| **Madurai** | Training — engine work happens |
| **Rameswaram** | Evaluation — metrics check |
| **HuggingFace Hub** (Destination) | Public artifact published |

Each station handoff = data format change (jsonl → tokenized → tensor → adapter weights → uploaded model card).

---

## 3. Layered architecture — Components view

### 🍰 Cake-layer view

```
┌─────────────────────────────────────────────────────────────┐
│  APPLICATION LAYER                                          │
│  ┌──────────────────────┐   ┌──────────────────────┐       │
│  │  Gradio UI           │   │  Jupyter notebooks   │       │
│  │  (Space + Colab)     │   │  (kaggle, demo)      │       │
│  └──────────────────────┘   └──────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATION LAYER                                        │
│  ┌──────────────────────────────────────────────────┐      │
│  │  scripts/run_*.sh + notebooks                    │      │
│  │  - run_scraping.sh                               │      │
│  │  - run_preprocessing.sh                          │      │
│  │  - run_training.sh                               │      │
│  │  - run_evaluation.sh                             │      │
│  └──────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  DOMAIN LAYER (src/)                                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ data_        │ │ preprocessing│ │ training/    │        │
│  │  collection/ │ │              │ │              │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
│  ┌──────────────┐ ┌──────────────┐                         │
│  │ evaluation/  │ │ utils/       │                         │
│  └──────────────┘ └──────────────┘                         │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  FRAMEWORK LAYER                                            │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │ transformers│ │   peft      │ │     trl     │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │bitsandbytes │ │  datasets   │ │ accelerate  │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  COMPUTE LAYER                                              │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐           │
│  │  PyTorch    │ │   CUDA      │ │  GPU (T4)   │           │
│  └─────────────┘ └─────────────┘ └─────────────┘           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  EXTERNAL SERVICES                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐        │
│  │ YouTube API  │ │  OpenAI API  │ │  HF Hub      │        │
│  └──────────────┘ └──────────────┘ └──────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### Separation of concerns

Each layer has **single responsibility**:

| Layer | What it does | What it doesn't do |
|---|---|---|
| Application | User interaction | Computation |
| Orchestration | Workflow sequencing | Business logic |
| Domain (src/) | Project-specific logic | Framework details |
| Framework | ML primitives | Project decisions |
| Compute | Hardware execution | High-level logic |
| External | Third-party services | Local state |

---

## 4. Module-by-module breakdown

### 📂 src/data_collection/

```
src/data_collection/
├── __init__.py
├── youtube_scraper.py      # YouTube Data API v3 client
├── synthetic_generator.py  # OpenAI gpt-4o-mini Tanglish QA generator
└── dataset_merger.py       # Combine sources, tag, dedup
```

#### `youtube_scraper.py`

**Responsibility:** Fetch Tanglish comments from configured Tamil tech YouTube channels.

**Public API:**
```python
class YouTubeScraper:
    def __init__(self, api_key: str, config: dict): ...
    def fetch_channel_videos(self, channel_id: str) -> List[Video]: ...
    def fetch_video_comments(self, video_id: str) -> List[Comment]: ...
    def run(self) -> Iterator[Comment]: ...  # main entry
```

**Key design choices:**
- **Retry with exponential backoff** (httpx + tenacity) — API flakiness common
- **Quota tracking** — YouTube gives 10,000 units/day free; ~100 unit/video search + comment fetch
- **State file** (`.scraped_videos.json`) — resume after partial failure
- **Streaming output** — write JSONL line-by-line, not all-at-end (crash safety)

**Quota math:**
```
search.list           = 100 units / call
videos.list (batch)   = 1 unit / video
commentThreads.list   = 1 unit / page

12 channels × 80 videos = 960 videos
960 × 1 (videos.list)   = 960 units
960 × 2 (comment pages) = 1,920 units
Channel search          = 12 × 100 = 1,200 units
Total: ~4,080 units ≪ 10,000 daily quota ✅
```

#### `synthetic_generator.py`

**Responsibility:** Generate Tanglish QA pairs via gpt-4o-mini when real data is sparse.

**Key prompt template:**
```python
SYSTEM = """You are an expert Tanglish (Tamil-English code-switched) 
technical content creator. Generate question-answer pairs about {topic}.

Rules:
1. Use Tanglish: Tamil scaffolding + English technical terms
2. Roman-script Tamil (no Tamil script)
3. Casual, natural tone — how real Tamil techies talk
4. Preserve English technical vocabulary verbatim

Output JSON with exactly this format:
{
  "pairs": [
    {"question": "...", "answer": "..."},
    ...
  ]
}
"""

USER = f"Generate {n} Tanglish technical QA pairs on {topic}."
```

**Why JSON mode is critical:**
```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},  # ← without this, 30% malformed
    messages=[...],
)
```

**Quality control:**
- Reject pairs with Tanglish ratio outside [0.05, 0.95]
- Reject duplicates (cosine similarity ≥ 0.85)
- Require technical keyword presence

**Cost calculation:**
```
gpt-4o-mini pricing: $0.15 / 1M input tokens, $0.60 / 1M output

Per topic: ~50 pairs × 50 tokens output × ~3 retries = ~7,500 output tokens
         + ~500 prompt tokens × 3 retries = ~1,500 input tokens

Per topic cost: (1,500/1M × $0.15) + (7,500/1M × $0.60) = $0.0047
10 topics: $0.047
With validation re-runs: ~$0.10 total ✅
```

#### `dataset_merger.py`

**Responsibility:** Combine YouTube + synthetic + existing HF datasets into unified format.

**Output format:**
```json
{
  "id": "yt_abc123_0042",
  "source": "youtube",
  "channel": "UCIFQgj1...",
  "video_id": "xYz...",
  "question": "Anna React hooks-a epdi use pannardhu?",
  "answer": "Hook na functional component-le state manage panrathuku...",
  "tanglish_ratio": 0.42,
  "topic": "web",
  "difficulty": "intermediate",
  "raw_text": "..." 
}
```

**Why these fields?**
- `id`: unique, deterministic (resume-safe)
- `source`: enable per-source ablation later
- `tanglish_ratio`: pre-computed (avoid re-tokenization)
- `topic + difficulty`: stratified split keys

---

### 📂 src/preprocessing/

```
src/preprocessing/
├── __init__.py
├── text_cleaner.py        # Unicode, whitespace, emoji handling
├── language_filter.py     # English/Tamil mix detection
├── tanglish_detector.py   # Calculate Tanglish ratio
└── qa_formatter.py        # Format as Alpaca/ChatML
```

#### `tanglish_detector.py`

**Core function:**
```python
def tanglish_ratio(text: str) -> float:
    """
    Returns fraction of tokens that are Tamil-Romanized.
    0.0 = pure English, 1.0 = pure Tamil-script Tamil.
    Natural Tanglish: 0.15 - 0.85
    """
    tokens = text.lower().split()
    tamil_count = sum(1 for t in tokens if is_tamil_romanized(t))
    return tamil_count / max(len(tokens), 1)


def is_tamil_romanized(token: str) -> bool:
    """
    A token is "Tamil-Romanized" if:
    1. Matches a Tamil-word lexicon (naan, pannu, sollu, etc.)
    2. Ends with Tamil suffix (-la, -ku, -aana, -aagum, etc.)
    3. Has Tamil consonant cluster patterns (ndr, ndh, kk, pp, etc.)
    """
    if token in TAMIL_LEXICON: return True
    if any(token.endswith(s) for s in TAMIL_SUFFIXES): return True
    if has_tamil_consonants(token): return True
    return False
```

**Why a 5%-95% band (loose)?**

| Band | Pros | Cons |
|---|---|---|
| 15%-85% (tight) | High purity | Drops too many real-user samples |
| **5%-95% (loose)** ✅ | Keeps natural variance | Includes some borderline samples |
| 0%-100% (no filter) | Maximum data | Pure English / pure Tamil pollution |

v0.1 used loose (5%-95%). v2 plan: revisit with better classifier.

#### `qa_formatter.py`

**Format converters:**
```python
def to_alpaca(sample: dict) -> str:
    return (
        "### Instruction:\n"
        + sample["instruction"] + "\n\n"
        + "### Response:\n"
        + sample["answer"]
    )

def to_chatml(sample: dict) -> List[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]
```

**Why ChatML?** Llama-3.1-Instruct uses ChatML-like format natively. Aligning to its training distribution → less re-learning.

---

### 📂 src/training/

```
src/training/
├── __init__.py
├── model_loader.py        # Load Llama in 4-bit + tokenizer setup
├── lora_config.py         # Build LoRA config from YAML
├── callbacks.py           # Memory logging, sample generation
└── trainer.py             # SFTTrainer orchestration
```

#### `model_loader.py`

```python
def load_model_and_tokenizer(config: dict):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=config["quantization"]["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"]["default"],
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"]["default"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer
```

**`prepare_model_for_kbit_training` does what?**
- Casts norm layers to fp32 (stability)
- Enables gradient checkpointing
- Freezes base model params
- Sets up gradient flow for LoRA injection

#### `lora_config.py`

```python
def build_lora_config(profile: str, config: dict) -> LoraConfig:
    p = config["lora_configs"][profile]  # e.g., "lora_r8"
    return LoraConfig(
        r=p["r"],
        lora_alpha=p["lora_alpha"],
        target_modules=p["target_modules"],
        lora_dropout=p["lora_dropout"],
        bias=p["bias"],
        task_type=p["task_type"],
    )
```

**Profile selection via YAML enables ablation:**
```yaml
# config/model_config.yaml
lora_configs:
  lora_r8: {r: 8, lora_alpha: 16, ...}   # default
  lora_r16: {r: 16, lora_alpha: 32, ...} # v2
  lora_r64: {r: 64, lora_alpha: 128,...} # ablation
```

#### `trainer.py`

```python
def train(config, model, tokenizer, train_ds, val_ds):
    args = SFTConfig(
        output_dir=config["training"]["output_root"],
        num_train_epochs=config["training"]["num_train_epochs"],
        per_device_train_batch_size=config["training"]["per_device_train_batch_size"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        warmup_ratio=config["training"]["warmup_ratio"],
        learning_rate=config["training"]["learning_rate"],
        bf16=True,
        fp16=False,  # ← critical for Llama-3.1
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=30,
        save_steps=60,
        save_total_limit=2,
        load_best_model_at_end=True,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        packing=False,
    )
    # max_seq_length not accepted directly in SFTConfig anymore
    # Set via tokenizer.model_max_length or via setattr fallback

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,  # new API; fallback try/except
        peft_config=lora_config,
        callbacks=[MemoryLoggingCallback(), SampleGenCallback()],
    )
    trainer.train()
    trainer.save_model()
```

---

### 📂 src/evaluation/

```
src/evaluation/
├── __init__.py
├── metrics.py             # PPL, BLEU, ROUGE, CSPS, TTR, TCF
├── ablation.py            # Base vs fine-tuned comparison
├── human_eval.py          # Side-by-side N=20 sample generation
└── visualizer.py          # Plot generation
```

#### `metrics.py` — Key functions

```python
def compute_perplexity(model, tokenizer, texts, device) -> float: ...
def compute_bleu(predictions, references) -> float: ...
def compute_rouge(predictions, references) -> dict: ...
def compute_csps(texts) -> float: ...    # novel
def compute_ttr(texts) -> float: ...     # novel
def compute_tcf(predictions, references) -> float: ...  # novel
```

#### `ablation.py`

Runs BOTH base model AND fine-tuned model on test set → computes all metrics → produces comparison table.

**Output:**
```json
{
  "base": {"ppl": 57.05, "bleu": 4.2, "csps": 1.2, ...},
  "finetuned": {"ppl": 12.40, "bleu": 4.5, "csps": 3.6, ...},
  "deltas": {"ppl_pct": -78, "csps_pct": +200, ...}
}
```

---

### 📂 src/utils/

```
src/utils/
├── __init__.py        # config loader with ${VAR} expansion
├── hf_uploader.py     # Push dataset/model/Space to HF Hub
├── logger.py          # Loguru wrapper
└── seed.py            # Reproducibility seeding
```

**`__init__.py` — Config loader:**
```python
def load_config(path: str) -> dict:
    with open(path) as f:
        raw = f.read()
    # Expand ${VAR} from environment
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)
```

**Enables:** YAML refers to `${OPENAI_API_KEY}` → resolved from `.env` at load time. Secrets stay out of YAML.

---

## 5. Sequence diagrams — Major flows

### 5.1 Data collection flow

```
User           run_scraping.sh    YouTubeScraper   YouTube API    Disk
 │                  │                  │              │              │
 │ ./run_scraping.sh│                  │              │              │
 │─────────────────▶│                  │              │              │
 │                  │ python scraper.py│              │              │
 │                  │─────────────────▶│              │              │
 │                  │                  │ search.list  │              │
 │                  │                  │─────────────▶│              │
 │                  │                  │◀─────────────│              │
 │                  │                  │ (12 channels)│              │
 │                  │                  │ for each ch: │              │
 │                  │                  │  videos.list │              │
 │                  │                  │─────────────▶│              │
 │                  │                  │◀─────────────│              │
 │                  │                  │  commentThreads.list        │
 │                  │                  │─────────────▶│              │
 │                  │                  │◀─────────────│              │
 │                  │                  │ append JSONL                │
 │                  │                  │────────────────────────────▶│
 │                  │                  │ update .scraped_videos.json │
 │                  │                  │────────────────────────────▶│
 │                  │ exit 0           │              │              │
 │                  │◀─────────────────│              │              │
 │ done             │                  │              │              │
 │◀─────────────────│                  │              │              │
```

### 5.2 Training flow (Kaggle)

```
Notebook       Kaggle Env    HF Hub       T4 GPU       Disk
 │                │             │           │             │
 │ load configs   │             │           │             │
 │───────────────▶│             │           │             │
 │ load datasets  │             │           │             │
 │───────────────▶│             │           │             │
 │ AutoTokenizer  │             │           │             │
 │ ┐              │  pull model │           │             │
 │ └────────────────────────────▶───────────▶             │
 │                │             │ cache to /tmp           │
 │                │             │─────────────────────────▶
 │ from_pretrained                                        │
 │ (4-bit)        │             │           │             │
 │───────────────▶│             │           │ load weights│
 │                │             │           │◀────────────│
 │                │             │           │ in 4-bit (4GB)
 │ build LoRA     │             │           │             │
 │───────────────▶│             │           │             │
 │ SFTTrainer.train()                                     │
 │───────────────▶│             │           │ forward     │
 │                │             │           │ backward    │
 │                │             │           │ optimizer step
 │                │             │           │ ┐           │
 │                │             │           │ └─loop......│
 │                │             │           │             │
 │ every 30 steps: eval_loss → log                        │
 │ every 60 steps: save checkpoint                        │
 │                │             │           │────────────▶│ /kaggle/working/
 │ final save     │             │           │             │
 │───────────────▶│             │           │────────────▶│
 │ Click "Save Version" ← CRITICAL                        │
 │───────────────▶│             │           │             │
```

### 5.3 Inference flow (live demo)

```
User       Gradio UI    Base Model   Fine-tuned   PEFT
 │            │            (4-bit)     adapter
 │ type input │            │            │            │
 │───────────▶│            │            │            │
 │            │ generate() │            │            │
 │            │───────────▶│            │            │
 │            │◀───────────│            │            │
 │            │            (Tamil-script output)     │
 │            │ generate() │            │            │
 │            │───────────────────────▶│           │
 │            │                        │ apply LoRA │
 │            │                        │───────────▶│
 │            │                        │◀───────────│
 │            │◀───────────────────────│            │
 │            │  (Tanglish output)     │            │
 │            │ render side-by-side    │            │
 │◀───────────│            │            │            │
```

---

## 6. Decision tree — Why this design

### Q: Why Llama-3.1-8B and not GPT or Gemini?

| Choice | Pros | Cons | Decision |
|---|---|---|---|
| **Llama-3.1-8B** ✅ | Open weights, free, fits T4 in 4-bit | Smaller than GPT-4 | **WIN** |
| GPT-4 | Highest quality | Closed, API-only, can't fine-tune | ❌ |
| Gemini | Free tier exists | Limited fine-tuning support | ❌ |
| Mistral-7B | Open, similar size | Less multilingual coverage | Backup choice |
| Llama-3.1-70B | Better quality | Won't fit T4 even in 4-bit | ❌ |

### Q: Why QLoRA and not full fine-tuning?

| Choice | Cost | Quality | Decision |
|---|---|---|---|
| Full FT | Needs A100 80GB ($$$$) | Best | ❌ for free tier |
| LoRA fp16 | Needs ~24GB | Very good | Still tight on T4 |
| **QLoRA** ✅ | Fits T4 16GB | ~99% of full FT | **WIN** |
| Adapters (other PEFT) | Similar | Slightly worse than LoRA | ❌ |

### Q: Why YouTube comments + synthetic, not just one?

| Source | Strength | Weakness |
|---|---|---|
| YouTube comments | **Real user Tanglish** — authentic distribution | Noisy, off-topic, profanity |
| Synthetic (gpt-4o-mini) | Clean, on-topic, structured QA | Slightly less natural, gpt-mini's "imagined" Tanglish |
| **Both combined** ✅ | Authenticity + structure | Need merge logic |

92% YouTube + 8% synthetic gave best signal-to-noise.

### Q: Why Kaggle and not Colab Pro?

| Platform | GPU | Time | Cost |
|---|---|---|---|
| Colab free | T4 (interruptible) | ~3-4 hr | Free but unreliable |
| Colab Pro | T4/A100 | Longer | $10/mo |
| **Kaggle free** ✅ | T4 dedicated | 12 hr/session, 30 hr/week | Free + reliable |
| Lambda Labs | A100 | Unlimited | $1-2/hr |

For 1-epoch QLoRA on 4,415 samples: Kaggle's 6-7 hours fit perfectly.

### Q: Why HuggingFace Hub and not S3 / personal site?

| Platform | Pros | Cons |
|---|---|---|
| **HF Hub** ✅ | Free, community standard, datasets+models+Spaces | Tied to HF ecosystem |
| GitHub LFS | Familiar | Bandwidth costs, no model viewer |
| S3 | Industrial | Costs money, no discovery |
| Personal site | Full control | Nobody finds it |

For a research artifact: discoverability > control → HF Hub.

### Q: Why Gradio static showcase + Colab live demo, instead of HF Space live?

| Option | Pros | Cons |
|---|---|---|
| HF Space free CPU | Free hosting | Llama-3.1-8B OOM on 16GB CPU |
| HF Space ZeroGPU | Live inference | Requires HF PRO ($9/mo) |
| **Static Space + Colab live** ✅ | Free, both work | Two paths to maintain |

Free-tier compromise.

---

## 7. Configuration architecture

### YAML hierarchy

```
config/
├── data_config.yaml      # Scraping, preprocessing, splits
├── model_config.yaml     # Base model, quantization, LoRA, training
└── eval_config.yaml      # Test set paths, metric configs
```

### Env variable expansion

YAML uses `${VAR}` syntax → resolved at load time from `.env`:

```yaml
# config/data_config.yaml
youtube:
  api_key: "${YOUTUBE_API_KEY}"

synthetic:
  openai_api_key: "${OPENAI_API_KEY}"
```

```bash
# .env (NEVER committed)
YOUTUBE_API_KEY=AIzaSy...
OPENAI_API_KEY=sk-proj-...
HF_TOKEN=hf_...
```

```python
# src/utils/__init__.py
def load_config(path):
    text = open(path).read()
    text = os.path.expandvars(text)  # ${VAR} → actual value
    return yaml.safe_load(text)
```

### Why this pattern?

- **Configs in repo** ✅ — reproducibility
- **Secrets out of repo** ✅ — security
- **No code change to swap profiles** ✅ — just edit YAML

---

## 8. Storage & artifact hierarchy

### 📁 Local filesystem

```
tamiltech-qa/
├── .env                        # secrets (gitignored)
├── data/
│   ├── raw/                    # scraped jsonl (gitignored, can regen)
│   ├── processed/              # post-cleanup (gitignored)
│   └── final/                  # train/val/test (uploaded to HF, gitignored locally)
├── outputs/
│   ├── checkpoints/            # ckpt-30, ckpt-60, ... (gitignored, ~50MB each)
│   ├── best/                   # symlink to best checkpoint
│   └── eval/
│       ├── eval_report.json    # quantitative metrics
│       ├── eval_report.md      # human-readable
│       └── plots/*.png         # loss curves, metric bars
```

### ☁️ HuggingFace Hub

```
hf.co/
├── datasets/dheepakkaran/TamilTech-QA/
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   ├── README.md
│   └── dataset_infos.json
├── dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA/
│   ├── adapter_model.safetensors    # ~50 MB
│   ├── adapter_config.json
│   ├── README.md (model card)
│   └── tokenizer files
└── spaces/dheepakkaran/TamilTech-QA-Demo/
    ├── app.py                       # Gradio app
    ├── requirements.txt
    └── README.md (Space metadata)
```

### 🐙 GitHub

```
github.com/dheepakkaran/TamilTech-QA/
├── README.md
├── LICENSE
├── .gitignore                       # protects .env, data/, outputs/
├── requirements.txt
├── setup.py
├── config/*.yaml
├── src/**/*.py
├── notebooks/**/*.ipynb
├── tests/**/*.py
└── scripts/run_*.sh
```

### 🧊 Kaggle (training only)

```
/kaggle/input/                       # read-only inputs
├── tamiltech-qa-repo/               # uploaded source code
└── tamiltech-qa-data/               # uploaded preprocessed data zips

/kaggle/working/                     # WIPED on timeout unless "Save Version"
├── trained_adapter/
│   ├── adapter_model.safetensors
│   └── adapter_config.json
└── eval_report.json
```

---

## 9. Failure modes & recovery paths

### 🔥 Failure matrix

| Failure | Symptom | Root cause | Recovery |
|---|---|---|---|
| **YouTube quota exceeded** | HTTP 403 | >10K units/day used | Wait 24 hr OR rotate API key |
| **OpenAI rate limit** | HTTP 429 | Too many requests/min | Exponential backoff (built-in) |
| **JSON parse error (synthetic)** | Malformed response | gpt-4o-mini drift | Use `response_format={"type":"json_object"}` |
| **CUDA OOM (training)** | `CUDA out of memory` | batch/seq too big | ↓batch, ↓max_seq, ↑grad_accum |
| **NotImplementedError bf16** | At first backward pass | fp16 grad scaler on bf16 weights | Set `bf16=True, fp16=False` |
| **Kaggle session timeout** | `/kaggle/working/` wiped | 12hr limit hit | "Save Version" before timeout |
| **HF push 401** | Auth failure | Token expired/wrong scope | Regenerate token with write scope |
| **README overwritten on push** | Default placeholder appears | `push_to_hub` regenerates README | Upload README separately via `huggingface_hub.upload_file` |
| **PowerShell zip rejected** | Backslash paths | Compress-Archive uses `\` | Use Python `zipfile` module |
| **Tanglish ratio = 0 always** | Wrong lexicon | Tokenization issue | Verify TAMIL_LEXICON has entries |

### 🔄 Recovery flows

#### Flow A: Kaggle session timeout recovery

```
[Session active, training in progress]
        ↓
[6 hours pass, no human action]
        ↓
[Kaggle: "Save Version" reminder OR direct timeout]
        ↓
[Outputs at risk!]
        ↓
[Manual: click "Save Version" → saves /kaggle/working/ snapshot]
        ↓
[New notebook session]
        ↓
[Attach saved version as "input dataset"]
        ↓
[Access at /kaggle/input/saved-version-N/]
        ↓
[Resume eval / pushing]
```

#### Flow B: HF model card overwritten

```
[trainer.push_to_hub() called]
        ↓
[HF generates default README → uploads alongside model]
        ↓
[Custom README lost!]
        ↓
[Recovery:]
        ↓
[huggingface_hub.upload_file(
   path_or_fileobj="README.md",
   path_in_repo="README.md",
   repo_id="user/repo",
   commit_message="Restore custom model card"
)]
```

#### Flow C: bf16 NotImplementedError

```
trainer.train() called
        ↓
First backward pass
        ↓
[FATAL] NotImplementedError: BFloat16 not supported in fp16 grad scaler
        ↓
[Root cause: Llama-3.1 stores bf16, fp16 scaler attempts conversion]
        ↓
Fix in TrainingArguments:
    bf16=True
    fp16=False
        ↓
Restart training
```

---

## 10. Workflow walkthrough — Cold start to publish

### 🛣️ Full journey

#### Day 0 (Hour 0-2): Setup

```bash
# 1. Clone repo (or create fresh)
git clone https://github.com/dheepakkaran/TamilTech-QA.git
cd TamilTech-QA

# 2. Create venv
python -m venv .venv
.venv\Scripts\activate           # PowerShell
pip install -r requirements.txt

# 3. Create .env
cp .env.example .env             # edit with your keys

# 4. Verify
python -c "from src.utils import load_config; print(load_config('config/data_config.yaml')['youtube']['api_key'][:10])"
```

#### Day 1 (Hour 3-8): Data collection

```bash
./scripts/run_scraping.sh
# Outputs:
#   data/raw/youtube_comments.jsonl  (~7,500 entries)
#   data/raw/synthetic_pairs.jsonl   (~500 entries)
#   data/raw/.scraped_videos.json    (state)
```

#### Day 2 (Hour 9-12): Preprocessing

```bash
./scripts/run_preprocessing.sh
# Outputs:
#   data/processed/cleaned.jsonl
#   data/final/train.jsonl  (3,536)
#   data/final/val.jsonl    (431)
#   data/final/test.jsonl   (447)
```

#### Day 3 (Hour 13-20): Training on Kaggle

```
1. Upload data/final/ + src/ as Kaggle dataset
2. Open notebook: notebooks/train_on_kaggle_v2.ipynb
3. Set HF_TOKEN secret in Kaggle Add-ons → Secrets
4. Run all cells (~6 hours)
5. Click "Save Version" BEFORE 12 hr limit
6. Download adapter from saved version OR push to HF directly
```

#### Day 4 (Hour 21-23): Evaluation

```bash
./scripts/run_evaluation.sh
# Outputs:
#   outputs/eval/eval_report.json
#   outputs/eval/eval_report.md
#   outputs/eval/plots/*.png
```

#### Day 5 (Hour 24-30): Publishing

```bash
# 1. Push dataset
python -m src.utils.hf_uploader --type dataset

# 2. Push model
python -m src.utils.hf_uploader --type model

# 3. Build & push Space
cd space/  # separate gradio app
git push hf main

# 4. GitHub push
git add . && git commit -m "v0.1 release" && git push origin main
```

#### Day 6: Demo + iteration

- Test Colab live demo notebook
- Manual quality checks (N=20)
- Update READMEs based on observations
- Tag release: `git tag v0.1`

---

## 11. Architecture trade-offs & alternatives

### 🤔 Things we deliberately did NOT do

| Considered but rejected | Why rejected |
|---|---|
| **Full fine-tuning** | Doesn't fit free GPU; ~99% of quality from QLoRA |
| **Multiple LoRA ranks training** | Time-budget; ablation deferred to v2 |
| **Curriculum learning (easy→hard)** | Adds complexity, marginal gains shown in lit |
| **RLHF/DPO** | Needs preference data we didn't have; v2 plan |
| **Tamil-script generation** | 0.05-0.95 band filter dropped Tamil-script samples; v2 |
| **Multi-task training (code + chat)** | Scope creep; focus on QA first |
| **Custom tokenizer** | Llama's BPE works; custom tokenizer = retraining from scratch |
| **GGUF quantization for inference** | v2 deployment plan; not v0.1 |
| **vLLM serving** | Not needed for demo; v2 if scale matters |

### 🎯 Things we did but could improve in v2

| v0.1 choice | Limitation | v2 plan |
|---|---|---|
| 1 epoch | Underfit on smaller subsets | 3-5 epochs with early stopping |
| `lora_r8` only | Limited capacity | `lora_r16` + add FFN targets |
| Length unconstrained | 14× length ratio in generation | DPO with length-controlled prefs |
| 5%-95% Tanglish band | Drops Tamil-script | Two-stage: separate Roman vs script models |
| Manual recovery from timeout | Stress | Auto-checkpoint to HF every N steps |
| BLEU/ROUGE flat | Style-insensitive | Add MAUVE, BERTScore, GPT-4 judge |
| Static Space demo | No live free inference | GGUF + llama.cpp on Spaces |

---

## ✅ End of Doc 2

**Architecture summary:**
- 5 main pipeline phases (Collect → Process → Train → Eval → Publish)
- 6-layer architecture (App → Orchestration → Domain → Framework → Compute → External)
- 4 main `src/` modules (data_collection, preprocessing, training, evaluation) + utils
- Config-driven, secret-safe, recovery-aware design

**Next up:** `03_local_setup_from_scratch.md` — build it line-by-line on your laptop.

**Padichu paaru:**
- Diagrams clear-a iruka?
- Module breakdown sufficient depth-a?
- Decision tree useful-a, or too much?
- Anything missing?

Confirm pannina apparam **Doc 3** start panren. 🚀
