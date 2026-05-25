# 📘 Document 1 — Basics & Concepts (Tanglish Edition)

> **Intha doc ku purpose:** TamilTech-QA project pannarom na, base level la ena ena therinjirukanum-nu start-to-finish-a explain pannardhu. Theory + analogy + maths + parameters + tuning intuition — ellam onna.
>
> **Reading time:** ~2 hours (slowly, with breaks).
> **Prereq:** Python basic, ML intro level. PyTorch deep knowledge optional.

---

## Table of Contents

1. [LLM (Large Language Model) — Adhu enna?](#1-llm-large-language-model--adhu-enna)
2. [Transformer architecture — Engine inside the LLM](#2-transformer-architecture--engine-inside-the-llm)
3. [Tokenization — Words epadi numbers aaguthu](#3-tokenization--words-epadi-numbers-aaguthu)
4. [Pre-training vs Fine-tuning — Two phases](#4-pre-training-vs-fine-tuning--two-phases)
5. [Full Fine-tuning vs PEFT vs LoRA vs QLoRA](#5-full-fine-tuning-vs-peft-vs-lora-vs-qlora)
6. [Quantization — 4-bit nu enna, math epdi work aaguthu](#6-quantization--4-bit-nu-enna-math-epdi-work-aaguthu)
7. [Tanglish / Code-switching — Linguistics view](#7-tanglish--code-switching--linguistics-view)
8. [Tech Stack — Full inventory](#8-tech-stack--full-inventory)
9. [Evaluation Metrics — Maths included](#9-evaluation-metrics--maths-included)
10. [Hyperparameters — Each one ena pannum](#10-hyperparameters--each-one-ena-pannum)
11. [Tuning principles — When to change what](#11-tuning-principles--when-to-change-what)
12. [Pipeline philosophy — ETL → Train → Eval → Deploy](#12-pipeline-philosophy--etl--train--eval--deploy)
13. [MLOps basics — Production readiness](#13-mlops-basics--production-readiness)
14. [Glossary — Quick reference](#14-glossary--quick-reference)

---

## 1. LLM (Large Language Model) — Adhu enna?

### 🧠 Definition

**LLM** = **Large Language Model**. Romba periya neural network, **billions of parameters** kondu, **next-token prediction** task-ku train aagirukku.

Example: Llama-3.1-8B → **8 billion** weights (parameters). GPT-4 → estimated 1.7 **trillion**.

### 🎯 Core task

Sentence kuduthurom na, **adutha word ena varum** nu predict pannardhu.

```
Input:  "Indha pasanga romba"
Output: "smart" / "lazy" / "padikira" / "azhaga" ...
        (probability distribution across vocabulary)
```

LLM oru word kuduthurukum-na, **vocabulary la ulla ella words ku oru probability** kuduthurukum. Highest probability word-a select pannum (or sample pannum).

### 🥥 Tamil-style analogy

Imagine: **"Anna kadai-le 100 customers vaaranga. Avanga ennana order pannuvanga-nu nee predict pannanum."**

Anna munaadi kadai panni 10 varushama irukaru. Avaru rendu cheeti vechirukaru:

| Customer type | Common orders |
|---|---|
| College pasanga | Sandwich, Maggi, Tea |
| Office uncle | Filter coffee, idli |
| Auto driver | Strong tea, beedi |

**Indha "experience" = Llama-3.1-8B oda 8 billion weights**.

Ovvoru weight um, **konjam patterns** memorize panni vechirukku — "intha word vandha apparam intha word vara chance jaasti". 8 billion weights = 8 billion micro-patterns.

### 📐 Probability math (informal)

Vocabulary V size = 128,000 tokens (Llama-3.1).
Input context = `x_1, x_2, ..., x_n`.

```
P(x_{n+1} = w | x_1, ..., x_n) for each w in V
```

LLM oru function:
```
f_θ(context) → vector of length 128,000
              → softmax → probabilities sum to 1
```

θ (theta) = model parameters = 8 billion numbers (weights).

### Why "large"?

Scale = capability. Empirical observation (Kaplan et al. 2020, Chinchilla 2022):
- **More params + more data + more compute → better performance** (scaling laws).
- 8B-le emergent reasoning kandupidikalam, 1B-le sariya varathu.

But: more params = more VRAM, more cost. Tradeoff.

---

## 2. Transformer architecture — Engine inside the LLM

### 🏗️ High-level structure

Llama-3.1-8B kulla **32 transformer layers** stack panni vechirukku. Ovvoru layer la:

```
Input embedding
   ↓
[Layer 1] → Attention → FeedForward → Output
   ↓
[Layer 2] → Attention → FeedForward → Output
   ↓
   ...
   ↓
[Layer 32] → Output
   ↓
LM head (project to vocabulary)
   ↓
Next-token probabilities
```

### 🔍 Attention — The killer feature

**Self-attention** = sentence la ulla ovvoru word, **vera ella words layam compare panni**, "yaaru yaaroda relate aaguthu" nu kandupidikum.

#### 🥘 Sambar analogy

Imagine sambar pannum bothu:
- **Vegetables** = words (tokens)
- **Spices** = attention weights
- **Each vegetable, vera vegetables oda taste exchange pannuthu** = attention mechanism

Drumstick (murungaikkaai) sambar la pottaa, **adhoda flavor surrounding ku spread aagum** — but onion ku konjam jaasti, tomato ku konjam kammi.

Attention exactly idhe pannum:
```
"Naan school ku ponen because adhu pidikkum"
```
"adhu" = **school**-a refer pannuthu, not "Naan". Attention mechanism idhula:
- "adhu" → "school": weight = 0.7
- "adhu" → "ponen": weight = 0.1
- "adhu" → "naan": weight = 0.05

### ⚙️ Math (simplified)

Each word has 3 vectors:
- **Q (Query)** — "naan enna search pannaren?"
- **K (Key)** — "naan yaarukku match aagaren?"
- **V (Value)** — "match aana naan ena kudukkaren?"

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

- `Q · K^T` = how much each word matches every other word
- `√d_k` = scaling (prevents softmax saturation)
- `softmax` = normalize to probabilities (sum=1)
- `· V` = weighted sum of "values"

In Llama-3.1: **32 attention heads per layer**. Each head learns different relationship patterns (one head = subject-verb, another = pronoun-antecedent, etc.).

### 📦 FeedForward block

Attention apparam, oru **MLP (multi-layer perceptron)** vandhu, each token-a independently process pannum. Bigger hidden dim — Llama-3.1 la 14,336.

```
x → Linear(4096 → 14336) → SwiGLU activation → Linear(14336 → 4096)
```

Idhula than **factual knowledge** store aaguthu (research suggests). Attention = relationships, FFN = facts.

### 🧮 Parameter count breakdown (Llama-3.1-8B)

| Component | Params |
|---|---|
| Token embeddings | 525M |
| 32 × Attention blocks | ~1B |
| 32 × FFN blocks | ~6B |
| LayerNorms + biases | <100M |
| LM head (tied with embeddings) | 0 (shared) |
| **Total** | **~8B** |

---

## 3. Tokenization — Words epadi numbers aaguthu

### 🔤 Problem

Neural network text-a understand pannathu. Numbers than process pannum. So we need: **text → integer sequence**.

### Naive approach (bad)

Each word → unique integer.
```
"naan school ponen"  →  [501, 8421, 222]
```

**Problem:**
- Vocabulary explode aagum (English ke ~170K words, all languages ke billions).
- New word vandha, model-a unknown token-a treat pannum.

### BPE (Byte-Pair Encoding) — Llama uses this

**Sub-word tokenization**: words-a chinna chinna pieces-a break pannum.

```
"unbelievable" → ["un", "believ", "able"]
```

Tanglish-le:
```
"romba super-a iruku" → ["rom", "ba", " super", "-", "a", " iru", "ku"]
```

#### 🍰 Cake-piece analogy

Imagine birthday cake. Whole cake = one word. But share pannum bothu, **slice-a cut pannum** — that's BPE. Each slice = sub-word token.

Frequent slices (common letter combos) kuduppe dedicated slot in vocabulary. Rare combos? They get composed from multiple sub-words.

### Llama-3.1 specifics

- Vocabulary size: **128,256 tokens**
- Tamil characters supported (Unicode-aware)
- **But:** Tamil-le ovvoru word um 3-5 tokens splits aagum (efficient illa)
- English: typically 1 word = 1-2 tokens

### ⚠️ Tanglish tokenization issue

```python
text = "Pasanga code-a debug pannarainga"
tokens = tokenizer.tokenize(text)
# → ['P', 'asang', 'a', ' code', '-', 'a', ' debug', ' pann', 'ar', 'aing', 'a']
```

Pasanga = `P` + `asang` + `a` (3 tokens!). English part "debug" = 1 token only.
**Implication:** Tanglish prompts use **more tokens** than pure English → costs more, fits less in context.

### 🔢 Special tokens

| Token | Meaning |
|---|---|
| `<\|begin_of_text\|>` | Start of sequence |
| `<\|end_of_text\|>` | End of generation |
| `<\|eot_id\|>` | End of turn (chat) |
| `<\|start_header_id\|>...<\|end_header_id\|>` | Role marker (system/user/assistant) |
| `<\|pad\|>` | Padding (if added) |

---

## 4. Pre-training vs Fine-tuning — Two phases

### Phase A — Pre-training

- **Data:** Internet-scale (15+ trillion tokens for Llama-3.1)
- **Task:** Predict next token (unsupervised)
- **Cost:** $$$$ (Llama-3.1 = millions of GPU-hours)
- **Result:** "Base model" — knows language, knows world, but doesn't follow instructions

### Phase B — Fine-tuning

- **Data:** Smaller, curated (thousands → millions samples)
- **Task:** Specific behavior — instruction following, domain adaptation, style change
- **Cost:** Affordable (hours-days on consumer GPUs)
- **Variants:**
  - **SFT** (Supervised Fine-Tuning) — input/output pairs
  - **RLHF** (Reinforcement Learning from Human Feedback) — preference-based
  - **DPO** (Direct Preference Optimization) — RLHF simpler version

### 🏏 Cricket analogy

- **Pre-training** = childhood la 10 varushama ground la oodi oodi practice. Basic skills, fitness, reflexes — ellam ready.
- **Fine-tuning** = T20 World Cup ku 3 months specific training. Specific shots, specific bowlers, specific match scenarios.

Childhood training illama T20 fine-tuning useless. T20 training illama, world cup la perform pannamudiyathu. **Both phases needed**.

### TamilTech-QA position

- Pre-training: Meta did it. Llama-3.1-8B base model.
- Fine-tuning: **We did it**. 4,415 Tanglish QA pairs, 1 epoch QLoRA on Kaggle T4.

---

## 5. Full Fine-tuning vs PEFT vs LoRA vs QLoRA

### Spectrum of approaches

| Approach | What changes | VRAM | Quality |
|---|---|---|---|
| Full FT | All 8B params | 64+ GB | Best |
| PEFT (Adapters, IA³, Prefix) | <1% params | 24 GB | Good |
| **LoRA** | Low-rank updates only | 24 GB | Very good |
| **QLoRA** | LoRA + 4-bit base | **<16 GB** ✅ | Surprisingly close to full FT |

### 🔧 LoRA — Low-Rank Adaptation

**Insight (Hu et al. 2021):** Fine-tuning updates have **low intrinsic rank**. You don't need to update the full weight matrix W; you can approximate the update as a low-rank product:

```
W_new = W_original + ΔW
ΔW ≈ B · A
```

Where:
- `W` is `(d × d)` — e.g., `4096 × 4096`
- `A` is `(r × d)` — e.g., `8 × 4096`
- `B` is `(d × r)` — e.g., `4096 × 8`
- `r` = rank, typically 4-64

#### Params comparison

Full update of one `4096 × 4096` matrix = **16,777,216 params** to train.
LoRA rank-8 update = `(8 × 4096) + (4096 × 8)` = **65,536 params** = **0.4%** of full.

### 🪡 Tailor analogy

You have a ready-made shirt (W = base model). It fits okay, but needs **specific small tweaks**: kollar konjam loose, sleeve konjam adjust.

- **Full FT** = entire shirt-a tear down panni, fresh stitching pannardhu. Costly, time-consuming.
- **LoRA** = ready shirt-oda mela **chinna patch** vekkardhu (B · A). Original shirt intact, patch maaridum vendaa-na remove panniralam.

### 🧊 QLoRA — Quantized LoRA (Dettmers et al. 2023)

Two key ideas combined:

1. **Freeze base model in 4-bit** — 32-bit floats-a 4-bit-a compress pannardhu → 8× memory reduction
2. **Train LoRA adapters in higher precision (fp16/bf16)** — quality preserve aagum

#### Memory math (Llama-3.1-8B example)

| Storage | Per param | Total for 8B |
|---|---|---|
| fp32 (full precision) | 4 bytes | 32 GB |
| fp16 / bf16 | 2 bytes | 16 GB |
| **int4 (QLoRA)** | **0.5 bytes** | **4 GB** ✅ |

Adapter params: ~3.4M params (r=8, q/k/v/o projections in 32 layers).
Adapter in fp16: ~7 MB.

**Total VRAM needed during training:**
- 4-bit base model: ~4 GB
- Adapter weights: <100 MB
- Gradients (LoRA only): <100 MB
- Optimizer states (AdamW for adapter): ~300 MB
- Activations (per microbatch, seq=512, bs=2): ~5 GB
- **Total: ~10 GB → fits T4 (16 GB) easy** ✅

### Why QLoRA preserves quality

- **NF4 quantization** = "NormalFloat 4-bit" — distribution-aware, not uniform
- **Double quantization** — quantization constants themselves get quantized
- **Paged optimizers** — handle memory spikes via CPU offload

Empirically: QLoRA matches full FT performance on most benchmarks. TamilTech-QA proved this on Tanglish.

---

## 6. Quantization — 4-bit nu enna, math epdi work aaguthu

### What does "4-bit" mean?

Each weight gets **only 4 bits** instead of 32. **2^4 = 16 possible values** per weight.

Naive approach:
```
fp32 weight: 0.1273
4-bit: scale to [0, 15] range → bucket 8 → store as 0b1000
```

### NF4 — NormalFloat 4-bit

LLM weights typically follow **normal distribution** (mean ≈ 0, small std). NF4 puts **more buckets near zero** (where most weights live) and fewer in tails.

```
NF4 bucket values (approximately):
[-1.0, -0.70, -0.53, -0.40, -0.28, -0.18, -0.09, 0.0,
  0.08, 0.16, 0.25, 0.34, 0.45, 0.58, 0.74, 1.0]
```

### Block-wise quantization

Per-tensor quantization (one scale for whole tensor) → bad, outliers ruin it.

**Block-wise:** split tensor into blocks of 64 weights. Each block gets its own `scale` (fp32) and `zero point`.

```python
# Per block of 64 weights:
scale = max(abs(block)) / 7  # for symmetric quant
q_block = round(block / scale)  # values in [-8, 7]
# Store: 64 × 4 bits + 1 × fp32 scale = 32 + 4 = 36 bytes per 64 weights
# vs fp32: 64 × 4 = 256 bytes
# 7.1× compression
```

### Double quantization

Block scales (fp32, 4 bytes each) themselves compressed → another 8-bit format.
Saves ~0.37 bits per param. Total: ~3.5 bits per param effective.

### 💾 Loading code (what `bitsandbytes` does)

```python
from transformers import BitsAndBytesConfig
import torch

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",         # NormalFloat-4 buckets
    bnb_4bit_use_double_quant=True,    # quantize the scales too
    bnb_4bit_compute_dtype=torch.float16,  # matmul in fp16
)

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    quantization_config=bnb,
    device_map="auto",
)
```

**During inference/training:**
- Weights stay 4-bit in VRAM (compact)
- When matmul happens: **dequantize on-the-fly to fp16** → multiply → result in fp16
- No quality loss visible in most tasks (Dettmers showed <1% drop)

### Tradeoffs

| | Pro | Con |
|---|---|---|
| 4-bit | 8× memory savings, fits consumer GPUs | Slightly slower than fp16 (dequant overhead) |
| NF4 vs FP4 | Better accuracy | Slightly more compute |
| Double quant | More memory saved | Marginal accuracy loss |

---

## 7. Tanglish / Code-switching — Linguistics view

### 📚 Definition (formal)

**Code-switching (CS)** = within a single utterance, speaker alternates between two or more languages. Common in multilingual societies.

**Tanglish** = Tamil + English code-switching, especially in:
- Tamil Nadu / Sri Lanka / Singapore diaspora
- Casual/digital communication (WhatsApp, YouTube, Twitter)
- Tech / education / professional contexts

### Linguistic types of CS

| Type | Example | Description |
|---|---|---|
| **Inter-sentential** | "I went to college. Aanaa enaku boring-a iruchu." | Whole sentences switch |
| **Intra-sentential** | "Naan **code-a** debug pannaren" | Within one sentence |
| **Tag-switching** | "It was good, **theriyumla**?" | Inserts/tags |
| **Romanized vs script** | "naan" vs "நான்" | Roman or Tamil script |

TamilTech-QA focuses on **intra-sentential, Roman-script Tanglish** — most common online.

### 🧬 Why is CS hard for LLMs?

1. **Vocabulary mismatch:** Pretrained on mostly monolingual data
2. **Tokenization inefficiency:** Romanized Tamil tokens-a learn pannumbothu sub-word splits chappa
3. **Register confusion:** Model defaults to one language (usually English) when uncertain
4. **Evaluation difficulty:** Standard metrics (BLEU/ROUGE) don't capture switching quality

### 🗣️ Tanglish patterns we observed

| Pattern | Frequency | Example |
|---|---|---|
| English noun + Tamil verb suffix | Very high | "code-a write panren" |
| English tech term + Tamil scaffold | High | "TCP/IP enna na athu network protocol" |
| Tamil discourse markers | Very high | "athu", "ippo", "apdi", "na" |
| Tamil pronouns + English content | High | "naan Python use panren" |

### TamilTech-QA's contribution

We define **three novel CS metrics** (covered in §9): CSPS, TTR, TCF. They quantify *how natural* the code-switching is, beyond word accuracy.

---

## 8. Tech Stack — Full inventory

### Core ML stack

| Library | Version | What it does |
|---|---|---|
| **Python** | 3.10/3.11 | Language |
| **PyTorch** | 2.1+ | Tensor ops, autograd, GPU compute |
| **transformers** | 4.40+ (HF) | Model loading, tokenizers, training utilities |
| **peft** | 0.10+ | LoRA, adapters — Parameter Efficient Fine-Tuning |
| **bitsandbytes** | 0.43+ | 4-bit quantization, 8-bit optimizers |
| **accelerate** | 0.30+ | Multi-GPU, mixed precision orchestration |
| **trl** | 0.8+ | SFTTrainer, DPOTrainer (HuggingFace TRL) |
| **datasets** | 2.18+ | HF datasets library — load, map, save |

### Data + utilities

| Library | What it does |
|---|---|
| `google-api-python-client` | YouTube Data API v3 access |
| `openai` | gpt-4o-mini for synthetic data |
| `pandas` | Tabular data manipulation |
| `polars` | Faster pandas alt (used optionally) |
| `pyyaml` | YAML config loading |
| `python-dotenv` | `.env` secret loading |
| `tqdm` | Progress bars |
| `loguru` | Better logging than stdlib |

### Evaluation

| Library | What it does |
|---|---|
| `sacrebleu` | BLEU score (translation metric) |
| `rouge-score` | ROUGE-1/2/L (summarization metric) |
| `nltk` | Tokenization for metrics |
| `numpy` | Math primitives for CSPS/TTR/TCF |

### Deployment

| Library | What it does |
|---|---|
| `huggingface_hub` | Push/pull models, datasets, Spaces |
| `gradio` | Web UI for demos |
| `streamlit` | Alt web UI |

### Why these specific choices?

- **transformers + peft + bitsandbytes + trl** = de facto standard for QLoRA in 2026. Tightly integrated, well-maintained.
- **HuggingFace Hub** = research community standard. Free public hosting.
- **Gradio** > Streamlit for ML demos (better widgets for chat/images, Spaces native support).

### Hardware

| Platform | GPU | VRAM | Cost | Used for |
|---|---|---|---|---|
| Kaggle Free | T4 | 16 GB | Free (30hr/wk) | Training |
| Colab Free | T4 | 16 GB | Free (limited) | Live demo |
| Local CPU | - | - | Free | Code dev, light inference |
| HF Spaces (free) | CPU | 16 GB RAM | Free | Static hosting (couldn't run 8B) |

---

## 9. Evaluation Metrics — Maths included

### 9.1 Perplexity (PPL)

#### Definition

**How "surprised" the model is by held-out text.** Lower = better.

```
PPL(X) = exp(-1/N × Σ log P(x_i | x_<i))
       = exp(cross-entropy loss)
```

For language model on test set of N tokens:
- Compute log-prob of each token given previous context
- Average them (negative)
- Exponentiate

#### Intuition

PPL = "the model is, on average, choosing among PPL words at each step"
- PPL=1: perfect prediction (impossible for natural language)
- PPL=10: 10-way confusion per token
- PPL=∞: random guessing

#### Code

```python
import torch, math
from torch.nn import CrossEntropyLoss

def compute_perplexity(model, tokenizer, text, device):
    encodings = tokenizer(text, return_tensors="pt").to(device)
    input_ids = encodings.input_ids
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    return math.exp(loss.item())
```

#### TamilTech-QA results

- Base Llama-3.1-8B on Tanglish test set: **PPL = 57.05**
- Fine-tuned (1 epoch QLoRA): **PPL = 12.40**
- **Reduction: 78%** ← Headline result

### 9.2 BLEU (Bilingual Evaluation Understudy)

#### Definition

**N-gram overlap between generated text and reference text.** Higher = better. Range: 0-100.

```
BLEU = BP × exp(Σ w_n × log p_n)
```

- `p_n` = n-gram precision (n=1,2,3,4 usually)
- `w_n` = weights (default 0.25 each)
- `BP` = brevity penalty (penalize too-short outputs)

#### Intuition

"Did the model produce words that the reference has?" Word overlap focused.

```python
from sacrebleu import corpus_bleu
bleu = corpus_bleu(predictions, [references]).score
```

#### Limitations on Tanglish

- BLEU assumes one "correct" answer. Tanglish answers can vary massively in form.
- Reference Tanglish + generated English → BLEU drops even if meaning matches.
- TamilTech-QA: BLEU barely changed after fine-tuning (limitation, not failure).

### 9.3 ROUGE (Recall-Oriented Understudy for Gisting Evaluation)

#### Variants

- **ROUGE-1**: unigram overlap (recall)
- **ROUGE-2**: bigram overlap
- **ROUGE-L**: longest common subsequence

```
ROUGE-1 = (count of common unigrams) / (count of unigrams in reference)
```

#### Use case

Originally for summarization, used for QA evaluation too. **Recall-focused** (BLEU is precision-focused).

### 9.4 CSPS — Code-Switch **Preservation** Score (Novel — Ours)

#### Definition

**How close** the prediction's Tanglish ratio is to the reference's Tanglish ratio. Per-sample similarity score in `[0, 1]`.

```
CSPS_i = 1 - |tanglish_ratio(pred_i) - tanglish_ratio(ref_i)|
CSPS  = mean over all samples
```

`tanglish_ratio` itself = fraction of Tamil-Romanized tokens in the text (computed by our `TanglishDetector`).

#### Why "preservation"?

A model that **preserves** the code-switch level of the reference scores high. A model that drifts (e.g., gives pure English when reference is mixed) scores low.

| Pred ratio | Ref ratio | CSPS_i |
|---|---|---|
| 0.40 | 0.40 | **1.00** (perfect match) |
| 0.30 | 0.40 | 0.90 |
| 0.10 | 0.40 | 0.70 (English-drift) |
| 0.00 | 0.40 | 0.60 (pure English) |

#### TamilTech-QA result

- Base: CSPS ≈ 0.62 (drifts English when uncertain)
- Fine-tuned: CSPS ≈ 0.66 (closer to ref distribution)
- **+3.7% absolute improvement** ← Significant for style metric

### 9.5 TTR — **Technical Term Retention** (Novel — Ours)

#### Definition

Fraction of **technical English keywords** present in the reference that **also appear in the prediction**.

```
TTR_i = |tech_terms(pred_i) ∩ tech_terms(ref_i)| / |tech_terms(ref_i)|
TTR  = mean over samples where reference has ≥1 tech term
```

#### Why it matters

In Tanglish technical QA, English **technical vocabulary must be preserved**. We don't want "function" translated to "செயல்பாடு" or "variable" to "மாறி". Those formal Tamil terms exist but **never** show up in real-world Tanglish discourse — preserving the English form is the whole point.

#### Lexicon

We maintain `technical_keywords` list in `config/data_config.yaml`:
function, array, class, error, algorithm, model, training, gradient, circuit, signal, database, api, loop, variable, memory, pointer, transistor, opamp, kernel, process, thread, regression, classification, neural, transformer, tensor, cnn, rnn, lstm, decorator, inheritance, polymorphism, graph, tree, hash, queue, stack, http, rest, sql, schema (~40 terms).

#### TamilTech-QA result

- Base: TTR ≈ 0.81 (81% retention — base model knows English already)
- Fine-tuned: TTR ≈ 0.89 (89% — explicitly preserved English tech terms)
- **+8% absolute improvement**

### 9.6 TCF — **Tamil Connector Fluency** (Novel — Ours)

#### Definition

Rule-based **naturalness** check. Of the Tamil discourse connectors used by the reference, **how many does the prediction also use**?

```
common(pred, ref) = set(tamil_connectors_in_pred) ∩ set(tamil_connectors_in_ref)

If ref uses K Tamil connectors:
   TCF_i = |common(pred, ref)| / max(K, 1)
Else (ref has 0 Tamil connectors):
   TCF_i = 1.0 if pred has any  else 0.5  (neutral baseline)
```

#### Connector lexicon

From `config/data_config.yaml`:
enna na, apdi patha, intha maari, sollunga, puriyutha, theriyuma, paaru, paatha, mathiri, vidu, kaata, solliyaachu, aanaa, aprm, aprum (~15 connectors).

#### Why this matters

Connectors carry the **conversational glue** of Tanglish. Without them you get robotic English-with-Tamil-words, which doesn't pass natural-Tanglish vibe check.

#### TamilTech-QA result

- Base: TCF ≈ 0.49 (uses few Tamil connectors)
- Fine-tuned: TCF ≈ 0.498 (slight improvement; 1-epoch training limit)
- **+1.8% — small but meaningful (didn't degrade)**

### Summary table

| Metric | Direction | Base | Fine-tuned | Δ |
|---|---|---|---|---|
| **Perplexity** ↓ | Lower better | 57.05 | 12.40 | **-78%** |
| BLEU ↑ | Higher better | 4.2 | 4.5 | +0.3 (noisy) |
| ROUGE-L ↑ | Higher better | 0.31 | 0.32 | +0.01 |
| **CSPS** ↑ | Higher better | 0.62 | 0.66 | **+3.7%** |
| **TTR** ↑ | Higher better | 0.81 | 0.89 | **+8%** |
| **TCF** ↑ | Higher better | 0.490 | 0.498 | +1.8% |

**Headline takeaway:** Quantitative metrics (PPL) AND qualitative metrics (CSPS, TTR) both improve. BLEU/ROUGE are insensitive to *style* changes — that's the gap we filled with CS metrics.

---

## 10. Hyperparameters — Each one ena pannum

### 10.1 Learning rate (LR)

**What it controls:** Step size during gradient descent. Big LR = big jumps, small LR = small jumps.

```
new_weight = old_weight − lr × gradient
```

#### TamilTech-QA setting: `2e-4`

- For LoRA, typical range: `1e-4` to `5e-4`
- Too low (`1e-6`): training doesn't converge in 1 epoch
- Too high (`1e-2`): loss explodes (NaN), model corruption
- Sweet spot for QLoRA + Llama: **2e-4** (standard)

#### 🧭 Driving analogy

Highway-le driving = full FT (big model, careful adjustments needed → small LR)
Narrow lane-le driving = LoRA (small adapter, can adjust faster → larger LR)

### 10.2 Batch size + Gradient Accumulation

#### Per-device batch size

How many samples processed in parallel on GPU. Limited by VRAM.

#### Gradient accumulation steps

Accumulate gradients over N mini-batches before updating weights → **effective batch size = batch_size × accum_steps**.

```python
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
# Effective batch size = 16
```

#### Why it matters

- Large effective batch → stable gradients, smoother loss curve
- Small batch → noisy gradients but lower memory
- 16-32 is typical for LLM fine-tuning

#### 🍱 Lunch box analogy

- batch_size=2 = "I can carry 2 lunch boxes at once"
- accum=8 = "I make 8 trips before reporting back"
- Effective batch=16 = "Total 16 lunches collected before update"

### 10.3 Epochs

Number of **complete passes** through training data.

- TamilTech-QA: **1 epoch** (Kaggle session time limit)
- Typical for LoRA fine-tuning: 1-3 epochs
- Too many: overfitting (model memorizes training set, fails on test)
- Too few: underfitting (model didn't learn enough)

### 10.4 LoRA rank `r`

**The size of the low-rank approximation matrix.**

- `r=4`: tiny adapter, fast, limited expressivity
- `r=8`: TamilTech-QA v0.1 — good balance ✅
- `r=16`: more expressive, more params, v2 plan
- `r=64`: ablation only, diminishing returns
- `r=256`: approaches full FT, defeats purpose

```python
lora_config = LoraConfig(
    r=8,                                  # rank
    lora_alpha=16,                        # scaling factor (typically 2×r)
    target_modules=["q_proj", "v_proj",   # which weights to adapt
                    "k_proj", "o_proj"],
    lora_dropout=0.05,
)
```

### 10.5 LoRA alpha

**Scaling factor** for the LoRA update: `ΔW = (α/r) × B @ A`

Convention: `α = 2r` (so scaling = 2). This is empirical — works well across tasks.

### 10.6 Target modules

Which model weights to apply LoRA to.

| Target | Effect |
|---|---|
| `q_proj`, `k_proj`, `v_proj`, `o_proj` (attention) | TamilTech-QA v0.1 — standard ✅ |
| Above + `gate_proj`, `up_proj`, `down_proj` (FFN) | All linear layers — more expressive, ablation |
| Just `q_proj`, `v_proj` | Original LoRA paper — minimal |

### 10.7 Warmup ratio

**Linearly increase LR from 0 to target over first X% of training.** Prevents early instability.

- TamilTech-QA: `warmup_ratio = 0.03` (first 3% of steps)
- Helps avoid loss spikes at step 0

### 10.8 LR scheduler

**How LR changes over training.**

| Scheduler | Behavior |
|---|---|
| `constant` | LR stays at peak | Quick experiments |
| `linear` | LR decreases linearly to 0 | Old-school |
| **`cosine`** ✅ | LR follows cosine curve | TamilTech-QA — smooth decay |

### 10.9 Weight decay

**L2 regularization** — penalize large weights.

- For LoRA: typically `0.0` (LoRA adapters are already constrained by low rank)
- TamilTech-QA: `weight_decay = 0.0`

### 10.10 Gradient checkpointing

**Trade compute for memory.** Don't store all activations during forward pass; recompute during backward.

- **Memory saved:** ~30-50%
- **Speed cost:** ~20-30% slower
- TamilTech-QA: **enabled** (else T4 OOM)

### 10.11 Mixed precision (bf16/fp16)

Store activations in 16-bit instead of 32-bit.

- **bf16** (brain-float): wider exponent, narrower mantissa. Llama-3.1 native format.
- **fp16**: wider mantissa, narrower exponent. Older standard.

**Critical:** Llama-3.1 weights stored in bf16 → fp16 grad scaler fails with `NotImplementedError`. **MUST use bf16=True, fp16=False**.

### 10.12 Optimizer

| Optimizer | Memory | Speed | Used in |
|---|---|---|---|
| AdamW (fp32) | 8 bytes/param state | Fast | Full FT |
| **AdamW 8-bit** ✅ | 2 bytes/param state | Fast | TamilTech-QA |
| SGD | 0-4 bytes | Slow | Rarely for LLM |

`paged_adamw_8bit` from bitsandbytes: page memory to CPU when GPU spikes. Critical for tight VRAM.

---

## 11. Tuning principles — When to change what

### Decision tree

```
Loss not decreasing?
  → Check data quality first
  → If data OK: ↑ learning rate (2× try)
  → Or: ↑ LoRA rank (8 → 16)

Loss decreasing but eval loss not improving?
  → Overfitting
  → ↓ epochs, OR
  → ↑ dropout (0.05 → 0.1), OR
  → ↑ data quantity

OOM (Out of Memory) crash?
  → ↓ batch_size (8 → 4 → 2)
  → ↓ max_seq_length (1024 → 512)
  → ↑ gradient_accumulation_steps to maintain effective batch
  → Enable gradient_checkpointing
  → Switch to 4-bit if not already

Loss spikes / NaN?
  → ↓ learning rate (2e-4 → 1e-4 → 5e-5)
  → ↑ warmup_ratio (0.03 → 0.1)
  → Check for bad samples (length outliers)

Quality good but not great?
  → ↑ LoRA rank (8 → 16 → 32)
  → ↑ target modules (add FFN: gate, up, down)
  → ↑ epochs (1 → 2 → 3)
  → ↑ data quality (curation, deduplication)
```

### General principles

1. **Always check data first** before blaming hyperparams. 90% of issues = data issues.
2. **Change one thing at a time** for ablation clarity.
3. **Start small, scale up** — small dataset + small model first, then scale.
4. **Validate on held-out set** every N steps. Loss curve alone misleading.
5. **Inspect samples manually** — quantitative metrics miss qualitative issues.

---

## 12. Pipeline philosophy — ETL → Train → Eval → Deploy

### Stage 1: Extract (data collection)

- **YouTube API** for real-user Tanglish comments (12 channels, ~80 videos each)
- **OpenAI gpt-4o-mini** for synthetic Tanglish QA augmentation
- **HuggingFace datasets** for any existing relevant corpora

### Stage 2: Transform (preprocessing)

```
Raw text
  ↓ encoding fix (UTF-8 BOM strip)
  ↓ Tanglish-ratio filter (5%-95% band)
  ↓ length filter (≥10 words)
  ↓ technical keyword presence check
  ↓ profanity / PII filter
  ↓ deduplication (exact + MinHash if scalable)
  ↓ stratified split (train/val/test by topic + difficulty)
Final QA pairs
```

### Stage 3: Load (training)

```
QA pairs
  ↓ apply chat template (system, user, assistant)
  ↓ tokenize with Llama tokenizer
  ↓ pad to max_seq=512
  ↓ DataLoader (batch_size=2)
  ↓ model.forward() — 4-bit base + LoRA fp16
  ↓ loss.backward() — only LoRA params get gradients
  ↓ optimizer.step() — paged AdamW 8-bit
  ↓ Every 30 steps: evaluate on val set
  ↓ Every 60 steps: save checkpoint
```

### Stage 4: Evaluate

- **Quantitative:** PPL, BLEU, ROUGE on test set
- **CS metrics:** CSPS, TTR, TCF
- **Qualitative:** N=20 manual side-by-side comparison

### Stage 5: Deploy

- Push **dataset** to HF Hub
- Push **model adapter** to HF Hub
- Build **Space** with Gradio (or static demo + Colab live)
- **GitHub** for code/notebooks

### Why this order?

Each stage **depends on previous stage's output quality**. Bad data → bad training. Bad training → bad eval. Bad eval → false claims.

---

## 13. MLOps basics — Production readiness

### 13.1 Reproducibility

```python
import torch, numpy, random
torch.manual_seed(42)
numpy.random.seed(42)
random.seed(42)
```

Plus: lock dependency versions (`requirements.txt` with exact pins).

### 13.2 Versioning

| What | Tool |
|---|---|
| Code | `git` + GitHub |
| Data | HF datasets (commit hashes) |
| Model | HF model repos (commit + revision pins) |
| Configs | YAML in repo, committed |

### 13.3 Checkpointing

- Save every N steps (`save_steps: 60`)
- Keep top-K (`save_total_limit: 2`)
- Best-checkpoint tracking (`load_best_model_at_end: true`)

### 13.4 Recovery

**Critical lesson learned:** Kaggle sessions can timeout silently. `/kaggle/working/` is wiped.

**Solution we built:** "Save Version" before timeout → adapter persists as a Kaggle dataset → attach to new notebook for eval-only run.

### 13.5 Monitoring

- **Training:** `loss`, `eval_loss` curves logged every step
- **System:** GPU memory, time per step
- Optional: W&B integration (`report_to: "wandb"`)

### 13.6 Cost tracking

| Resource | Cost for TamilTech-QA v0.1 |
|---|---|
| Kaggle T4 | Free (used ~6 hours of 30/wk) |
| OpenAI gpt-4o-mini (synthetic data) | ~$0.10 |
| YouTube API | Free |
| HF Hub hosting | Free |
| Colab T4 (demos) | Free (limited) |
| **Total** | **~$0.10** ✅ |

---

## 14. Glossary — Quick reference

| Term | Tanglish explanation |
|---|---|
| **Activation** | Layer-oda output (intermediate result) |
| **Adapter** | Small trainable module added to frozen base model |
| **Alignment** | Model behavior-a human preferences-oda align pannardhu |
| **Attention** | Tokens between relationships-a learn pannum mechanism |
| **Autoregressive** | Previous tokens-a vechi next token predict pannum |
| **Backprop** | Gradient-a output-la irundhu input-ku-kondu carry pannum |
| **bf16** | Brain-float-16 — wider range than fp16, used in Llama |
| **Causal LM** | Left-to-right generation (no peek at future tokens) |
| **Checkpoint** | Saved model state at specific training step |
| **Code-switching** | Multiple languages within one utterance |
| **Context window** | Max input tokens model can process (Llama-3.1 = 128K) |
| **Cross-entropy** | Loss function for classification / LM tasks |
| **Decoder-only** | Architecture used in GPT/Llama (no separate encoder) |
| **Dropout** | Random neuron deactivation during training (regularization) |
| **Embedding** | Token → vector mapping (continuous representation) |
| **FFN** | Feed-Forward Network — MLP block in transformer |
| **Fine-tuning** | Adapting pretrained model to specific task |
| **fp16/fp32** | Floating-point 16/32-bit |
| **Gradient** | Derivative of loss w.r.t. parameters (direction to update) |
| **Hallucination** | Model generating false but confident output |
| **Hyperparameter** | Config not learned by model (LR, batch size, etc.) |
| **In-context learning** | Learning from examples in prompt (no weight update) |
| **Inference** | Using trained model to generate (no training) |
| **KV cache** | Stored keys/values for fast autoregressive generation |
| **Logits** | Pre-softmax scores from model output |
| **Loss** | How wrong the prediction is (scalar to minimize) |
| **MLP** | Multi-Layer Perceptron = stacked linear layers + activation |
| **NF4** | NormalFloat-4-bit quantization scheme |
| **Optimizer** | Algorithm that updates weights (Adam, SGD, etc.) |
| **Overfitting** | Memorizing train set, failing on test |
| **PEFT** | Parameter-Efficient Fine-Tuning (umbrella term) |
| **Perplexity** | exp(loss) — model's "surprise" on text |
| **Prompt** | Input text to LLM |
| **QLoRA** | Quantized LoRA — 4-bit base + fp16 adapters |
| **Quantization** | Reducing precision of weights (32→8→4 bits) |
| **RLHF** | Reinforcement Learning from Human Feedback |
| **Scheduler** | Function that adjusts LR over training |
| **Softmax** | Normalize logits to probabilities (sum=1) |
| **Temperature** | Sampling param — higher = more random |
| **Token** | Sub-word unit; LLM input/output element |
| **Tokenizer** | Text → token IDs (and back) |
| **Top-k / Top-p** | Sampling strategies — restrict to top-k or top-p% mass |
| **TRL** | Transformer Reinforcement Learning library (HuggingFace) |
| **VRAM** | Video RAM (GPU memory) |
| **Warmup** | LR increases from 0 to target at training start |

---

## ✅ End of Doc 1

**Next up:** `02_architecture_workflow.md` — full system architecture with diagrams.

**Ungal-a check pannanum:**
- Concepts purinjucha?
- Maths section overwhelming-a illa?
- Glossary helpful-a iruka?
- Style — analogies more vendum-a, less vendum-a?

Confirm pannina apparam Doc 2 start panren.
