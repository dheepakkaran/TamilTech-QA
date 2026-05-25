# TamilTech-QA

> **The first publicly released Tanglish (Tamil-English code-switched) technical question-answering dataset, a QLoRA fine-tuned Llama-3.1-8B model, and three novel evaluation metrics for code-switched generation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![HuggingFace Dataset](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/dheepakkaran/TamilTech-QA)
[![HuggingFace Model](https://img.shields.io/badge/🤗-Model-yellow)](https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo)

---

## What is Tanglish?

Tamil engineering students don't consume technical content in pure Tamil or pure English — they use **Tanglish**: Tamil grammatical scaffolding in Roman script, English technical vocabulary kept in English.

> "indha gradient descent epdi work pannuthu sollunga"  
> "konjam intha for-loop-a optimize panna mudiyumaa"  
> "REST API enna mathiri work pannuthu intha concept-a clear-a sollunga"

This register is universal in Tamil tech YouTube, college classrooms, and Indian dev communities — yet no open dataset captured it for technical QA. **TamilTech-QA closes that gap.**

---

## Headline results

We compared **base Llama-3.1-8B-Instruct (zero-shot)** vs. **our QLoRA fine-tuned model** on 447 held-out Tanglish test samples.

| Metric | Base Llama-3.1-8B | TamilTech-QA (ours) | Change |
|---|---|---|---|
| **Perplexity** | 57.05 | **12.40** | **-78%** ⭐ |
| **CSPS** (Code-Switch Preservation Score, novel) | 0.7268 | **0.7539** | **+3.7%** |
| **TTR** (Technical Term Retention, novel) | 0.388 | **0.419** | **+8%** |
| **TCF** (Tamil Connector Fluency, novel) | 0.499 | 0.508 | +1.8% |
| BLEU-4 | 0.0012 | 0.0012 | ~ |
| ROUGE-L | 0.021 | 0.020 | ~ |
| BERTScore F1 | 0.525 | 0.502 | -4% |

**Headline takeaway:** Perplexity drops **78%**, meaning the model has internalised the Tanglish distribution at the language-modeling level. The novel code-switching metrics (CSPS, TTR, TCF) all improve consistently. Surface metrics (BLEU/ROUGE) are limited by 1-epoch training and short reference answers — see [Limitations](#limitations).

### Qualitative comparison — same question, both models

> **Q:** *"indha gradient descent enna na konjam explain pannunga"*

**Base Llama-3.1-8B:**

> Gradient Descent, yaar! Gradient Descent, வேகமாக நம்முடைய லர்னிங் மாதிரியில் பயன்படும் ஒரு அல்கோரிதம். இது நம்முடைய லர்னிங் மாதிரியில் உள்ள பரம்பரை பயன்படுத்துவதை குறைக்க உதவும்...

❌ Switches to *Tamil script* (வேகமாக, நம்முடைய) — **wrong register** for a Tamil engineering student.

**TamilTech-QA (fine-tuned):**

> Gradient Descent-la, optimization process-a use panna, we update parameters to minimize loss function. Indha process-la, we use iterative methods like stochastic gradient descent (SGD) or batch gradient descent (BGD). SGD-la, each iteration-la one sample use panna, while BGD-la, entire dataset use panna. Indha methods-la, we use learning rate control to adjust step size...

✅ **Correct Tanglish** — Roman-script Tamil suffixes (`-la`, `-a`, `panna`) + English technical terms preserved (`gradient descent`, `optimization`, `learning rate`, `parameters`).

This register gap is the headline qualitative result.

---

## Quick links

| Artifact | URL |
|---|---|
| Dataset (4,415 samples) | https://huggingface.co/datasets/dheepakkaran/TamilTech-QA |
| Model (LoRA adapter, 26 MB) | https://huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA |
| Showcase Space (metrics + analyzer) | https://huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo |
| Live demo (Colab, free T4 GPU) | [`live_demo.ipynb`](live_demo.ipynb) |
| Base model | https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct |

---

## Repository structure

```
TamilTech-QA/
├── README.md                    # this file
├── LICENSE                      # MIT
├── .gitignore                   # excludes secrets + generated data
├── requirements.txt             # pinned Python deps
├── setup.py                     # installable package
│
├── config/                      # YAML configs (single source of truth)
│   ├── data_config.yaml         # YouTube channels, synthetic params, preprocessing
│   ├── model_config.yaml        # base model, LoRA, training hyperparameters
│   └── eval_config.yaml         # eval suite settings
│
├── src/
│   ├── data_collection/
│   │   ├── youtube_scraper.py       # YouTube Data API v3, resumable ledger
│   │   ├── synthetic_generator.py   # OpenAI gpt-4o-mini with JSON-mode
│   │   └── dataset_merger.py        # merge YouTube + synthetic + public datasets
│   ├── preprocessing/
│   │   ├── tanglish_detector.py     # lexicon-based code-switch detection
│   │   ├── text_cleaner.py          # URL/HTML strip + technical-keyword gate + dedup
│   │   ├── language_filter.py       # apply 0.05–0.95 Tanglish band filter
│   │   └── qa_formatter.py          # Alpaca + ChatML rendering, stratified split
│   ├── training/
│   │   ├── model_loader.py          # 4-bit quantized Llama loader
│   │   ├── lora_config.py           # 4 LoRA configurations
│   │   ├── trainer.py               # trl SFTTrainer orchestration
│   │   └── callbacks.py             # memory + sample-gen + early-stopping
│   ├── evaluation/
│   │   ├── metrics.py               # BLEU, ROUGE, BERTScore + novel CSPS/TTR/TCF
│   │   ├── ablation.py              # rank / data-size / source / per-topic studies
│   │   ├── visualizer.py            # publication-quality plots
│   │   └── human_eval.py            # blinded rating sheet builder/scorer
│   └── utils/
│       ├── logger.py                # loguru config
│       ├── seed.py                  # reproducible seeding
│       └── hf_uploader.py           # HuggingFace Hub publishing
│
├── notebooks/
│   ├── train_on_kaggle.ipynb        # end-to-end Kaggle pipeline (T4, single session)
│   ├── 01_data_exploration.ipynb    # corpus exploration
│   ├── 02_preprocessing_analysis.ipynb
│   ├── 03_training_monitoring.ipynb
│   └── 04_results_visualization.ipynb
│
├── live_demo.ipynb              # Colab notebook — both models live, side-by-side
│
├── scripts/                     # one-line bash entrypoints per phase
│   ├── run_scraping.sh
│   ├── run_preprocessing.sh
│   ├── run_training.sh
│   └── run_evaluation.sh
│
└── tests/                       # unit tests (preprocessing + metrics)
    ├── test_language_filter.py
    ├── test_qa_formatter.py
    └── test_metrics.py
```

---

## How it was built (the proven pathway)

This is the **shortest path that actually works** end-to-end on free hardware. Each step distils what we learned the hard way.

### Phase 1 — Data collection (~50 min, ~$0.10)

**Source mix:** 92% real-user YouTube comments + 8% synthetic.

1. **YouTube scrape** — 12 Tamil tech channels (Brototype Tamil, A2D Channel, Tamil Tech MrTT, AI Coach John, AI with Thiru, Karthik's Show, Tamil Coding Wizard, Tamil Programmer, Data Engineering Tamil, Endless Knowledge, Curious Freaks, The AI Dude). Use the YouTube Data API v3 free tier (10K units/day).
2. **Synthetic generation** — OpenAI **gpt-4o-mini** with `response_format={"type": "json_object"}`. Costs ~$0.10 for 500 pairs. **Use JSON mode** — without it, 30% of responses are malformed.
3. **Merge** — unify under one schema with `source` tag for ablation.

Result: ~43,000 raw records merged.

### Phase 2 — Preprocessing (~5 min on CPU)

1. **Normalise** — strip URLs, HTML, NFC unicode.
2. **Technical-keyword gate** — drop comments without any technical keyword (curated list in `config/data_config.yaml`).
3. **Length gate** — keep records with 10–512 words.
4. **Exact dedup** — by `(question, answer)` lowercased. (Skip MinHash on >10K records — it doesn't scale; exact dedup catches the duplicates that matter.)
5. **Tanglish band filter** — keep only records with `0.05 ≤ tanglish_ratio ≤ 0.95`. This drops pure-Tamil and pure-English to leave only genuine code-switching.
6. **Stratified split** — 80/10/10 train/val/test by `(topic, difficulty)`.

Result: **4,415 final samples** (3,536 / 431 / 447).

### Phase 3 — QLoRA fine-tuning (~5 hours on free Kaggle T4)

The crucial settings that make this fit one Kaggle session:

```yaml
# config/model_config.yaml (excerpt)
quantization:
  load_in_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true

lora_configs:
  lora_r8:
    r: 8
    lora_alpha: 16
    target_modules: [q_proj, v_proj, k_proj, o_proj]

training:
  num_train_epochs: 1
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8     # effective batch = 16
  learning_rate: 2.0e-4
  fp16: false                        # required — see below
  bf16: true                         # required — see below
  optim: paged_adamw_8bit

tokenizer:
  max_seq_length: 512
```

**Why `bf16: true` and `fp16: false`:** Llama-3.1's native weight dtype is bf16. Training with `fp16=True` causes a `NotImplementedError: ... not implemented for BFloat16` inside the fp16 grad scaler when it encounters bf16 gradients (e.g. from the input-embeddings hook added by `prepare_model_for_kbit_training`). The fix is to use bf16 mixed-precision throughout — no grad scaler needed.

Training command:

```bash
python -m src.training.trainer --config lora_r8 --dataset data/final/
```

Result: training loss **1.55 → 0.97**; eval loss tracks training closely (no overfitting).

### Phase 4 — Evaluation (~3 hours per model on T4)

Compare base zero-shot vs. fine-tuned on 447 test samples. Compute the full metric suite:

- Standard: BLEU-1/2/4, ROUGE-1/2/L, BERTScore F1, exact-match, token F1, answer-length ratio, perplexity.
- **Novel code-switching metrics** (defined in `src/evaluation/metrics.py`):
  - **CSPS** — `1 − |tanglish_ratio_pred − tanglish_ratio_ref|`, averaged.
  - **TTR** — fraction of reference technical keywords retained in the prediction.
  - **TCF** — natural Tamil connector usage match.

```bash
python -m src.evaluation.metrics --models all
```

### Phase 5 — Publishing

Push dataset and adapter to HuggingFace Hub, with proper model and dataset cards (metrics, examples, citation):

```bash
python -m src.utils.hf_uploader --dataset --repo <user>/TamilTech-QA
python -m src.utils.hf_uploader --model outputs/lora_r8_TIMESTAMP --repo <user>/TamilTech-QA-Llama3.1-8B-QLoRA
```

---

## Reproduce it yourself

### Prerequisites

| Requirement | How to get it |
|---|---|
| Python 3.10 or 3.11 | python.org (avoid 3.12 — bitsandbytes compatibility) |
| HuggingFace account | huggingface.co |
| Llama-3.1 access | Request at https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct (uses your HF email — institutional `.edu` mail gets fast approval) |
| YouTube Data API v3 key | console.cloud.google.com → enable YouTube Data API → create credentials → API key |
| OpenAI API key (optional) | platform.openai.com — costs ~$0.10 for the synthetic step with gpt-4o-mini |
| GPU | Kaggle T4 (free, 12-hr session) or Colab Pro+ A100 |

Create `.env` at project root:

```env
YOUTUBE_API_KEY=AIza...
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

### Install

```bash
git clone https://github.com/<your-username>/TamilTech-QA.git
cd TamilTech-QA
python -m venv .venv
.venv\Scripts\activate              # Windows
# source .venv/bin/activate         # macOS / Linux
pip install -r requirements.txt
pip install -e .
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

> **Windows users:** `bitsandbytes` requires WSL2 for training. Preprocessing and evaluation work natively on Windows. The simplest path is to run the training in Kaggle (see below).

### Option A — full pipeline on Kaggle (recommended, free)

1. Upload this repository as a Kaggle **Dataset** (`tamiltech-qa-repo`).
2. Attach `meta-llama/Llama-3.1` from Kaggle **Models** (or accept the HuggingFace fallback in the notebook).
3. Add Kaggle **Secrets**: `YOUTUBE_API_KEY`, `OPENAI_API_KEY`, `HF_TOKEN`.
4. Open `notebooks/train_on_kaggle.ipynb` in a new Kaggle notebook (Settings → GPU T4 x2, Internet on).
5. Run cells top to bottom. Total wall time: about 6 hours.

**Save Version every 30 minutes** during training — Kaggle's `/kaggle/working/` is wiped on session timeout otherwise.

### Option B — local run

```bash
# Phase 1 — collect data (~50 min)
bash scripts/run_scraping.sh

# Phase 2 — clean + filter + split (~5 min, CPU)
bash scripts/run_preprocessing.sh

# Phase 3 — QLoRA fine-tune (~5 hrs on T4)
bash scripts/run_training.sh lora_r8

# Phase 4 — full evaluation (~1 hr per model)
bash scripts/run_evaluation.sh
```

### Option C — live demo only (no training)

If you just want to interact with the fine-tuned model:

1. Open `live_demo.ipynb` in Google Colab.
2. Runtime → Change runtime type → T4 GPU.
3. Add your `HF_TOKEN` via Colab → 🔑 Secrets sidebar.
4. Run all. After ~5 min a public `gradio.live` URL appears with both models loaded side-by-side.

---

## Methodology notes

### Novel metrics (defined in this work)

Standard generation metrics (BLEU, ROUGE) penalise lexical paraphrasing of Tamil scaffolding even when the technical content is preserved. We propose three lightweight, reference-based metrics that isolate code-switching quality:

**CSPS — Code-Switch Preservation Score**

$$\text{CSPS}_i = 1 - \left| r(\text{pred}_i) - r(\text{ref}_i) \right|$$

where `r(·)` is the Tanglish ratio (Tamil-token fraction). Corpus CSPS is the mean over all samples. Range: `[0, 1]`. Higher = the prediction matches the reference's code-switching register.

**TTR — Technical Term Retention**

For each (pred, ref) pair, let `T_ref` be the set of English technical keywords (from the curated list) present in the reference. TTR_i is the fraction of `T_ref` that also appears in the prediction. Samples with zero reference keywords are skipped.

**TCF — Tamil Connector Fluency**

Rule-based check for natural Tamil connectors (`enna na`, `apdi patha`, `sollunga`, `puriyutha`, `theriyuma`, `indha maari`). Score per sample is `c_pred / max(1, c_ref)`, with a neutral 0.5 fallback when the reference has no connectors.

### Why surface metrics underestimate code-switching quality

BLEU/ROUGE require lexical overlap. In Tanglish, the Tamil scaffolding can be paraphrased freely while keeping the English technical vocabulary fixed; this yields high CSPS/TTR but low BLEU. **Perplexity** is the most reliable indicator that the model has learnt the distribution — our model's 78% perplexity drop is the strongest evidence of register acquisition.

### Data sources (full attribution)

12 Tamil tech YouTube channels were scraped under YouTube's Terms of Service for publicly available comments:

| # | Channel | Channel ID |
|---|---|---|
| 1 | Brototype Tamil | UCIFQgj1Rhx-FFgyo0zzPSfw |
| 2 | Tamil Coding Wizard | UCKOob5-7sMljgW3f4pO_Dyg |
| 3 | Tamil Programmer | UCEKv3WR3HUVuIHV2eXFyGYg |
| 4 | Tamil Tech (MrTT) | UC20sXo8ReewkzNKBFgzVCPA |
| 5 | A2D Channel | UCvyZS6W6zMJCZBVzF-Ei6sw |
| 6 | AI Coach John (Tamil) | UCmCAY_mStg1POKUWgMg-aGQ |
| 7 | AI with Thiru | UCY8kgTLO7GitoKuxz4Cw3uQ |
| 8 | Karthik's Show | UCBF5i6PogoMwnoAP0LFiCmQ |
| 9 | Data Engineering Tamil | UC9xghV-TcBwGvK-aEMhpt5w |
| 10 | Endless Knowledge | UCApUMSkgDT8ayJZU8jBweYw |
| 11 | Curious Freaks | UCvhU9qF1xtUsFXdKrcJxbFA |
| 12 | The AI Dude — Tamil | UCsq38VCprHHb_0rwBNfWFYw |

The synthetic 8% was generated with OpenAI gpt-4o-mini using structured JSON-mode prompting over 10 topic categories (Python, DSA, ML, ECE, web, OS, networking, databases, algorithms, debugging).

---

## Limitations

Being explicit so future work is well-targeted:

1. **Single epoch.** Trained on one epoch due to free-tier compute. Surface generation metrics (BLEU/ROUGE) need more epochs to move meaningfully. Perplexity already shows the model has learnt the distribution.
2. **Output length still verbose.** Length-ratio vs. reference is around 14× — the model has learnt Tanglish at the LM level but not the conciseness of natural Tanglish answers. Style alignment (DPO/preference learning) is the natural v2.
3. **Roman-script only.** Tamil-script Tanglish is currently filtered out by the band filter — left for v2.
4. **Domain skew.** Heavy on Python / ML / gadget reviews (driven by the source channels). Thinner on ECE and networking.
5. **No human evaluation yet.** All metrics are automatic. Human preference rating is the most impactful next step.
6. **Not safety-tuned.** Don't deploy in user-facing safety-critical settings without red-teaming.

---

## Future work (v2 roadmap)

| Phase | Focus | Effort | Expected gain |
|---|---|---|---|
| A | More data (15+ channels, Telegram, hand-curated gold) | 1–2 weeks | 4× corpus size |
| B | 3–5 epochs on `lora_r16`, longer sequences | 3–5 days | BLEU/ROUGE actually move |
| C | DPO style alignment | 2 weeks | Fixes verbose-output issue |
| D | Tamil-script Tanglish support | 1 week | Broader coverage |
| E | GGUF quantisation + streaming inference | 1 week | Production-grade |

Target venue for v2 paper: ACL / EMNLP / NeurIPS workshop.

---

## Citation

```bibtex
@misc{karan2026tamiltechqa,
  author       = {Karan E S, Dheepak},
  title        = {{TamilTech-QA}: A Tamil-English Code-Switched Benchmark for
                  Technical Question Answering with QLoRA Fine-Tuning and
                  Novel Code-Switching Evaluation Metrics},
  year         = {2026},
  month        = may,
  publisher    = {HuggingFace Hub},
  url          = {https://huggingface.co/datasets/dheepakkaran/TamilTech-QA},
  note         = {Dataset, fine-tuned model, and code. Code repository:
                  \url{https://github.com/<your-username>/TamilTech-QA}.}
}
```

## License

Code: MIT (see [LICENSE](LICENSE)).  
Dataset license: see the dataset card on HuggingFace.  
Model weights inherit Meta's Llama 3.1 Community License.

## Acknowledgements

- Base model courtesy of Meta AI (Llama 3.1).
- Training infrastructure courtesy of Kaggle (free T4 GPU).
- Live demo hosting courtesy of Google Colab.
- Source data from 12 Tamil tech YouTube channels with public-comment attribution.
- HuggingFace for the open-source ecosystem (`transformers`, `peft`, `trl`, `datasets`, `huggingface_hub`).

---

*Built by **Dheepak Karan E S** as part of independent research on code-switched NLP for Indian low-resource languages.*
