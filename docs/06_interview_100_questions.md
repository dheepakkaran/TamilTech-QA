# 🎯 Document 6 — 100 Interview Questions on TamilTech-QA (Tanglish Edition)

> **Intha doc-ku purpose:** GitHub + HuggingFace + resume la TamilTech-QA project paartha **senior technical recruiter / ML interviewer** ena ena kepparu — full 100 questions, **detailed Tanglish answers**, plus **"interviewer-ena-expect-pandraru" hint** for each.
>
> **Calibrated for:** FAANG ML interviews, AI research scientist screens, ML engineer interviews, startup ML founder-led interviews.
>
> **Reading time:** ~6 hours for full pass. Better to **practice 10-15 questions per day**.
> **Prereq:** Docs 1-5 padichirukanum. Memorization illa — **understanding** matters.

---

## How to use this doc

For each question:

1. 🗣️ **Read the question** — imagine interviewer asking it
2. 🧠 **Mentally answer in 60 seconds** — out loud, in Tanglish (English fine for tech terms)
3. 📖 **Read the suggested answer** — compare structure + depth
4. 🎯 **Read interviewer expectation** — what they're really probing
5. ✏️ **Note gaps** — concepts you fumbled, write them in margin

**Don't memorize.** Internalize the **structure**: what claim → what evidence → what tradeoff.

---

## Question Categories

| # | Category | Questions |
|---|---|---|
| **A** | Project Overview & Motivation | Q1-Q10 |
| **B** | Data Collection & Curation | Q11-Q20 |
| **C** | Preprocessing & Filtering | Q21-Q30 |
| **D** | Model & Transformer Architecture | Q31-Q40 |
| **E** | QLoRA & Fine-tuning Mechanics | Q41-Q55 |
| **F** | Evaluation & Metrics | Q56-Q70 |
| **G** | Engineering & MLOps | Q71-Q80 |
| **H** | Publishing & Deployment | Q81-Q87 |
| **I** | Research, Limitations & Future Work | Q88-Q95 |
| **J** | Behavioral & Design Defense | Q96-Q100 |

---

# 🅰️ Category A: Project Overview & Motivation (Q1–Q10)

---

## Q1. "Tell me about TamilTech-QA in 90 seconds."

**Answer (Tanglish):**

TamilTech-QA na, naan build pannina **first open-source Tanglish technical QA system**. Three components: (1) **4,415 Tanglish QA pairs** — 92% real-user YouTube comments from 12 Tamil tech channels + 8% gpt-4o-mini synthetic, (2) **QLoRA fine-tuned Llama-3.1-8B** adapter — trained on free Kaggle T4 in 6 hours, (3) **three novel evaluation metrics** — CSPS, TTR, TCF — specifically designed for code-switched output quality.

Headline result: **perplexity 78% reduction** (57 → 12.4) on Tanglish test set. Base Llama-3.1 paartha pothu, Tanglish prompt-ku Tamil-script Tamil output kuduthuchu — **register mismatch**. Fine-tune apparam, Roman-script Tanglish with English technical terms preserved — **production-grade output**.

Published on HuggingFace (dataset + model + Space) and GitHub. Total compute cost: **$0.10** (gpt-4o-mini synthetic only).

🎯 **Interviewer expectation:** They want a **structured story** — problem → solution → result → why it matters. Time-box yourself to 90s; rambling = red flag. End with a concrete metric.

---

## Q2. "Why Tanglish specifically? Why not pure Tamil or pure English?"

**Answer:**

Two reasons — **demand + gap**.

**Demand:** 80M+ Tamil-English bilingual speakers in Tamil Nadu, Sri Lanka, Singapore, Malaysia, and diaspora. Casual digital communication la **Tanglish than dominant** — WhatsApp, YouTube comments, Twitter — pure Tamil-script almost never used by under-35 Tamil techies. Production assistants (banking, customer support) need to **match this register** to feel natural.

**Gap:** Existing models fail in three ways:
1. Pure-Tamil models (IndicBERT, etc.) understand formal Tamil-script Tamil — wrong register
2. Pure-English models (Llama, GPT) ignore Tamil scaffolding — context loss
3. Code-switching research focuses on **Hindi-English** (much larger community) — Tamil-English **publicly under-resourced**

So Tanglish = real-world need + research opportunity = clean PhD-level problem statement.

🎯 **Interviewer expectation:** Show that you didn't pick the topic randomly. **Market sizing + literature gap** = research maturity.

---

## Q3. "Why technical QA as the domain? Why not casual conversation or news?"

**Answer:**

Three reasons:

1. **Evaluation tractability** — Technical QA has **objective correctness** (function explained correctly or not), unlike casual chat (no ground truth). Easier to measure improvement.

2. **English-vocab preservation requirement** — Tech domain forces the model to keep English terms (`function`, `array`, `gradient`) verbatim. This **stress-tests** the code-switching property. Casual chat la English terms minimal, kandupidikathu.

3. **Real audience** — Tamil tech YouTube has 5M+ subscribers across Brototype Tamil, Tamil Coding Wizard, etc. Comments la **authentic Tanglish technical questions** every day. Free, ethically scrapeable data source.

Casual chat la also Tanglish heavy but answer-evaluation difficult (many "right" answers). Technical QA = clean experimental setup.

🎯 **Interviewer expectation:** Show **research design thinking** — picked domain for **measurability**, not vibe. Mention "evaluation tractability" — that's grad-level reasoning.

---

## Q4. "What's the research gap you're filling?"

**Answer:**

Three specific gaps:

1. **Dataset gap:** No public Tanglish technical QA corpus existed. Dravidian-CodeMix exists but **sentiment**-focused; Cognitive-Lab Aya Tamil = monolingual. Mine = **first technical QA**.

2. **Method gap:** Code-switching fine-tuning literature is mostly **Hindi-English**. Tamil-English on **Llama-3.1 with QLoRA on free hardware** — not published before.

3. **Metric gap:** Standard NLG metrics (BLEU/ROUGE) are **insensitive to code-switching quality**. Two outputs with the same word overlap can have very different code-switching patterns. I propose **CSPS, TTR, TCF** to measure that gap.

Plus: full **reproducibility pipeline** — anyone with a Kaggle account can replicate end-to-end.

🎯 **Interviewer expectation:** They want to see you've done **literature search** and can pinpoint the specific contribution. Don't say "no work exists" — name the closest related work, then differentiate.

---

## Q5. "Who's the target end-user?"

**Answer:**

Two primary user groups:

1. **Production builders** — companies building Tamil-English chatbots, voice assistants, customer support tools. Bank-le call panrachu pothu pothu Tamil-English casual switch panra customer-ku natural reply venum. Current models default to formal Tamil-script Tamil — sounds robotic. TamilTech-QA model gives **register-matched** replies.

2. **Researchers** — anyone working on code-switching, low-resource Indic NLP, multilingual LLM evaluation. They get a benchmark + baseline + reproducible pipeline.

Secondary: **content creators** building Tanglish coding education (similar to Brototype Tamil, the dominant channel) — can integrate as Q&A backend.

🎯 **Interviewer expectation:** Distinguish between **research artifact** vs **product**. A grad-school project doesn't need a billion users — but it should serve a clearly-defined community.

---

## Q6. "How is this different from prior code-switching work?"

**Answer:**

Three differentiators:

1. **Language pair:** Most published CS work = Hindi-English (Devanagari + Latin), Spanglish (two-Latin), or Chinese-English. **Tamil-English with Roman-script Tamil** is structurally distinct — Tamil has agglutinative suffixes and a different syllable structure that breaks naïve BPE tokenizers.

2. **Method:** Earlier CS papers used full fine-tuning of mBERT or XLM-R (sequence classification). I'm doing **generative QA with QLoRA on Llama-3.1-8B** — closer to the LLM era state of the art.

3. **Novel metrics for CS quality:** Most prior work used BLEU/accuracy. I propose **three metrics specifically targeting code-switch preservation, technical term retention, and connector fluency** — capture style, not just word overlap.

Plus mine is fully open (dataset + adapter + code + demo).

🎯 **Interviewer expectation:** Show literature awareness without listing 50 papers. Pick 2-3 reference points and **explicitly contrast**.

---

## Q7. "What was your hypothesis going in?"

**Answer:**

Three hypotheses, in order of confidence:

1. **(High confidence)** A QLoRA fine-tune on a small (~5K) Tanglish dataset would shift the **output register** from Tamil-script to Roman-script Tanglish, even though Llama-3.1-8B already understands both.

2. **(Medium confidence)** Perplexity will drop substantially (>50%), but **BLEU/ROUGE will move very little** — because those metrics are style-insensitive.

3. **(Lower confidence)** Novel CS metrics (CSPS/TTR/TCF) will show **statistically meaningful** improvements that BLEU/ROUGE miss — proving the value of the metrics themselves.

🎯 **Interviewer expectation:** Show that you predicted outcomes **in advance**, not just rationalized after results came in. Stating confidence levels for each hypothesis = mature researcher.

---

## Q8. "Was the hypothesis confirmed?"

**Answer:**

Mostly yes, with one partial.

- **H1 (register shift):** ✅ Confirmed strongly. Base output: `ரியாக்ட் ஹூக்ஸ் என்பது...`. Fine-tuned: `React hooks na functional component-le state manage panrathuku...`. Dramatic register change.

- **H2 (PPL ↓, BLEU/ROUGE flat):** ✅ Confirmed exactly. PPL: 57 → 12.4 (-78%). BLEU: 0.042 → 0.045. ROUGE-L: 0.31 → 0.32. Just like predicted.

- **H3 (novel metrics show improvement BLEU misses):** ✅ **Partial**. CSPS +3.7%, TTR +8%, TCF +1.8%. Real improvements but **smaller than I'd expected**. Probably because 1 epoch is undertrained — v2 plan addresses with 3-5 epochs.

🎯 **Interviewer expectation:** They want to see **scientific honesty**. Don't oversell. Mentioning H3 as "partial" + giving the reason is **far stronger** than claiming full success.

---

## Q9. "Your headline number is '78% perplexity reduction.' Why is that the right headline?"

**Answer:**

Three reasons:

1. **It's the metric that moves the most** — and the metric directly tied to "is the model learning the distribution?". Big movement = real learning, not noise.

2. **PPL is interpretable across language pairs** — readers familiar with English LLMs immediately understand "57 → 12 means the model now finds the test set much less surprising". BLEU/ROUGE numbers (0.04 → 0.045) need explanation.

3. **It's the **conservative** claim** — if reviewer doesn't buy CSPS/TTR/TCF (because novel = unproven), PPL still stands as a defensible quantitative result.

Counter — what I **don't** lead with: "BLEU improved 7%" would mislead. BLEU went from 0.042 to 0.045 — noise. PPL = real signal.

🎯 **Interviewer expectation:** Show you can **distinguish signal from noise** and pick the right headline number ethically. Selling 7% on a noisy metric = red flag for a researcher.

---

## Q10. "Pitch this project to me as if I'm a VC."

**Answer:**

"500M+ Indians switch languages mid-sentence — and every existing LLM gets that wrong. Customer-support chatbots, banking IVRs, ed-tech tutors all need to sound **natural** in Tanglish/Hinglish, but current models output Tamil-script Tamil when the user typed Roman-script — instant disconnect.

TamilTech-QA is the **first open-source Tanglish technical assistant**, built for $0.10 of compute. We fine-tuned Llama-3.1-8B with QLoRA on real Tamil tech YouTube discourse and developed **three novel metrics** that quantify the gap that BLEU/ROUGE miss. Result: 78% perplexity drop and visibly fluent Tanglish output.

Next steps: scale to 20K samples, add **Hinglish** and **Tenglish** (Telugu-English), and partner with an Indian fintech / ed-tech to deploy in production. **Initial market: $200B India digital services**, addressable through better CS NLP. Looking for $250K for a year of compute + 1 research engineer to ship v2."

🎯 **Interviewer expectation:** Show you can speak **business language** without losing technical credibility. Numbers (500M, $200B, $250K) anchor the pitch. Don't be hand-wavy.

---

# 🅱️ Category B: Data Collection & Curation (Q11–Q20)

---

## Q11. "Why YouTube comments specifically? Why not Twitter or Reddit?"

**Answer:**

Three reasons YouTube wins for this domain:

1. **Tamil tech community lives on YouTube** — Brototype Tamil, Tamil Coding Wizard, AI with Thiru — 5M+ combined subscribers, comments la authentic Tanglish technical questions every video.

2. **API access** — YouTube Data API v3 has **10K free units/day**, structured pagination, plain-text comments. Twitter/X locked behind paid API tier (Basic $100/mo). Reddit Tamil tech subreddits = small + spam-heavy.

3. **Topic-anchored comments** — videos have a clear technical topic, so comments cluster around `python`, `web`, `ml`. Twitter is firehose, no topic structure.

Trade-off: YouTube comments are **noisy** (off-topic, profanity, emoji-only). I added a Tanglish-ratio band filter + technical-keyword presence filter to handle this.

🎯 **Interviewer expectation:** Show you considered alternatives + explicitly traded off. Not "I picked YouTube because I knew it" but "I picked YouTube because X, even though Y."

---

## Q12. "Why these 12 channels and not random Tamil channels?"

**Answer:**

Three selection criteria:

1. **Technical focus** — exclude entertainment/news. I want channels where viewers actually **discuss code** in comments.

2. **Audience size** — minimum 100K subscribers. Below that, comment volume is too low for batch processing (need ~80 videos × 200 comments to get useful samples).

3. **Active recent uploads** — within last 12 months. Stale channels = stale tech vocabulary (no GPT, no LLM references).

Final list: Brototype Tamil, Tamil Coding Wizard, Tamil Programmer, MrTT, A2D, AI Coach John, AI with Thiru, Karthik's Show, Data Engineering Tamil, Endless Knowledge, Curious Freaks, The AI Dude Tamil.

Bias caveat: All channels are **male-led**. Future v2 should add Tamil women technical creators for register diversity.

🎯 **Interviewer expectation:** Show **deliberate sampling**, not random. Bias caveat is a strong signal of statistical maturity.

---

## Q13. "Why did you also need synthetic data? Wasn't real data enough?"

**Answer:**

92% real, 8% synthetic — synthetic served two purposes:

1. **Topic coverage gaps** — YouTube heavily favors **web dev + AI** topics. **OS, networking, databases** had thin coverage. gpt-4o-mini generated balanced synthetic pairs for those.

2. **Format consistency** — YouTube comments are casual, often only the answer (no clear question). Synthetic gives structured `{question, answer}` pairs that the model needs for instruction tuning.

Why only 8%? Three concerns about pure synthetic:
- **Distribution drift**: gpt-4o-mini's idea of Tanglish is "imagined" — slightly off from real user patterns
- **Style collapse**: too much synthetic = model learns to mimic synthetic style, not real users
- **Cost** at scale: each pair = ~$0.0001, scales but adds up

So real data anchors the distribution; synthetic fills gaps.

🎯 **Interviewer expectation:** Show **calibrated trust in synthetic data**. Pure synthetic enthusiasts = naive. Anti-synthetic absolutists = also naive. The mature answer is **proportional use with explicit reasoning**.

---

## Q14. "Why gpt-4o-mini and not gpt-4o or Claude?"

**Answer:**

Three reasons gpt-4o-mini won:

1. **Cost** — 4o-mini is **~16× cheaper** than 4o. ~500 synthetic pairs at 4o-mini = $0.10; at 4o = $1.60; at Claude 3.5 Sonnet = $2-3.

2. **Sufficient quality for Tanglish generation** — 4o-mini handles instruction following + JSON mode well. Quality difference vs 4o is marginal for **simple structured generation**. The gap shows up in deep reasoning, not this.

3. **JSON mode availability** — 4o-mini supports `response_format={"type": "json_object"}` reliably. This eliminates ~30% parse-error rate I'd otherwise hit.

What I'd avoid: gpt-3.5-turbo — produces less natural Tanglish, sometimes drops Tamil tokens entirely.

🎯 **Interviewer expectation:** Show **cost-aware engineering**. ML grad students who reach for the most expensive model = red flag for production roles. Production reality = **cheapest model that meets quality bar**.

---

## Q15. "JSON mode — what is it and why was it critical?"

**Answer:**

JSON mode is an OpenAI API feature: `response_format={"type": "json_object"}` forces the model to **constrain output to valid JSON** at the decoding step.

Without it: gpt-4o-mini produces output that looks like JSON ~70% of the time but has issues:
- Trailing commas (`{"a": 1,}`)
- Missing closing braces
- Stray markdown wrappers (` ```json ... ``` `)
- Mixed Tamil-script chars breaking string escaping

With JSON mode: the model **cannot** emit invalid JSON — sampler restricts to legal next-tokens. Parse success rate ~99%+.

**Critical caveat:** JSON mode requires you to mention "JSON" explicitly in the prompt **and** request a top-level **object** (not bare array). Bare array fails — wrap as `{"pairs": [...]}`.

Without JSON mode I'd waste ~30% of my $0.10 budget on retries.

🎯 **Interviewer expectation:** Show familiarity with **production LLM tricks**, not just academic stuff. JSON mode is a real-world detail interviewers love.

---

## Q16. "How did you handle PII / personal info in YouTube comments?"

**Answer:**

Four-layer protection:

1. **API design eliminates most** — YouTube Data API returns `authorDisplayName` (public anyway) but no email, location, account history.

2. **Filtering at preprocessing** — regex-based PII scrubbing:
   - Phone numbers (`\d{10}` or `+91\s*\d{10}`)
   - Email addresses
   - URLs that look like personal profile links
   - "@username" mentions

3. **Anonymization in dataset** — I don't include `authorDisplayName` in the final published dataset. Only the comment text, video ID, topic, and Tanglish ratio.

4. **Licensing compliance** — YouTube comments are publicly posted under YouTube's ToS. Aggregating for research is consistent with Section 5 of ToS for non-commercial use.

What I could improve in v2: ML-based PII detector (presidio) for edge cases the regex misses.

🎯 **Interviewer expectation:** Show you **thought about ethics**, not just compliance. Mentioning what you'd improve = honesty signal.

---

## Q17. "How did you manage YouTube API quota?"

**Answer:**

Free tier = **10,000 units/day**. Quota costs:

| Action | Cost |
|---|---|
| `search.list` (channel discovery) | 100 units |
| `videos.list` (1 video) | 1 unit |
| `commentThreads.list` (1 page) | 1 unit |

Per channel: ~1 search + 80 videos + ~160 comment pages = ~241 units. 12 channels = **~2,892 units**. Well under 10K. ✅

Engineering:
- **State file** (`.scraped_videos.json`) — resume after partial run, no double-fetching
- **Exponential backoff** via `tenacity` — retry on 5xx
- **Per-channel throttling** — sleep 1-2s between channel scrapes

If quota hits limit: HTTP 403 → log + skip → resume next day at midnight Pacific (quota reset time).

🎯 **Interviewer expectation:** Show you **planned within a free-tier budget**. "I'd just upgrade" = junior. "Here's my quota math + state file design" = senior.

---

## Q18. "How do you know your synthetic data is actually good?"

**Answer:**

Three validation layers:

1. **Tanglish-ratio band check** — synthetic pairs must score within 0.05-0.95 like real data. Reject if outside.

2. **Technical keyword presence** — each pair must contain ≥1 keyword from the technical lexicon (function, array, gradient, etc.). Rejects vapid output.

3. **Manual review of N=50 random samples** — I read every 10th generated pair to spot drift. Found:
   - 5/50 too English-heavy (synthetic style collapse) → tightened prompt
   - 2/50 factually wrong technical content → added "Output must be technically accurate" to system prompt
   - 0/50 toxic / inappropriate → safe

After tuning prompts: re-generated, re-validated.

Limitation: I haven't done a **rigorous A/B** — train two models (with vs without synthetic) and compare. v2 will.

🎯 **Interviewer expectation:** Show **multiple validation strategies** + acknowledge ablation as a gap. Saying "I haven't done the A/B yet, here's the plan" is **stronger than claiming full validation**.

---

## Q19. "Any licensing concerns with YouTube comments in a published dataset?"

**Answer:**

Yes — I thought hard about this.

**YouTube ToS Section 5** says users grant YouTube a worldwide license to host their content. **Researcher's status:** allowed to extract publicly visible content for non-commercial research, but **redistribution** of the verbatim comments is grayer.

What I did:
1. **Aggregated as a derived dataset** — not raw comments. Each entry is (cleaned question, cleaned answer, topic, ratio) — strips formatting cues that identify the original poster.
2. **No author names** — anonymized at the data layer
3. **MIT license** with explicit attribution to YouTube as the source
4. **Removability** — the dataset README has a takedown email address; if a creator objects, I'll remove their content within 7 days
5. **Non-commercial framing** — research artifact, not commercial product

For v2 going to formal datasets like LDC: I'd consult IRB and get formal consent. v0.1 is fair-use research scope.

🎯 **Interviewer expectation:** Show you **considered legality + ethics**, not just "I scraped because it was available." Removal policy = mature signal.

---

## Q20. "Could you scale this to 100K samples? What would change?"

**Answer:**

Yes, with three changes:

1. **Data sources expansion** — current 12 channels would top out around 15K. Add:
   - Telegram Tamil tech groups (~20+ active)
   - Twitter/X Tamil tech accounts
   - Stack Overflow Tamil tags
   - Discord Tamil dev servers (with permission)
   - Reach 100K-200K raw comments → ~30-50K after filtering

2. **Synthetic generation scale-up** — current $0.10 budget gave 500 pairs. To reach 70K synthetic = ~$15. Trivial cost.

3. **Engineering changes:**
   - Move scraping from local Python to a Beam/Spark pipeline
   - Replace exact dedup with **MinHash LSH** (now justified at 100K scale)
   - Add a **classifier-based Tanglish detector** (current lexicon-based hits precision ceiling)
   - Stratified sampling to maintain topic balance

Quality risk at scale: distribution drift toward whatever new sources I add. Mitigation: hold out a "real-user golden test set" (~500 hand-curated) — measure on this regardless of training data growth.

🎯 **Interviewer expectation:** Show **scalability thinking**. Don't just say "yes I'd scale" — name the **specific bottleneck** (current = exact dedup, lexicon detector) and the upgrade.

---

# 🅲️ Category C: Preprocessing & Filtering (Q21–Q30)

---

## Q21. "How do you compute the Tanglish ratio for a sample?"

**Answer:**

Algorithm (simplified):

```python
def tanglish_ratio(text):
    tokens = re.findall(r"[A-Za-z஀-௿][A-Za-z஀-௿\-']*", text)
    if not tokens: return 0.0
    tamil_count = sum(1 for t in tokens if is_tamil(t))
    return tamil_count / len(tokens)

def is_tamil(token):
    if re.search(r"[஀-௿]", token): return True       # Tamil script
    if token.lower() in TAMIL_LEXICON: return True   # Roman-script Tamil word
    if any(token.lower().endswith(s) for s in TAMIL_SUFFIXES):
        return token.lower() not in ENGLISH_GUARD    # ends with -a, -na, etc.
    return False
```

**Three-criteria union:** script, lexicon, suffix. Union (OR) maximizes recall; `ENGLISH_GUARD` (schema, java, lambda, alpha, data...) prevents false positives.

**Lexicon source:** Hand-curated ~60 most-frequent Roman-script Tamil function words and connectors (naan, indha, panrathu, athu, ippo...). I prioritized coverage of **function words** over content words because Tamil content words vary a lot but function words anchor sentence structure.

🎯 **Interviewer expectation:** Show **algorithmic detail without code-dumping**. Mention precision vs recall tradeoff for the lexicon design.

---

## Q22. "Why a 5%-95% band? Why not tighter (15%-85%)?"

**Answer:**

I tried both. The **tight band (15%-85%)** comes from common code-switching literature heuristics. **Loose band (5%-95%)** is what I shipped in v0.1.

**Why loose won:**

| Band | Train size | Notes |
|---|---|---|
| Tight (15-85%) | ~2,400 samples | Drops too much real-user data |
| Loose (5-95%) | **4,415 samples** ✅ | Keeps borderline natural samples |
| None (0-100%) | ~7,500 raw | Includes pure English + pure Tamil pollution |

**Trade-off:** Looser = more data + more diversity but also some samples that are 95% English with one Tamil word. Tighter = purer Tanglish but throws away real users who casually write 90% English with 2-3 Tanglish injections.

**Empirical evidence for loose:** Test-set evaluations were better with model trained on loose. More data > more purity at this scale.

v2 plan: train two models — one on each band — and ablate properly.

🎯 **Interviewer expectation:** Show you **made a data-driven choice**, not "I just picked a number." Mentioning the planned ablation = transparent about what you haven't yet proven.

---

## Q23. "Why drop Tamil-script samples? Aren't they also Tanglish?"

**Answer:**

Strictly: Tamil-script (`நான்`, `வணக்கம்`) is **pure Tamil**, not Tanglish. Tanglish in this project = **Roman-script Tamil + English**.

Two reasons:

1. **Distribution focus** — real-user Tanglish on YouTube/WhatsApp/Twitter is **overwhelmingly Roman-script** for users under 40. Targeting that population.

2. **Tokenization efficiency** — Llama-3.1's BPE handles Latin chars efficiently but **Tamil-script breaks into 3-5 tokens per word**. Mixing would degrade training efficiency.

But: real users **do** sometimes mix Tamil-script (older users, formal contexts). v2 plan: build a **second model** specifically for Tamil-script Tanglish, then optionally route based on user input.

**Limitation acknowledgment** I include in the model card: "Tamil-script outputs not supported in v0.1."

🎯 **Interviewer expectation:** Show **scoping discipline**. Trying to handle both scripts in v0.1 = scope creep. Acknowledging the limitation = research honesty.

---

## Q24. "Lexicon vs ML classifier for Tanglish detection — why lexicon?"

**Answer:**

Three reasons for lexicon-first:

1. **Bootstrapping problem** — to train an ML classifier I'd need labeled Tanglish data, which is exactly what I'm collecting. Chicken-and-egg.

2. **Latency** — lexicon lookup is O(1) per token. At 50K samples × 100 tokens each = 5M ops, instant. ML classifier (BERT-based) would be hundreds of GPU-seconds.

3. **Interpretability** — when a sample is rejected, I can point to **which tokens classified it** as Tamil/English. ML classifier = black box, harder to debug.

**Lexicon limits:**
- Misses **rare** Tamil words not in the ~60-entry list
- Can't handle **transliteration variants** ("naan" vs "naanu" vs "naangal")
- False positives on English words ending in Tamil-suffix-like patterns

v2 plan: **hybrid** — use lexicon as fast first pass, then a small XLM-R classifier for borderline cases (e.g., ratios in 0.05-0.15 or 0.85-0.95).

🎯 **Interviewer expectation:** Show **classical NLP + modern ML mix**. ML-only enthusiasts = naive. Lexicon-only also naive. Hybrid for the right reason = mature.

---

## Q25. "Walk me through your deduplication strategy."

**Answer:**

Two-phase strategy:

**Phase 1: Exact dedup** by key `(question.strip().lower(), answer.strip().lower())`:

```python
seen = set()
deduped = []
for r in records:
    key = (r["question"].strip().lower(), r["answer"].strip().lower())
    if key in seen: continue
    seen.add(key)
    deduped.append(r)
```

Removed ~5% (350-ish exact duplicates — same comment re-uploaded or reposted across videos).

**Phase 2 (deferred for v2): Near-dedup via MinHash LSH** for paraphrases. Threshold 0.85 Jaccard similarity. Would remove "Same question reworded" pairs.

**Why I skipped MinHash in v0.1:** At 7,500 records, MinHash LSH init was **slow on Kaggle CPU** (>30 min hang) due to library overhead. Exact dedup at this scale = 95% of the value at 5% of the time.

v2 with larger data: MinHash justified.

🎯 **Interviewer expectation:** Show **simplest tool that works**. Reaching for fancy MinHash on 7K records = premature optimization. Saving it for the 100K scale = right call.

---

## Q26. "Why exact dedup and not just MinHash?"

**Answer:** (See Q25)

Three reasons exact was right for v0.1:

1. **Scale** — 7,500 records; MinHash justifies itself only above ~50K (LSH index has fixed overhead).
2. **Speed** — exact dedup runs in milliseconds; MinHash LSH took 30+ min on Kaggle CPU.
3. **Sufficient signal** — most "dupes" in YouTube comments are exact reposts, not creative paraphrases.

Edge case I missed: **exact** dedup ignores whitespace/punctuation variations. "Anna" vs "Anna." Could be near-dupes. I lowered + stripped both fields before hashing, which catches some but not all.

🎯 **Interviewer expectation:** Same as Q25 — **scale-appropriate tool choice**.

---

## Q27. "Why min_words ≥ 10 in the length filter?"

**Answer:**

Three reasons for the floor:

1. **Below 10 words = no real QA structure** — typical: "+1", "good video", "thank you anna". No question or answer content.

2. **Pre-training of Llama already covers short texts** — the value-add from fine-tuning comes from **longer, structured discourse**. Short snippets don't shift the distribution.

3. **Tokenization minimum** — short text → too few tokens for meaningful gradient signal. Effective batch of 16 samples with 5 tokens each = noisy.

I picked 10 empirically. Bumping to 15 dropped useful samples; lowering to 5 added noise. 10 sweet-spotted.

Upper bound: I didn't set an explicit one, but `max_seq_length=512` during training truncates ultra-long samples. p95 of my training set was ~380 tokens, comfortably under 512.

🎯 **Interviewer expectation:** Show **filter thresholds are chosen, not guessed**. Mentioning the p95 = real engineering rigor.

---

## Q28. "How did you split train/val/test?"

**Answer:**

80/10/10 stratified split:

- **Train:** 3,536 (80%)
- **Validation:** 431 (10%) — used during training for eval_loss, early stopping
- **Test:** 447 (10%) — held out, only touched at final eval

Stratification keys: `topic` (python, ml, web, ...) + `difficulty` (beginner, intermediate, advanced — assigned during synthesis or estimated for YouTube).

**Why stratify:** Avoid the failure mode where train has 90% python and test has 70% networking → metrics look bad just due to distribution mismatch, not actual model quality.

**Implementation:** `sklearn.model_selection.train_test_split` with `stratify=df[["topic", "difficulty"]]` (joint stratification).

Could improve: **temporal stratification** — newer YouTube comments in test set to simulate "future drift". I didn't do this in v0.1.

🎯 **Interviewer expectation:** Show **proper experimental hygiene** — stratification, no peek at test. Bonus for mentioning temporal split as future improvement.

---

## Q29. "What's the risk of data leakage between train and test?"

**Answer:**

Three leakage vectors I controlled for:

1. **Exact duplicates** — dedup ensures no sample appears in both train and test. Done via tuple-key hash before splitting.

2. **Near-paraphrase leakage** — slightly differently-worded versions of the same QA could land in different splits. v0.1 doesn't catch this — flagged as v2 improvement (MinHash will catch).

3. **Source leakage** — same YouTube video's comments could span splits. **Risk:** model overfits to that video's vocabulary. **Mitigation:** stratification is by topic, not video — so a video's comments **can** span splits. Not perfect; v2 = video-level split.

4. **Synthetic-template leakage** — if synthetic uses similar templates across pairs and pairs span splits, model could learn the template rather than concepts. **Mitigation:** I varied prompt phrasing for synthetic and limited to 8% of total. Risk = low.

🎯 **Interviewer expectation:** Show **proactive leak hunting**, not just reactive fixes after metrics look suspicious. Naming **specific vectors** is far stronger than vague "I made sure no leakage."

---

## Q30. "Walk me through one full sample row in your dataset."

**Answer:**

```json
{
  "id": "yt_xYzAbC_0042",
  "source": "youtube",
  "channel_id": "UCIFQgj1Rhx-FFgyo0zzPSfw",
  "video_id": "xYzAbC",
  "topic": "web",
  "difficulty": "intermediate",
  "question": "Anna React la useEffect hook eppo run aagum?",
  "answer": "useEffect na component render aana udane run aagum. Default-a every render-ku run aagum. Dependency array [] potha first render only-le run aagum. [count] potha count change pannina pothu only run aagum. Cleanup function-a return panna unmount time-le clear aagum.",
  "tanglish_ratio": 0.42,
  "tanglish_confidence": 0.78,
  "language_tags": ["ta", "en"],
  "tamil_word_count": 18,
  "total_word_count": 43
}
```

**Per-field provenance:**
- `id`: deterministic, resume-safe
- `source`: enables per-source ablation
- `topic`/`difficulty`: stratification keys
- `tanglish_ratio`: pre-computed (avoids re-tokenization downstream)
- `language_tags`: lightweight signal for filtering

🎯 **Interviewer expectation:** Show **schema design thinking**. Each field exists for a reason.

---

# 🅳️ Category D: Model & Transformer Architecture (Q31–Q40)

---

## Q31. "Why Llama-3.1-8B and not Mistral-7B?"

**Answer:**

Both viable; Llama-3.1-8B won on three axes:

1. **Multilingual coverage** — Llama-3.1 pre-training included **~5% non-English** (multiple Indic languages). Mistral-7B is more English-skewed. For Tanglish, Llama starts with **better Tamil token coverage**.

2. **Context length** — Llama-3.1 supports **128K context** natively (vs Mistral's 32K via sliding-window). Future-proofs for long-context tasks.

3. **Tokenizer** — Llama-3.1's 128K vocab handles Tamil characters; Mistral's 32K vocab is tighter.

But Mistral has advantages I didn't need:
- More efficient inference (smaller hidden dim)
- Smaller footprint at deployment

**For v0.1**, the multilingual advantage of Llama mattered most. Mistral remains a candidate for v2 ablation.

🎯 **Interviewer expectation:** Show **multiple-axis tradeoff** thinking. Single-axis reasoning ("Llama is bigger") = junior.

---

## Q32. "Why not just use GPT-4 directly?"

**Answer:**

Five reasons:

1. **No fine-tuning openly available** — OpenAI doesn't allow you to download GPT-4 weights, so you can't truly **fine-tune for register**. They have a fine-tuning API for some models, but limited.

2. **Cost at inference** — GPT-4 = $30/1M input tokens. A Tanglish chatbot serving 100K queries/day = ~$3K/day. My fine-tuned Llama-3.1-8B = self-hosted, $0 per query.

3. **Vendor lock-in / IP** — research artifact has to be **reproducible by others** without paying OpenAI. Open-weight model = reproducible science.

4. **Latency** — API calls add 200-500ms; self-hosted Llama on a small GPU = 50-100ms.

5. **Privacy** — production users in India may not want their queries going to OpenAI servers in the US.

GPT-4 served a different role here — as a **synthetic data generator** (gpt-4o-mini for cost) — but not as the deployed system.

🎯 **Interviewer expectation:** Show **research vs production framing**. GPT-4 is great for some use cases; for an **open research artifact**, open-weight is the principled choice.

---

## Q33. "Explain the transformer architecture in 2 minutes."

**Answer:**

Transformer = stack of identical **layers**, each with two sub-blocks: **self-attention** + **feed-forward MLP**, with residual connections + layer norm.

**Input:** tokens → embeddings (lookup table). Positional info added (RoPE in Llama).

**Each layer:**
1. **Multi-head self-attention** — each token computes 3 vectors (Query, Key, Value). Attention(Q, K, V) = softmax(QK^T / √d) · V. Multiple heads (32 in Llama-3.1) attend to different patterns.
2. **Feed-forward** — token-wise MLP with intermediate dim 14,336. Uses SwiGLU activation in Llama.
3. **Residual + norm** around each sub-block.

**Output of final layer:** projected to vocabulary size (128K in Llama-3.1) → softmax → next-token probability distribution.

**Training objective:** predict the next token at every position. Standard cross-entropy loss.

Llama-3.1-8B has **32 layers** stacked. ~8B total params.

🎯 **Interviewer expectation:** Show **architectural fluency** without being a textbook. End with a concrete number (32 layers, 8B) to anchor the abstract description.

---

## Q34. "How does attention work in Llama specifically?"

**Answer:**

Llama-3.1 uses **Grouped Query Attention (GQA)** with **Rotary Position Embedding (RoPE)**.

**Standard multi-head attention (32 heads):** Each head has its own Q, K, V projection. Total params = 32 × 3 × hidden_dim × head_dim = a lot.

**GQA optimization in Llama-3.1:** 32 query heads share **8 KV head groups** (4:1 ratio). Memory for KV cache during inference is 4× smaller while quality stays close to full MHA.

**RoPE (Rotary Position Embedding):** Instead of adding positional embeddings, RoPE **rotates** Q and K vectors as a function of position. Two key properties:
- Relative position info is preserved in `Q · K^T`
- Extrapolates to longer contexts than training (essential for 128K context)

**Attention computation:**
```
Q_rot = rotate(Q, position)
K_rot = rotate(K, position)
Attention = softmax(Q_rot · K_rot^T / √d_head) · V
```

For Tanglish specifically: nothing special — same attention. Code-switching is handled by **learned weights**, not architecture.

🎯 **Interviewer expectation:** Show **deep familiarity with Llama specifically** — GQA + RoPE are non-trivial details. Acknowledging "nothing special for Tanglish" = honesty.

---

## Q35. "How does the tokenizer handle Tanglish?"

**Answer:**

Llama-3.1's tokenizer is **BPE (Byte-Pair Encoding)** with 128,256 tokens.

For pure English: typically **1 word = 1-2 tokens** efficient.

For Tanglish:
- **Roman-script Tamil words** break into 2-4 sub-tokens. "panrathu" might tokenize as `["pan", "rath", "u"]`.
- **English tech terms** stay as 1-2 tokens.
- **Tamil-script** (which we don't use much) is **3-5 tokens per word**.

**Implication:**
- Tanglish prompts use **20-30% more tokens** than equivalent English prompts.
- Context window fills faster.
- Cost (if using API) is higher per query.

**No special tokens added** for Tanglish in v0.1 — chose to rely on Llama's existing vocab. v2 idea: train a small **vocabulary extension** with the 200 most-frequent Tanglish sub-strings.

🎯 **Interviewer expectation:** Show **quantitative understanding** of tokenization cost. Mention the 20-30% overhead — production teams care about this.

---

## Q36. "Why did you disable KV cache during training (`use_cache=False`)?"

**Answer:**

KV cache is an inference-time optimization — stores K and V vectors from previous tokens to avoid recomputing.

During **training**:
- KV cache **conflicts with gradient checkpointing**. Gradient checkpointing **drops** activations to save memory; KV cache wants to **keep** them. Incompatible.
- You only need cache for autoregressive generation (one new token at a time). During training, the full sequence is processed in parallel; cache provides no speedup.

Therefore: `model.config.use_cache = False` is **mandatory** before enabling gradient checkpointing.

At **inference time** (after training): re-enable cache for fast generation.

🎯 **Interviewer expectation:** Show you know **why one flag interacts with another**. ML engineers who don't understand these interactions hit subtle bugs.

---

## Q37. "Llama-3.1 has no pad token. How did you handle padding?"

**Answer:**

Llama-3.1 ships with no `<pad>` token in its vocabulary — unlike BERT etc. This breaks batching by default.

**Three options:**

1. ✅ **Reuse EOS as pad** (my choice): `tokenizer.pad_token = tokenizer.eos_token`. Works because attention mask still tells the model which positions are real.

2. **Add a new `<pad>` token:** `tokenizer.add_special_tokens({"pad_token": "<|pad|>"})`. Forces `model.resize_token_embeddings(len(tokenizer))`. Adds a row to the embedding matrix that has random init — needs training to be meaningful.

3. **Use UNK as pad:** unreliable; UNK is rare but not pad-specific.

**Why option 1 is fine:** EOS during training is masked from the loss (via `pad_token_id`); model never confuses it. Common HF idiom.

Padding side: `right` (Llama default). Left padding is for generation; right padding is for training/loss computation.

🎯 **Interviewer expectation:** Show you understand the **interaction** between pad token, attention mask, and loss computation. Bonus: explain why right vs left padding matters.

---

## Q38. "Why ChatML format and not just instruction templates?"

**Answer:**

Llama-3.1-Instruct's pre-training **used ChatML-like format with specific special tokens** (`<|begin_of_text|>`, `<|start_header_id|>`, `<|end_header_id|>`, `<|eot_id|>`).

If I use a custom format (e.g., Alpaca: `### Instruction:\n...\n\n### Response:\n...`):
- Model wastes capacity re-learning a new format
- Special tokens still appear in pre-training distribution and conflict
- Inference at deployment time has to use the same custom format = friction

If I use **the tokenizer's native chat template**:
- Aligns with pre-training distribution → faster convergence
- `tokenizer.apply_chat_template()` handles all the special tokens correctly
- Deployment-time inference matches what the model expects

I used `tokenizer.apply_chat_template()` — never hand-wrote the format. **Best practice for instruct-tuned models.**

🎯 **Interviewer expectation:** Show **modeling discipline** — match training distribution. Custom formats are an anti-pattern for instruct-tuned models.

---

## Q39. "Llama-3.1 has 128K vocab. Why so big? Does it help Tanglish?"

**Answer:**

Llama-3.1 expanded vocab from Llama-2's 32K to **128,256 tokens** primarily for **multilingual coverage**.

**Helps Tanglish in two ways:**

1. **More Tamil sub-tokens** in the vocab → fewer pieces per Tamil word. Llama-2 might split `panrathu` as `[p, an, ra, th, u]` (5 tokens); Llama-3.1 likely `[pan, rathu]` (2 tokens). Roughly **2-3× efficiency gain** for Tamil.

2. **Latin sub-words like "ku", "la", "na"** (common Tanglish suffixes) get their own tokens in Llama-3.1. Better composition.

**Cost:** Embedding matrix grows from 32K × 4096 → 128K × 4096. ~525M params just in embeddings (out of 8B total).

But it's **shared with the LM head output projection** (weight tying), so practical overhead is reasonable.

For Tanglish, 128K vocab is a **direct architecture-level win**.

🎯 **Interviewer expectation:** Connect architecture choice (large vocab) → downstream effect (Tanglish efficiency). Mentioning weight tying = bonus depth.

---

## Q40. "What's new in Llama-3.1 vs Llama-3.0 or Llama-2?"

**Answer:**

Key Llama-3.1 changes:

1. **128K context window** (vs 8K in Llama-3.0, 4K in Llama-2). Achieved via RoPE scaling tricks.
2. **More multilingual training data** — explicit improvements for ~8 non-English languages, including some Indic coverage.
3. **Improved instruction tuning** — better at structured output, tool use, JSON.
4. **Tokenizer extended** — same 128K size as Llama-3.0 but with refined merges.
5. **License change** — Llama-3.1's license permits commercial use up to 700M MAU (similar to Llama-3.0).

For TamilTech-QA, the **multilingual data** + **larger context** mattered most. 128K context lets me potentially scale to **document-level Tanglish QA** in v2 (currently capped at 512).

vs Llama-2: vocab change alone makes Llama-3.1 significantly better for Indic. Llama-2 has 32K vocab; Tamil words split into far more sub-tokens.

🎯 **Interviewer expectation:** Show you **track model progress** at a technical level. Mentioning license + commercial usability = product-savvy researcher.

---

# 🅴️ Category E: QLoRA & Fine-tuning Mechanics (Q41–Q55)

---

## Q41. "What is QLoRA in one minute?"

**Answer:**

QLoRA = **Quantized Low-Rank Adaptation**. Two ideas combined:

1. **Quantize the base model to 4-bit** — Llama-3.1-8B drops from 16 GB (fp16) to ~4 GB (4-bit NF4). Frozen, never updated.

2. **Train small LoRA adapter modules** alongside the frozen base — in fp16. Adapters insert into attention layers: `ΔW = (α/r) · B @ A` where A is (r × d), B is (d × r). Only ~3.4M params trainable (0.04% of 8B).

**Training:** forward pass dequantizes 4-bit weights on-the-fly for compute, applies adapter, backprop only flows through adapter weights.

**Memory profile during training:**
- Base 4-bit: 4 GB
- Adapter: <100 MB
- Gradients + optimizer states (8-bit AdamW): ~400 MB
- Activations (with checkpointing): ~5 GB
- **Total ~10 GB** → fits Kaggle T4 (16 GB) ✅

**Result:** ~99% of full fine-tuning quality at ~10% of the memory.

🎯 **Interviewer expectation:** Tight, technical, end with the magic memory number. Don't ramble through history.

---

## Q42. "QLoRA vs full fine-tuning — when would you pick which?"

**Answer:**

| Scenario | Pick |
|---|---|
| Free / cheap GPU | QLoRA |
| Adapting to new domain (register, style) | QLoRA — sufficient |
| Adapting to brand-new task (multi-step reasoning) | Full FT may be needed |
| Adapter must be lightweight (deploy with base) | QLoRA wins |
| Have A100s and unlimited budget | Either |
| Need to **change facts** the model knows | Full FT |
| Need to **change behavior** | QLoRA |

**TamilTech-QA's case:** Llama-3.1 already **knows English technical content** and **understands Roman-script Tamil tokens**. I'm changing **output style/register**, not adding new knowledge. → **QLoRA is sufficient**.

If I were training Llama-3.1 to do, say, novel multi-step symbolic reasoning it had never seen → full FT might be needed.

🎯 **Interviewer expectation:** Show **decision criteria**, not just a preference. Linking the choice to "what's changing — behavior vs knowledge" is the senior-level framing.

---

## Q43. "Why r=8 for LoRA rank? Why not r=4 or r=64?"

**Answer:**

r is the **intrinsic dimensionality** of the fine-tuning update. Larger r = more capacity but more params.

| Rank | Params trainable | Pros | Cons |
|---|---|---|---|
| r=4 | ~1.7M | Fastest, lowest mem | Risk under-capacity |
| **r=8** ✅ | **3.4M** | Sweet spot empirically | — |
| r=16 | 6.8M | Better quality typically | More mem, longer train |
| r=64 | 27M | Diminishing returns | Often no better than r=16 |

I picked r=8 because:
1. **Dettmers QLoRA paper** showed r=8-32 covers most use cases without major quality gap.
2. **Tanglish is a register shift, not a hard task** — capacity needs aren't huge.
3. **Kaggle T4 time budget** — r=8 trains in ~6 hrs; r=16 estimated ~8 hrs (less safety margin within 12 hr session).

v2 plan: train r=16 and ablate. If quality jumps significantly → adopt r=16 + also targeting FFN modules.

🎯 **Interviewer expectation:** Cite the paper + give the time argument. Pure intuition arguments are weaker than time-budget + literature.

---

## Q44. "What does `lora_alpha` do?"

**Answer:**

`lora_alpha` scales the LoRA update:
```
ΔW = (α / r) × B @ A
```

`α` and `r` together control the **magnitude** of the update.

**Convention from the LoRA paper:** `α = 2r` (so scaling factor = 2). This is **empirically stable** across many tasks — adopted by HuggingFace defaults and most published work.

I used `α=16` with `r=8` (i.e., 2r). No experimentation needed.

**Why does scaling matter?**
- Too low: adapter contribution diluted; model behaves like base
- Too high: adapter overpowers base; training unstable

The `α/r = 2` scaling keeps the update magnitude roughly in the same range across different rank choices — so you can change r without re-tuning learning rate.

🎯 **Interviewer expectation:** Show you know the convention (`α = 2r`) and **why** it exists, not just that it's a magic constant.

---

## Q45. "Why apply LoRA to q_proj, k_proj, v_proj, o_proj — and not the FFN?"

**Answer:**

These are the **attention projection matrices** in each transformer layer:
- `q_proj` (Query)
- `k_proj` (Key)
- `v_proj` (Value)
- `o_proj` (Output projection from attention)

Why these in v0.1:

1. **LoRA paper found attention projections capture most of the gain** — adding FFN modules helps marginally but doubles trainable params.

2. **Memory budget** — adding FFN (`gate_proj`, `up_proj`, `down_proj`) would push trainable params from 3.4M → ~10M. Tight on T4.

3. **Style/register tasks tend to live in attention** — relationships between tokens are what get re-tuned. Factual knowledge lives more in FFN.

For Tanglish, register shift = attention re-weighting (which tokens to attend more to) → attention LoRA suffices.

v2 plan: ablate `lora_r16_full` profile that also includes FFN — see if technical content improves.

🎯 **Interviewer expectation:** Show **which modules carry which signal** (attention = relationships, FFN = facts). It's a finding from recent interpretability work.

---

## Q46. "Walk me through 4-bit quantization math."

**Answer:**

4-bit per weight → 2^4 = 16 possible values per weight. **Block-wise** quantization:

For each block of 64 weights:
```
1. Find block's value range (typically uses absolute max)
2. Compute scale: scale = max(|block|) / 7   (for symmetric 4-bit in [-8, 7])
3. Quantize: q_w = round(w / scale)  → 4-bit integer
4. Store: 64 × 4 bits + 1 × fp32 scale = 32 + 4 = 36 bytes per 64 weights
```

vs fp32 = 64 × 4 = 256 bytes → **~7× compression**.

**Dequantization at compute time:**
```
w_approx = q_w × scale  →  fp16 for matmul
```

QLoRA's NF4 variant uses **non-uniform buckets** matched to the normal distribution of LLM weights (more buckets near zero where most weights live, fewer in tails). Better quality than uniform 4-bit.

**Double quantization** = quantize the block scales themselves into 8-bit + a secondary scale. Saves another ~0.4 bits/param.

🎯 **Interviewer expectation:** Math-fluent answer. Bonus: mention NF4 is **distribution-aware** — that's the QLoRA-specific innovation.

---

## Q47. "NF4 vs FP4 — what's the difference?"

**Answer:**

Both are 4-bit but different value distributions:

**FP4 (Float4):** Uniform spacing of 16 buckets. Equally spaced from -∞ to ∞. Naive.

**NF4 (NormalFloat-4):** Buckets **placed using quantiles of a standard normal distribution**. More buckets near zero (where most LLM weights are), fewer in tails.

NF4 bucket values (approximate):
```
[-1.0, -0.70, -0.53, -0.40, -0.28, -0.18, -0.09, 0.0,
  0.08, 0.16, 0.25, 0.34, 0.45, 0.58, 0.74, 1.0]
```

**Why NF4 works:** LLM weight distributions are roughly Gaussian. Putting more quantization granularity where weights cluster → less rounding error → less quality loss.

Empirically (Dettmers paper): NF4 yields **measurably lower perplexity drop** vs FP4 on standard benchmarks.

**Trade-off:** NF4 dequantization is slightly more expensive (lookup table) but negligible at training time.

🎯 **Interviewer expectation:** Connect **distribution awareness** to quantization quality. This is recent research detail — knowing it = staying current.

---

## Q48. "Why double quantization?"

**Answer:**

In block-wise 4-bit, each block of 64 weights has its **own fp32 scale** (4 bytes).

For Llama-3.1-8B: ~8B weights / 64 per block = 125M blocks. Each scale = 4 bytes → **500 MB just for scales**.

**Double quantization** = quantize those scales themselves:
1. Group 256 block scales into a "scale block"
2. Quantize them with **8-bit** precision + a secondary fp32 scale per group of 256

Memory savings:
- Before: 500 MB of fp32 scales
- After: 125 MB of 8-bit scales + tiny secondary scales

Total saving: ~375 MB on Llama-3.1-8B. ~0.4 bits per param effective.

**Quality cost:** Negligible — secondary scale precision is much finer than needed since first-level scales are already small magnitudes.

So **effective bits per param ≈ 3.6** instead of 4. Free win.

🎯 **Interviewer expectation:** Show **multi-level optimization thinking**. Senior engineers compress the compression.

---

## Q49. "Paged AdamW 8-bit — what does it do?"

**Answer:**

AdamW optimizer stores **3 fp32 states per param**: momentum (m), variance (v), and the param itself for updates. For Llama-3.1-8B, that's 24 GB of fp32 state for full FT.

For QLoRA, we only train **3.4M LoRA params**, so optimizer state is ~40 MB in fp32. Small.

**8-bit AdamW** (bitsandbytes): stores m and v in 8-bit using block-wise quantization. Saves ~75% on optimizer state memory. 40 MB → 10 MB.

**Paging** (Linux memory paging concept applied to GPU):
- When GPU runs low on memory, optimizer state can be **swapped to CPU** transparently
- When needed, swapped back in
- Smooths out memory spikes

For TamilTech-QA with already-tight VRAM, paged AdamW provides a **safety margin** against transient memory peaks during backward pass.

```python
optim="paged_adamw_8bit",  # in TrainingArguments
```

🎯 **Interviewer expectation:** Connect optimizer state to **memory budget** discussions. Mentioning paging = bonus depth.

---

## Q50. "Gradient checkpointing — what is it and why does it help?"

**Answer:**

Normal training stores **all activations** during forward pass for use during backward. Memory grows linearly with depth × seq_len × hidden_dim.

For Llama-3.1-8B with seq=512, batch=2:
- Activations ≈ 32 layers × 512 × 4096 × 4 bytes × 2 = ~270 MB / batch (before factoring in attention scores)
- Plus attention scores: 32 × 32 × 512 × 512 × 2 bytes ≈ another ~500 MB
- Total: ~5 GB on T4

**Gradient checkpointing:** Don't store all activations. Instead:
- Forward pass stores only a **subset** of intermediate activations (typically at layer boundaries)
- During backward: **recompute** missing activations from the nearest checkpoint

**Trade-off:**
- Memory: ~30-50% reduction
- Compute: ~20-30% slower (recompute cost)

For T4 with 16 GB, the speed-for-memory trade is non-negotiable. Enable it:
```python
model.gradient_checkpointing_enable()
```

🎯 **Interviewer expectation:** Time-vs-memory trade is one of the most important LLM training concepts. Numbers (30-50% mem, 25% slower) anchor the answer.

---

## Q51. "Why bf16 and not fp16 — and what bug did this cause?"

**Answer:**

bf16 (brain float 16) and fp16 are both 16-bit floats but differ in **mantissa vs exponent allocation**:

| Format | Sign | Exponent | Mantissa | Range | Precision |
|---|---|---|---|---|---|
| fp16 | 1 | 5 | 10 | smaller | higher |
| bf16 | 1 | 8 | 7 | larger (same as fp32) | lower |

bf16 has the **same exponent range as fp32** → no underflow issues. Quality loss is minimal for LLMs.

**The Llama-3.1 bug:** Llama-3.1 weights are stored in **bf16** natively. If I configure training with `fp16=True`:
- The fp16 GradScaler is invoked to prevent gradient underflow
- It tries to convert bf16 gradients to fp16 for scaling
- **Raises `NotImplementedError: BFloat16 is not supported in fp16 grad scaler`**

**Fix:** Set `bf16=True, fp16=False` — uses bf16 native mixed precision, no scaler needed (bf16 range avoids underflow).

🎯 **Interviewer expectation:** Show **understanding of the bug at the format level**, not just "set bf16=True". Knowing the gradient scaler's role = production-ready.

---

## Q52. "How did you choose learning rate 2e-4?"

**Answer:**

I didn't experiment — used the **community-standard QLoRA LR**.

**Reasoning behind the standard:**

- Full fine-tuning of large models: **1e-5 to 5e-5** (small steps, all params)
- LoRA: **1e-4 to 5e-4** (only a few % of params train; can take larger steps without instability)
- QLoRA: same range as LoRA — **2e-4 is the most-cited starting point**

**Why not higher (e.g., 1e-3):** Loss can spike / NaN with high LR on long sequences.

**Why not lower (e.g., 1e-5):** Slower convergence; in 1 epoch wouldn't move enough.

**Did I validate it was right?** Indirectly: loss curve was smooth (no spikes), final eval_loss dropped 1.55 → 0.97 monotonically. Looked healthy.

For v2 I plan to sweep LR over {1e-4, 2e-4, 5e-4} to find optimum for this dataset specifically.

🎯 **Interviewer expectation:** Acknowledge using a **community default** but justify it. Naïve answer: "I just picked 2e-4." Mature answer: "Standard, validated by smooth loss curve, will sweep for v2."

---

## Q53. "Why batch size 2 with grad accum 8 (effective 16)?"

**Answer:**

Decoupled into two constraints:

1. **Per-device batch limited by VRAM** — at seq=512 with all the activations, 16 GB T4 fits **batch=2** safely. Batch=4 = OOM (Error #1 from Doc 4).

2. **Effective batch should be 16+ for stable LLM gradients** — empirically, batch < 8 = noisy gradient signal; 16-32 = sweet spot for instruction tuning.

**Solution:** `gradient_accumulation_steps=8` means trainer accumulates gradients over 8 microbatches before stepping the optimizer. Effective batch = 2 × 8 = 16.

```python
per_device_train_batch_size=2,
gradient_accumulation_steps=8,    # effective batch = 16
```

**Compute cost:** Same total compute as batch=16 directly (just split into 8 microbatches). Memory savings come from never materializing 16 samples worth of activations at once.

🎯 **Interviewer expectation:** Connect **hardware constraint** (VRAM) to **statistical requirement** (stable gradients) and resolve via grad accum. This is **the** classic LLM-training tradeoff.

---

## Q54. "Why warmup_ratio = 0.03?"

**Answer:**

Warmup = linearly ramp LR from 0 to peak over the first X% of steps.

**Why warmup matters:**
- At step 0, gradients are noisy (untrained adapter)
- High LR + noisy gradients = parameter explosion / NaN loss
- Gradually increasing LR lets the model settle before applying full LR

**3% = standard for fine-tuning:**
- Pre-training uses larger warmup (10%) because much bigger LR + longer schedule
- Fine-tuning is shorter; 3% is enough to avoid early spikes
- Total steps ≈ 220 for 1 epoch on 3,536 samples / batch 16. 3% = ~7 warmup steps.

If warmup is too long: wastes compute at small LR. Too short: risks early instability.

For Tanglish, I never observed loss spikes — warmup_ratio=0.03 worked first try.

🎯 **Interviewer expectation:** Show **why** warmup, not just the number. The instability argument is the key insight.

---

## Q55. "Why cosine LR scheduler? Why not constant or linear?"

**Answer:**

| Scheduler | Behavior | When to use |
|---|---|---|
| `constant` | LR stays at peak | Short experiments, quick tests |
| `linear` | LR decreases linearly to 0 | Older standard; tends to overshoot |
| **`cosine`** ✅ | LR follows cosine curve to 0 | Modern standard for fine-tuning |

**Cosine specifics:**
```
lr(t) = lr_max × 0.5 × (1 + cos(π × t / T))
```

Where t = current step, T = total steps.

**Why cosine wins for fine-tuning:**
1. **Smooth decay** — no sharp transitions; gradient flow stays stable
2. **Slower at end** — at the end of training, decreasing LR slowly helps "polish" the model around a good minimum without bouncing
3. **Standard** — HF Trainer's most-cited choice; well-validated across tasks

vs linear: linear decreases LR too sharply mid-training, can lose convergence.

vs constant: model can oscillate around the minimum forever; never settles.

🎯 **Interviewer expectation:** Show **scheduler intuition**, not just the name. Mentioning the "polish" effect at the end = practitioner-level.

---

# 🅵️ Category F: Evaluation & Metrics (Q56–Q70)

---

## Q56. "What does perplexity measure?"

**Answer:**

**Perplexity = exp(average negative log-likelihood)** of held-out test data under the model.

Intuitively: "The model is, on average, choosing among **PPL words** at each step."

- PPL = 1: perfect prediction (impossible for natural language)
- PPL = 10: 10-way confusion per token
- PPL = ∞: random guessing

**Math:**
```
PPL(X) = exp(-1/N × Σ log P(x_i | x_<i))
       = exp(cross-entropy loss)
```

For TamilTech-QA on Tanglish test set:
- Base Llama-3.1-8B: PPL = 57.05 ("the model finds Tanglish surprising")
- Fine-tuned: PPL = 12.40 ("the model has learned the distribution")

**Lower = better.** 78% reduction means the model is no longer confused by Tanglish patterns.

**Caveat:** PPL only measures distributional fit. It doesn't say if the output is **correct** or **helpful** — just that it matches the test distribution.

🎯 **Interviewer expectation:** Definition + math + intuition + your numbers + caveat. Full coverage in <90 seconds.

---

## Q57. "78% PPL drop sounds huge. Is that an honest claim or cherry-picked?"

**Answer:**

Honest, but with caveats I disclose:

**Why it's real:**
- Test set was **held out** before training began (never seen by model)
- Same evaluation protocol for both base and fine-tuned (same tokenizer, same context length, same loss computation)
- 78% drop is **monotonic** — every checkpoint between step 30 and final showed decreasing eval_loss
- Reproducible — anyone can run the same eval

**Why the headline could mislead:**
- Test set distribution **resembles** train set (same channels, same topics) — could be domain-specific overfitting to this kind of Tanglish, not generic Tanglish
- PPL drop **doesn't translate proportionally** to BLEU/ROUGE — the model improved at predicting next tokens but not at matching reference word-for-word
- Sample size: ~447 test samples → not millions. Confidence intervals exist.

**Mitigation v2:** Hold out a "golden" hand-curated test set from **different sources** (Twitter, Telegram) — measure transfer.

🎯 **Interviewer expectation:** Show **statistical maturity**. "Yes my headline is real" is fine — but adding "here's why it could mislead" earns trust.

---

## Q58. "Why didn't BLEU and ROUGE improve much?"

**Answer:**

BLEU and ROUGE measure **n-gram overlap** with reference text. They're **style-insensitive**.

Two reasons they're flat for our setup:

1. **No single "correct" answer in Tanglish** — for the question "Explain React hooks", many natural Tanglish answers are valid. Model could generate a perfectly fluent Tanglish answer that has **zero 4-gram overlap** with the reference. BLEU score: low. Quality: high.

2. **Length divergence** — fine-tuned model generates **~14× longer** outputs than reference (we observed). Long outputs hurt BLEU precision (long denominator).

**This is a known limitation** of BLEU/ROUGE for open-ended generation. It's why machine translation moved to **BERTScore, COMET, MAUVE**, and human eval over the last 3 years.

**My contribution:** **CSPS, TTR, TCF** specifically capture the **style-level** quality that BLEU misses. They moved (+3.7%, +8%, +1.8%). PPL moved (-78%). Together they tell a coherent improvement story.

🎯 **Interviewer expectation:** Don't say "BLEU is dumb." Say "BLEU is **incomplete for this task**, here's why specifically, here's what I added to fill the gap." Critique with respect.

---

## Q59. "Walk me through CSPS — definition, math, why novel."

**Answer:**

**CSPS = Code-Switch Preservation Score.**

**Definition:** How close the prediction's Tanglish ratio is to the reference's Tanglish ratio.

**Math:**
```
CSPS_i = 1 - |tanglish_ratio(pred_i) - tanglish_ratio(ref_i)|
CSPS   = mean(CSPS_i) across all test samples
```

Range: [0, 1]. Higher = better preservation.

**Example:**
- Ref ratio = 0.42, Pred ratio = 0.40 → CSPS_i = 0.98 (great)
- Ref ratio = 0.42, Pred ratio = 0.05 → CSPS_i = 0.63 (model dropped Tamil)
- Ref ratio = 0.42, Pred ratio = 0.99 → CSPS_i = 0.43 (model went all-Tamil)

**Why novel:**
- Standard metrics (BLEU/ROUGE) ignore **style**. CSPS targets it directly.
- Previous code-switching metrics (CMI, M-index) measure switches **within** sentence; mine measures **alignment with reference** — better for evaluation of generation models.

**TamilTech-QA result:** CSPS 0.62 → 0.66 (+3.7% absolute). Small but meaningful — confirms style shifted toward reference.

🎯 **Interviewer expectation:** Definition + math + intuition + your numbers + novelty justification. Bonus: cite prior CS metrics (CMI from Das & Gambäck 2014) and contrast.

---

## Q60. "Walk me through TTR — definition, math, intuition."

**Answer:**

**TTR = Technical Term Retention.**

**Definition:** Fraction of English technical keywords appearing in the **reference** that also appear in the **prediction**.

**Math:**
```
ref_terms_i = set(tokens in ref_i that are in TECH_KEYWORDS)
pred_terms_i = set(tokens in pred_i that are in TECH_KEYWORDS)
TTR_i = |ref_terms_i ∩ pred_terms_i| / |ref_terms_i|   (skip if ref has 0)
TTR  = mean across samples with at least one ref tech term
```

**Why it matters:**
- Tanglish technical writing **must preserve English technical vocabulary** verbatim (function, array, gradient).
- A model that "translates" `function → செயல்பாடு` or `variable → மாறி` is **wrong for the register** even if semantically equivalent.
- BLEU/ROUGE don't penalize this style mismatch.

**TECH_KEYWORDS list:** ~40 terms from `config/data_config.yaml` — function, array, class, error, model, gradient, neural, transformer, etc.

**TamilTech-QA result:** TTR 0.81 → 0.89 (+8%). Fine-tuned model explicitly preserves English tech terms while adding Tamil scaffolding.

🎯 **Interviewer expectation:** Make the **register argument** clearly — translation vs preservation. This is the key linguistic insight.

---

## Q61. "Walk me through TCF — what does it measure?"

**Answer:**

**TCF = Tamil Connector Fluency.**

**Definition:** Rule-based **naturalness** check. Of the Tamil discourse connectors (na, athula, ippadi, mathiri, etc.) used by the reference, how many does the prediction also use?

**Math:**
```
TAMIL_CONNECTORS = ["enna na", "apdi patha", "intha maari", "sollunga", "puriyutha",
                    "theriyuma", "paaru", "mathiri", "aanaa", ...]  # ~15 connectors

ref_connectors_i = {c for c in TAMIL_CONNECTORS if c in ref_i.lower()}
pred_connectors_i = {c for c in TAMIL_CONNECTORS if c in pred_i.lower()}

If ref_connectors_i is non-empty:
    TCF_i = |ref_c ∩ pred_c| / |ref_c|
Else:
    TCF_i = 1.0 if pred_connectors_i  else 0.5  (neutral baseline)
```

**Why it matters:**
- Tamil connectors are the **conversational glue** of Tanglish discourse.
- Without them you get robotic English-with-Tamil-words ("Function variable la value store pannum"), missing the rhythm of natural Tanglish ("Function na, athula variable irukku, athula value store pannum athu").

**TamilTech-QA result:** TCF 0.490 → 0.498 (+1.8%). Small. Reflects that **1 epoch is undertrained** for connector usage. v2 with more epochs should improve this most.

🎯 **Interviewer expectation:** Make the **rhythm of natural speech** argument. Bonus: acknowledge the small improvement and tie it to undertraining.

---

## Q62. "Why these three metrics specifically? What does each one isolate?"

**Answer:**

Each targets a different facet of code-switching quality:

| Metric | What it isolates | Failure mode it catches |
|---|---|---|
| **CSPS** | **Overall style preservation** | Model drifts toward one language |
| **TTR** | **Domain vocabulary** | Model translates English tech terms to Tamil |
| **TCF** | **Conversational fluency** | Model writes "robot Tanglish" without connectors |

A model could:
- Have high CSPS but low TTR (preserves ratio, but loses tech terms) → vague but Tanglish-y
- Have high TTR but low CSPS (keeps tech terms but in English-only output) → not Tanglish
- Have high CSPS + high TTR but low TCF (right ratio + right terms but robotic flow) → Mid quality

Reporting all three forces a holistic view. Improving in only one is partial.

**Together they form an "interpretability triangle"** for code-switching quality.

🎯 **Interviewer expectation:** Show **why three, not one**. Each catches what others miss — that's the metric design principle.

---

## Q63. "What other metrics would you add in v2?"

**Answer:**

Five additions, in priority order:

1. **MAUVE** — distribution-level metric using token statistics. Measures whole-text similarity beyond n-grams. Good for open-ended generation.

2. **BERTScore (multilingual)** — semantic similarity via XLM-R embeddings. Handles paraphrases and Tanglish variation that BLEU misses.

3. **GPT-4 / Claude judge** — pairwise win-rate (model A vs model B) judged by GPT-4 with specific Tanglish-quality criteria. Expensive (~$0.01/judgment) but high signal.

4. **Length ratio** — already computed (14× too long); make it a first-class metric for fluency.

5. **Human eval (N=200)** — recruit 5 native Tanglish speakers, blind side-by-side rating on naturalness/accuracy/style. Gold standard.

**Priority for v2:** GPT-4 judge first (fast, automatable). Human eval after publishing v0.2 for the actual paper.

🎯 **Interviewer expectation:** Show familiarity with **the modern evaluation toolkit**. Don't say "I'd just add BLEU-5" — show research-current options.

---

## Q64. "Did you do human eval? If not, why not?"

**Answer:**

**Not formally in v0.1.** I did informal N=20 manual side-by-side reading to verify the qualitative direction. That's it.

**Why no formal human eval:**

1. **Time/cost** — getting 5+ Tanglish-native annotators on a structured rating task is a 2-week effort with stipends. v0.1 was a 5-week solo project.

2. **Bootstrapping issue** — to design a rating rubric you need pilot data; my pilot data was the eval I'd be rating.

3. **Pre-publication scope** — v0.1 is a research artifact for community feedback. Human eval planned for v0.2 paper.

**What I'd ask 5 evaluators:**
- Rate **naturalness** (1-5): "Does this sound like a real Tamil techie wrote it?"
- Rate **accuracy** (1-5): "Is the technical content correct?"
- Rate **style match** (1-5): "Does it match the question's register?"
- Pairwise: "Which response (A or B) would you trust more for your child's coding question?"

**Sample size for statistical power:** ~200 samples × 3 raters = 600 ratings, Krippendorff's α target ≥ 0.6.

🎯 **Interviewer expectation:** Honest about gaps + concrete plan for v2. "I don't have human eval but here's the protocol I'd design" >> "yes I did human eval" (which would be obvious if you didn't).

---

## Q65. "Your test set is 447 samples. Is that enough for reliable metrics?"

**Answer:**

**For perplexity:** Yes — PPL is computed at **token level**, not sample level. 447 samples × ~150 tokens average = ~67K test tokens. Statistically robust for PPL claims.

**For BLEU/ROUGE:** Borderline. Corpus-level BLEU on 447 samples gives a number, but **confidence intervals are wide**. Improvements <5% are noise-level.

**For CSPS/TTR/TCF:** Decent — they're per-sample averages, central limit theorem applies. Standard error ≈ σ / √447. Reasonable.

**For human eval (planned v2):** 200 samples × 3 raters is standard. 447 is plenty.

**What I'd do differently for stronger claims:**
- **Bootstrap confidence intervals** (resample 447 with replacement, 1000 times, report 95% CI for each metric)
- Right now I report point estimates; CIs would be more rigorous

For v0.1 research artifact: 447 is adequate. For peer review: need CIs.

🎯 **Interviewer expectation:** Show **statistical literacy** (CLT, bootstrap CIs) without being pedantic. Mention what you didn't yet do = honesty.

---

## Q66. "Could your model just be overfitting to YouTube comment style?"

**Answer:**

Real risk — and partially yes.

**Evidence of overfit risk:**
- 92% of training data = YouTube. Model learned YouTube-comment register.
- Test set also from YouTube → looks great on this distribution.
- Untested: Twitter Tanglish, Telegram Tanglish, Stack Overflow Tanglish.

**Why "partial overfit" is okay for v0.1:**
- YouTube comments span 12 channels = diverse-ish within YouTube.
- Synthetic 8% adds out-of-distribution variety.
- Goal of v0.1 = prove the basic approach works.

**Where overfit matters:** **deployment generalization**. If someone deployed this for a banking chatbot, the YouTube-comment register might not match formal customer-support register.

**v2 fix:**
- Add Twitter + Telegram + StackOverflow as data sources
- Hold out a **"golden test set"** of hand-curated samples from a different distribution (e.g., banking customer queries from open datasets)
- Report metrics on both in-distribution and out-of-distribution test sets

🎯 **Interviewer expectation:** Acknowledge the risk **upfront**, don't get defensive. Naming the specific failure mode = mature.

---

## Q67. "How do you know you didn't overfit?"

**Answer:**

Three signals:

1. **Eval_loss curve** — monotonically decreased to ~0.97 with no late-stage uptick. Classic overfitting pattern (train_loss falling, eval_loss rising) didn't happen.

2. **Train vs eval gap** — final train_loss 0.95, eval_loss 0.97. Tiny gap (~2%). Consistent with under-fitting / well-fit, not over-fitting.

3. **Sample inspection** — generated outputs on novel test questions look coherent, not memorized. No verbatim test-set leakage detected.

**Why 1 epoch helps:** With LoRA and a single pass over modest data, overfitting is **structurally unlikely**. Capacity is small (3.4M trainable on 8B), data is small (~3.5K samples), epochs is 1.

**v2 risk:** at 5 epochs, overfitting **could** appear. I'll add:
- `EarlyStoppingCallback` with patience=3
- `load_best_model_at_end=True`
- Monitor eval_loss every 30 steps

🎯 **Interviewer expectation:** Show **multiple evidence types** + structural argument (LoRA + 1 epoch = unlikely overfit). Don't just say "loss looked good."

---

## Q68. "How is your test set designed? Could there be leakage?"

**Answer:**

**Design:**
- 10% of total data (447 samples)
- **Stratified** by topic and difficulty
- **Random** within strata, seed=42

**Leakage controls:**
1. **Exact dedup applied before split** → no sample appears in both train and test
2. **No knowledge of test labels during training** — eval_steps uses val set, not test
3. **Test only touched at final eval phase** — discipline ensures no peeking

**Possible residual leakage:**
- **Paraphrase leakage** — slightly reworded question/answer pairs could span splits. v2 will use MinHash to catch.
- **Video-level overlap** — comments from the same YouTube video can span splits. Theoretically model could overfit to that video's vocabulary. v2 = video-level stratification.

**Stress test I did:** Plot Tanglish-ratio distribution for train vs test → confirmed they're similar (no obvious distribution shift introduced by split).

🎯 **Interviewer expectation:** Show **proactive leak hunting** + acknowledge residuals. "I've controlled for X, X, X but Y is still possible" is the senior framing.

---

## Q69. "Quantitative vs qualitative — which matters more?"

**Answer:**

**Both, but qualitative often dominates for style tasks like this.**

**For TamilTech-QA specifically:**

- **Quantitative** (PPL/BLEU/CSPS/TTR/TCF) tells me **whether to publish**. Sets a defensible numerical baseline. Without this, no peer review.

- **Qualitative** tells me **whether the model is useful**. PPL could drop with the model outputting gibberish-but-confident text — qualitative reading catches this.

**What I do:**
- Compute all quantitative metrics first
- Then read **20 random side-by-side samples** (base vs fine-tuned)
- Then read **10 hand-picked edge cases** (long queries, ambiguous queries, queries with mostly English)
- Only declare a model "ready" if **both** quant + qual look good

**Failure mode I avoid:** "Loss dropped 50%, ship it!" — without reading actual outputs, you can ship terrible models.

For papers: report quant. For shipping: validate qual.

🎯 **Interviewer expectation:** Show **balanced view**. Pure quant = naive. Pure qual = unscientific. Both, in order, with discipline.

---

## Q70. "You mentioned a 14× length ratio. What does that mean and is it bad?"

**Answer:**

**Length ratio = len(prediction) / len(reference)** averaged across test set.

For TamilTech-QA: ~14× — fine-tuned model generates outputs ~14 times longer than reference answers.

**Why this happened:**
- YouTube reference answers are short — comment-style, 1-3 sentences.
- Llama-3.1-Instruct's pre-training was on **long, expansive explanations** (English internet text, books).
- Fine-tuning on 4,400 samples didn't override the length prior.

**Is it bad?** Mixed:

| Use case | Bad? |
|---|---|
| Quick chatbot reply | Yes, too verbose |
| Tutorial-style assistant | No, longer is okay |
| Comment-style mimicry | Yes, broke the register |

**For Tanglish authenticity:** Bad. Real Tanglish on YouTube is **concise**. 14× is a register breach.

**v2 plan:** **DPO (Direct Preference Optimization)** training where preference pairs penalize long outputs. OR: include length penalty in beam-search decoding.

**v0.1 mitigation:** Set `max_new_tokens=120` at inference, which hard-caps.

🎯 **Interviewer expectation:** Show **honest limitation** + concrete fix path (DPO). Mentioning DPO specifically = current-research awareness.

---

# 🅶️ Category G: Engineering & MLOps (Q71–Q80)

---

## Q71. "Why Kaggle? Why not Colab Pro / cloud A100?"

**Answer:**

Three reasons Kaggle won at v0.1:

1. **Cost** — Kaggle free T4 = $0. Lambda A100 = $1-2/hr × 6 hrs = $6-12 per run. Colab Pro = $10/mo subscription.

2. **Session reliability** — Kaggle gives **12-hour dedicated sessions**; Colab free kicks you off after 3-4 hrs. For a 6-hour training run, Kaggle is the only free reliable option.

3. **Pre-installed ML stack** — Kaggle ships with PyTorch, transformers, and most of what I need. Colab is similar but Kaggle's environment is more stable across sessions.

**Trade-offs:**
- Kaggle's 30 hr/week budget vs unlimited paid GPUs → limits experimentation cadence
- T4 is slow vs A100 → 6 hrs on T4 = ~1 hr on A100
- For a paper-scale project I'd graduate to paid compute; v0.1 is fine on Kaggle

**Cost summary:** Whole project = **$0.10** (just gpt-4o-mini synthetic). Anywhere else = $50-100 minimum.

🎯 **Interviewer expectation:** Show **resource-constrained problem solving**. Senior engineers respect $0 results.

---

## Q72. "What happened with the session timeout? How did you recover?"

**Answer:**

**Incident:** First training run completed at ~hour 6, but session continued idle. At hour 11.5, session ended. I had **not clicked "Save Version"**. `/kaggle/working/` wiped. **6 hours of training lost.**

**Recovery for round 2:**

1. **Re-ran training** (~6 hours again)
2. Built **3-layer recovery infrastructure**:
   - Auto-save every 60 steps via `save_steps=60` in `SFTConfig`
   - "Save Version → Quick Save" at hour 4 (mid-run)
   - Eval-only notebook (`eval_recovery.ipynb`) that loads adapter from a saved Kaggle dataset for evaluation/push

**Lesson learned:**
- Click "Save Version" early and often
- Don't trust session lifetime
- Build infrastructure assuming worst case

This loss taught me **MLOps discipline** for real — every training run since then has had explicit checkpointing.

🎯 **Interviewer expectation:** Show **incident response maturity**. Mentioning what you learned > masking the failure.

---

## Q73. "Walk me through the VRAM budget on T4 (16 GB)."

**Answer:**

```
Total T4 VRAM:                  16 GB
─────────────────────────────────────
Llama-3.1-8B in 4-bit (NF4):    4 GB
LoRA adapters (fp16):           <100 MB
Optimizer states (8-bit AdamW): ~300 MB
Activations (bs=2, seq=512,
  with gradient checkpointing): ~5 GB
Gradients (LoRA only):          <100 MB
CUDA reserved overhead:         ~1 GB
─────────────────────────────────────
Total in use:                   ~10-11 GB
Buffer:                         ~5 GB ✅
```

**Each optimization's contribution:**

| Technique | Memory saved |
|---|---|
| 4-bit base | 12 GB |
| LoRA (vs full FT) | 15 GB worth of grads + optimizer states |
| 8-bit optimizer | ~1 GB |
| Gradient checkpointing | ~3 GB |
| bf16 activations | ~50% |
| seq_len=512 (vs 1024) | ~50% activations |

**If I removed any one of these:** OOM. They stack.

🎯 **Interviewer expectation:** Show **memory math at component level**. Senior engineers compute this **before** training, not after OOM.

---

## Q74. "How do you ensure reproducibility?"

**Answer:**

Five layers:

1. **Seed everything**:
```python
torch.manual_seed(42); numpy.random.seed(42); random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
os.environ["PYTHONHASHSEED"] = "42"
```

2. **Pin dependencies** in `requirements.txt` — exact versions for `transformers`, `peft`, `trl`, `bitsandbytes`. Different versions = different APIs = different behavior.

3. **Version-controlled configs** — every hyperparameter in `config/*.yaml`, committed to git.

4. **Deterministic data splits** — random_state=42 in train_test_split. Same split every time.

5. **Frozen base model** — `meta-llama/Llama-3.1-8B-Instruct` is a specific HF revision; I pin commit if needed.

**Caveat — what's not 100% reproducible:**
- CUDA matmul ops are technically non-deterministic by default; `cudnn.deterministic=True` enables deterministic but ~10% slower
- gpt-4o-mini synthetic generation is **not** reproducible (temperature 0.8, OpenAI's sampling)

For the synthetic data, I **archive the generated jsonl** so others can reuse the exact same synthetic pairs.

🎯 **Interviewer expectation:** Show **multi-layer reproducibility**. Bonus for acknowledging what isn't perfectly reproducible (cuda non-determinism) and why.

---

## Q75. "Why .env for secrets and not just environment variables?"

**Answer:**

`.env` is **the standard for local development**, and `python-dotenv` automatically loads it. Three benefits over raw env vars:

1. **Project-scoped** — different projects can have different `.env` files, no global pollution
2. **Self-documenting** — `.env.example` ships in the repo as a template
3. **Tool-friendly** — `python-dotenv`, `direnv`, and most IDEs natively understand it

**In production:** I'd use:
- **Cloud secret managers** (AWS Secrets Manager, GCP Secret Manager, HashiCorp Vault)
- **Kaggle Secrets / Colab userdata** for notebook contexts
- **K8s Secrets** for containerized deployments

`.env` is for **local dev only**. Production secrets get rotated, audited, encrypted at rest — `.env` files do none of that.

**Critical:** `.env` MUST be in `.gitignore` from day 1. If accidentally committed, the secret is permanently public — git history retains it. Revoke and rotate immediately.

🎯 **Interviewer expectation:** Show **stage-appropriate tool choice**. Dev uses `.env`; prod uses managed secrets.

---

## Q76. "Why a config-driven design? Why not hardcode hyperparameters?"

**Answer:**

Three reasons:

1. **Ablation experiments** — switching `lora_r8 → lora_r16` is a 1-line YAML change, not a code change. Code stays clean; config tracks experiment.

2. **Reproducibility** — config files are committed to git. The exact training run is encoded in the YAML + commit hash. Hardcoded params get lost over time.

3. **Cross-environment portability** — Kaggle, Colab, local can all load the same config. Hardcoded paths/params would diverge.

**My config layers:**
- `config/data_config.yaml` — data collection + preprocessing + split
- `config/model_config.yaml` — model + LoRA + training
- `config/eval_config.yaml` — test set paths + metric configs
- `.env` — secrets (NOT in YAML)

**${VAR} expansion** — YAML refers to `${OPENAI_API_KEY}`, resolved at runtime from `.env`. Keeps secrets out of YAML.

```python
def load_config(path):
    raw = open(path).read()
    expanded = os.path.expandvars(raw)
    return yaml.safe_load(expanded)
```

🎯 **Interviewer expectation:** Show **clean separation of concerns**: code vs config vs secrets. This is a hallmark of senior ML engineering.

---

## Q77. "Walk me through your logging strategy."

**Answer:**

I use **`loguru`** (not stdlib `logging`) for three reasons:

1. **Zero config** to look pretty — colors, timestamps, file:line — out of the box
2. **f-string-style** formatting: `log.info("Got {} items in {:.2f}s", n, t)` — much cleaner than `%s` formatting
3. **Better tracebacks** — full local variable inspection on exceptions

**Centralized setup:**
```python
from loguru import logger
logger.add(sys.stderr, level="INFO",
           format="<green>{time}</green> | <level>{level}</level> | {message}",
           colorize=True)
```

**Per-module pattern:**
```python
from src.utils.logger import get_logger
log = get_logger(__name__)
log.info("Processing {} samples", len(data))
```

**What I log:**
- DATA: file paths, row counts, filter drop rates
- TRAINING: loss every 5 steps, GPU memory every 30 steps
- EVAL: per-metric value, run time
- ERROR: with full stacktrace via `log.exception()`

**What I don't log:** secrets, full tensors, model outputs >100 chars (use sampling).

🎯 **Interviewer expectation:** Show **practical logging discipline** — not just "I use print". Mention loguru = current Python ecosystem awareness.

---

## Q78. "What's your error handling philosophy?"

**Answer:**

**Layered approach:**

1. **Fail fast at boundaries** — at script start, verify all configs load, all paths exist, all required env vars set. Crash early with clear message.

2. **Retry transient failures** — API rate limits, network blips, 5xx errors get **exponential backoff retry** (via `tenacity`):
```python
@retry(stop=stop_after_attempt(5),
       wait=wait_exponential(multiplier=2, min=2, max=60))
def fetch_video_comments(video_id): ...
```

3. **Log + skip permanent failures** — a single video's comments unfetchable? Log it, skip, continue. Don't tank the whole run for one bad input.

4. **Don't swallow exceptions silently** — `except Exception: pass` is forbidden. At minimum log + re-raise.

5. **Cleanup on failure** — file handles closed, temp files deleted via `try/finally` or context managers.

**Where I'm strict:** training loop. NaN loss → halt + alert. Better to kill a bad run than waste 6 hours.

**Where I'm lenient:** data collection. Some YouTube videos return 403; log + skip; don't crash.

🎯 **Interviewer expectation:** Show **graduated responses** based on failure type. Senior engineers know when to retry, when to log+skip, when to halt.

---

## Q79. "What did you test? What's the testing philosophy?"

**Answer:**

**Unit tests in `tests/`:**

1. `test_language_filter.py` — Tanglish detector
   - Pure English → rejected
   - Pure Tamil-script → rejected
   - Natural Tanglish → accepted
   - Edge cases: empty string, single word

2. `test_metrics.py` — CSPS/TTR/TCF
   - Perfect match → score 1.0
   - Total mismatch → score < 0.5
   - Empty input → graceful handling

3. `test_qa_formatter.py` — Alpaca/ChatML formatters
   - Schema correctness
   - Special token escaping

**Run:** `pytest tests/ -v`. CI would auto-run on push.

**Philosophy:**
- Test **business logic** (filter rules, metric formulas)
- Don't test **framework code** (transformers, peft) — they have their own tests
- Test **edge cases** explicitly (empty, single, oversize)
- **No mock training** — too brittle, low value vs end-to-end smoke tests on small data

**Coverage gap I'd close in v2:** integration test that runs the full pipeline on 10 synthetic samples end-to-end.

🎯 **Interviewer expectation:** Show **proportional testing** — not "100% coverage" mantras, but tests targeting **business logic + edge cases**.

---

## Q80. "If this went to production, what would you add for CI/CD?"

**Answer:**

**Build pipeline (CI):**

```yaml
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - pip install -r requirements.txt
      - pytest tests/ --cov=src --cov-fail-under=70
      - black --check src/
      - mypy src/
```

**Add:**
- **Pre-commit hooks** scanning for secrets (`detect-secrets`)
- **Type checking** (`mypy`) on `src/`
- **Linting** (`ruff` or `black`)
- **Dependency vulnerability scan** (`pip-audit`)
- **Notebook nbformat check** (no execution output committed)

**Deploy pipeline (CD):**
- On main push: smoke-test inference on 10 samples
- On tag (e.g., `v0.2`): push adapter to HF Hub + update README + create GitHub release

**Monitoring:**
- HF Hub model card analytics (downloads, likes)
- Latency dashboard if hosted (Prometheus + Grafana)
- Drift detection on user queries vs training distribution

**Rollback strategy:**
- Tag each adapter version on HF Hub
- If v0.2 has issues, revert deployment to v0.1 tag

🎯 **Interviewer expectation:** Show you **think beyond the model** to the lifecycle. Mention rollback specifically — that's a production-grade detail.

---

# 🅷️ Category H: Publishing & Deployment (Q81–Q87)

---

## Q81. "Why HuggingFace Hub vs alternatives?"

**Answer:**

Three reasons HF dominates for ML research artifacts:

1. **Discoverability** — HF Hub has ~5M weekly visitors, organic search/recommendations. GitHub repos don't surface to ML researchers the same way.

2. **Tool integration** — `huggingface_hub` library lets users **load my model with 3 lines**:
```python
PeftModel.from_pretrained(base, "dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA")
```
Versus a custom S3 bucket = users have to download + extract + figure out structure.

3. **Free + LFS-aware** — Git LFS for large files is HF's bread and butter. No bandwidth costs. S3 charges egress.

**Trade-offs:**
- HF Hub is **opinionated** — README has YAML frontmatter rules, dataset cards have specific format. Some friction.
- Tied to HF ecosystem — if HF disappears, my links break.

**Mitigation:** I also publish to **GitHub** for the code + Colab notebooks; if HF goes down, the code is still available.

🎯 **Interviewer expectation:** Show **community/ecosystem awareness**. ML research has **distribution norms** — HF is the norm.

---

## Q82. "Why MIT license?"

**Answer:**

Three reasons:

1. **Permissive** — anyone can use, fork, modify, sell. Maximum adoption.
2. **Compatibility** — works with most other licenses, no copyleft restrictions.
3. **Researcher-friendly** — common in ML research; no friction for adoption.

**Considered alternatives:**

| License | Pros | Cons |
|---|---|---|
| **MIT** ✅ | Maximum adoption | No copyleft protection |
| Apache-2.0 | Patent grant, MIT-ish | Slightly more legalese |
| GPL-3 | Copyleft (derivatives must be open) | Hostile to commercial adopters |
| CC-BY | Good for datasets | Less standard for code |

**Special case — the model:** Llama-3.1's license **transmits** to fine-tuned adapters. Adapters must comply with Llama 3.1 Community License (commercial use allowed up to 700M MAU). My **dataset** and **code** are MIT; my **adapter** is Llama-3.1 license.

🎯 **Interviewer expectation:** Show you read licenses, especially Llama's clause that transmits to derivatives. Most people miss this.

---

## Q83. "Why publish the dataset separately from the model?"

**Answer:**

Three reasons:

1. **Independence** — someone might use my dataset with a **different base model** (Mistral, Gemma). Bundling them couples uses.

2. **HF Hub conventions** — datasets have their own UI (preview tab, schema view), card format, audience (`datasets` library users). Models have a different UI and audience.

3. **Storage** — dataset is ~few MB; model is ~50 MB. Different release cadences likely (dataset grows; model retrains less often).

**Linking strategy:**
- Model card explicitly lists `datasets: [dheepakkaran/TamilTech-QA]` in YAML frontmatter → HF shows the dataset link on the model page
- Dataset card mentions the model as "the example fine-tuned baseline"
- Both link to GitHub for code

🎯 **Interviewer expectation:** Show you know **the three artifacts have three audiences** with different update cycles. Coupling them is anti-pattern.

---

## Q84. "Why static demo + Colab instead of live HF Space?"

**Answer:**

**Constraint:** HF Space free tier = **CPU only, 16 GB RAM**. Llama-3.1-8B even in 4-bit = ~4 GB weights but inference overhead pushes total to ~8-10 GB. **Loading works**, but inference is **3+ minutes per response** due to CPU bottleneck — unusable.

**ZeroGPU tier** = T4 access, but requires HF PRO ($9/mo).

**Compromise:**

| Component | Hosts where | Purpose |
|---|---|---|
| **Static Space** | Free HF Space CPU | Showcase: pre-computed base vs FT comparison, interactive Tanglish-ratio analyzer |
| **Live demo** | Colab T4 free | Real inference, both models loaded, Gradio UI |

User flow:
1. Visit Space → see comparison snapshots + try Tanglish analyzer (no model)
2. Click "Open in Colab" → launch interactive demo with real model
3. Free GPU = 30-60 min sessions

**Cost:** $0 for both paths. Trade is **two clicks** to reach live inference vs one click.

🎯 **Interviewer expectation:** Show **constraint-driven design**. Couldn't afford ZeroGPU → built the compromise. Honest about the UX cost.

---

## Q85. "How did you cross-link the three platforms?"

**Answer:**

**Linking matrix:**

| Source | Links to | Where |
|---|---|---|
| GitHub README | HF Dataset, Model, Space | Top of README + bottom "Links" section |
| HF Model card | GitHub, HF Dataset | YAML `datasets:` field + README link section |
| HF Dataset card | GitHub, HF Model | README "Use with" section |
| HF Space README | All three | Footer credits |
| GitHub "About" / Website | HF Model | Single primary link (most representative) |

**The standard footer** I paste in every README:
```markdown
## 🔗 Project Links
- 💻 Code (GitHub):  github.com/dheepakkaran/TamilTech-QA
- 📊 Dataset:        huggingface.co/datasets/dheepakkaran/TamilTech-QA
- 🤖 Model:          huggingface.co/dheepakkaran/TamilTech-QA-Llama3.1-8B-QLoRA
- 🎮 Space:          huggingface.co/spaces/dheepakkaran/TamilTech-QA-Demo
```

**Why this matters:** Discoverability is **cumulative**. Visitor lands on Model card → clicks GitHub → reads code → stars repo → opens Colab → tries the live demo. Multi-hop journey.

🎯 **Interviewer expectation:** Show **funnel thinking** — visitors enter at any platform, every platform must route them to the rest.

---

## Q86. "Tell me about the README override bug."

**Answer:**

**Bug:** When you call `model.push_to_hub("user/repo")` or `trainer.push_to_hub()`, HuggingFace **silently auto-generates a default README and uploads it**, overwriting any custom README you uploaded first.

**How I got bitten:** Wrote a beautiful 400-line model card → uploaded it → pushed model via `trainer.push_to_hub()` → reloaded the page → my custom README was replaced by default placeholder.

**Three fixes (in order of preference):**

1. ✅ **Use `upload_folder` instead of `push_to_hub`** — `upload_folder` is a passive sync, no auto-generation.

2. **Upload README AFTER push** — let push_to_hub overwrite, then push your README on top:
```python
api.upload_file(path_or_fileobj="model_card.md",
                path_in_repo="README.md",
                repo_id=REPO_ID, repo_type="model",
                commit_message="Restore custom card")
```

3. **Edit on website** — quick fix; not automatable.

I used Fix 3 for v0.1 (quick), then automated with Fix 1 for subsequent runs.

🎯 **Interviewer expectation:** Show **practical platform knowledge**. This bug is **not in any tutorial** — knowing it = hands-on experience.

---

## Q87. "What can't your demo do that you'd want it to?"

**Answer:**

Live demo (Colab) limitations:

1. **Cold start latency** — 10-15 min for setup (HF login, model download, both base + FT load in 4-bit). Frustrating UX.

2. **No persistent state** — closing tab loses session. Next user pays full setup cost.

3. **Free Colab kicks idle sessions** — after ~90 min of inactivity, session dies. Not robust for sharing.

4. **Single-user** — Colab is per-Google-account. Can't share one running demo URL with multiple visitors simultaneously.

5. **No analytics** — I don't know how many people actually try the demo, what queries they enter, where the model fails.

**v2 deployment plan:**
- **GGUF quantization** for llama.cpp → maybe runnable on HF Space free CPU after all
- **Modal.com or Replicate** for paid (~$0.50/hr) reliable always-on hosting if budget permits
- **Analytics** via Sentry or Posthog for free
- **Caching** for repeat queries

🎯 **Interviewer expectation:** Show you **understand the gap** between research demo and production. Be honest about what real users experience.

---

# 🅸️ Category I: Research, Limitations, Future Work (Q88–Q95)

---

## Q88. "What are the main limitations of v0.1?"

**Answer:**

Five honest limitations:

1. **1 epoch undertrains some patterns** — TCF improved only 1.8% because connector patterns need more exposure. v2 = 3-5 epochs.

2. **Length divergence (14× too long)** — fine-tuning didn't shorten outputs. v2 = DPO with length-controlled preferences.

3. **Roman-script only** — Tamil-script Tanglish dropped by filter. Some real users prefer Tamil-script. v2 = second model for that.

4. **YouTube distribution bias** — 92% from YouTube comments. Not validated on Twitter / Telegram / Stack Overflow Tanglish.

5. **No human eval yet** — informal N=20 reading only. Need formal study for paper-grade claims.

**What's stable:**
- Approach is sound (QLoRA on Llama-3.1)
- Dataset is publicly reusable
- Code is reproducible
- Metrics framework (CSPS/TTR/TCF) is documented and shareable

🎯 **Interviewer expectation:** Show **structured limitations** — what's broken, why, what's planned. Each limitation has a v2 path. Nothing hand-waved.

---

## Q89. "Why only 1 epoch? Was that enough?"

**Answer:**

Three reasons:

1. **Kaggle session time limit** — 12 hr sessions, T4 speed. 1 epoch on ~3.5K samples with batch=16, seq=512 = ~6 hours. 2 epochs would risk session timeout for safety margin.

2. **Eval signal during training** — eval_loss dropped smoothly from 1.55 → 0.97 by end of epoch 1. No early-stopping triggered. The model **was** learning.

3. **Risk of overfit** — at 3-5 epochs on 3.5K samples with QLoRA, risk increases. v0.1 = conservative.

**Was 1 enough?** Quantitative metrics (PPL -78%) say yes for the headline. Qualitative metrics (TCF +1.8%) suggest **no** for connector style — more training helps.

**v2:** 3-5 epochs with `EarlyStoppingCallback` (patience=3, threshold=0.0001 on eval_loss). Auto-stop when no improvement. Expect TCF and length-ratio to improve most.

🎯 **Interviewer expectation:** Show **time-budget-driven decision** + acknowledge what it cost (TCF stayed flat). Don't pretend 1 epoch was optimal.

---

## Q90. "The 14× length ratio problem — how would you fix it?"

**Answer:**

Three approaches, increasing in sophistication:

1. **Inference-time hack** — `max_new_tokens=120` + early stopping on `<|eot_id|>`. Easy. Doesn't fix the underlying preference.

2. **Add length-penalized samples** — augment training data with shorter-style answers. The model learns "match the question style." Limited because Llama-3.1's pre-training prior is strong.

3. **DPO with length-controlled preferences** — for each question, generate 2 responses, mark the **shorter** one as preferred. Train DPO. This **directly teaches** the model to prefer brevity. Best long-term fix.

**Why DPO works:**
- Llama-3.1's base prior says "longer answer = more helpful" (from instruction-tuning data).
- DPO **overrides** that prior with **explicit preference signal**.
- ~500-1000 preference pairs sufficient (faster than collecting more SFT data).

**v2 plan:** DPO is phase 3 of the v2 roadmap. After getting more data + more epochs of SFT, then DPO on style.

🎯 **Interviewer expectation:** Name **DPO specifically** — current best technique for style/preference alignment. Mentioning the prior-override insight = research-level depth.

---

## Q91. "Top 3 priorities for v2?"

**Answer:**

In order:

1. **Data expansion to ~15K samples** — Telegram + Twitter + 500 hand-curated gold pairs. Address the YouTube-only bias. Expected: better generalization, validated on held-out diverse test set.

2. **Better training**: 3-5 epochs, `lora_r16`, add FFN target modules (`gate_proj`, `up_proj`, `down_proj`). Expected: TCF and length-ratio improvements.

3. **Style alignment via DPO** — 500-1000 preference pairs penalizing verbose, English-drifting outputs. Expected: 14× length ratio drops to ~2× (still longer than reference but acceptable).

Optional phase 4: Tamil-script Tanglish second model. Phase 5: production deployment (GGUF, streaming inference).

**Estimated effort:** 5-7 weeks for phases 1-3. Phase 4-5 = stretch.

**Target venue:** ACL or EMNLP workshop on multilingual NLP.

🎯 **Interviewer expectation:** Show **prioritized roadmap** with rationale + outcomes per phase. "Just do everything" = junior. "Phase 1, Phase 2, Phase 3 because..." = mature.

---

## Q92. "Why DPO specifically and not RLHF or PPO?"

**Answer:**

| Approach | Complexity | Compute | Reward model needed? |
|---|---|---|---|
| **RLHF (PPO)** | High | High | Yes — train separately |
| **DPO** ✅ | Lower | Low | No — direct from prefs |
| **ORPO** | Lowest | Low | No — combined with SFT |
| **IPO** | Medium | Low | No |

**DPO advantages for my use case:**

1. **No reward model needed** — DPO reformulates RLHF math to skip the reward-modeling step. ~2× less compute.

2. **Stable training** — PPO is notoriously finicky; DPO uses standard cross-entropy.

3. **Reproducibility** — DPO's training loop is simpler; easier for community to replicate.

4. **Sufficient quality** — DPO papers show comparable/better quality than PPO for alignment tasks.

**Why not ORPO?** Newer; less battle-tested. DPO is well-supported in TRL.

**What DPO needs:** Preference dataset of `(prompt, chosen_response, rejected_response)` triples. For my case: ~1000 such triples should suffice.

🎯 **Interviewer expectation:** Show **awareness of post-PPO landscape**. ORPO/DPO/IPO are all current research. Knowing which is mature vs experimental = up-to-date.

---

## Q93. "How would you handle Tamil-script Tanglish in v2?"

**Answer:**

Two architectural choices:

**Option A: Single multi-script model**

- Add Tamil-script samples to training data (currently filtered out)
- Model learns to **switch scripts** based on prompt cues
- Risk: tokenizer inefficiency for Tamil-script reduces useful context

**Option B: Two separate models + classifier**

- Model A: Roman-script Tanglish (current v0.1)
- Model B: Tamil-script Tanglish (new)
- **Classifier** decides which to call based on user input
- Clean separation; each model optimal for its script

**My pick: Option B for v2**, because:
1. Cleaner evaluation (can measure each model on its own register)
2. Tokenization is optimized per model
3. Failure mode is local (Model B doesn't hurt Model A)

**Routing classifier:** Light XLM-R based — detect Tamil-script presence in input → route accordingly. Latency overhead negligible.

🎯 **Interviewer expectation:** Show **system design thinking** — when to use one model vs many. Mentioning the routing classifier specifically = production-aware.

---

## Q94. "Could you train Llama-3.1-70B for this instead?"

**Answer:**

**Compute infeasibility on free hardware:**

- Llama-3.1-70B in 4-bit = ~35 GB
- Plus activations + adapter + optimizer = ~50 GB
- Free T4 = 16 GB → won't fit

**Where it could work:**

- Multi-GPU A100 (80 GB × 2-4) — would need ~$50-100 / training run
- ZeRO-3 distributed training across CPU + GPU — possible but slow
- Quantized 2-bit (experimental) — fits but quality unclear

**Would 70B help for Tanglish?**

Marginally, in my opinion. Reasoning:
- Tanglish is a **style shift**, not a knowledge task. 70B's extra knowledge doesn't directly help.
- 8B already understands English tech vocab + Tamil tokens.
- The bottleneck is **training data + register alignment**, not model capacity.

**Where 70B would shine:** Long-form Tanglish reasoning (multi-step debugging chats), complex code generation. Not v0.1 scope.

**My v2 plan stays at 8B.** Better data + more epochs first; consider 70B only if quality plateaus.

🎯 **Interviewer expectation:** Show **judgment about scale** — bigger isn't always better. Tying scale decision to **what's limiting quality** = research maturity.

---

## Q95. "What would production deployment look like?"

**Answer:**

**Stack:**

1. **Model serving** — GGUF quantized Llama-3.1-8B + LoRA merged into base. Served via `llama.cpp` (CPU) or `vLLM` (GPU).

2. **API layer** — FastAPI with streaming endpoints. ~100-200ms first-token latency target.

3. **Caching** — Redis for common-query LLM response cache. ~50% hit rate expected.

4. **Rate limiting** — per-user (1 RPS), per-IP (10 RPS).

5. **Monitoring**:
   - Prometheus metrics: latency p50/p95/p99, error rates, GPU utilization
   - Grafana dashboards
   - Sentry for errors
   - Logging: each query (with PII scrubbed) for drift detection

6. **A/B testing** — feature flag for routing % of traffic to new model versions. Rollback on metric regression.

**Cost estimate** (modest traffic = 100K queries/day):
- GPU (A10g, 24/7): ~$15/day
- Redis (small): $20/mo
- API host: $20/mo
- Total: ~$500/mo

**SLA target:** 99% uptime, p95 latency < 500ms, error rate < 0.1%.

**Drift detection:** Daily job comparing user queries to training distribution; alert if KL divergence spikes.

🎯 **Interviewer expectation:** Show **end-to-end production thinking** — not just "deploy to AWS." Latency targets, cost estimates, SLAs — practical detail.

---

# 🅹️ Category J: Behavioral & Design Defense (Q96–Q100)

---

## Q96. "Why did you build this project?"

**Answer:**

Three motivations, in increasing order of depth:

1. **Tactical:** Strong Northeastern grad-school portfolio piece. Resume-friendly. Shows end-to-end ownership of an ML research project.

2. **Career:** I want to work in **multilingual NLP or Indic AI** post-grad. This project signals I can identify a research gap, build a dataset, fine-tune an LLM, and publish artifacts publicly.

3. **Personal:** I'm a native Tanglish speaker. Every existing model irritates me by replying in formal Tamil-script Tamil when I type Roman-script. Wanted to **prove the gap was fixable** with current tools.

**What I learned that surprised me:**
- Free tools can do real research (Kaggle + HF + OpenAI = $0.10 total)
- Tokenization is everything for low-resource languages
- BLEU/ROUGE are insufficient — designing custom metrics is **necessary**, not optional

🎯 **Interviewer expectation:** Show **three-layer motivation** — tactical, career, personal. Pure tactical = mercenary. Pure personal = unfocused. Three together = grounded researcher.

---

## Q97. "What was the hardest bug you faced?"

**Answer:**

**The bf16/fp16 grad scaler bug.**

**Symptom:** `NotImplementedError: BFloat16 is not supported in fp16 grad scaler` — raised at the **first backward pass**, hours into a Kaggle session, after model loading + dataset processing already completed.

**Why hard:**
1. **Error message was misleading** — it pointed at GradScaler, not the actual config conflict.
2. **First search results suggested updating PyTorch** — false lead, wasted an hour.
3. **Required understanding Llama-3.1's native dtype** — most tutorials use `fp16=True` because they trained on Llama-2 (which is fp16-native). Llama-3.1 = bf16-native is a recent change.

**How I solved:**
1. Read `model.config.torch_dtype` → saw `torch.bfloat16`
2. Connected: GradScaler is fp16-specific; bf16 has different precision properties (no underflow)
3. Realized: bf16 needs **no scaler at all**
4. Fix: `bf16=True, fp16=False` in `TrainingArguments`

**Lesson:** Match training precision to the model's native dtype. **Read the model config before configuring training**.

🎯 **Interviewer expectation:** Show **debugging story** with explicit steps. Bonus for naming the misleading first search result — proves real experience.

---

## Q98. "If you could reverse one decision, what would it be?"

**Answer:**

**Not using W&B (Weights & Biases) for experiment tracking.**

**What I did:** Used `report_to="none"` and relied on Kaggle's console logs.

**Why I'd reverse:**
1. **Lost training curves** — when v1 crashed mid-run, I had no persistent record of how loss was evolving. Had to retrain to see the curve.
2. **No ablation tracking** — couldn't easily compare runs side-by-side. Each run was isolated.
3. **No shareable artifact** — for a paper-grade project, having a public W&B dashboard would be a strong supplement to the README.

**Counterargument I had at the time:** "Adds complexity, free Kaggle integration is flaky" — partially true but overweighted.

**What I'd do now:** Enable `report_to="wandb"` with W&B API key in Kaggle secrets. ~5 min setup, massive long-term value.

**Other decision considered:** Not buying Colab Pro for $10. In hindsight, that $10 would've saved ~10 hours of session-timeout pain. Cheap insurance.

🎯 **Interviewer expectation:** Show **regret with reasoning**, not just any regret. Naming **two** candidates and picking one shows comparison thinking.

---

## Q99. "What did you learn that surprised you most?"

**Answer:**

**Three surprises, increasing depth:**

1. **Surprise 1 (technical):** QLoRA is **almost as good as full FT**. I'd expected a noticeable quality gap; instead, register shift was clearly visible after just 1 epoch on free hardware. The Dettmers paper claim that QLoRA = ~99% of FT quality is real.

2. **Surprise 2 (research):** **Designing metrics is harder than designing models.** I spent ~2 days iterating on CSPS / TTR / TCF definitions — what counts as a "Tamil token" alone took 3 rewrites. The model training is mechanical; the **measurement** is creative work.

3. **Surprise 3 (personal):** **Free tools are sufficient for real research.** Kaggle T4 + HuggingFace + OpenAI gpt-4o-mini synthetic = $0.10 total project cost. The barrier to ML research isn't compute anymore — it's **knowing what to do** with it.

**Implication for my career:** The bottleneck is **research taste**, not access. Tools democratized; problem-selection is the differentiator.

🎯 **Interviewer expectation:** Three surprises = depth. Ending on the "research taste" insight = grad-school-level reflection. Most candidates give one shallow surprise; the layered answer stands out.

---

## Q100. "If you had unlimited compute, what would you do?"

**Answer:**

**Five projects in priority order:**

1. **Scale to 100K+ samples across 5+ Indic CS pairs** — Tanglish, Hinglish, Tenglish, Manglish (Malayalam-English), Kanglish (Kannada-English). Build a **CS benchmark suite** for Indic NLP.

2. **Train Llama-3.1-70B QLoRA on the full multi-CS dataset** — 8B is enough for proof-of-concept; 70B for paper-grade quality claims.

3. **DPO + RLHF on style preferences** — collect ~10K human preferences for length/naturalness/accuracy. Multi-stage alignment.

4. **Pre-train a custom Tamil-English tokenizer** — extend Llama's 128K vocab with 5K-10K Tanglish-specific merges. Re-train embeddings only (LoRA on embeddings).

5. **Production deployment with retrieval augmentation** — connect to live Tamil tech docs (StackOverflow Tamil, Tamil dev forums) via RAG. Tanglish answers grounded in real-time content.

**Estimated compute:** ~$50K-100K total (4×A100 × 6 months).

**Estimated impact:** ACL/EMNLP main conference paper + 10K+ HF downloads + production partnerships with Indian fintech/edtech.

**Personal stretch:** Apply this approach to **endangered Indic scripts** — e.g., Toda, Kota. Use code-switching framework as a bridge for ultra-low-resource languages.

🎯 **Interviewer expectation:** Show **ambition with structure** — not pie-in-sky but **prioritized phases**. Naming **concrete venues** (ACL/EMNLP) + concrete partners (fintech/edtech) = grounded ambition.

---

# 🎓 Final notes — How to actually prep for the interview

### 1. **Don't memorize. Internalize.**

For every answer, identify:
- **Claim** (what I'm saying is true)
- **Evidence** (number, observation, paper citation)
- **Tradeoff** (what I gave up to get this)

If you can articulate **all three** without scripts, you'll handle any rephrasing.

### 2. **The 60-second rule**

Most answers should fit in 60-90 seconds verbally. If you need more, structure with "Three reasons: first... second... third..." — gives natural pacing and exit cue for interviewer to dig in.

### 3. **Practice in Tanglish AND English**

- For Indian interviewers, Tanglish is **rapport-building** (especially South Indian interviewers)
- For US/EU interviewers, pure English with **occasional Tamil terms** for color
- Have both versions ready for your top 20 Q&A

### 4. **The "I don't know" answer**

If asked something you don't know:
- ✅ "I haven't measured that specifically — my guess is X because Y, but I'd need to run the experiment."
- ❌ "I don't know."
- ❌ Fabricated numbers.

Confidence + honest uncertainty wins.

### 5. **Lead with the headline number**

For any "tell me about results" question, **start with the number**:
- "78% perplexity reduction"
- "$0.10 total compute cost"
- "4,415 samples, 92% real / 8% synthetic"

Numbers anchor; explanations follow.

### 6. **Practice the "and then?" continuation**

Interviewers often follow up with "And then?" or "Why?" Practice extending each answer by **one level deeper** without panicking. The most common interview failure is freezing on follow-ups.

---

## ✅ End of Doc 6 — End of Series

**Padichi paaru, slowly:**
- 10-15 questions per day for ~10 days = full internalization
- Highlight gaps where you fumble
- Re-read corresponding source doc (Docs 1-5) to fill them

**Folder state:**
```
01_basics_concepts.md            37 KB
02_architecture_workflow.md      52 KB
03_local_setup_from_scratch.md   49 KB
04_kaggle_training_walkthrough.md 39 KB
05_huggingface_github_publishing.md 29 KB
06_interview_100_questions.md    [this file]
```

🚀 **Best of luck on every interview, mapla. You built something real — own it.**
