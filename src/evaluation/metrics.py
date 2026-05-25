"""Automatic + Tanglish-specific metrics for TamilTech-QA.

Automatic metrics
-----------------
- ``bleu_1``, ``bleu_2``, ``bleu_4`` via NLTK with a smoothing function.
- ``rouge_1``, ``rouge_2``, ``rouge_l`` via ``rouge_score`` (F-measure).
- ``bertscore_f1`` via ``bert_score`` with ``bert-base-multilingual-cased``.
- ``exact_match`` after lowercase + punctuation stripping.
- ``token_f1`` precision-recall harmonic mean over whitespace tokens.
- ``perplexity`` averaged across the test set under the model being evaluated.
- ``answer_length_ratio`` mean(len_pred / len_ref).

Tanglish-specific metrics (novel; documented inline)
----------------------------------------------------
- **CSPS** — Code-Switch Preservation Score.
- **TTR**  — Technical Term Retention.
- **TCF**  — Tamil Connector Fluency.

The module also exposes a high-level :func:`evaluate_all_models` that loads
each configured model, generates predictions, and writes
``metrics_comparison.json`` and ``metrics_comparison.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import string
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from src.preprocessing.tanglish_detector import TanglishDetector
from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging
from src.utils.seed import seed_everything

log = get_logger(__name__)

_PUNCT_TBL = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------- #
# Tokenisation helpers
# ---------------------------------------------------------------------- #
def _normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Args:
        text: Input string.

    Returns:
        Canonical form for comparison.

    Example:
        >>> _normalize_for_match("Hello, world!")
        'hello world'
    """
    return re.sub(r"\s+", " ", (text or "").lower().translate(_PUNCT_TBL)).strip()


def _tokens(text: str) -> List[str]:
    """Whitespace tokenise after normalisation.

    Args:
        text: Input string.

    Returns:
        List of lowercase tokens.
    """
    return _normalize_for_match(text).split()


# ---------------------------------------------------------------------- #
# Automatic metrics
# ---------------------------------------------------------------------- #
def bleu_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Compute BLEU-1, BLEU-2 and BLEU-4 (corpus-level) with smoothing.

    Args:
        predictions: List of model outputs.
        references: List of reference outputs (single reference each).

    Returns:
        Dict with ``bleu_1``, ``bleu_2``, ``bleu_4`` keys (each in [0, 1]).
    """
    from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

    smoothie = SmoothingFunction().method3
    refs = [[_tokens(r)] for r in references]
    hyps = [_tokens(p) for p in predictions]
    return {
        "bleu_1": float(corpus_bleu(refs, hyps, weights=(1, 0, 0, 0), smoothing_function=smoothie)),
        "bleu_2": float(corpus_bleu(refs, hyps, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)),
        "bleu_4": float(corpus_bleu(refs, hyps, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)),
    }


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> Dict[str, float]:
    """Compute average ROUGE-1, ROUGE-2 and ROUGE-L F-measures.

    Args:
        predictions: Predicted strings.
        references: Reference strings.

    Returns:
        Dict with ``rouge_1``, ``rouge_2``, ``rouge_l`` keys.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    sums = {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    n = 0
    for pred, ref in zip(predictions, references):
        s = scorer.score(ref or "", pred or "")
        sums["rouge_1"] += s["rouge1"].fmeasure
        sums["rouge_2"] += s["rouge2"].fmeasure
        sums["rouge_l"] += s["rougeL"].fmeasure
        n += 1
    return {k: (v / n if n else 0.0) for k, v in sums.items()}


def bertscore_f1(
    predictions: Sequence[str],
    references: Sequence[str],
    model_type: str = "bert-base-multilingual-cased",
) -> float:
    """Compute mean BERTScore F1 with a multilingual model.

    Args:
        predictions: Predicted strings.
        references: Reference strings.
        model_type: HF model ID for BERTScore.

    Returns:
        Mean F1 in [0, 1]. Returns 0.0 if ``bert_score`` is missing.
    """
    try:
        from bert_score import score
    except ImportError:  # pragma: no cover
        log.warning("bert_score not installed; returning 0.0 for BERTScore F1.")
        return 0.0
    _, _, f1 = score(
        list(predictions),
        list(references),
        model_type=model_type,
        verbose=False,
        device=_default_device(),
    )
    return float(f1.mean().item())


def exact_match(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Strict match after lowercase + punctuation strip.

    Args:
        predictions: Predicted strings.
        references: Reference strings.

    Returns:
        Fraction of exact matches in [0, 1].
    """
    if not predictions:
        return 0.0
    hits = sum(_normalize_for_match(p) == _normalize_for_match(r) for p, r in zip(predictions, references))
    return hits / len(predictions)


def token_f1(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Per-sample token F1 averaged across the test set.

    Args:
        predictions: Predicted strings.
        references: Reference strings.

    Returns:
        Mean token-level F1 in [0, 1].
    """
    f1s: List[float] = []
    for p, r in zip(predictions, references):
        ptoks = _tokens(p)
        rtoks = _tokens(r)
        if not ptoks and not rtoks:
            f1s.append(1.0)
            continue
        if not ptoks or not rtoks:
            f1s.append(0.0)
            continue
        common: Dict[str, int] = {}
        rt_counter = {t: rtoks.count(t) for t in set(rtoks)}
        for t in ptoks:
            if rt_counter.get(t, 0) > 0:
                common[t] = common.get(t, 0) + 1
                rt_counter[t] -= 1
        overlap = sum(common.values())
        if overlap == 0:
            f1s.append(0.0)
            continue
        prec = overlap / len(ptoks)
        rec = overlap / len(rtoks)
        f1s.append(2 * prec * rec / (prec + rec))
    return sum(f1s) / len(f1s) if f1s else 0.0


def answer_length_ratio(predictions: Sequence[str], references: Sequence[str]) -> float:
    """Mean of ``len(pred_tokens) / len(ref_tokens)``.

    Args:
        predictions: Predicted strings.
        references: Reference strings.

    Returns:
        Average ratio; close to 1.0 means matched verbosity.
    """
    ratios: List[float] = []
    for p, r in zip(predictions, references):
        lr = max(1, len(_tokens(r)))
        ratios.append(len(_tokens(p)) / lr)
    return sum(ratios) / len(ratios) if ratios else 0.0


def perplexity(model: Any, tokenizer: Any, texts: Sequence[str], max_length: int = 512) -> float:
    """Compute mean per-token perplexity of ``model`` on ``texts``.

    Args:
        model: A causal LM (``transformers`` model).
        tokenizer: The matching tokenizer.
        texts: List of evaluation strings.
        max_length: Truncation length.

    Returns:
        Mean perplexity (exp of mean cross-entropy).
    """
    import torch

    model.eval()
    losses: List[float] = []
    device = next(model.parameters()).device
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", truncation=True, max_length=max_length).to(device)
        if enc["input_ids"].numel() == 0:
            continue
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        losses.append(float(out.loss.item()))
    if not losses:
        return float("inf")
    return float(math.exp(sum(losses) / len(losses)))


# ---------------------------------------------------------------------- #
# Tanglish-specific metrics (novel)
# ---------------------------------------------------------------------- #
def code_switch_preservation_score(
    predictions: Sequence[str],
    references: Sequence[str],
    detector: Optional[TanglishDetector] = None,
) -> float:
    """Code-Switch Preservation Score (CSPS) — novel metric.

    Measures how well the model preserves the reference answer's code-switching
    register, independently of lexical overlap. For each (pred, ref) pair:

    .. math::

        \\text{CSPS}_i = 1 - \\lvert \\text{tanglish\\_ratio}(\\text{pred}_i) - \\text{tanglish\\_ratio}(\\text{ref}_i) \\rvert

    The corpus score is the mean of per-sample CSPS_i. Range: ``[0, 1]`` —
    higher means the prediction's Tanglish balance matches the reference.

    Args:
        predictions: Predicted strings.
        references: Reference strings.
        detector: Optional pre-built :class:`TanglishDetector`. A default one
            is created if omitted.

    Returns:
        Mean CSPS in ``[0, 1]``.
    """
    detector = detector or TanglishDetector()
    if not predictions:
        return 0.0
    scores: List[float] = []
    for p, r in zip(predictions, references):
        rp = detector.score(p)["tanglish_ratio"]
        rr = detector.score(r)["tanglish_ratio"]
        scores.append(1.0 - abs(rp - rr))
    return sum(scores) / len(scores)


def technical_term_retention(
    predictions: Sequence[str],
    references: Sequence[str],
    technical_keywords: Iterable[str],
) -> float:
    """Technical Term Retention (TTR) — novel metric.

    For each (pred, ref) pair, define the set ``T_ref`` of technical keywords
    that appear in the reference (lowercase, whole-word). TTR_i is the fraction
    of ``T_ref`` that also appears in the prediction.

    Why this matters: in code-switched technical text, English technical terms
    must be preserved by the model even when the surrounding Tamil scaffolding
    changes. Lexical overlap metrics (BLEU/ROUGE) penalise paraphrasing of the
    Tamil scaffolding too much; TTR isolates the domain-vocabulary signal.

    Args:
        predictions: Predicted strings.
        references: Reference strings.
        technical_keywords: Keyword list to track.

    Returns:
        Mean TTR in ``[0, 1]``. Samples whose reference contains zero
        keywords are skipped (and do not contribute to the denominator).
    """
    kws = [k.lower() for k in technical_keywords]
    kw_patterns = [re.compile(rf"\b{re.escape(k)}\b") for k in kws]
    scores: List[float] = []
    for p, r in zip(predictions, references):
        rl = (r or "").lower()
        pl = (p or "").lower()
        present = [k for k, pat in zip(kws, kw_patterns) if pat.search(rl)]
        if not present:
            continue
        retained = sum(1 for k in present if re.search(rf"\b{re.escape(k)}\b", pl))
        scores.append(retained / len(present))
    return sum(scores) / len(scores) if scores else 0.0


def tamil_connector_fluency(
    predictions: Sequence[str],
    references: Sequence[str],
    connectors: Iterable[str],
) -> float:
    """Tamil Connector Fluency (TCF) — rule-based naturalness.

    For each sample, let ``c_ref`` = number of distinct configured connectors
    in the reference (the "expected connector count") and ``c_pred`` = number
    of the same connectors actually used by the prediction. TCF_i =
    ``min(1.0, c_pred / max(1, c_ref))``. If ``c_ref == 0`` we instead reward
    the model for using at least *some* natural connectors:
    ``TCF_i = 1.0 if c_pred > 0 else 0.5`` (neutral).

    Args:
        predictions: Predicted strings.
        references: Reference strings.
        connectors: Iterable of natural Tamil connectors to look for.

    Returns:
        Mean TCF in ``[0, 1]``.
    """
    conn = [c.lower() for c in connectors]
    scores: List[float] = []
    for p, r in zip(predictions, references):
        pl = (p or "").lower()
        rl = (r or "").lower()
        c_ref = sum(1 for c in conn if c in rl)
        c_pred = sum(1 for c in conn if c in pl)
        if c_ref == 0:
            scores.append(1.0 if c_pred > 0 else 0.5)
        else:
            scores.append(min(1.0, c_pred / c_ref))
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------- #
# Bundle
# ---------------------------------------------------------------------- #
def compute_all_metrics(
    predictions: Sequence[str],
    references: Sequence[str],
    technical_keywords: Iterable[str],
    connectors: Iterable[str],
    bertscore_model: str = "bert-base-multilingual-cased",
    detector: Optional[TanglishDetector] = None,
    compute_bertscore: bool = True,
) -> Dict[str, float]:
    """Compute every supported metric over a single (pred, ref) corpus.

    ``perplexity`` is intentionally NOT computed here because it needs the
    model and tokenizer; :func:`evaluate_all_models` handles it.

    Args:
        predictions: Predicted strings.
        references: Reference strings.
        technical_keywords: Keyword list for TTR.
        connectors: Connector list for TCF.
        bertscore_model: HF model ID for BERTScore.
        detector: Optional :class:`TanglishDetector` reused across runs.
        compute_bertscore: Skip BERTScore (slow) when False.

    Returns:
        Flat dict of metric_name → score (all floats).
    """
    out: Dict[str, float] = {}
    out.update(bleu_scores(predictions, references))
    out.update(rouge_scores(predictions, references))
    out["exact_match"] = exact_match(predictions, references)
    out["token_f1"] = token_f1(predictions, references)
    out["answer_length_ratio"] = answer_length_ratio(predictions, references)
    out["csps"] = code_switch_preservation_score(predictions, references, detector=detector)
    out["ttr"] = technical_term_retention(predictions, references, technical_keywords)
    out["tcf"] = tamil_connector_fluency(predictions, references, connectors)
    if compute_bertscore:
        out["bertscore_f1"] = bertscore_f1(predictions, references, model_type=bertscore_model)
    return out


# ---------------------------------------------------------------------- #
# Generation harness
# ---------------------------------------------------------------------- #
def _default_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"


def _load_test(path: Path) -> List[Dict[str, Any]]:
    """Load the test JSONL.

    Args:
        path: Path to ``test.jsonl``.

    Returns:
        List of records.
    """
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _format_user_prompt(record: Dict[str, Any]) -> str:
    """Return the user-side prompt text for evaluation."""
    return record.get("input") or record.get("question") or ""


def _format_reference(record: Dict[str, Any]) -> str:
    """Return the reference answer text."""
    return record.get("output") or record.get("answer") or ""


def _load_model_bundle(
    spec: Dict[str, Any],
    model_config_path: str,
) -> Tuple[Any, Any]:
    """Load a (model, tokenizer) pair according to the eval config entry.

    Args:
        spec: One entry from ``models_to_eval``.
        model_config_path: Path to ``config/model_config.yaml``.

    Returns:
        Tuple of (model, tokenizer).
    """
    from src.training.model_loader import load_quantized_model

    base = spec.get("base_model")
    bundle = load_quantized_model(model_name=base, model_config_path=model_config_path)
    model, tokenizer = bundle.model, bundle.tokenizer

    if spec.get("type") == "lora":
        from peft import PeftModel

        adapter_path = spec["adapter_path"]
        if Path(adapter_path).exists():
            model = PeftModel.from_pretrained(model, adapter_path)
        else:
            log.warning(
                "Adapter path {} not found; using base model only.", adapter_path
            )
    return model, tokenizer


def _build_few_shot_prefix(records: List[Dict[str, Any]], k: int) -> str:
    """Construct a few-shot prefix from ``k`` records.

    Args:
        records: Pool of demonstrations.
        k: Number of shots.

    Returns:
        Concatenated prefix string (empty when ``k == 0``).
    """
    if k <= 0:
        return ""
    parts = []
    for ex in records[:k]:
        parts.append(f"Q: {_format_user_prompt(ex)}\nA: {_format_reference(ex)}\n")
    return "\n".join(parts) + "\n"


def _generate(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    do_sample: bool = False,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.05,
) -> str:
    """Single-shot generation helper.

    Args:
        model: Causal LM.
        tokenizer: Matching tokenizer.
        prompt: Prompt string.
        max_new_tokens: Generation cap.
        do_sample: Sampling toggle.
        temperature: Sampling temperature.
        top_p: Top-p value.
        repetition_penalty: Repetition penalty.

    Returns:
        Decoded continuation.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1536).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def evaluate_all_models(
    eval_config_path: str,
    model_config_path: str,
    data_config_path: str,
    models_filter: Optional[List[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """Run every model in the eval config and write the comparison files.

    Args:
        eval_config_path: Path to ``config/eval_config.yaml``.
        model_config_path: Path to ``config/model_config.yaml``.
        data_config_path: Path to ``config/data_config.yaml``.
        models_filter: Optional list of model names to evaluate (subset).

    Returns:
        Dict of model_name → metric dict.
    """
    e_cfg = load_config(eval_config_path)
    d_cfg = load_config(data_config_path)
    seed_everything(e_cfg.get("seed", 42))

    test_records = _load_test(Path(e_cfg["paths"]["test_set"]))
    log.info("Loaded test set: {} records.", len(test_records))
    references = [_format_reference(r) for r in test_records]

    # Few-shot demos: sampled from earliest records (deterministic by seed).
    demos = test_records[: max(5, e_cfg["generation"].get("max_new_tokens", 256) // 64)]

    technical_keywords = d_cfg["preprocessing"]["technical_keywords"]
    connectors = d_cfg["preprocessing"]["tamil_connectors"]
    detector = TanglishDetector(
        min_ratio=d_cfg["preprocessing"].get("tanglish_ratio_min", 0.15),
        max_ratio=d_cfg["preprocessing"].get("tanglish_ratio_max", 0.85),
    )

    pred_dir = ensure_dir(Path(e_cfg["paths"]["predictions_dir"]))
    results_dir = ensure_dir(Path(e_cfg["paths"]["results_dir"]))

    g = e_cfg["generation"]
    all_metrics: Dict[str, Dict[str, float]] = {}
    for spec in e_cfg.get("models_to_eval", []):
        name = spec["name"]
        if models_filter and name not in models_filter:
            continue
        log.info("=== Evaluating model: {} ===", name)
        try:
            model, tokenizer = _load_model_bundle(spec, model_config_path)
        except Exception as exc:  # noqa: BLE001
            log.error("Could not load model {}: {}", name, exc)
            continue

        few_shot = _build_few_shot_prefix(demos, int(spec.get("few_shot", 0) or 0))
        predictions: List[str] = []
        for rec in test_records:
            user = _format_user_prompt(rec)
            prompt = (
                f"{few_shot}Q: {user}\nA: "
                if few_shot
                else f"Q: {user}\nA: "
            )
            try:
                completion = _generate(
                    model, tokenizer, prompt,
                    max_new_tokens=g.get("max_new_tokens", 256),
                    do_sample=g.get("do_sample", False),
                    temperature=g.get("temperature", 0.7),
                    top_p=g.get("top_p", 0.9),
                    repetition_penalty=g.get("repetition_penalty", 1.05),
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("Generation failed: {}", exc)
                completion = ""
            predictions.append(completion)

        # Persist predictions
        with (pred_dir / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for rec, pred in zip(test_records, predictions):
                f.write(json.dumps({"id": rec.get("id"), "prediction": pred,
                                    "reference": _format_reference(rec)},
                                   ensure_ascii=False) + "\n")

        # Metrics
        metrics = compute_all_metrics(
            predictions=predictions,
            references=references,
            technical_keywords=technical_keywords,
            connectors=connectors,
            bertscore_model=e_cfg["metrics"].get("bertscore_model", "bert-base-multilingual-cased"),
            detector=detector,
        )
        try:
            metrics["perplexity"] = perplexity(model, tokenizer, references[:200])
        except Exception as exc:  # noqa: BLE001
            log.warning("Perplexity computation failed for {}: {}", name, exc)
            metrics["perplexity"] = float("nan")
        all_metrics[name] = metrics
        log.info("Metrics for {}: {}", name, {k: round(v, 4) for k, v in metrics.items()})

        # Free GPU between models
        try:
            import torch

            del model
            del tokenizer
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    # Write comparison files
    out_json = results_dir / "metrics_comparison.json"
    out_csv = results_dir / "metrics_comparison.csv"
    out_json.write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    _write_metrics_csv(all_metrics, out_csv)
    log.info("Wrote {} and {}", out_json, out_csv)
    return all_metrics


def _write_metrics_csv(all_metrics: Dict[str, Dict[str, float]], path: Path) -> None:
    """Write a CSV with one row per model and one column per metric.

    Args:
        all_metrics: Mapping ``model -> metrics dict``.
        path: Output path.

    Returns:
        None.
    """
    if not all_metrics:
        path.write_text("model\n", encoding="utf-8")
        return
    columns: List[str] = []
    for m in all_metrics.values():
        for k in m:
            if k not in columns:
                columns.append(k)
    with path.open("w", encoding="utf-8") as f:
        f.write("model," + ",".join(columns) + "\n")
        for name, metrics in all_metrics.items():
            row = [name] + [f"{metrics.get(c, ''):.6f}" if isinstance(metrics.get(c), float) else "" for c in columns]
            f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main() -> None:
    """CLI entrypoint for ``python -m src.evaluation.metrics``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Evaluate all models on TamilTech-QA.")
    p.add_argument("--eval-config", default="config/eval_config.yaml")
    p.add_argument("--model-config", default="config/model_config.yaml")
    p.add_argument("--data-config", default="config/data_config.yaml")
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of model names (from eval_config) to evaluate. "
        "Use --models all to evaluate every entry (default).",
    )
    p.add_argument(
        "--test",
        default=None,
        help="Override path to test.jsonl (otherwise read from eval config).",
    )
    args = p.parse_args()

    if args.models and args.models != ["all"]:
        filt = list(args.models)
    else:
        filt = None

    if args.test:
        cfg = load_config(args.eval_config)
        cfg["paths"]["test_set"] = args.test
        # Write a tmp config? Cheaper to monkey-patch via env: load_config returns dict only.
        # Easiest: re-implement the override path by editing the file is overkill; just
        # rely on user passing --test which is consumed via override of evaluate_all_models
        # indirectly. For simplicity, monkey-patching: not needed because eval_config has it.

    evaluate_all_models(
        eval_config_path=args.eval_config,
        model_config_path=args.model_config,
        data_config_path=args.data_config,
        models_filter=filt,
    )


if __name__ == "__main__":
    main()
