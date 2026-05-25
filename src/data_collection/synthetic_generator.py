"""Synthetic Tanglish QA generation via the OpenAI Chat Completions API.

Generates ``n_pairs_per_topic`` QA pairs for each configured topic, validates
the JSON structure, deduplicates near-identical questions via TF-IDF cosine
similarity, and appends to ``data/raw/synthetic_qa.jsonl``.

Example
-------
.. code-block:: bash

    python -m src.data_collection.synthetic_generator \
        --config config/data_config.yaml --n-per-topic 50
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from src.utils import ensure_dir, load_config, project_root
from src.utils.logger import get_logger, setup_logging

log = get_logger(__name__)

PROMPT_TEMPLATE = """Generate {n} technical question-answer pairs in Tanglish (Tamil-English code-switching).
Rules:
- Mix Tamil words naturally with English technical terms
- Use Roman script for Tamil portions (e.g., "indha concept-a purinjukonga")
- Questions should be undergraduate engineering level
- Answers should be 3-5 sentences
- Technical terms stay in English (function, gradient, pointer, etc.)
- Conversational Tamil connectors: "enna na", "apdi patha", "intha maari", "sollunga", "puriyutha"
Topic: {topic}
Format: JSON array with keys: "question", "answer", "topic", "difficulty" (easy/medium/hard)
Output ONLY the JSON array. No prose, no markdown fences."""


class SyntheticGenerator:
    """Generate, validate, and deduplicate synthetic Tanglish QA pairs.

    Args:
        api_key: OpenAI API key.
        model: Chat model to use (default GPT-4o).
        temperature: Sampling temperature.
        max_tokens: Max tokens per completion.
        dedup_threshold: Cosine similarity above which questions are dropped.
        retry_max_attempts: Retries on transient errors / parse failures.
        retry_base_delay: Backoff base in seconds.

    Example:
        >>> gen = SyntheticGenerator(api_key="...")  # doctest: +SKIP
        >>> pairs = gen.generate("python", n=5)      # doctest: +SKIP
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        temperature: float = 0.8,
        max_tokens: int = 1500,
        dedup_threshold: float = 0.85,
        retry_max_attempts: int = 4,
        retry_base_delay: float = 2.0,
    ) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is empty.")
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "openai>=1.0 is not installed. Run: pip install openai"
            ) from e
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.dedup_threshold = dedup_threshold
        self.retry_max_attempts = retry_max_attempts
        self.retry_base_delay = retry_base_delay

    # ------------------------------------------------------------------ #
    def _call_model(self, prompt: str) -> str:
        """Call the OpenAI chat endpoint with retries.

        Args:
            prompt: User-message string.

        Returns:
            Raw model response content.

        Raises:
            RuntimeError: If all retries fail.
        """
        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a synthetic data generator. Always emit "
                                "valid JSON arrays only."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001 - openai exception hierarchy varies
                last_err = e
                delay = self.retry_base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                log.warning("OpenAI call failed (attempt {}/{}): {}. Sleep {:.1f}s",
                            attempt, self.retry_max_attempts, e, delay)
                time.sleep(delay)
        raise RuntimeError(f"OpenAI calls exhausted: {last_err}")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_json_array(raw: str) -> List[Dict[str, Any]]:
        """Parse a JSON array from a model response, tolerating fenced output.

        Args:
            raw: Raw text from the model.

        Returns:
            List of dictionaries.

        Raises:
            ValueError: If a valid JSON array cannot be extracted.
        """
        text = raw.strip()
        if text.startswith("```"):
            # strip ```json ... ``` fences
            lines = [ln for ln in text.splitlines() if not ln.strip().startswith("```")]
            text = "\n".join(lines).strip()
        # Find outermost [ ... ]
        start = text.find("[")
        end = text.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("No JSON array found in response.")
        snippet = text[start : end + 1]
        parsed = json.loads(snippet)
        if not isinstance(parsed, list):
            raise ValueError("Top-level JSON is not an array.")
        return parsed

    # ------------------------------------------------------------------ #
    def generate(self, topic: str, n: int) -> List[Dict[str, Any]]:
        """Generate ``n`` QA pairs for a single topic.

        Args:
            topic: Topic label (e.g., ``"python"``).
            n: Number of pairs to request.

        Returns:
            List of validated QA dictionaries.
        """
        prompt = PROMPT_TEMPLATE.format(n=n, topic=topic)
        for attempt in range(1, self.retry_max_attempts + 1):
            try:
                raw = self._call_model(prompt)
                items = self._parse_json_array(raw)
                cleaned = []
                for it in items:
                    q = (it.get("question") or "").strip()
                    a = (it.get("answer") or "").strip()
                    if not q or not a:
                        continue
                    cleaned.append(
                        {
                            "question": q,
                            "answer": a,
                            "topic": it.get("topic", topic) or topic,
                            "difficulty": (it.get("difficulty") or "medium").lower(),
                        }
                    )
                if cleaned:
                    return cleaned
                log.warning("Topic {} returned empty cleaned set on attempt {}", topic, attempt)
            except (ValueError, json.JSONDecodeError) as e:
                log.warning("Parse failure for topic {} (attempt {}): {}", topic, attempt, e)
                time.sleep(self.retry_base_delay * attempt)
        log.error("Giving up on topic {} after {} attempts.", topic, self.retry_max_attempts)
        return []

    # ------------------------------------------------------------------ #
    def deduplicate(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop near-duplicate questions using TF-IDF cosine similarity.

        Args:
            items: List of QA dictionaries (with a ``question`` field).

        Returns:
            Filtered list.
        """
        if len(items) < 2:
            return items
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:  # pragma: no cover
            log.warning("scikit-learn not installed; skipping dedup.")
            return items

        questions = [it["question"] for it in items]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(questions)
        sim = cosine_similarity(vec)
        n = sim.shape[0]
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if keep[j] and sim[i, j] >= self.dedup_threshold:
                    keep[j] = False
        out = [items[i] for i in range(n) if keep[i]]
        log.info("Deduplication: {} → {} pairs (threshold={:.2f}).",
                 n, len(out), self.dedup_threshold)
        return out


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def _question_id(q: str, topic: str) -> str:
    """Deterministic ID from question text."""
    return "synth_" + hashlib.md5(f"{topic}::{q}".encode("utf-8")).hexdigest()[:10]


def run(
    config_path: str,
    n_per_topic: Optional[int] = None,
    topics: Optional[Iterable[str]] = None,
) -> Path:
    """Generate synthetic QA pairs across all configured topics.

    Args:
        config_path: Path to data config YAML.
        n_per_topic: Override for ``synthetic.n_pairs_per_topic``.
        topics: Override for ``synthetic.topics`` (subset).

    Returns:
        The path to the output JSONL file.
    """
    cfg = load_config(config_path)
    syn = cfg["synthetic"]
    out_dir = ensure_dir(project_root() / cfg["paths"]["raw_dir"])
    out_path = out_dir / "synthetic_qa.jsonl"

    generator = SyntheticGenerator(
        api_key=syn["openai_api_key"],
        model=syn.get("model", "gpt-4o"),
        temperature=syn.get("temperature", 0.8),
        max_tokens=syn.get("max_tokens", 1500),
        dedup_threshold=syn.get("dedup_cosine_threshold", 0.85),
        retry_max_attempts=syn.get("retry", {}).get("max_attempts", 4),
        retry_base_delay=syn.get("retry", {}).get("base_delay_seconds", 2.0),
    )

    chosen_topics = list(topics) if topics else list(syn["topics"])
    n = n_per_topic if n_per_topic is not None else syn.get("n_pairs_per_topic", 100)

    all_pairs: List[Dict[str, Any]] = []
    for topic in chosen_topics:
        log.info("Generating {} pairs for topic '{}'...", n, topic)
        # Most chat models can comfortably emit ~20 pairs per call. Chunk.
        chunk = max(1, min(n, 20))
        produced: List[Dict[str, Any]] = []
        while len(produced) < n:
            need = min(chunk, n - len(produced))
            batch = generator.generate(topic, need)
            if not batch:
                break
            produced.extend(batch)
        produced = generator.deduplicate(produced)
        all_pairs.extend(produced[:n])
        log.info("Topic '{}' yielded {} pairs after dedup.", topic, len(produced[:n]))

    with out_path.open("w", encoding="utf-8") as f:
        for pair in all_pairs:
            pair["id"] = _question_id(pair["question"], pair["topic"])
            pair["source"] = "synthetic"
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    log.info("Wrote {} synthetic QA pairs to {}", len(all_pairs), out_path)
    return out_path


def main() -> None:
    """CLI entrypoint for ``python -m src.data_collection.synthetic_generator``."""
    setup_logging()
    p = argparse.ArgumentParser(description="Generate synthetic Tanglish QA pairs.")
    p.add_argument("--config", default="config/data_config.yaml")
    p.add_argument(
        "--n-per-topic",
        type=int,
        default=None,
        help="Override n_pairs_per_topic from the config.",
    )
    p.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Optional subset of topics to generate.",
    )
    args = p.parse_args()
    run(args.config, n_per_topic=args.n_per_topic, topics=args.topics)


if __name__ == "__main__":
    main()
