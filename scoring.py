"""Scoring functions for prompt reliability evaluation."""

import re
import math
from collections import Counter


def score_exact(response: str, expected: str) -> float:
    """Case-insensitive, whitespace-trimmed exact match. Returns 1.0 or 0.0."""
    if expected is None:
        return -1.0  # not applicable
    return 1.0 if response.strip().lower() == expected.strip().lower() else 0.0


def score_regex(response: str, pattern: str) -> float:
    """Regex search match. Returns 1.0 if pattern found, 0.0 otherwise."""
    if pattern is None:
        return -1.0
    return 1.0 if re.search(pattern, response) else 0.0


def score_semantic(response: str, anchor: str, client) -> float:
    """Cosine similarity using OpenAI text-embedding-3-small.

    Args:
        response: The model's response text.
        anchor: The reference text to compare against.
        client: An initialized openai.OpenAI client.

    Returns:
        Cosine similarity in [0, 1]. Returns -1.0 if scoring fails.
    """
    try:
        result = client.embeddings.create(
            model="text-embedding-3-small",
            input=[response, anchor]
        )
        vec_a = result.data[0].embedding
        vec_b = result.data[1].embedding
        return _cosine_similarity(vec_a, vec_b)
    except Exception as e:
        print(f"  [warn] Semantic scoring failed: {e}")
        return -1.0


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_consistency(responses: list[str]) -> float:
    """Fraction of responses matching the most common (modal) response.

    Args:
        responses: List of response strings (across all variants and runs).

    Returns:
        Consistency score in [0, 1]. 1.0 means all responses identical.
    """
    if not responses:
        return 0.0
    normalized = [r.strip().lower() for r in responses]
    counter = Counter(normalized)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(normalized)


def compute_flip_rate(baseline_responses: list[str], variant_responses: list[str]) -> float:
    """Fraction of variant responses that differ from the baseline modal answer.

    Args:
        baseline_responses: Responses from the baseline variant.
        variant_responses: Responses from a non-baseline variant.

    Returns:
        Flip rate in [0, 1]. 0.0 means variant always matches baseline mode.
    """
    if not baseline_responses or not variant_responses:
        return 0.0
    baseline_norm = [r.strip().lower() for r in baseline_responses]
    baseline_mode = Counter(baseline_norm).most_common(1)[0][1]
    baseline_modal = Counter(baseline_norm).most_common(1)[0][0]

    variant_norm = [r.strip().lower() for r in variant_responses]
    flips = sum(1 for r in variant_norm if r != baseline_modal)
    return flips / len(variant_norm)
