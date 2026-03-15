"""
similarity.py
─────────────
Multi-metric similarity engine for comparing student answers to model answers.

Metrics used and why:
  1. Semantic Similarity (Sentence Transformers + Cosine)
      -> Catches paraphrasing and synonyms
      -> "Plants use sunlight" ~= "Photosynthesis uses solar energy"

  2. RapidFuzz Token-Sort Ratio
      -> Handles word reordering

  3. RapidFuzz Partial Ratio
      -> More forgiving when OCR adds/removes words

  4. Jaccard Word Overlap
      -> Vocabulary coverage sanity check

For noisy OCR, scores are computed on both:
  - full extracted text
  - best-matching text window
and the stronger value is used per metric.
"""

import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from rapidfuzz import fuzz
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# ── Model (loaded once) ────────────────────────────────────────────────────────

_MODEL_NAME = "all-MiniLM-L6-v2"    # Best balance of speed vs accuracy
_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading Sentence Transformer model: %s", _MODEL_NAME)
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


# ── Individual metrics ─────────────────────────────────────────────────────────

def semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings (0–1)."""
    if not text_a.strip() or not text_b.strip():
        return 0.0
    model = _get_model()
    emb = model.encode([text_a, text_b])
    score = float(sk_cosine([emb[0]], [emb[1]])[0][0])
    return max(0.0, min(1.0, score))


def token_sort_similarity(text_a: str, text_b: str) -> float:
    """RapidFuzz token_sort_ratio normalised to 0–1."""
    return fuzz.token_sort_ratio(text_a, text_b) / 100.0


def partial_similarity(text_a: str, text_b: str) -> float:
    """RapidFuzz partial_ratio normalised to 0–1."""
    return fuzz.partial_ratio(text_a, text_b) / 100.0


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard index (0–1)."""
    set_a = set(text_a.split())
    set_b = set(text_b.split())
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union


# ── Weighted ensemble ─────────────────────────────────────────────────────────

WEIGHTS = {
    "semantic":   0.58,
    "token_sort": 0.14,
    "partial":    0.18,
    "jaccard":    0.10,
}

NOISY_OCR_WEIGHTS = {
    "semantic":   0.66,
    "token_sort": 0.08,
    "partial":    0.20,
    "jaccard":    0.06,
}


def _tokens(text: str) -> list[str]:
    return [t for t in text.split() if t]


def _best_matching_window(student_answer: str, model_answer: str) -> str:
    """Pick the student text span that best overlaps model tokens."""
    s_tokens = _tokens(student_answer)
    m_tokens = _tokens(model_answer)

    if not s_tokens or not m_tokens:
        return student_answer
    if len(s_tokens) <= len(m_tokens) + 5:
        return student_answer

    m_set = set(m_tokens)
    window_size = max(8, len(m_tokens))
    stride = max(1, window_size // 3)

    best_start = 0
    best_score = -1.0

    starts = list(range(0, max(1, len(s_tokens) - window_size + 1), stride))
    last_start = max(0, len(s_tokens) - window_size)
    if last_start not in starts:
        starts.append(last_start)

    for start in starts:
        chunk = s_tokens[start:start + window_size]
        overlap = len(set(chunk) & m_set)
        score = overlap / max(1, len(m_set))
        if score > best_score:
            best_score = score
            best_start = start

    return " ".join(s_tokens[best_start:best_start + window_size])


def calculate_similarity(
    student_answer: str,
    model_answer: str,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a weighted ensemble similarity score.

    Args:
        student_answer : Cleaned student text.
        model_answer   : Cleaned model/reference text.

    Returns:
        final_score    : Float in [0, 1].
        breakdown      : Dict with individual metric scores.
    """
    if not student_answer.strip() or not model_answer.strip():
        logger.warning("Empty input detected — returning 0.0")
        return 0.0, {}

    best_window = _best_matching_window(student_answer, model_answer)

    full_scores: Dict[str, float] = {
        "semantic":   semantic_similarity(student_answer, model_answer),
        "token_sort": token_sort_similarity(student_answer, model_answer),
        "partial":    partial_similarity(student_answer, model_answer),
        "jaccard":    jaccard_similarity(student_answer, model_answer),
    }

    window_scores: Dict[str, float] = {
        "semantic":   semantic_similarity(best_window, model_answer),
        "token_sort": token_sort_similarity(best_window, model_answer),
        "partial":    partial_similarity(best_window, model_answer),
        "jaccard":    jaccard_similarity(best_window, model_answer),
    }

    scores: Dict[str, float] = {
        k: max(full_scores[k], window_scores[k]) for k in full_scores
    }

    s_len = len(_tokens(student_answer))
    m_len = len(_tokens(model_answer))
    length_ratio = (s_len / m_len) if m_len else 1.0
    weights = NOISY_OCR_WEIGHTS if length_ratio > 1.8 else WEIGHTS

    final = sum(scores[k] * weights[k] for k in scores)
    final = max(0.0, min(1.0, final))

    logger.info(
        "Similarity scores -> Semantic: %.2f | TokenSort: %.2f | "
        "Partial: %.2f | Jaccard: %.2f | FINAL: %.2f | Ratio: %.2f",
        scores["semantic"], scores["token_sort"],
        scores["partial"], scores["jaccard"], final, length_ratio,
    )

    return final, scores


# ── Keyword coverage bonus (optional) ────────────────────────────────────────

def keyword_coverage(student_answer: str, keywords: list[str]) -> float:
    """
    Fraction of expected key terms found in the student answer (0–1).
    Used to add a small bonus for domain-critical vocabulary.
    """
    if not keywords:
        return 1.0
    student_words = set(student_answer.lower().split())
    found = sum(1 for kw in keywords if kw.lower() in student_words)
    return found / len(keywords)