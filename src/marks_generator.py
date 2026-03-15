"""
marks_generator.py
──────────────────
Converts a similarity score (0–1) into marks with:
  • Grade band classification
  • Keyword-coverage bonus (up to 5% of max marks)
  • Feedback comment generation
"""

from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    raw_similarity: float          # 0–1 from similarity engine
    semantic_score: float          # 0–1 semantic-only similarity
    semantic_marks: float          # Marks equivalent for semantic score
    keyword_coverage: float        # 0–1 fraction of keywords found
    final_score: float             # 0–1 combined score
    marks: float                   # Awarded marks
    max_marks: float
    percentage: float
    grade: str
    feedback: str


GRADE_BANDS = [
    (0.88, "A+", "Outstanding answer - excellent understanding demonstrated."),
    (0.78, "A",  "Strong answer - key concepts covered well."),
    (0.66, "B",  "Good answer - most concepts present with minor gaps."),
    (0.55, "C",  "Adequate answer - core ideas are present."),
    (0.44, "D",  "Partial answer - relevant content found, but major gaps remain."),
    (0.33, "E",  "Weak answer - limited relevant content found."),
    (0.00, "F",  "Insufficient answer — little to no relevant content."),
]


def _grade(score: float):
    for threshold, grade, feedback in GRADE_BANDS:
        if score >= threshold:
            return grade, feedback
    return "F", GRADE_BANDS[-1][2]


def generate_marks(
    similarity: float,
    max_marks: float = 10.0,
    semantic_score: Optional[float] = None,
    keyword_coverage: Optional[float] = None,
    keyword_weight: float = 0.05,
) -> EvaluationResult:
    """
    Convert similarity score to final marks.

    Args:
        similarity        : Weighted ensemble similarity (0–1).
        max_marks         : Maximum possible marks for this question.
        semantic_score    : Optional semantic-only similarity (0–1).
        keyword_coverage  : Optional fraction of keywords found (0–1).
        keyword_weight    : How much of max_marks the keyword bonus can add.

    Returns:
        EvaluationResult dataclass.
    """
    sem_score = semantic_score if semantic_score is not None else similarity

    # If keyword coverage is not provided, keep it neutral instead of assuming perfect.
    kw_cov = keyword_coverage if keyword_coverage is not None else similarity

    # Combine: primarily similarity, with a small keyword term.
    base_weight = 1.0 - keyword_weight
    final_score = (similarity * base_weight) + (kw_cov * keyword_weight)

    # Mild teacher-like calibration:
    # - Give slight credit to relevant mid-range answers.
    # - Keep very weak answers from being inflated.
    if 0.45 <= final_score <= 0.75:
        final_score += 0.03
    elif final_score < 0.30:
        final_score *= 0.95

    final_score = max(0.0, min(1.0, final_score))
    sem_score = max(0.0, min(1.0, sem_score))

    marks = round(final_score * max_marks, 2)
    semantic_marks = round(sem_score * max_marks, 2)
    percentage = round(final_score * 100, 2)
    grade, feedback = _grade(final_score)

    logger.info(
        "Marks awarded: %.2f / %.2f  |  Grade: %s  |  Score: %.2f%%",
        marks, max_marks, grade, percentage,
    )

    return EvaluationResult(
        raw_similarity=round(similarity, 4),
        semantic_score=round(sem_score, 4),
        semantic_marks=semantic_marks,
        keyword_coverage=round(kw_cov, 4),
        final_score=round(final_score, 4),
        marks=marks,
        max_marks=max_marks,
        percentage=percentage,
        grade=grade,
        feedback=feedback,
    )