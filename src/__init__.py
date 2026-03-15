"""
src/__init__.py
───────────────
Package initialiser for the Handwritten Answer Evaluation System.

Ensures:
  1. The project root is on sys.path so all `src.*` imports resolve
     regardless of where Python is launched from.
  2. Logging is configured before any sub-module is imported.
  3. All public API objects are re-exported for convenience so callers
     can do:
         from src import load_image, preprocess_image, ...
     instead of importing each sub-module individually.
"""

import sys
import os
import logging

# ── 1. Add project root to sys.path ──────────────────────────────────────────
# __file__ = .../HandwrittenEvaluationSystem/src/__init__.py
# parent   = .../HandwrittenEvaluationSystem/
_SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# ── 2. Package-level logger ───────────────────────────────────────────────────
logging.getLogger(__name__).addHandler(logging.NullHandler())

# ── 3. Re-export public API ───────────────────────────────────────────────────
try:
    from src.image_loader import (
        select_image,
        load_image,
        get_image_info,
        resize_if_needed,
    )

    from src.preprocess import (
        preprocess_image,
        deskew,
        denoise,
        enhance_contrast,
        binarise,
        morphological_cleanup,
        remove_border,
    )

    from src.ocr_reader import (
        extract_text,
        get_reader,
    )

    from src.text_cleaner import (
        clean_text,
        normalise_unicode,
        fix_ocr_errors,
        expand_contractions,
        spell_correct_word,
        tokenise,
        remove_stopwords,
        lemmatise,
    )

    from src.similarity import (
        calculate_similarity,
        semantic_similarity,
        token_sort_similarity,
        partial_similarity,
        jaccard_similarity,
        keyword_coverage,
    )

    from src.marks_generator import (
        generate_marks,
        EvaluationResult,
        GRADE_BANDS,
    )

    from src.report_generator import generate_report

except ImportError as _e:
    # Gracefully skip re-exports if a dependency is not yet installed.
    # The individual modules will raise a clear error when called.
    import warnings
    warnings.warn(
        f"src package: could not pre-load all modules ({_e}). "
        "Run `pip install -r requirements.txt` to install dependencies.",
        ImportWarning,
        stacklevel=2,
    )

# ── 4. Version ────────────────────────────────────────────────────────────────
__version__ = "2.0.0"
__author__  = "Handwritten Evaluation System"
__all__ = [
    # image_loader
    "select_image", "load_image", "get_image_info", "resize_if_needed",
    # preprocess
    "preprocess_image", "deskew", "denoise", "enhance_contrast",
    "binarise", "morphological_cleanup", "remove_border",
    # ocr_reader
    "extract_text", "get_reader",
    # text_cleaner
    "clean_text", "normalise_unicode", "fix_ocr_errors",
    "expand_contractions", "spell_correct_word",
    "tokenise", "remove_stopwords", "lemmatise",
    # similarity
    "calculate_similarity", "semantic_similarity", "token_sort_similarity",
    "partial_similarity", "jaccard_similarity", "keyword_coverage",
    # marks_generator
    "generate_marks", "EvaluationResult", "GRADE_BANDS",
    # report_generator
    "generate_report",
]