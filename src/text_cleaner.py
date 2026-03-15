"""
text_cleaner.py
───────────────
NLP-grade text cleaning pipeline for OCR output.

Steps:
  1. Unicode normalisation
  2. Lowercase
  3. Remove non-alphanumeric characters (keep spaces)
  4. Expand common contractions
  5. Fix common OCR substitutions  (0↔O, 1↔l/I, etc.)
  6. Spell correction via SymSpell (fast, dictionary-based)
  7. Tokenise
  8. Remove NLTK stopwords (optional — two modes: 'full' or 'light')
  9. Lemmatise with WordNet (optional)
  10. Re-join tokens
"""

import re
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)

# ── NLTK resources (download once) ───────────────────────────────────────────

def _ensure_nltk():
    import nltk
    resources = {
        "punkt": "tokenizers/punkt",
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords",
        "wordnet": "corpora/wordnet",
        "omw-1.4": "corpora/omw-1.4",
    }
    for resource, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource, quiet=True)

_ensure_nltk()

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_STOPWORDS_EN = set(stopwords.words("english"))
_LEMMATIZER = WordNetLemmatizer()

# ── Contractions map ─────────────────────────────────────────────────────────

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "isn't": "is not",
    "aren't": "are not", "wasn't": "was not", "weren't": "were not",
    "don't": "do not", "doesn't": "does not", "didn't": "did not",
    "it's": "it is", "that's": "that is", "there's": "there is",
    "they're": "they are", "we're": "we are", "you're": "you are",
    "i'm": "i am", "i've": "i have", "we've": "we have",
    "they've": "they have", "he's": "he is", "she's": "she is",
    "let's": "let us", "what's": "what is",
}

# ── OCR substitution fixes ────────────────────────────────────────────────────

OCR_FIXES = [
    (r"\b0(?=[a-z])", "o"),        # 0ther → other
    (r"(?<=[a-z])0\b", "o"),       # als0 → also
    (r"\b1(?=[a-z])", "l"),        # 1ight → light
    (r"\bI(?=[a-z])", "l"),        # Ilght → light  (I misread as l)
    (r"(?<!\w)rn(?!\w)", "m"),     # 'rn' → 'm' (classic OCR confusion)
    (r"(?<!\w)vv(?!\w)", "w"),     # 'vv' → 'w'
]


# ── SymSpell spell correction (lazy init) ─────────────────────────────────────

_symspell = None

def _get_symspell():
    global _symspell
    if _symspell is not None:
        return _symspell
    try:
        from symspellpy import SymSpell
        import pkg_resources

        sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        sym.load_dictionary(dict_path, term_index=0, count_index=1)
        _symspell = sym
        logger.info("SymSpell dictionary loaded.")
    except Exception as e:
        logger.warning("SymSpell not available (%s). Skipping spell correction.", e)
        _symspell = False          # Mark as unavailable
    return _symspell


def spell_correct_word(word: str) -> str:
    """Correct a single word using SymSpell (edit distance ≤ 2)."""
    sym = _get_symspell()
    if not sym:
        return word
    suggestions = sym.lookup(word, verbosity=0, max_edit_distance=2)
    return suggestions[0].term if suggestions else word


# ── Core cleaning functions ───────────────────────────────────────────────────

def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def fix_ocr_errors(text: str) -> str:
    for pattern, replacement in OCR_FIXES:
        text = re.sub(pattern, replacement, text)
    return text


def expand_contractions(text: str) -> str:
    for contraction, expanded in CONTRACTIONS.items():
        text = text.replace(contraction, expanded)
    return text


def remove_noise(text: str) -> str:
    """Keep only letters, digits, and spaces."""
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenise(text: str) -> List[str]:
    try:
        return word_tokenize(text)
    except LookupError:
        import nltk

        # Newer NLTK versions may require punkt_tab for sentence tokenization.
        nltk.download("punkt_tab", quiet=True)
        try:
            return word_tokenize(text)
        except LookupError:
            logger.warning("NLTK punkt resources unavailable; falling back to whitespace tokenization.")
            return text.split()


def remove_stopwords(tokens: List[str], mode: str = "light") -> List[str]:
    """
    mode='light'  → remove only very common function words
    mode='full'   → remove entire NLTK stopword list (aggressive)
    """
    if mode == "full":
        return [t for t in tokens if t not in _STOPWORDS_EN]
    # Light: keep informative stopwords like 'not', 'no', numbers
    light_stop = {
        "a", "an", "the", "is", "are", "was", "were", "be",
        "been", "being", "have", "has", "had", "do", "does", "did",
        "will", "would", "shall", "should", "may", "might", "can",
        "could", "of", "in", "on", "at", "to", "for", "with", "by",
        "from", "up", "about", "into", "through", "during",
        "this", "that", "these", "those", "it", "its",
    }
    return [t for t in tokens if t not in light_stop]


def lemmatise(tokens: List[str]) -> List[str]:
    return [_LEMMATIZER.lemmatize(t) for t in tokens]


# ── Main entry point ──────────────────────────────────────────────────────────

def clean_text(
    text: str,
    spell_check: bool = True,
    remove_stops: bool = True,
    lemmatize: bool = True,
    stopword_mode: str = "light",
) -> str:
    """
    Full cleaning pipeline.

    Args:
        text          : Raw OCR output.
        spell_check   : Apply SymSpell spell correction.
        remove_stops  : Remove stopwords.
        lemmatize     : Lemmatise tokens.
        stopword_mode : 'light' or 'full'.

    Returns:
        Cleaned text string ready for similarity comparison.
    """
    if not text or not text.strip():
        return ""

    text = normalise_unicode(text)
    text = text.lower()
    text = fix_ocr_errors(text)
    text = expand_contractions(text)
    text = remove_noise(text)

    tokens = tokenise(text)

    if spell_check:
        tokens = [spell_correct_word(t) if len(t) > 2 else t for t in tokens]

    if remove_stops:
        tokens = remove_stopwords(tokens, mode=stopword_mode)

    if lemmatize:
        tokens = lemmatise(tokens)

    # Remove empty / single-char residue
    tokens = [t for t in tokens if len(t) > 1]

    return " ".join(tokens)