"""Small, review-only vocabulary suggestions from verified master notes."""

import re
from difflib import get_close_matches


def _vocabulary_map(text: str) -> dict[str, str]:
    """Extract unique words mapped from their lowercased form to original casing."""
    vocab = {}
    for word in re.findall(r"\b[A-Za-z][A-Za-z'-]*[A-Za-z]\b", text):
        clean_word = word.strip("'-")
        if len(clean_word) >= 3:
            key = clean_word.lower()
            if key not in vocab:
                vocab[key] = clean_word
    return vocab


def suggest_master_terms(
    transcription: str,
    master_text: str,
    cutoff: float = 0.82,
    limit: int = 15,
) -> list[tuple[str, str]]:
    """Suggest verified master-note terms for misspelled or noisy OCR tokens.

    Returns:
        A list of tuples: (original_ocr_word, suggested_master_word)
    """
    if not transcription.strip() or not master_text.strip():
        return []

    word_map = _vocabulary_map(master_text)
    known_keys = list(word_map.keys())

    suggestions: list[tuple[str, str]] = []
    already_checked: set[str] = set()

    # Find candidate words from the raw OCR transcription
    candidates = re.findall(r"\b[A-Za-z][A-Za-z'-]*[A-Za-z]\b", transcription)

    for token in candidates:
        word = token.strip("'-")
        if len(word) < 4:
            continue

        key = word.lower()
        if key in word_map or key in already_checked:
            continue

        already_checked.add(key)
        matches = get_close_matches(key, known_keys, n=1, cutoff=cutoff)

        if matches:
            best_match_cased = word_map[matches[0]]
            suggestions.append((word, best_match_cased))

            if len(suggestions) >= limit:
                break

    return suggestions