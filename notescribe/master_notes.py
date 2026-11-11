"""Small, review-only vocabulary suggestions from verified master notes."""

import re
from difflib import get_close_matches


def _vocabulary(text: str) -> set[str]:
    """Extract words suitable for lightweight spelling suggestions."""
    return {word.lower() for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", text)}


def suggest_master_terms(transcription: str, master_text: str) -> list[tuple[str, str]]:
    """Suggest similar master-note words without automatically editing text."""
    known_words = _vocabulary(master_text)
    suggestions = []
    already_checked: set[str] = set()

    for word in re.findall(r"[A-Za-z][A-Za-z'-]{3,}", transcription):
        key = word.lower()
        if key in known_words or key in already_checked:
            continue
        already_checked.add(key)
        match = get_close_matches(key, known_words, n=1, cutoff=0.82)
        if match:
            suggestions.append((word, match[0]))
    return suggestions
