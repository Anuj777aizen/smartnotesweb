"""Gemini-powered correction, English master notes, and translation."""

import os
from functools import lru_cache

from google import genai


GEMINI_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-4-argon",
]

_last_working_model = None


class GeminiError(Exception):
    """Raised when a Gemini request cannot be completed."""


@lru_cache(maxsize=1)
def get_gemini_client():
    """Create and cache a Gemini client."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise GeminiError(
            "GEMINI_API_KEY is not set. Set it in your terminal, "
            "then restart the app."
        )

    try:
        return genai.Client(api_key=api_key)
    except Exception as error:
        raise GeminiError(f"Could not initialize Gemini: {error}") from error


def _can_try_another_model(error: Exception) -> bool:
    """Return whether a different model might handle this error."""
    message = str(error).upper()

    if any(term in message for term in ("NO CREDITS", "BILLING", "API_KEY_INVALID")):
        return False

    return any(
        term in message
        for term in (
            "503",
            "UNAVAILABLE",
            "429",
            "RESOURCE_EXHAUSTED",
            "404",
            "NOT_FOUND",
            "MODEL",
        )
    )


def _generate(prompt: str) -> str:
    """Generate text, automatically falling back to another Gemini model."""
    global _last_working_model

    client = get_gemini_client()
    models_to_try = GEMINI_MODELS.copy()

    if _last_working_model in models_to_try:
        models_to_try.remove(_last_working_model)
        models_to_try.insert(0, _last_working_model)

    errors = []

    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )
            result = (response.text or "").strip()

            if not result:
                errors.append(f"{model}: empty response")
                continue

            _last_working_model = model
            return result

        except Exception as error:
            if not _can_try_another_model(error):
                raise GeminiError(f"Gemini request failed: {error}") from error
            errors.append(f"{model}: {error}")

    details = "\n".join(errors)
    raise GeminiError(
        "None of the configured Gemini models could complete the request. "
        "Check your API key, quota, and model availability.\n"
        f"{details}"
    )


def correct_grammar(text: str) -> str:
    """Correct OCR text and format clearly identified math as LaTeX."""
    if not text.strip():
        raise GeminiError("There is no transcription to correct.")

    prompt = f"""
Correct grammar, spelling, punctuation, and obvious OCR errors in these notes.
Pay special attention to mathematical formulas and fractions.

Rules:
- Preserve the source meaning, values, variables, and operations.
- Correct a mathematical symbol only when the surrounding text makes it clear.
- Do not solve equations or invent missing values.
- Write fractions in LaTeX using \\frac{{numerator}}{{denominator}}.
- Put inline formulas between single dollar signs, like $\\frac{{x+1}}{{2}}$.
- Put standalone formulas between double dollar signs on separate lines, like:
$$
\\frac{{x+1}}{{2}} = 4
$$
- If a formula is unclear, preserve what was recognized rather than guessing.
- Return only the corrected notes.

NOTES:
{text}
"""
    return _generate(prompt)


def generate_master_note(text: str) -> str:
    """Generate one concise master note in English."""
    if not text.strip():
        raise GeminiError("There are no notes to summarize.")

    prompt = f"""
Create one concise master note from the source text in English.

Rules:
- Write the master note in English only.
- Capture the main ideas and important details.
- Use headings and bullet points where helpful.
- Preserve mathematical formulas, symbols, and values.
- Write fractions using LaTeX \\frac{{numerator}}{{denominator}}.
- Put inline formulas between single dollar signs, such as $\\frac{{x+1}}{{2}}$.
- Put standalone equations between double dollar signs on separate lines.
- Do not solve formulas or invent missing values.
- Do not add facts that are not in the source.
- Return only the English master note.

SOURCE:
{text}
"""
    return _generate(prompt)


def translate_notes(text: str, language: str) -> str:
    """Translate notes into the selected language."""
    if not text.strip():
        raise GeminiError("There are no notes to translate.")
    if not language.strip():
        raise GeminiError("Choose a language for the notes.")

    prompt = f"""
Translate these notes into {language}.

Rules:
- Preserve the original meaning, facts, names, numbers, and structure.
- Preserve mathematical formulas and LaTeX exactly.
- Do not add explanations or new facts.
- Return only the translated notes.

NOTES:
{text}
"""
    return _generate(prompt)