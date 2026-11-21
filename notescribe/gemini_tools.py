"""Gemini API helper functions for NoteScribe with multi-model fallback."""

from __future__ import annotations

import os
import streamlit as st

try:
    from google import genai
except ImportError:
    genai = None


class GeminiError(Exception):
    """Raised when a Gemini API request cannot be completed."""


# Primary models tried in order (fastest first, then stronger/latest aliases)
MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.1-pro-preview",
    "gemini-flash-latest",
    "gemini-pro-latest",
)


def _get_api_key() -> str:
    """Read GEMINI_API_KEY from environment variables or .streamlit/secrets.toml."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    try:
        secret_key = st.secrets.get("GEMINI_API_KEY", "")
        if isinstance(secret_key, str) and secret_key.strip():
            return secret_key.strip()
    except Exception:
        pass

    raise GeminiError(
        "GEMINI_API_KEY is not set. Add it to your environment variables or "
        ".streamlit/secrets.toml, then restart the app."
    )


def _discover_fallback_models(client, already_tried: set[str]) -> list[str]:
    """Query the API for any other available Gemini text-generation models."""
    discovered: list[str] = []
    excluded_keywords = ("embedding", "imagen", "veo", "tts", "audio", "robotics")

    try:
        for model in client.models.list():
            raw_name = getattr(model, "name", "") or ""
            short_name = raw_name.replace("models/", "").strip()
            actions = getattr(model, "supported_actions", None) or []

            if not short_name or short_name in already_tried:
                continue
            if "gemini" not in short_name.lower():
                continue
            if any(word in short_name.lower() for word in excluded_keywords):
                continue
            if actions and "generateContent" not in actions:
                continue

            discovered.append(short_name)
    except Exception:
        return []

    # Prefer flash models first for speed, then pro models
    discovered.sort(key=lambda name: ("flash" not in name.lower(), name))
    return discovered[:5]


def _generate_text(prompt: str) -> str:
    """Call Gemini across multiple models until one succeeds."""
    if genai is None:
        raise GeminiError(
            "The 'google-genai' package is not installed. Run: pip install google-genai"
        )

    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)

    errors: list[str] = []
    tried: set[str] = set()

    def try_models(model_list: list[str] | tuple[str, ...]) -> str | None:
        for model_name in model_list:
            if model_name in tried:
                continue
            tried.add(model_name)
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = (getattr(response, "text", None) or "").strip()
                if text:
                    return text
                errors.append(f"{model_name}: Returned an empty response.")
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        return None

    # 1. Try configured multi-model list
    result = try_models(MODELS)
    if result:
        return result

    # 2. Auto-discover other active Gemini models on this API key
    dynamic_models = _discover_fallback_models(client, tried)
    if dynamic_models:
        result = try_models(dynamic_models)
        if result:
            return result

    raise GeminiError(
        "None of the configured Gemini models could complete the request. "
        "Check your API key, quota, and model availability: "
        + " | ".join(errors)
    )


def process_and_generate_master_note(raw_transcription: str) -> tuple[str, str]:
    """Correct OCR text and generate a structured English master note."""
    cleaned_input = (raw_transcription or "").strip()
    if not cleaned_input:
        raise GeminiError("Cannot generate a master note from empty OCR text.")

    prompt = f"""You are an expert academic note editor.
You are given raw OCR text extracted from handwritten student notes.

Perform two tasks:
1. Produce a clean, corrected version of the OCR transcription (fix spelling, broken words, and misread symbols without inventing unrelated content).
2. Produce a well-structured English Master Note using clean Markdown:
   - Use `#`, `##`, and `###` headings
   - Use `- ` bullet points and `**bold**` key terms
   - Format mathematical formulas using standard LaTeX (`$...$` for inline and `$$...$$` for display equations)

Return your response in this exact format with both markers included:

<<<CORRECTED_TRANSCRIPTION>>>
(put corrected transcription here)

<<<MASTER_NOTE>>>
(put structured English master note here)

Raw OCR Text:
{cleaned_input}
"""

    output = _generate_text(prompt)

    corrected_marker = "<<<CORRECTED_TRANSCRIPTION>>>"
    master_marker = "<<<MASTER_NOTE>>>"

    if corrected_marker in output and master_marker in output:
        _, remainder = output.split(corrected_marker, 1)
        corrected_part, master_part = remainder.split(master_marker, 1)
        return corrected_part.strip(), master_part.strip()

    return cleaned_input, output


def generate_study_questions(
    notes_text: str,
    num_short: int = 4,
    num_long: int = 2,
) -> str:
    """Generate short and long conceptual revision questions from notes."""
    cleaned_notes = (notes_text or "").strip()
    if not cleaned_notes:
        raise GeminiError("Please provide notes or a topic to generate questions.")

    prompt = f"""You are an expert exam coach and subject-matter tutor.
Based on the study notes below, generate conceptual revision questions that test real understanding rather than rote memorization.

Requirements:
- Section 1: `## Short Conceptual Questions` — Generate exactly {num_short} short-answer questions (2–3 marks each) with a brief 1–2 line key concept hint under each.
- Section 2: `## Long Conceptual Questions` — Generate exactly {num_long} long-answer/analytical questions (5–8 marks each) with bulleted key points to cover in the answer.
- Use clean Markdown (`##`, `###`, `- `, `**bold**`) and standard LaTeX (`$...$` / `$$...$$`) for any equations.

Study Notes:
{cleaned_notes}
"""

    return _generate_text(prompt)


def translate_notes(master_note: str, target_language: str) -> str:
    """Translate the master note into the selected language while preserving formatting."""
    cleaned_note = (master_note or "").strip()
    if not cleaned_note:
        raise GeminiError("No master note available to translate.")

    prompt = f"""Translate the following academic study notes into natural, accurate {target_language}.

Rules:
- Preserve all Markdown headings (`#`, `##`, `###`), bullet points (`- `), and `**bold**` formatting.
- Keep all LaTeX mathematical expressions (`$...$` and `$$...$$`) completely unchanged.
- Keep standard technical acronyms (e.g., LED, BLDC, HVAC, DC, AC) intact alongside the translation where appropriate.
- Output only the translated notes with no extra commentary.

Master Note:
{cleaned_note}
"""

    return _generate_text(prompt)