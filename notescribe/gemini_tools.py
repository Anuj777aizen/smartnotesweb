"""Gemini API helper functions for NoteScribe with multi-model fallback."""

from __future__ import annotations

import os
import time
import streamlit as st

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from google import genai
except ImportError:
    genai = None


class GeminiError(Exception):
    """Raised when a Gemini API request cannot be completed."""


# Active Free-Tier text models tried in order
MODELS: tuple[str, ...] = (
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-4-hydrogen",
    "gemini-3.1-flash-lite",
)


def _get_api_key() -> str:
    """Read GEMINI_API_KEY from environment variables, .env, or .streamlit/secrets.toml."""
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
        "GEMINI_API_KEY is not set. Add it to your environment variables, .env file, or "
        ".streamlit/secrets.toml, then restart the app."
    )


def _discover_fallback_models(client, already_tried: set[str]) -> list[str]:
    """Query the API for other active text-generation Gemini models."""
    discovered: list[str] = []
    excluded_keywords = (
        "embedding",
        "imagen",
        "image",
        "banana",
        "veo",
        "tts",
        "audio",
        "robotics",
        "live",
        "transcribe",
        "omni",
        "2.0",
        "2.5",
        "argon",
        "carbon",
        "pro-preview",
    )

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

    # Prefer newest Flash / Flash-Lite models first
    discovered.sort(reverse=True)
    discovered.sort(key=lambda name: "flash" not in name.lower())
    return discovered[:5]


def _is_temporary_overload(exc: Exception) -> bool:
    """Check if an exception is a temporary 503 high-demand error."""
    msg = str(exc).upper()
    return "503" in msg or "UNAVAILABLE" in msg


def _generate_text(prompt: str) -> str:
    """Call Gemini across active text models with safe retry on temporary 503 spikes."""
    if genai is None:
        raise GeminiError(
            "The 'google-genai' package is not installed. Run: pip install google-genai"
        )

    api_key = _get_api_key()
    client = genai.Client(api_key=api_key)

    errors: list[str] = []
    tried: set[str] = set()
    overloaded_models: list[str] = []

    def try_models(
        model_list: list[str] | tuple[str, ...],
        allow_retry: bool = False,
    ) -> str | None:
        for model_name in model_list:
            if not allow_retry and model_name in tried:
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
                if not allow_retry and _is_temporary_overload(exc):
                    overloaded_models.append(model_name)
        return None

    # 1. Try primary active free-tier text models
    result = try_models(MODELS)
    if result:
        return result

    # 2. Try any newly discovered non-image text models
    dynamic_models = _discover_fallback_models(client, tried)
    if dynamic_models:
        result = try_models(dynamic_models)
        if result:
            return result

    # 3. If valid models only failed due to a brief 503 spike, wait once and retry
    if overloaded_models:
        time.sleep(3)
        result = try_models(overloaded_models, allow_retry=True)
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


def generate_group_master_note(
    group_name: str,
    topic: str,
    submissions: list,
) -> str:
    """Merge OCR notes from multiple class/group members into one English Master Note."""
    if not submissions:
        raise GeminiError("No student submissions found in this room yet.")

    blocks = []
    for index, sub in enumerate(submissions, start=1):
        blocks.append(
            f"=== Submission #{index} (Member: {sub['member_name']}) ===\n"
            f"{sub['extracted_text']}"
        )
    combined_notes = "\n\n".join(blocks)

    prompt = f"""You are an expert academic note synthesizer for a class or study group.
Class / Group: {group_name}
Topic: {topic}
Total Member Submissions: {len(submissions)}

Below are OCR-extracted notes uploaded by different classmates covering the same lecture/topic.

Tasks:
1. Cross-verify across submissions to fix OCR errors, broken words, and misread symbols.
2. Deduplicate overlapping points so every concept is explained clearly once without repetition.
3. Preserve every unique definition, formula, derivation, edge case, or example captured by even a single student.
4. If students wrote conflicting facts or values, list them briefly at the bottom under `## Needs Clarification`.
5. Output a clean, exam-ready English Master Note using standard Markdown:
   - Use `#`, `##`, and `###` headings
   - Use `- ` bullet points and `**bold**` key terms
   - Format mathematical formulas using standard LaTeX (`$...$` for inline and `$$...$$` for display equations)

Classmate Notes:
{combined_notes}
"""

    return _generate_text(prompt)


def generate_probable_question_paper(
    pyq_content: str,
    subject_name: str = "",
    total_marks: int = 70,
    duration: str = "3 Hours",
    difficulty: str = "Match PYQ Level",
    include_hints: bool = True,
    master_note_context: str = "",
) -> str:
    """Analyze 3–4+ PYQs and generate a probable future practice question paper."""
    cleaned_pyqs = (pyq_content or "").strip()
    if not cleaned_pyqs:
        raise GeminiError("Please upload or paste at least one PYQ paper/question set first.")

    hints_rule = (
        "- Section 3: `## Quick Answer Hints & Key Formulas` — Provide concise 2–3 line "
        "solution approaches or key formulas for each question."
        if include_hints
        else "- Do NOT include answers or solution hints; output only the analysis and question paper."
    )

    notes_block = ""
    if (master_note_context or "").strip():
        notes_block = f"""
Optional Class Master Notes (prioritize predicting questions that align with both the PYQ patterns AND these studied topics):
{master_note_context.strip()}
"""

    prompt = f"""You are a senior university professor and exam-paper setter.
Analyze the Previous Year Questions (PYQs) provided below to identify:
- Frequently repeated concepts and high-weightage topics
- Question framing patterns (derivations, numericals, compare/contrast, short theory)
- Typical mark distribution and internal choices (e.g., OR questions)

Then create a realistic **Future Probable Practice Question Paper** for students to practice.
Do NOT just copy-paste old questions word-for-word; create fresh, exam-probable variations, closely related numericals, and logical follow-up questions based on the PYQ trends.

Paper Specifications:
- Subject / Exam Title: {subject_name.strip() or "Auto-detect from PYQs"}
- Maximum Marks: {total_marks}
- Time Allowed: {duration}
- Difficulty Target: {difficulty}

Required Output Format (clean Markdown with `#`, `##`, `###`, `- `, `**bold**`, and standard LaTeX `$...$` / `$$...$$` for math):
1. `## PYQ Trend & High-Probability Topics` — A brief bulleted summary of the 4–6 most repeated/high-probability topics spotted across the PYQs.
2. `# Probable Practice Question Paper` — A complete, ready-to-solve paper with:
   - Header showing Subject, Time Allowed, and Maximum Marks
   - Properly grouped sections (e.g., Section A: Short Answer, Section B: Medium Conceptual/Numerical, Section C: Long Analytical/Derivation) whose marks add up to exactly **{total_marks} marks**
   - Clear mark tags next to every question (e.g., **[5 Marks]**)
{hints_rule}

Previous Year Questions (PYQs):
{cleaned_pyqs}
{notes_block}
"""

    return _generate_text(prompt)