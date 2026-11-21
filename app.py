"""Streamlit entry point for NoteScribe."""

from io import BytesIO
from pathlib import Path
import html
import importlib
import re

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image

from notescribe.gemini_tools import (
    GeminiError,
    generate_study_questions,
    process_and_generate_master_note,
    translate_notes,
)
from notescribe.image_processing import split_into_lines
from notescribe.ocr import OCRError, recognize_with_easyocr


def _load_suggest_master_terms():
    """Optionally load suggest_master_terms if notescribe.masternotes exists."""
    try:
        module = importlib.import_module("notescribe.masternotes")
        return getattr(module, "suggest_master_terms", None)
    except Exception:
        return None


suggest_master_terms = _load_suggest_master_terms()

LANGUAGES = ["Hindi", "Tamil", "Assamese"]

st.set_page_config(
    page_title="NoteScribe | Handwritten Notes",
    page_icon="✍️",
    layout="wide",
)


def load_styles() -> None:
    """Load the optional app stylesheet."""
    css_path = Path(__file__).parent / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def render_brand_header() -> None:
    """Show the NoteScribe heading."""
    st.markdown(
        """
        <div class="hero">
          <div class="logo">✍️</div>
          <div>
            <h1>NoteScribe</h1>
            <p>Turn handwritten pages into clear, editable notes and revision questions.</p>
          </div>
          <div class="hero-tag">AI NOTE DIGITIZER</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_uploaded_pages(uploaded_files) -> list[tuple[str, int, Image.Image]]:
    """Convert uploaded images and PDF pages into PIL images."""
    pages = []

    for uploaded in uploaded_files:
        uploaded.seek(0)
        file_bytes = uploaded.getvalue()
        filename = uploaded.name.lower()

        if filename.endswith(".pdf"):
            document = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                for page_number, pdf_page in enumerate(document, start=1):
                    pixmap = pdf_page.get_pixmap(
                        matrix=fitz.Matrix(2, 2),
                        alpha=False,
                    )
                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    pages.append((uploaded.name, page_number, image))
            finally:
                document.close()
        else:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
            pages.append((uploaded.name, 1, image))

    return pages


def _read_latex_group(source: str, position: int) -> tuple[str, int]:
    """Read a braced LaTeX group and return its contents and ending position."""
    if position >= len(source) or source[position] != "{":
        if position >= len(source):
            return "", position
        return source[position], position + 1

    depth = 0
    start = position + 1

    for index in range(position, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[start:index], index + 1

    return source[start:], len(source)


def _latex_to_pdf_html(source: str) -> str:
    """Convert common LaTeX formulas into HTML for PDF rendering."""
    output = []
    index = 0

    replacements = {
        "cdot": "&middot;",
        "times": "&times;",
        "div": "&divide;",
        "leq": "&le;",
        "geq": "&ge;",
        "neq": "&ne;",
        "pi": "&pi;",
        "theta": "&theta;",
        "alpha": "&alpha;",
        "beta": "&beta;",
        "sqrt": "&radic;",
    }

    while index < len(source):
        if source.startswith(r"\frac", index):
            numerator, next_index = _read_latex_group(source, index + 5)
            denominator, next_index = _read_latex_group(source, next_index)

            output.append(
                '<table class="fraction"><tbody>'
                '<tr><td class="numerator">'
                f"{_latex_to_pdf_html(numerator)}"
                "</td></tr>"
                '<tr><td class="denominator">'
                f"{_latex_to_pdf_html(denominator)}"
                "</td></tr>"
                "</tbody></table>"
            )
            index = next_index
            continue

        if source.startswith(r"\left", index):
            index += len(r"\left")
            continue

        if source.startswith(r"\right", index):
            index += len(r"\right")
            continue

        if source.startswith(r"\text", index):
            text_value, next_index = _read_latex_group(source, index + 5)
            output.append(html.escape(text_value))
            index = next_index
            continue

        if source[index] == "\\":
            command_end = index + 1
            while command_end < len(source) and source[command_end].isalpha():
                command_end += 1

            command = source[index + 1 : command_end]
            output.append(replacements.get(command, html.escape(command)))
            index = command_end
            continue

        if source[index] in ("_", "^"):
            tag = "sub" if source[index] == "_" else "sup"
            value, next_index = _read_latex_group(source, index + 1)
            output.append(f"<{tag}>{_latex_to_pdf_html(value)}</{tag}>")
            index = next_index
            continue

        if source[index] in "{}":
            index += 1
            continue

        output.append(html.escape(source[index]))
        index += 1

    return "".join(output)


def _notes_to_pdf_html(text: str) -> str:
    """Convert note headings, bullets, and LaTeX formulas into HTML."""
    math_pattern = re.compile(r"(\$\$.*?\$\$|\$.*?\$)", re.DOTALL)
    html_parts = []

    def render_plain_text(plain_text: str) -> str:
        rendered_lines = []

        for line in plain_text.splitlines():
            escaped = html.escape(line)
            escaped = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", escaped)

            if line.startswith("### "):
                rendered_lines.append(f"<h3>{html.escape(line[4:])}</h3>")
            elif line.startswith("## "):
                rendered_lines.append(f"<h2>{html.escape(line[3:])}</h2>")
            elif line.startswith("# "):
                rendered_lines.append(f"<h1>{html.escape(line[2:])}</h1>")
            elif line.startswith("- "):
                rendered_lines.append(f"<p class='bullet'>• {escaped[2:]}</p>")
            elif line.strip():
                rendered_lines.append(f"<p>{escaped}</p>")
            else:
                rendered_lines.append("<br>")

        return "".join(rendered_lines)

    position = 0
    for match in math_pattern.finditer(text):
        html_parts.append(render_plain_text(text[position : match.start()]))

        expression = match.group(0)
        is_display_math = expression.startswith("$$")
        delimiter_size = 2 if is_display_math else 1
        latex = expression[delimiter_size:-delimiter_size].strip()
        rendered_formula = _latex_to_pdf_html(latex)

        if is_display_math:
            html_parts.append(f'<div class="display-math">{rendered_formula}</div>')
        else:
            html_parts.append(f'<span class="inline-math">{rendered_formula}</span>')

        position = match.end()

    html_parts.append(render_plain_text(text[position:]))
    return "".join(html_parts)


def make_pdf_bytes(text: str, title: str) -> bytes:
    """Create a paginated PDF with typeset stacked fractions."""
    safe_title = html.escape(title)
    note_html = _notes_to_pdf_html(text)

    story = fitz.Story(
        html=f"<h1>{safe_title}</h1>{note_html}",
        user_css="""
            body {
                font-family: sans-serif;
                font-size: 11pt;
                line-height: 1.5;
            }
            h1 {
                font-size: 18pt;
                margin-bottom: 18pt;
            }
            h2 {
                font-size: 15pt;
                margin-top: 14pt;
            }
            h3 {
                font-size: 13pt;
                margin-top: 10pt;
            }
            p {
                margin: 4pt 0;
            }
            .bullet {
                margin-left: 14pt;
            }
            .display-math {
                text-align: center;
                margin: 12pt 0;
                font-size: 14pt;
            }
            .inline-math {
                font-size: 12pt;
            }
            .fraction {
                display: inline-table;
                width: auto;
                border-collapse: collapse;
                text-align: center;
                vertical-align: middle;
                line-height: 1.05;
                margin: 0 2pt;
            }
            .fraction td {
                padding: 0 0.12em;
                text-align: center;
                white-space: nowrap;
            }
            .fraction .numerator {
                border-bottom: 0.7pt solid black;
            }
        """,
    )

    buffer = BytesIO()
    writer = fitz.DocumentWriter(buffer)
    page_rect = fitz.Rect(0, 0, 595, 842)
    content_rect = fitz.Rect(50, 55, 545, 787)

    more = True
    while more:
        device = writer.begin_page(page_rect)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()

    writer.close()
    return buffer.getvalue()


def main() -> None:
    load_styles()
    render_brand_header()

    st.session_state.setdefault("raw_ocr_text", "")
    st.session_state.setdefault("ocr_transcription", "")
    st.session_state.setdefault("master_note", "")
    st.session_state.setdefault("study_questions", "")
    st.session_state.setdefault("translated_notes", "")

    tab_notes, tab_questions, tab_translate = st.tabs(
        [
            "📄 1. Digitizer & Master Note",
            "🎯 2. Revision Questions",
            "🌐 3. Translation",
        ]
    )

    with tab_notes:
        uploaded_files = st.file_uploader(
            "Choose page images or PDF files",
            type=["png", "jpg", "jpeg", "webp", "pdf"],
            accept_multiple_files=True,
            help="Upload images or multi-page PDFs.",
        )

        if uploaded_files:
            pages = []
            try:
                pages = get_uploaded_pages(uploaded_files)
            except Exception as error:
                st.error(f"Could not read uploaded files: {error}")

            if pages:
                with st.expander(f"Preview uploaded pages ({len(pages)})", expanded=True):
                    for filename, page_number, image in pages:
                        st.image(
                            image,
                            caption=f"{filename} — page {page_number}",
                            use_container_width=True,
                        )

                col1, col2 = st.columns([3, 1])
                with col2:
                    use_segmentation = st.checkbox(
                        "Adaptive line segmentation",
                        value=False,
                        help="Slice handwriting into horizontal lines before OCR.",
                    )

                with col1:
                    run_btn = st.button(
                        "✦ Create English master note",
                        type="primary",
                        use_container_width=True,
                    )

                if run_btn:
                    try:
                        page_results = []
                        progress = st.progress(0, text="Starting OCR…")

                        with st.spinner("Extracting handwritten text and formulating note…"):
                            for index, (filename, page_number, image) in enumerate(
                                pages, start=1
                            ):
                                if use_segmentation:
                                    line_crops = split_into_lines(image)
                                    recognized_lines = [
                                        recognize_with_easyocr(crop)
                                        for crop in line_crops
                                    ]
                                    recognized_text = "\n".join(
                                        line.strip()
                                        for line in recognized_lines
                                        if line.strip()
                                    )
                                else:
                                    recognized_text = recognize_with_easyocr(image)

                                page_results.append(
                                    f"--- {filename} — page {page_number} ---\n"
                                    f"{recognized_text}"
                                )
                                progress.progress(
                                    index / len(pages),
                                    text=f"Processed page {index} of {len(pages)}",
                                )

                            raw_transcription = "\n\n".join(page_results)
                            if not raw_transcription.strip():
                                progress.empty()
                                st.warning(
                                    "EasyOCR did not find readable text on the page."
                                )
                            else:
                                st.session_state["raw_ocr_text"] = raw_transcription
                                corrected_text, master_note_text = (
                                    process_and_generate_master_note(raw_transcription)
                                )
                                st.session_state["ocr_transcription"] = corrected_text
                                st.session_state["master_note"] = master_note_text
                                st.session_state["study_questions"] = ""
                                st.session_state["translated_notes"] = ""
                                progress.empty()
                                st.rerun()

                    except OCRError as error:
                        st.error(f"OCR Error: {error}")
                    except GeminiError as error:
                        st.error(f"Gemini API Error: {error}")
                    except Exception as error:
                        st.error(f"Execution Error: {error}")

        master_note = st.session_state.get("master_note", "")
        if master_note:
            st.markdown("---")
            st.markdown("## 📝 Generated Master Note")
            st.markdown(master_note)

            raw_text = st.session_state.get("raw_ocr_text", "")
            if raw_text and suggest_master_terms:
                suggestions = suggest_master_terms(raw_text, master_note)
                if suggestions:
                    with st.expander(f"Review Term Corrections ({len(suggestions)})"):
                        for ocr_term, verified_term in suggestions:
                            st.markdown(
                                f"- OCR: `{ocr_term}` → Suggestion: **{verified_term}**"
                            )

            st.download_button(
                "↓ Download English master note as PDF",
                data=make_pdf_bytes(master_note, "Master note — English"),
                file_name="master-note-english.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_master_note",
            )
        elif not uploaded_files:
            st.info("Upload page images or PDFs above to begin.")

    with tab_questions:
        st.markdown("## 🎯 Conceptual Revision & Practice Questions")
        st.write(
            "Generate short and long conceptual questions based on your master note or topic."
        )

        master_note = st.session_state.get("master_note", "")

        if master_note:
            st.success(
                "✅ Ready to generate questions directly from your active Master Note."
            )
            active_text = master_note
        else:
            st.info(
                "💡 Generate a Master Note in Tab 1 first, or paste your topic/notes below:"
            )
            manual_text = st.text_area(
                "Notes / Topic Content",
                placeholder="Paste any theoretical notes, formulas, or study topics here...",
                height=130,
                key="manual_questions_input",
            )
            active_text = manual_text.strip()

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            num_short_q = st.number_input(
                "Number of Short Conceptual Questions",
                min_value=1,
                max_value=10,
                value=4,
                step=1,
                key="tab_short_q",
            )
        with col_q2:
            num_long_q = st.number_input(
                "Number of Long Conceptual Questions",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key="tab_long_q",
            )

        if st.button(
            "✦ Generate Conceptual Questions",
            type="primary",
            use_container_width=True,
        ):
            if not active_text:
                st.warning(
                    "Please generate a Master Note in Tab 1 or enter notes above first."
                )
            else:
                try:
                    with st.spinner(
                        f"Generating {num_short_q} short and {num_long_q} long conceptual questions…"
                    ):
                        questions = generate_study_questions(
                            active_text,
                            num_short=int(num_short_q),
                            num_long=int(num_long_q),
                        )
                        st.session_state["study_questions"] = questions
                        st.rerun()
                except GeminiError as error:
                    st.error(f"Gemini API Error: {error}")
                except Exception as error:
                    st.error(f"Error: {error}")

        study_questions = st.session_state.get("study_questions", "")
        if study_questions:
            st.markdown("---")
            st.markdown("### 📋 Generated Revision Questions")
            st.markdown(study_questions)
            st.download_button(
                "↓ Download Revision Questions as PDF",
                data=make_pdf_bytes(
                    study_questions, "NoteScribe — Conceptual Questions"
                ),
                file_name="conceptual-questions.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_practice_questions",
            )

    with tab_translate:
        st.markdown("## 🌐 Translate Master Note")
        master_note = st.session_state.get("master_note", "")

        if not master_note:
            st.info("Generate a Master Note in Tab 1 first to use translation.")
        else:
            target_language = st.selectbox(
                "Choose translation language",
                LANGUAGES,
                key="target_lang_select",
            )

            if st.button("Translate master note", use_container_width=True):
                try:
                    with st.spinner(
                        f"Translating master note into {target_language}…"
                    ):
                        st.session_state["translated_notes"] = translate_notes(
                            master_note,
                            target_language,
                        )
                        st.rerun()
                except GeminiError as error:
                    st.error(str(error))

            translated_notes = st.session_state.get("translated_notes", "")
            if translated_notes:
                st.markdown(f"### Master note translation — {target_language}")
                st.markdown(translated_notes)

                st.download_button(
                    "↓ Download translated master note as PDF",
                    data=make_pdf_bytes(
                        translated_notes,
                        f"Master note translation — {target_language}",
                    ),
                    file_name=f"master-note-{target_language.lower()}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_translated_notes",
                )


if __name__ == "__main__":
    main()