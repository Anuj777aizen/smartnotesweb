"""Streamlit entry point for NoteScribe."""

from io import BytesIO
from pathlib import Path
import html
import re

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image

from notescribe.gemini_tools import (
    GeminiError,
    correct_grammar,
    generate_master_note,
    translate_notes,
)
from notescribe.ocr import OCRError, recognize_with_easyocr


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
            <p>Turn handwritten pages into clear, editable notes.</p>
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
                f'{_latex_to_pdf_html(numerator)}'
                '</td></tr>'
                '<tr><td class="denominator">'
                f'{_latex_to_pdf_html(denominator)}'
                '</td></tr>'
                '</tbody></table>'
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

            command = source[index + 1:command_end]
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
        html_parts.append(render_plain_text(text[position:match.start()]))

        expression = match.group(0)
        is_display_math = expression.startswith("$$")
        delimiter_size = 2 if is_display_math else 1
        latex = expression[delimiter_size:-delimiter_size].strip()
        rendered_formula = _latex_to_pdf_html(latex)

        if is_display_math:
            html_parts.append(
                f'<div class="display-math">{rendered_formula}</div>'
            )
        else:
            html_parts.append(
                f'<span class="inline-math">{rendered_formula}</span>'
            )

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

    st.markdown(
        "<div class='section-label'>WORKSPACE</div>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Choose page images or PDF files",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        help="Upload images or multi-page PDFs.",
    )

    st.session_state.setdefault("ocr_transcription", "")
    st.session_state.setdefault("master_note", "")
    st.session_state.setdefault("translated_notes", "")

    if not uploaded_files:
        st.markdown(
            """
            <div class="empty-state">
              <div class="empty-icon">📄</div>
              <h3>Your notes, ready to be digitized</h3>
              <p>Upload page images or PDFs to create a master note.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    try:
        pages = get_uploaded_pages(uploaded_files)
    except Exception as error:
        st.error(f"Could not read an uploaded file: {error}")
        return

    if not pages:
        st.warning("No pages were found in the uploaded files.")
        return

    with st.expander(f"Preview uploaded pages ({len(pages)})"):
        for filename, page_number, image in pages:
            st.image(
                image,
                caption=f"{filename} — page {page_number}",
                use_container_width=True,
            )

    if st.button(
        "✦  Create English master note",
        type="primary",
        use_container_width=True,
    ):
        try:
            page_results = []
            progress = st.progress(0, text="Starting OCR…")

            with st.spinner("Reading pages and creating the English master note…"):
                for index, (filename, page_number, image) in enumerate(
                    pages,
                    start=1,
                ):
                    recognized_text = recognize_with_easyocr(image)
                    page_results.append(
                        f"--- {filename} — page {page_number} ---\n"
                        f"{recognized_text}"
                    )
                    progress.progress(
                        index / len(pages),
                        text=f"Read page {index} of {len(pages)}",
                    )

                raw_transcription = "\n\n".join(page_results)
                if not raw_transcription.strip():
                    progress.empty()
                    st.warning(
                        "No text was recognized. Try clearer, "
                        "higher-resolution pages."
                    )
                    return

                corrected_transcription = correct_grammar(raw_transcription)
                st.session_state["ocr_transcription"] = corrected_transcription
                st.session_state["master_note"] = generate_master_note(
                    corrected_transcription
                )
                st.session_state["translated_notes"] = ""

            progress.empty()

        except OCRError as error:
            st.error(str(error))
        except GeminiError as error:
            st.error(f"Could not correct the OCR or create the note: {error}")
        except Exception as error:
            st.error(f"Could not process the pages: {error}")

    master_note = st.session_state.get("master_note", "")
    if master_note:
        st.markdown("## Master note — English")
        st.markdown(master_note)

        st.download_button(
            "↓  Download English master note as PDF",
            data=make_pdf_bytes(master_note, "Master note — English"),
            file_name="master-note-english.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_master_note",
        )

        st.markdown("## Translate master note")

        target_language = st.selectbox(
            "Choose translation language",
            LANGUAGES,
            key="target_language",
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
            except GeminiError as error:
                st.error(str(error))

    translated_notes = st.session_state.get("translated_notes", "")
    if translated_notes:
        target_language = st.session_state.get("target_language", "Hindi")
        st.markdown(f"### Master note translation — {target_language}")
        st.markdown(translated_notes)

        st.download_button(
            "↓  Download translated master note as PDF",
            data=make_pdf_bytes(
                translated_notes,
                f"Master note translation — {target_language}",
            ),
            file_name=f"master-note-{target_language.lower()}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_translated_notes",
        )

    if st.session_state.get("ocr_transcription"):
        st.caption(
            f"Processed {len(pages)} page(s) with EasyOCR. "
            "OCR text is corrected in the background and used to create "
            "the English master note."
        )


if __name__ == "__main__":
    main()