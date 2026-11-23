"""Streamlit entry point for NoteScribe (Solo + Multi-Room Wi-Fi Server + PYQ Predictor)."""

from io import BytesIO
from pathlib import Path
import html
import importlib
import re
import socket

import fitz  # PyMuPDF
import streamlit as st
from PIL import Image

import db
from notescribe.gemini_tools import (
    GeminiError,
    generate_group_master_note,
    generate_probable_question_paper,
    generate_study_questions,
    process_and_generate_master_note,
    translate_notes,
)
from notescribe.image_processing import split_into_lines
from notescribe.ocr import OCRError, recognize_with_easyocr

try:
    import qrcode
except ImportError:
    qrcode = None


def _load_suggest_master_terms():
    """Optionally load suggest_master_terms from notescribe.master_notes or notescribe.masternotes."""
    for module_name in ("notescribe.master_notes", "notescribe.masternotes"):
        try:
            module = importlib.import_module(module_name)
            func = getattr(module, "suggest_master_terms", None)
            if func is not None:
                return func
        except Exception:
            continue
    return None


suggest_master_terms = _load_suggest_master_terms()

LANGUAGES = ["Hindi", "Tamil", "Assamese"]

st.set_page_config(
    page_title="NoteScribe | Handwritten Notes",
    page_icon="✍️",
    layout="wide",
)

db.init_db()


def get_local_ip() -> str:
    """Detect this machine's local Wi-Fi IPv4 address."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def make_qr_png(url: str) -> bytes | None:
    """Generate a QR code PNG for quick phone access over Wi-Fi."""
    if qrcode is None:
        return None
    img = qrcode.make(url)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


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


def run_ocr_on_pages(
    pages: list[tuple[str, int, Image.Image]],
    use_segmentation: bool,
) -> str:
    """Run EasyOCR (with optional line segmentation) across all uploaded pages."""
    page_results = []
    progress = st.progress(0, text="Starting OCR…")

    try:
        for index, (filename, page_number, image) in enumerate(pages, start=1):
            if use_segmentation:
                line_crops = split_into_lines(image)
                recognized_lines = [
                    recognize_with_easyocr(crop) for crop in line_crops
                ]
                recognized_text = "\n".join(
                    line.strip() for line in recognized_lines if line.strip()
                )
            else:
                recognized_text = recognize_with_easyocr(image)

            page_results.append(
                f"--- {filename} — page {page_number} ---\n{recognized_text}"
            )
            progress.progress(
                index / len(pages),
                text=f"Processed page {index} of {len(pages)}",
            )
    finally:
        progress.empty()

    return "\n\n".join(page_results)


def extract_text_from_pyq_files(uploaded_files) -> str:
    """Extract text from PYQ PDFs (direct text first, OCR fallback) and images."""
    extracted_papers = []
    progress = st.progress(0, text="Reading PYQ files…")

    try:
        total_files = len(uploaded_files)
        for file_idx, uploaded in enumerate(uploaded_files, start=1):
            uploaded.seek(0)
            file_bytes = uploaded.getvalue()
            filename = uploaded.name
            file_parts = []

            if filename.lower().endswith(".pdf"):
                document = fitz.open(stream=file_bytes, filetype="pdf")
                try:
                    for page_num, pdf_page in enumerate(document, start=1):
                        direct_text = (pdf_page.get_text("text") or "").strip()
                        if len(direct_text) >= 40:
                            page_text = direct_text
                        else:
                            pixmap = pdf_page.get_pixmap(
                                matrix=fitz.Matrix(2, 2), alpha=False
                            )
                            image = Image.frombytes(
                                "RGB",
                                (pixmap.width, pixmap.height),
                                pixmap.samples,
                            )
                            page_text = recognize_with_easyocr(image)

                        if page_text.strip():
                            file_parts.append(f"[Page {page_num}]\n{page_text.strip()}")
                finally:
                    document.close()
            else:
                image = Image.open(BytesIO(file_bytes)).convert("RGB")
                ocr_text = recognize_with_easyocr(image)
                if ocr_text.strip():
                    file_parts.append(ocr_text.strip())

            if file_parts:
                extracted_papers.append(
                    f"=== PYQ File #{file_idx}: {filename} ===\n"
                    + "\n\n".join(file_parts)
                )

            progress.progress(
                file_idx / total_files,
                text=f"Processed PYQ file {file_idx} of {total_files}",
            )
    finally:
        progress.empty()

    return "\n\n".join(extracted_papers)


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


def _enter_room(room_code: str) -> None:
    """Switch dashboard into a class room and sync its saved Master Note."""
    code = room_code.strip().upper()
    st.session_state["active_room"] = code
    st.query_params["room"] = code

    saved_master = db.get_master_note(code)
    if saved_master:
        st.session_state["master_note"] = saved_master["content"]
        st.session_state["loaded_room_note_ts"] = saved_master["updated_at"]
    else:
        st.session_state["master_note"] = ""
        st.session_state["loaded_room_note_ts"] = ""

    st.session_state["raw_ocr_text"] = ""
    st.session_state["ocr_transcription"] = ""
    st.session_state["study_questions"] = ""
    st.session_state["translated_notes"] = ""


def _leave_room() -> None:
    """Exit the active class room back to Solo Mode."""
    st.session_state["active_room"] = ""
    st.session_state["loaded_room_note_ts"] = ""
    st.session_state["master_note"] = ""
    st.session_state["raw_ocr_text"] = ""
    st.session_state["ocr_transcription"] = ""
    st.session_state["study_questions"] = ""
    st.session_state["translated_notes"] = ""
    if "room" in st.query_params:
        del st.query_params["room"]


def render_room_sidebar(base_url: str) -> None:
    """Render Multi-Room Creation, Room Directory, Join Room, and Wi-Fi QR controls."""
    active_room = st.session_state.get("active_room", "")
    all_groups = db.list_all_groups()

    with st.sidebar:
        st.header("👥 Class Room Server")
        st.caption("Same Wi-Fi / Hotspot Link:")
        st.code(base_url, language=None)

        if active_room:
            group, submissions = db.get_group_details(active_room)
            if group:
                room_url = f"{base_url}/?room={active_room}"
                st.success(
                    f"**{group['topic']}** (`{active_room}`) — {len(submissions)} note(s)"
                )
                qr_png = make_qr_png(room_url)
                if qr_png:
                    st.image(qr_png, caption="Scan on phone to join room", width=180)
                col_ref, col_exit = st.columns(2)
                with col_ref:
                    if st.button("🔄 Refresh", use_container_width=True):
                        st.rerun()
                with col_exit:
                    if st.button("Exit Room", use_container_width=True):
                        _leave_room()
                        st.rerun()
        else:
            qr_png = make_qr_png(base_url)
            if qr_png:
                st.image(qr_png, caption="Scan on phone to open server", width=165)

        st.divider()
        st.subheader("Create Room(s)")

        existing_classes = sorted({row["group_name"] for row in all_groups})
        with st.form("create_room_form", clear_on_submit=True):
            if existing_classes:
                class_choice = st.selectbox(
                    "Add under existing Class / Group",
                    ["+ New Class / Group"] + existing_classes,
                )
            else:
                class_choice = "+ New Class / Group"

            new_class_name = st.text_input(
                "Or Enter New Class / Group Name",
                placeholder="e.g., EEE 2nd Year",
            )
            topics_input = st.text_area(
                "Topic(s) — one per line to create multiple rooms",
                placeholder=(
                    "Unit 1: DC Machines\n"
                    "Unit 2: Single-Phase Transformers\n"
                    "Unit 3: AC & DC Bridges"
                ),
                height=105,
            )
            submitted = st.form_submit_button(
                "Create Room(s)", type="primary", use_container_width=True
            )

            if submitted:
                target_class = (
                    new_class_name.strip()
                    if class_choice == "+ New Class / Group" or new_class_name.strip()
                    else class_choice
                )
                topic_lines = [
                    line.strip() for line in topics_input.splitlines() if line.strip()
                ]

                if not target_class or not topic_lines:
                    st.warning("Enter a Class/Group Name and at least one Topic.")
                else:
                    created_rooms = db.create_multiple_groups(target_class, topic_lines)
                    st.session_state["last_created_rooms"] = created_rooms
                    if created_rooms:
                        _enter_room(created_rooms[0]["group_code"])
                        st.rerun()

        last_created = st.session_state.get("last_created_rooms", [])
        if len(last_created) > 1:
            with st.expander(
                f"✅ {len(last_created)} Rooms Created (Codes)", expanded=True
            ):
                for item in last_created:
                    st.markdown(f"- **{item['topic']}** → `{item['group_code']}`")

        st.divider()
        st.subheader(f"📂 All Rooms ({len(all_groups)})")
        if not all_groups:
            st.caption("No rooms created yet.")
        else:
            grouped_rooms: dict[str, list] = {}
            for row in all_groups:
                grouped_rooms.setdefault(row["group_name"], []).append(row)

            for class_name, rooms in grouped_rooms.items():
                has_active = any(r["group_code"] == active_room for r in rooms)
                with st.expander(
                    f"📁 {class_name} ({len(rooms)})", expanded=has_active
                ):
                    for room in rooms:
                        code = room["group_code"]
                        is_current = code == active_room
                        status_icon = "✅" if room["has_master"] else "📝"
                        label_prefix = "▶ " if is_current else ""

                        col_open, col_del = st.columns([4, 1])
                        with col_open:
                            if st.button(
                                f"{label_prefix}{room['topic']} ({code}) • {status_icon} {room['submission_count']}",
                                key=f"open_room_{code}",
                                use_container_width=True,
                                type="primary" if is_current else "secondary",
                            ):
                                _enter_room(code)
                                st.rerun()
                        with col_del:
                            if st.button(
                                "🗑️",
                                key=f"del_room_{code}",
                                help=f"Delete room {code}",
                                use_container_width=True,
                            ):
                                db.delete_group(code)
                                st.session_state["last_created_rooms"] = [
                                    r
                                    for r in st.session_state.get(
                                        "last_created_rooms", []
                                    )
                                    if r["group_code"] != code
                                ]
                                if is_current:
                                    _leave_room()
                                st.rerun()

        st.divider()
        st.subheader("Join by Room Code")
        join_code = (
            st.text_input(
                "Room Code",
                placeholder="e.g., A4K9Z2",
                key="sidebar_join_code",
            )
            .strip()
            .upper()
        )
        if st.button("Join Room", use_container_width=True):
            if not join_code:
                st.warning("Enter a 6-character room code.")
            else:
                group, _ = db.get_group_details(join_code)
                if group:
                    _enter_room(join_code)
                    st.rerun()
                else:
                    st.error("Room code not found.")


def render_class_room_workspace(active_room: str, base_url: str) -> None:
    """Render collaborative multi-member upload and Master Note merge inside Tab 1."""
    group, submissions = db.get_group_details(active_room)
    if not group:
        st.error(f"Room `{active_room}` was not found.")
        if st.button("Return to Solo Mode"):
            _leave_room()
            st.rerun()
        return

    # Auto-sync if another classmate generated or updated the room's Master Note
    saved_master = db.get_master_note(active_room)
    if saved_master and saved_master["updated_at"] != st.session_state.get(
        "loaded_room_note_ts", ""
    ):
        st.session_state["master_note"] = saved_master["content"]
        st.session_state["loaded_room_note_ts"] = saved_master["updated_at"]
        st.session_state["study_questions"] = ""
        st.session_state["translated_notes"] = ""

    room_url = f"{base_url}/?room={active_room}"
    st.info(
        f"**Class Room:** {group['group_name']} — **{group['topic']}**  |  "
        f"**Code:** `{active_room}`  |  **Notes Uploaded:** `{len(submissions)}`  |  "
        f"**Join Link:** `{room_url}`"
    )

    st.markdown("### 📤 Submit Your Notes to This Room")
    member_name = st.text_input(
        "Your Name",
        placeholder="e.g., Anuj / Rahul / Priya",
        key="room_member_name",
    )
    uploaded_files = st.file_uploader(
        "Upload your handwritten note images or PDFs",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        accept_multiple_files=True,
        key="room_file_uploader",
    )
    extra_text = st.text_area(
        "Optional typed points / formulas",
        placeholder="Type any extra points the professor mentioned...",
        height=90,
        key="room_extra_text",
    )

    col_sub, col_seg = st.columns([3, 1])
    with col_seg:
        use_segmentation = st.checkbox(
            "Adaptive line segmentation",
            value=False,
            key="room_use_seg",
            help="Slice handwriting into horizontal lines before OCR.",
        )
    with col_sub:
        submit_btn = st.button(
            "➕ Submit My Notes to Room",
            type="primary",
            use_container_width=True,
        )

    if submit_btn:
        if not member_name.strip():
            st.warning("Please enter your name before submitting.")
        elif not uploaded_files and not extra_text.strip():
            st.warning("Upload at least one image/PDF or type some notes.")
        else:
            try:
                parts = []
                if uploaded_files:
                    pages = get_uploaded_pages(uploaded_files)
                    with st.spinner("Running OCR on your uploaded pages…"):
                        ocr_text = run_ocr_on_pages(pages, use_segmentation)
                    if ocr_text.strip():
                        parts.append(ocr_text.strip())
                if extra_text.strip():
                    parts.append(extra_text.strip())

                final_text = "\n\n".join(parts)
                if not final_text.strip():
                    st.warning("EasyOCR did not find readable text on the uploaded pages.")
                else:
                    db.add_submission(active_room, member_name, final_text)
                    st.success(
                        f"Saved notes from **{member_name.strip()}** to room `{active_room}`!"
                    )
                    st.rerun()
            except OCRError as error:
                st.error(f"OCR Error: {error}")
            except Exception as error:
                st.error(f"Submission Error: {error}")

    st.markdown("---")
    st.markdown(f"### 📚 Shared Class Submissions ({len(submissions)})")
    if not submissions:
        st.info("No group notes submitted yet. Share the QR code or Room Code in the sidebar.")
    else:
        for idx, sub in enumerate(submissions, start=1):
            with st.expander(f"#{idx} — {sub['member_name']}"):
                st.text(sub["extracted_text"])

        if st.button(
            f"✦ Generate / Refresh Class Master Note (from {len(submissions)} submission(s))",
            type="primary",
            use_container_width=True,
        ):
            try:
                with st.spinner(
                    f"Merging important points from {len(submissions)} classmate note(s)…"
                ):
                    merged_note = generate_group_master_note(
                        group["group_name"],
                        group["topic"],
                        submissions,
                    )
                    db.save_master_note(active_room, merged_note, len(submissions))
                    updated = db.get_master_note(active_room)
                    st.session_state["master_note"] = merged_note
                    st.session_state["loaded_room_note_ts"] = (
                        updated["updated_at"] if updated else ""
                    )
                    st.session_state["study_questions"] = ""
                    st.session_state["translated_notes"] = ""
                    st.rerun()
            except GeminiError as error:
                st.error(f"Gemini API Error: {error}")
            except Exception as error:
                st.error(f"Merge Error: {error}")

    master_note = st.session_state.get("master_note", "")
    if master_note:
        st.markdown("---")
        st.markdown("## 📝 Class Master Note")
        if saved_master:
            st.caption(
                f"Synthesized from {saved_master['source_count']} member submission(s) • "
                "Switch to Tab 2 for Revision Questions or Tab 3 to Translate."
            )
        st.markdown(master_note)
        st.download_button(
            "↓ Download Class Master Note as PDF",
            data=make_pdf_bytes(
                master_note,
                f"{group['group_name']} — {group['topic']} (Master Note)",
            ),
            file_name=f"class-master-note-{active_room}.pdf",
            mime="application/pdf",
            use_container_width=True,
            key="download_room_master_note",
        )


def main() -> None:
    load_styles()
    render_brand_header()

    st.session_state.setdefault("raw_ocr_text", "")
    st.session_state.setdefault("ocr_transcription", "")
    st.session_state.setdefault("master_note", "")
    st.session_state.setdefault("study_questions", "")
    st.session_state.setdefault("translated_notes", "")
    st.session_state.setdefault("probable_paper", "")
    st.session_state.setdefault("active_room", "")
    st.session_state.setdefault("loaded_room_note_ts", "")
    st.session_state.setdefault("last_created_rooms", [])

    # Auto-join room when a classmate scans a QR code (?room=XXXXXX)
    url_room = st.query_params.get("room", "")
    if isinstance(url_room, list):
        url_room = url_room[0] if url_room else ""
    url_room = url_room.strip().upper()

    if url_room and url_room != st.session_state["active_room"]:
        group, _ = db.get_group_details(url_room)
        if group:
            _enter_room(url_room)

    port = st.get_option("server.port") or 8501
    base_url = f"http://{get_local_ip()}:{port}"
    render_room_sidebar(base_url)

    active_room = st.session_state.get("active_room", "")
    tab1_label = (
        f"👥 1. Class Room ({active_room})"
        if active_room
        else "📄 1. Digitizer & Master Note"
    )

    tab_notes, tab_questions, tab_translate, tab_pyq = st.tabs(
        [
            tab1_label,
            "🎯 2. Revision Questions",
            "🌐 3. Translation",
            "📊 4. PYQ Paper Predictor",
        ]
    )

    with tab_notes:
        if active_room:
            render_class_room_workspace(active_room, base_url)
        else:
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
                    with st.expander(
                        f"Preview uploaded pages ({len(pages)})", expanded=True
                    ):
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
                            with st.spinner(
                                "Extracting handwritten text and formulating note…"
                            ):
                                raw_transcription = run_ocr_on_pages(
                                    pages, use_segmentation
                                )

                                if not raw_transcription.strip():
                                    st.warning(
                                        "EasyOCR did not find readable text on the page."
                                    )
                                else:
                                    st.session_state["raw_ocr_text"] = raw_transcription
                                    corrected_text, master_note_text = (
                                        process_and_generate_master_note(
                                            raw_transcription
                                        )
                                    )
                                    st.session_state["ocr_transcription"] = (
                                        corrected_text
                                    )
                                    st.session_state["master_note"] = master_note_text
                                    st.session_state["study_questions"] = ""
                                    st.session_state["translated_notes"] = ""
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
                        with st.expander(
                            f"Review Term Corrections ({len(suggestions)})"
                        ):
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
                st.info(
                    "Upload page images or PDFs above for Solo Mode, or Create/Join a Class Room in the sidebar."
                )

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

    with tab_pyq:
        st.markdown("## 📊 PYQ-Based Probable Question Paper Generator")
        st.write(
            "Upload 3–4 previous year question papers (PDFs or photos) or paste PYQ questions "
            "below to predict a probable practice question paper."
        )

        pyq_files = st.file_uploader(
            "Upload PYQ papers (PDF, PNG, JPG — upload 3 to 4 papers for best accuracy)",
            type=["pdf", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            key="pyq_file_uploader",
        )

        with st.expander(
            "Or Paste / Type PYQ Questions Manually (Optional)",
            expanded=not bool(pyq_files),
        ):
            col_p1, col_p2 = st.columns(2)
            with col_p1:
                pyq_1 = st.text_area("PYQ 1 (e.g., 2022)", height=110, key="pyq_text_1")
                pyq_3 = st.text_area("PYQ 3 (e.g., 2024)", height=110, key="pyq_text_3")
            with col_p2:
                pyq_2 = st.text_area("PYQ 2 (e.g., 2023)", height=110, key="pyq_text_2")
                pyq_4 = st.text_area("PYQ 4 (e.g., 2025)", height=110, key="pyq_text_4")

        st.markdown("### ⚙️ Practice Paper Settings")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            pyq_subject = st.text_input(
                "Subject / Exam Name (Optional)",
                placeholder="e.g., Electrical Machines / Data Structures",
                key="pyq_subject",
            )
        with col_s2:
            pyq_marks = st.number_input(
                "Total Marks",
                min_value=10,
                max_value=200,
                value=70,
                step=5,
                key="pyq_marks",
            )
        with col_s3:
            pyq_duration = st.selectbox(
                "Time Allowed",
                ["1 Hour", "1.5 Hours", "2 Hours", "3 Hours"],
                index=3,
                key="pyq_duration",
            )

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            pyq_difficulty = st.selectbox(
                "Difficulty Level",
                [
                    "Match PYQ Level",
                    "Slightly Harder (Exam-Ready)",
                    "Conceptual / Analytical Focus",
                ],
                key="pyq_difficulty",
            )
        with col_opt2:
            include_hints = st.checkbox(
                "Include Answer Hints & Key Formulas",
                value=True,
                key="pyq_include_hints",
            )
        with col_opt3:
            active_master = st.session_state.get("master_note", "")
            align_with_notes = st.checkbox(
                "Also align with active Master Note",
                value=bool(active_master),
                disabled=not bool(active_master),
                key="pyq_align_notes",
            )

        if st.button(
            "✦ Generate Probable Question Paper",
            type="primary",
            use_container_width=True,
            key="btn_generate_pyq_paper",
        ):
            manual_entries = [
                f"=== Manual PYQ #{idx} ===\n{txt.strip()}"
                for idx, txt in enumerate([pyq_1, pyq_2, pyq_3, pyq_4], start=1)
                if txt.strip()
            ]

            if not pyq_files and not manual_entries:
                st.warning("Please upload PYQ files or paste at least one PYQ set above.")
            else:
                try:
                    combined_pyq_parts = []
                    if pyq_files:
                        with st.spinner("Extracting text from uploaded PYQ papers…"):
                            file_text = extract_text_from_pyq_files(pyq_files)
                            if file_text.strip():
                                combined_pyq_parts.append(file_text.strip())

                    if manual_entries:
                        combined_pyq_parts.extend(manual_entries)

                    full_pyq_text = "\n\n".join(combined_pyq_parts)
                    if not full_pyq_text.strip():
                        st.warning("Could not extract readable text from the uploaded PYQs.")
                    else:
                        with st.spinner(
                            "Analyzing PYQ patterns and generating probable question paper…"
                        ):
                            paper = generate_probable_question_paper(
                                pyq_content=full_pyq_text,
                                subject_name=pyq_subject,
                                total_marks=int(pyq_marks),
                                duration=pyq_duration,
                                difficulty=pyq_difficulty,
                                include_hints=include_hints,
                                master_note_context=active_master if align_with_notes else "",
                            )
                            st.session_state["probable_paper"] = paper
                            st.rerun()
                except OCRError as error:
                    st.error(f"OCR Error: {error}")
                except GeminiError as error:
                    st.error(f"Gemini API Error: {error}")
                except Exception as error:
                    st.error(f"Error: {error}")

        probable_paper = st.session_state.get("probable_paper", "")
        if probable_paper:
            st.markdown("---")
            st.markdown(probable_paper)
            paper_title = (
                f"Probable Question Paper — {pyq_subject.strip()}"
                if pyq_subject.strip()
                else "NoteScribe — Probable Practice Question Paper"
            )
            st.download_button(
                "↓ Download Probable Question Paper as PDF",
                data=make_pdf_bytes(probable_paper, paper_title),
                file_name="probable-question-paper.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="download_probable_paper",
            )


if __name__ == "__main__":
    main()