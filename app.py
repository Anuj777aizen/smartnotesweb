def get_uploaded_pages(uploaded_files) -> list[tuple[str, int, Image.Image]]:
    """Convert uploaded images and PDF pages into memory-safe PIL images."""
    pages = []

    for uploaded in uploaded_files:
        file_bytes = uploaded.getvalue()
        filename = uploaded.name.lower()

        if filename.endswith(".pdf"):
            document = fitz.open(stream=file_bytes, filetype="pdf")
            try:
                for page_number, pdf_page in enumerate(document, start=1):
                    pixmap = pdf_page.get_pixmap(
                        matrix=fitz.Matrix(1.5, 1.5),
                        alpha=False,
                    )
                    image = Image.frombytes(
                        "RGB",
                        (pixmap.width, pixmap.height),
                        pixmap.samples,
                    )
                    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    pages.append((uploaded.name, page_number, image))
            finally:
                document.close()
        else:
            image = Image.open(BytesIO(file_bytes)).convert("RGB")
            image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            pages.append((uploaded.name, page_number, image))

    return pages