"""EasyOCR integration for NoteScribe."""

from functools import lru_cache

import numpy as np
from PIL import Image


ENGINES = {
    "EasyOCR": {
        "description": "Local OCR for printed text and neat handwriting.",
    },
}


class OCRError(Exception):
    """Raised when EasyOCR cannot process an image."""


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Create and cache the CPU EasyOCR reader."""
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False)
    except Exception as error:
        raise OCRError(f"Could not initialize EasyOCR: {error}") from error


def recognize_with_easyocr(image: Image.Image) -> str:
    """Recognize a full page with EasyOCR."""
    try:
        detections = get_easyocr_reader().readtext(
            np.asarray(image.convert("RGB")),
            detail=0,
            paragraph=True,
        )
        return "\n".join(
            text.strip() for text in detections if text.strip()
        )
    except OCRError:
        raise
    except Exception as error:
        raise OCRError(f"EasyOCR failed: {error}") from error