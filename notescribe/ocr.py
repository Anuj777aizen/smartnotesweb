"""EasyOCR integration for NoteScribe."""

from functools import lru_cache
import threading
import numpy as np
from PIL import Image, ImageOps


ENGINES = {
    "EasyOCR": {
        "description": "Local OCR for printed text and neat handwriting.",
    },
}

# Maximum dimension (width or height) to prevent PyTorch CPU memory exhaustion
MAX_IMAGE_DIMENSION = 1600
_OCR_LOCK = threading.Lock()


class OCRError(Exception):
    """Raised when EasyOCR cannot process an image."""


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Create and cache the CPU EasyOCR reader."""
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False, download_enabled=True)
    except Exception as error:
        raise OCRError(f"Could not initialize EasyOCR reader: {error}") from error


def _preprocess_image(image: Image.Image) -> np.ndarray:
    """Resize large images and optimize contrast for OCR."""
    rgb_image = image.convert("RGB")

    width, height = rgb_image.size
    max_dim = max(width, height)
    if max_dim > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / float(max_dim)
        new_size = (int(width * scale), int(height * scale))
        rgb_image = rgb_image.resize(new_size, Image.Resampling.LANCZOS)

    try:
        rgb_image = ImageOps.autocontrast(rgb_image, cutoff=1)
    except Exception:
        pass

    return np.ascontiguousarray(np.asarray(rgb_image))


def recognize_with_easyocr(image: Image.Image) -> str:
    """Recognize a full page with EasyOCR safely and efficiently."""
    try:
        processed_array = _preprocess_image(image)
        with _OCR_LOCK:
            reader = get_easyocr_reader()
            detections = reader.readtext(
                processed_array,
                detail=0,
                paragraph=True,
            )

        extracted_lines = [text.strip() for text in detections if text.strip()]
        return "\n".join(extracted_lines)

    except OCRError:
        raise
    except Exception as error:
        raise OCRError(f"EasyOCR failed during text recognition: {error}") from error