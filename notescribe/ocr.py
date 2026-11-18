"""EasyOCR integration for NoteScribe (Optimized for 512MB - 1GB Cloud RAM)."""

import gc
from functools import lru_cache

import numpy as np
from PIL import Image


ENGINES = {
    "EasyOCR": {
        "description": "Local OCR for printed text and neat handwriting.",
    },
}

MAX_IMAGE_DIMENSION = 1024


class OCRError(Exception):
    """Raised when EasyOCR cannot process an image."""


@lru_cache(maxsize=1)
def get_easyocr_reader():
    """Create and cache the CPU EasyOCR reader with minimal RAM footprint."""
    try:
        import torch
        import easyocr

        torch.set_num_threads(1)
        return easyocr.Reader(
            ["en"],
            gpu=False,
            quantize=True,
        )
    except Exception as error:
        raise OCRError(f"Could not initialize EasyOCR: {error}") from error


def _resize_for_ocr(image: Image.Image) -> Image.Image:
    """Downscale large images in-memory so PyTorch tensors do not crash RAM."""
    rgb_image = image.convert("RGB")
    width, height = rgb_image.size

    if max(width, height) > MAX_IMAGE_DIMENSION:
        rgb_image.thumbnail(
            (MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION),
            Image.Resampling.LANCZOS,
        )
    return rgb_image


def recognize_with_easyocr(image: Image.Image) -> str:
    """Recognize a full page with EasyOCR using bounded memory."""
    try:
        import torch

        safe_image = _resize_for_ocr(image)
        image_array = np.asarray(safe_image)

        reader = get_easyocr_reader()
        with torch.no_grad():
            detections = reader.readtext(
                image_array,
                detail=0,
                paragraph=True,
                canvas_size=1024,
                batch_size=1,
            )

        del image_array
        del safe_image
        gc.collect()

        return "\n".join(
            text.strip() for text in detections if text.strip()
        )
    except OCRError:
        raise
    except Exception as error:
        gc.collect()
        raise OCRError(f"EasyOCR failed: {error}") from error