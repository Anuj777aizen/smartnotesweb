"""Image cleanup and handwritten line segmentation."""

import cv2
import numpy as np
from PIL import Image

ENGINES = {
    "Tesseract": {
        "description": "Local OCR using the Tesseract engine.",
    },
}

def split_into_lines(image: Image.Image) -> list[Image.Image]:
    """Apply adaptive thresholding and return page crops, one per text line.

    This is a simple prototype segmenter. It works best when the writing has
    visible gaps between lines; it can merge or split crowded handwriting.
    """
    rgb = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )

    # Count ink per horizontal row; rows with enough ink are likely part of text.
    row_ink = (ink > 0).sum(axis=1)
    active = row_ink > max(2, int(ink.shape[1] * 0.008))

    # Bridge tiny gaps between parts of the same handwritten line.
    active = np.convolve(active.astype(np.uint8), np.ones(9, dtype=np.uint8), mode="same") > 0
    bands: list[tuple[int, int]] = []
    start = None
    for y, is_active in enumerate(active):
        if is_active and start is None:
            start = y
        elif not is_active and start is not None:
            if y - start >= 5:
                bands.append((start, y))
            start = None
    if start is not None and len(active) - start >= 5:
        bands.append((start, len(active)))

    # Join bands whose gap is small, to avoid splitting a single line.
    merged: list[list[int]] = []
    for top, bottom in bands:
        if merged and top - merged[-1][1] < 10:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])

    padding = max(8, int(gray.shape[1] * 0.012))
    lines = []
    for top, bottom in merged:
        y0 = max(0, top - padding)
        y1 = min(gray.shape[0], bottom + padding)
        crop = rgb[y0:y1, :, :]
        if crop.shape[0] >= 12:
            lines.append(Image.fromarray(crop))

    # If line detection finds nothing, let the OCR engine try the whole page.
    return lines or [image.convert("RGB")]
