"""Image cleanup and handwritten line segmentation."""

import cv2
import numpy as np
from PIL import Image

ENGINES = {
    "Tesseract": {
        "description": "Local OCR using the Tesseract engine.",
    },
}

# Maximum dimension threshold for segmentation stability
MAX_IMAGE_DIM = 1800


def preprocess_image(image: Image.Image) -> tuple[np.ndarray, np.ndarray]:
    """Resize safely and create clean grayscale + binary ink representations."""
    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]

    # Downscale if excessively large to keep convolution & thresholding fast
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(h, w))
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Adaptive thresholding to extract ink
    ink = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        13,
    )

    # Morphological noise removal (filter out single-pixel noise and faint ruled lines)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    return rgb, ink


def split_into_lines(image: Image.Image) -> list[Image.Image]:
    """Segment handwritten page into separate line crops.

    Falls back cleanly to the full image if no clear text lines exist.
    """
    try:
        rgb, ink = preprocess_image(image)
        h, w = ink.shape[:2]

        # Count ink per horizontal row
        row_ink = (ink > 0).sum(axis=1)
        active = row_ink > max(5, int(w * 0.015))

        # Bridge small vertical gaps across text strokes using int32
        active_smooth = np.convolve(
            active.astype(np.int32),
            np.ones(11, dtype=np.int32),
            mode="same"
        ) > 0

        bands: list[tuple[int, int]] = []
        start = None

        for y, is_active in enumerate(active_smooth):
            if is_active and start is None:
                start = y
            elif not is_active and start is not None:
                if y - start >= 8:  # Minimum line height threshold
                    bands.append((start, y))
                start = None

        if start is not None and (len(active_smooth) - start) >= 8:
            bands.append((start, len(active_smooth)))

        if not bands:
            return [image.convert("RGB")]

        # Merge bands with small vertical gaps
        merged: list[list[int]] = []
        for top, bottom in bands:
            if merged and (top - merged[-1][1]) < 12:
                merged[-1][1] = bottom
            else:
                merged.append([top, bottom])

        # If segmentation created too many fragments (likely noise/screenshot artifacts),
        # return the full image instead of dozens of broken slices
        if len(merged) > 60:
            return [Image.fromarray(rgb)]

        padding = max(6, int(h * 0.008))
        lines: list[Image.Image] = []

        for top, bottom in merged:
            y0 = max(0, top - padding)
            y1 = min(h, bottom + padding)
            crop = rgb[y0:y1, :, :]
            if crop.shape[0] >= 10 and crop.shape[1] >= 10:
                lines.append(Image.fromarray(crop))

        return lines or [Image.fromarray(rgb)]

    except Exception:
        # Fallback to base RGB image if any OpenCV step fails
        return [image.convert("RGB")]