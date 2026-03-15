"""
preprocess.py
─────────────
Advanced image preprocessing pipeline to maximise OCR accuracy on
handwritten answer sheets.

Pipeline:
  1. Resize (if huge)
  2. Deskew (correct tilt)
  3. Grayscale conversion
  4. Denoise (Non-Local Means)
  5. Contrast enhancement (CLAHE)
  6. Adaptive thresholding (Sauvola-style via OpenCV)
  7. Morphological cleanup
  8. Border removal
"""

import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


# ── 1. Deskew ────────────────────────────────────────────────────────────────

def deskew(image: np.ndarray) -> np.ndarray:
    """
    Detect and correct the skew angle of a document image.
    Works on grayscale or colour images.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # Threshold to find text regions
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image  # Not enough text to determine skew

    angle = cv2.minAreaRect(coords)[-1]

    # Normalise angle to the range (-45°, 45°)
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) < 0.3:          # Skip tiny corrections
        return image

    h, w = image.shape[:2]
    centre = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    rotated = cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.info("Deskewed by %.2f°", angle)
    return rotated


# ── 2. Denoise ───────────────────────────────────────────────────────────────

def denoise(gray: np.ndarray) -> np.ndarray:
    """
    Remove noise with Non-Local Means denoising — much better than
    GaussianBlur for preserving handwriting stroke edges.
    """
    return cv2.fastNlMeansDenoising(gray, h=15, templateWindowSize=7, searchWindowSize=21)


# ── 3. CLAHE contrast enhancement ───────────────────────────────────────────

def enhance_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation).
    Handles uneven lighting across the page.
    """
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ── 4. Adaptive threshold ────────────────────────────────────────────────────

def binarise(gray: np.ndarray) -> np.ndarray:
    """
    Adaptive Gaussian thresholding: handles shadows and lighting gradients
    far better than global Otsu alone.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10,
    )
    return binary


# ── 5. Morphological cleanup ─────────────────────────────────────────────────

def morphological_cleanup(binary: np.ndarray) -> np.ndarray:
    """
    Remove tiny noise specks (open) and close small gaps in strokes (close).
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel2, iterations=1)
    return closed


# ── 6. Border removal ────────────────────────────────────────────────────────

def remove_border(image: np.ndarray, margin: int = 10) -> np.ndarray:
    """
    Whiten a thin margin around the image to strip scanning artefacts.
    """
    result = image.copy()
    result[:margin, :] = 255
    result[-margin:, :] = 255
    result[:, :margin] = 255
    result[:, -margin:] = 255
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def preprocess_image(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    Returns:
        processed  - binary image ready for OCR
        debug_gray - enhanced grayscale for OCR fallback/debugging
    """
    # Step 1: Deskew on the original colour image
    deskewed = deskew(image)

    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(deskewed, cv2.COLOR_BGR2GRAY) if deskewed.ndim == 3 else deskewed.copy()

    # Step 3: Denoise
    denoised = denoise(gray)

    # Step 4: Contrast enhancement
    enhanced = enhance_contrast(denoised)

    # Step 5: Binarise
    binary = binarise(enhanced)

    # Step 6: Morphological cleanup
    cleaned = morphological_cleanup(binary)

    # Step 7: Remove border artefacts
    final = remove_border(cleaned)

    logger.info("Preprocessing complete. Output shape: %s", final.shape)
    return final, enhanced      # (OCR-ready binary, OCR-friendly grayscale)