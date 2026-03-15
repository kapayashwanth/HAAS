"""
image_loader.py
───────────────
Handles image input: file selection, loading, validation, and basic diagnostics.
"""

import cv2
import numpy as np
from tkinter import Tk, filedialog
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"]


def select_image() -> str | None:
    """Open a file dialog to let the user pick an answer sheet image."""
    root = Tk()
    root.withdraw()
    root.lift()
    file_path = filedialog.askopenfilename(
        title="Select Handwritten Answer Sheet",
        filetypes=[
            ("Image Files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif *.webp"),
            ("All Files", "*.*"),
        ],
    )
    root.destroy()
    return file_path if file_path else None


def load_image(path: str) -> np.ndarray:
    """
    Load an image from disk, validate it, and return as a numpy array.

    Raises:
        FileNotFoundError: if the path doesn't exist.
        ValueError: if the format is unsupported or the file is corrupt.
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    if p.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{p.suffix}'. Supported: {SUPPORTED_FORMATS}"
        )

    image = cv2.imread(str(p))

    if image is None:
        raise ValueError(f"OpenCV could not decode the image: {path}")

    logger.info("Loaded image: %s | shape: %s | dtype: %s", path, image.shape, image.dtype)
    return image


def get_image_info(image: np.ndarray) -> dict:
    """Return basic metadata about the loaded image."""
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    return {
        "height": h,
        "width": w,
        "channels": channels,
        "resolution_mp": round((h * w) / 1_000_000, 2),
    }


def resize_if_needed(image: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """
    Resize large images to prevent slow OCR without losing readability.
    Maintains aspect ratio.
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image

    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    logger.info("Resized image from (%d×%d) to (%d×%d)", w, h, new_w, new_h)
    return resized