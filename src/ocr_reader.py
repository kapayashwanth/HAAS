"""
ocr_reader.py
─────────────
Robust text extraction using EasyOCR with confidence filtering,
multi-pass reading, and structured output.

Strategy:
  • Pass 1 : Read the preprocessed (binary) image.
  • Pass 2 : Read the enhanced-grayscale image.
  • Merge results, keeping the higher-confidence detection for each region.
  • Filter detections below a confidence threshold (default 0.3).
  • Sort detections top-to-bottom, left-to-right to reconstruct reading order.
"""

import easyocr
import numpy as np
import logging
from typing import List, Tuple
import cv2

logger = logging.getLogger(__name__)

# Singleton reader — initialising is expensive; reuse across calls
_reader: easyocr.Reader | None = None


def get_reader(languages: List[str] = None) -> easyocr.Reader:
    """Return a shared EasyOCR reader, creating it on first call."""
    global _reader
    if _reader is None:
        langs = languages or ["en"]
        logger.info("Initialising EasyOCR with languages: %s", langs)
        _reader = easyocr.Reader(langs, gpu=False)   # set gpu=True if CUDA available
    return _reader


# ── Detection helpers ─────────────────────────────────────────────────────────

def _top_left_y(detection) -> float:
    """Return the top-left y coordinate of a detection bounding box."""
    bbox = detection[0]          # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    return min(pt[1] for pt in bbox)


def _top_left_x(detection) -> float:
    bbox = detection[0]
    return min(pt[0] for pt in bbox)


def _sort_detections(detections) -> list:
    """
    Sort detections into reading order (top-to-bottom, then left-to-right).
    Groups lines by proximity in the y-axis (±15 px tolerance).
    """
    if not detections:
        return []

    sorted_by_y = sorted(detections, key=_top_left_y)
    median_h = float(np.median([_box_height(d) for d in sorted_by_y])) if sorted_by_y else 20.0
    line_tol = max(14.0, median_h * 0.55)

    lines: List[list] = []
    current_line: list = [sorted_by_y[0]]
    current_y = _top_left_y(sorted_by_y[0])

    for det in sorted_by_y[1:]:
        y = _top_left_y(det)
        if abs(y - current_y) <= line_tol:
            current_line.append(det)
        else:
            lines.append(sorted(current_line, key=_top_left_x))
            current_line = [det]
            current_y = y
    lines.append(sorted(current_line, key=_top_left_x))

    return [det for line in lines for det in line]


# ── Multi-pass extraction ─────────────────────────────────────────────────────

def _read_image(image: np.ndarray, confidence_threshold: float) -> list:
    """Run EasyOCR on a single image and return filtered detections."""
    reader = get_reader()
    raw = reader.readtext(
        image,
        detail=1,
        paragraph=False,
        decoder="beamsearch",
        beamWidth=8,
        mag_ratio=1.8,
        canvas_size=3200,
        text_threshold=0.60,
        low_text=0.20,
        link_threshold=0.35,
        contrast_ths=0.10,
        adjust_contrast=0.55,
        allowlist="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;!?'-() ",
    )
    filtered = [d for d in raw if d[2] >= confidence_threshold]
    return filtered


def _make_ocr_variants(processed_image: np.ndarray, gray_image: np.ndarray | None) -> List[tuple[str, np.ndarray, float]]:
    """Create OCR variants with tuned confidence thresholds for handwriting."""
    variants: List[tuple[str, np.ndarray, float]] = [("binary", processed_image, 0.42)]

    if gray_image is not None:
        variants.append(("enhanced_gray", gray_image, 0.35))

        up = cv2.resize(gray_image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        variants.append(("enhanced_gray_upscaled", up, 0.38))

        otsu = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        variants.append(("gray_otsu", otsu, 0.45))

    return variants


def extract_text(
    processed_image: np.ndarray,
    gray_image: np.ndarray | None = None,
    confidence_threshold: float = 0.30,
) -> Tuple[str, List[dict]]:
    """
    Extract text using a multi-pass strategy for higher recall.

    Args:
        processed_image  : Binary / preprocessed image (primary pass).
        gray_image       : Grayscale image (secondary pass, optional).
        confidence_threshold : Minimum OCR confidence to accept (0–1).

    Returns:
        text        : Reconstructed full text string.
        detections  : List of dicts with keys {bbox, text, confidence}.
    """
    all_detections: list = []
    for variant_name, variant_img, variant_min_conf in _make_ocr_variants(processed_image, gray_image):
        threshold = max(confidence_threshold, variant_min_conf)
        logger.info("OCR pass on %s (conf >= %.2f)", variant_name, threshold)
        detections = _read_image(variant_img, threshold)
        logger.info("%s: kept %d detections", variant_name, len(detections))
        all_detections.extend(detections)

    # Deduplicate by bounding-box centre proximity
    merged = _deduplicate(all_detections)

    # Sort into reading order
    ordered = _sort_detections(merged)

    # Build output
    detections_out = [
        {"bbox": d[0], "text": d[1], "confidence": round(float(d[2]), 3)}
        for d in ordered
    ]

    line_groups: List[List[str]] = []
    current_line: List[str] = []
    prev_y = None
    line_gap = max(18.0, float(np.median([_box_height(d) for d in ordered])) * 0.65) if ordered else 20.0
    for d in ordered:
        y = _top_left_y(d)
        if prev_y is not None and (y - prev_y) > line_gap:
            if current_line:
                line_groups.append(current_line)
            current_line = []
        current_line.append(d[1])
        prev_y = y
    if current_line:
        line_groups.append(current_line)

    text = "\n".join(" ".join(tokens) for tokens in line_groups).strip()
    logger.info("Extracted %d word/phrase tokens", len(ordered))
    return text, detections_out


def _bbox_centre(detection) -> Tuple[float, float]:
    bbox = detection[0]
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return (sum(xs) / 4, sum(ys) / 4)


def _bbox_rect(detection) -> Tuple[float, float, float, float]:
    bbox = detection[0]
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    return min(xs), min(ys), max(xs), max(ys)


def _box_height(detection) -> float:
    y1, y2 = _bbox_rect(detection)[1], _bbox_rect(detection)[3]
    return max(1.0, y2 - y1)


def _bbox_iou(det_a, det_b) -> float:
    ax1, ay1, ax2, ay2 = _bbox_rect(det_a)
    bx1, by1, bx2, by2 = _bbox_rect(det_b)

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _normalise_text(text: str) -> str:
    return "".join(ch.lower() for ch in text if ch.isalnum())


def _deduplicate(detections: list, dist_threshold: float = 20.0) -> list:
    """Remove near-duplicate detections (same region detected in both passes)."""
    kept = []
    for det in detections:
        cx, cy = _bbox_centre(det)
        h = _box_height(det)
        duplicate = False
        for k in kept:
            kx, ky = _bbox_centre(k)
            kh = _box_height(k)
            adaptive_dist = max(dist_threshold, 0.45 * min(h, kh))
            close_by_center = abs(cx - kx) < adaptive_dist and abs(cy - ky) < adaptive_dist
            overlap = _bbox_iou(det, k) > 0.35
            same_text = _normalise_text(det[1]) and _normalise_text(det[1]) == _normalise_text(k[1])

            if close_by_center or overlap or same_text:
                # Keep the higher-confidence one
                if det[2] > k[2]:
                    kept.remove(k)
                    kept.append(det)
                duplicate = True
                break
        if not duplicate:
            kept.append(det)
    return kept