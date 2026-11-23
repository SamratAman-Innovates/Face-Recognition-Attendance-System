"""Face preprocessing: crop, align (when eyes are found), resize, sharpness check, normalize."""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from core.config import SETTINGS
from core.face_detection import FaceBox

_eye_cascade: cv2.CascadeClassifier | None = None


def _get_eye_cascade() -> cv2.CascadeClassifier | None:
    """Load the eye cascade once (used for optional alignment)."""
    global _eye_cascade
    if _eye_cascade is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_eye.xml"
        cascade = cv2.CascadeClassifier(str(path))
        _eye_cascade = cascade if not cascade.empty() else None
    return _eye_cascade


def crop_face(frame_bgr: np.ndarray, box: FaceBox, margin: float = 0.15) -> np.ndarray:
    """Crop the face with a small margin, clipped to the image borders."""
    height, width = frame_bgr.shape[:2]
    pad_x, pad_y = int(box.w * margin), int(box.h * margin)
    x1, y1 = max(0, box.x - pad_x), max(0, box.y - pad_y)
    x2, y2 = min(width, box.x + box.w + pad_x), min(height, box.y + box.h + pad_y)
    return frame_bgr[y1:y2, x1:x2].copy()


def align_face(face_bgr: np.ndarray) -> np.ndarray:
    """Rotate the face so both eyes are level.

    Alignment is a best-effort step: if two plausible eyes are not found, or the
    detected tilt looks unrealistic, the crop is returned unchanged.
    """
    cascade = _get_eye_cascade()
    if cascade is None or face_bgr.size == 0:
        return face_bgr

    height, width = face_bgr.shape[:2]
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    upper_half = gray[: int(height * 0.6), :]  # eyes are in the upper part of the face

    min_eye = max(10, width // 10)
    eyes = cascade.detectMultiScale(upper_half, 1.1, 5, minSize=(min_eye, min_eye))
    if len(eyes) < 2:
        return face_bgr

    # Keep the two biggest detections, ordered left -> right.
    eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
    (x1, y1, w1, h1), (x2, y2, w2, h2) = sorted(eyes, key=lambda e: e[0])
    dx = (x2 + w2 / 2) - (x1 + w1 / 2)
    dy = (y2 + h2 / 2) - (y1 + h1 / 2)

    if abs(dx) < width * 0.2:      # eyes almost on top of each other -> false detection
        return face_bgr
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle) > 25:            # unrealistic head tilt -> false detection
        return face_bgr

    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        face_bgr, matrix, (width, height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_face(
    frame_bgr: np.ndarray,
    box: FaceBox,
    size: int | None = None,
    align: bool = True,
) -> np.ndarray:
    """Crop -> (align) -> resize. Returns a uint8 BGR image of size x size."""
    size = size or SETTINGS.face_size
    face = crop_face(frame_bgr, box)
    if align:
        face = align_face(face)
    return cv2.resize(face, (size, size), interpolation=cv2.INTER_AREA)


def is_sharp(face_bgr: np.ndarray, threshold: float | None = None) -> bool:
    """True if the face crop is sharp enough (variance of the Laplacian)."""
    threshold = SETTINGS.blur_threshold if threshold is None else threshold
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= threshold


def normalize_face(face_bgr: np.ndarray) -> np.ndarray:
    """Convert to RGB float32 and scale pixels to roughly [-1, 1] (FaceNet standardization)."""
    rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    return (rgb - 127.5) / 128.0
