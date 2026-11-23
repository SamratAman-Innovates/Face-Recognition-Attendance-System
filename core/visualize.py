"""Drawing helpers: boxes and labels on camera frames (OpenCV)."""
from __future__ import annotations

from typing import Dict, Iterable

import cv2
import numpy as np

from core.recognition import FaceResult

GREEN = (80, 200, 120)   # BGR
RED = (70, 70, 230)
AMBER = (40, 180, 250)


def draw_label(frame: np.ndarray, x: int, y: int, w: int, h: int, label: str, color) -> None:
    """Draw a face rectangle with a filled label above it."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    top = max(0, y - th - baseline - 8)
    cv2.rectangle(frame, (x, top), (x + tw + 10, top + th + baseline + 8), color, -1)
    cv2.putText(frame, label, (x + 5, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (15, 15, 15), 1, cv2.LINE_AA)


def draw_results(frame: np.ndarray, results: Iterable[FaceResult], names: Dict[str, str]) -> np.ndarray:
    """Annotate recognised (green) and unknown (red) faces."""
    for res in results:
        b = res.box
        if res.is_known:
            label = f"{names.get(res.student_id, res.student_id)} {res.score:.2f}"
            draw_label(frame, b.x, b.y, b.w, b.h, label, GREEN)
        else:
            draw_label(frame, b.x, b.y, b.w, b.h, f"Unknown {res.score:.2f}", RED)
    return frame


def draw_banner(frame: np.ndarray, text: str, color=AMBER) -> np.ndarray:
    """Draw a status message across the top of the frame."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 28), (20, 20, 20), -1)
    cv2.putText(frame, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
    return frame
