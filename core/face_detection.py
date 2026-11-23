"""Face detection using OpenCV's built-in Haar cascade (no downloads needed)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

from core.config import SETTINGS


@dataclass(frozen=True)
class FaceBox:
    """A detected face rectangle in the coordinates of the ORIGINAL frame."""
    x: int
    y: int
    w: int
    h: int

    @property
    def area(self) -> int:
        return self.w * self.h


class FaceDetector:
    """Detects frontal faces in BGR frames."""

    def __init__(
        self,
        scale_factor: float | None = None,
        min_neighbors: int | None = None,
        min_face_size: int | None = None,
        max_detect_width: int = 640,
    ) -> None:
        self.scale_factor = scale_factor or SETTINGS.detect_scale_factor
        self.min_neighbors = min_neighbors or SETTINGS.detect_min_neighbors
        self.min_face_size = min_face_size or SETTINGS.min_face_size
        self.max_detect_width = max_detect_width

        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Could not load the face cascade file: {cascade_path}")

    def detect(self, frame_bgr: np.ndarray) -> List[FaceBox]:
        """Return all faces found in the frame, largest first."""
        if frame_bgr is None or frame_bgr.size == 0:
            return []

        width = frame_bgr.shape[1]
        # Detect on a smaller copy for speed, then map boxes back to full size.
        scale = min(1.0, self.max_detect_width / float(width))
        small = cv2.resize(frame_bgr, None, fx=scale, fy=scale) if scale < 1.0 else frame_bgr

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # more robust to lighting changes

        min_size = max(20, int(self.min_face_size * scale))
        rects = self._cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(min_size, min_size),
        )
        if len(rects) == 0:
            return []

        boxes = [
            FaceBox(int(x / scale), int(y / scale), int(w / scale), int(h / scale))
            for (x, y, w, h) in rects
        ]
        boxes.sort(key=lambda b: b.area, reverse=True)
        return boxes
