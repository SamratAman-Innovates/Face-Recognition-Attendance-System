"""Webcam helpers built on OpenCV, with friendly error messages."""
from __future__ import annotations

import platform
import time

import cv2
import numpy as np

from core.config import SETTINGS


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or stops delivering frames."""


def open_camera(index: int | None = None) -> cv2.VideoCapture:
    """Open the webcam and return an opened cv2.VideoCapture.

    On Windows the DirectShow backend (CAP_DSHOW) opens much faster than the
    default one, so we try it first and fall back to OpenCV's default backend.
    """
    index = SETTINGS.camera_index if index is None else int(index)

    cap = None
    try:
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(index)
    except Exception as exc:  # OpenCV can raise on some driver problems
        raise CameraError(f"Camera {index} failed to initialise: {exc}") from exc

    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera {index}. Make sure a webcam is connected, "
            "no other app (Zoom, Teams, browser tab) is using it, and that "
            "camera access is allowed for desktop apps in Windows Privacy settings. "
            "You can also try a different camera index (1, 2, ...)."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, SETTINGS.frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, SETTINGS.frame_height)

    # Webcams often return dark/empty first frames while auto-exposure settles.
    for _ in range(3):
        cap.read()
    return cap


def read_frame(cap: cv2.VideoCapture, retries: int = 5) -> np.ndarray:
    """Read one BGR frame, retrying briefly before giving up."""
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.05)
    raise CameraError("Lost the camera feed (could not read frames). Is the webcam still connected?")


def release_camera(cap: cv2.VideoCapture | None) -> None:
    """Release the camera, ignoring errors."""
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass
