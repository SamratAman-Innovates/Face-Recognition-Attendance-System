"""
Central configuration.

Defaults live in the `Settings` dataclass. Any value can be overridden by
editing `config.json` in the project root. A broken or missing config file is
never fatal - the defaults are used instead.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = PROJECT_ROOT / "config.json"

logger = logging.getLogger(__name__)

# Settings that hold file-system paths (converted to Path objects when loaded).
_PATH_FIELDS = {"database_path", "faces_dir", "exports_dir"}


@dataclass
class Settings:
    # --- Storage -----------------------------------------------------------
    database_path: Path = PROJECT_ROOT / "data" / "attendance.db"
    faces_dir: Path = PROJECT_ROOT / "data" / "faces"
    exports_dir: Path = PROJECT_ROOT / "data" / "exports"

    # --- Camera ------------------------------------------------------------
    camera_index: int = 0
    frame_width: int = 640
    frame_height: int = 480

    # --- Face detection / preprocessing -------------------------------------
    face_size: int = 160              # FaceNet input size (pixels)
    detect_scale_factor: float = 1.1  # Haar cascade pyramid step
    detect_min_neighbors: int = 5     # higher = fewer false detections
    min_face_size: int = 80           # smallest face (px) that is detected

    # --- Recognition -------------------------------------------------------
    similarity_threshold: float = 0.65  # cosine similarity needed to accept a match
    min_margin: float = 0.03            # best match must beat the runner-up by this much
    confirm_frames: int = 3             # consecutive recognitions before marking attendance
    process_every_n_frames: int = 3     # run the CNN on every n-th frame (speed)
    max_faces_per_frame: int = 5        # recognize at most this many faces per frame

    # --- Registration ------------------------------------------------------
    registration_samples: int = 15
    min_samples: int = 5
    blur_threshold: float = 25.0        # Laplacian variance; lower = blurrier
    capture_interval_sec: float = 0.4   # pause between captured samples
    capture_timeout_sec: int = 90

    def ensure_directories(self) -> None:
        """Create the data folders if they do not exist yet."""
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)


def load_settings(config_file: Path = CONFIG_FILE) -> Settings:
    """Build a Settings object from defaults + config.json overrides."""
    settings = Settings()
    raw: dict = {}

    if config_file.exists():
        try:
            raw = json.loads(config_file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("config.json must contain a JSON object")
        except (OSError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("Could not read %s (%s). Using defaults.", config_file, exc)
            raw = {}

    for field in fields(Settings):
        if field.name not in raw:
            continue
        value = raw[field.name]
        try:
            if field.name in _PATH_FIELDS:
                value = Path(value)
                if not value.is_absolute():
                    value = PROJECT_ROOT / value
            else:
                value = type(getattr(settings, field.name))(value)
        except (TypeError, ValueError):
            logger.warning("Invalid value for '%s' in config.json - ignored.", field.name)
            continue
        setattr(settings, field.name, value)

    try:
        settings.ensure_directories()
    except OSError as exc:
        logger.warning("Could not create data folders: %s", exc)
    return settings


# Shared, importable settings object.
SETTINGS = load_settings()
