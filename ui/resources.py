"""Heavy objects (face detector, FaceNet model) created once and shared by all pages."""
from __future__ import annotations

from typing import Optional

import streamlit as st

from core.embeddings import EmbeddingModelError, FaceEmbedder
from core.face_detection import FaceDetector


@st.cache_resource(show_spinner=False)
def get_detector() -> FaceDetector:
    return FaceDetector()


@st.cache_resource(show_spinner="Loading FaceNet model... (the first run downloads ~100 MB once)")
def _load_embedder() -> FaceEmbedder:
    return FaceEmbedder()


def get_embedder() -> Optional[FaceEmbedder]:
    """Return the FaceNet embedder, or show an error and return None."""
    try:
        return _load_embedder()
    except EmbeddingModelError as exc:
        st.error(f"🧠 {exc}")
        return None
