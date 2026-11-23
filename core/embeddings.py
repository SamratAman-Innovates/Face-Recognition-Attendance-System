"""FaceNet embeddings (InceptionResnetV1 pretrained on VGGFace2) via facenet-pytorch.

The pretrained weights (~107 MB) are downloaded once from GitHub the first time
the model is created and cached in your user folder. After that the app works
fully offline. No API key or paid service is involved.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np

from core.preprocessing import normalize_face

EMBEDDING_DIM = 512


class EmbeddingModelError(RuntimeError):
    """Raised when the FaceNet model cannot be imported or loaded."""


class FaceEmbedder:
    """Turns preprocessed face crops into L2-normalised 512-d vectors."""

    model_name = "facenet-vggface2"

    def __init__(self) -> None:
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise EmbeddingModelError(
                "FaceNet is not installed. Run:  pip install torch==2.2.2 torchvision==0.17.2 "
                "facenet-pytorch==2.6.0  (use Python 3.10 or 3.11)."
            ) from exc

        try:
            self._torch = torch
            self._model = InceptionResnetV1(pretrained="vggface2").eval()
            torch.set_num_threads(max(1, min(4, os.cpu_count() or 2)))
        except Exception as exc:
            raise EmbeddingModelError(
                "Could not load the FaceNet weights. The first run needs an internet "
                f"connection to download them once. Details: {exc}"
            ) from exc

    def embed_batch(self, faces_bgr: List[np.ndarray]) -> np.ndarray:
        """Embed several face crops at once. Returns an array of shape (n, 512)."""
        if not faces_bgr:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)

        batch = np.stack([normalize_face(face) for face in faces_bgr])           # N,H,W,3
        tensor = self._torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous()  # N,3,H,W
        with self._torch.no_grad():
            output = self._model(tensor).cpu().numpy().astype(np.float32)

        norms = np.linalg.norm(output, axis=1, keepdims=True)
        return output / np.maximum(norms, 1e-10)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        """Embed a single face crop. Returns a vector of shape (512,)."""
        return self.embed_batch([face_bgr])[0]
