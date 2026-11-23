"""Face recognition by cosine similarity against the stored embedding gallery."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from core.config import SETTINGS
from core.face_detection import FaceBox, FaceDetector
from core.preprocessing import preprocess_face


@dataclass(frozen=True)
class MatchResult:
    student_id: Optional[str]   # None => unknown face
    score: float                # best cosine similarity found

    @property
    def is_known(self) -> bool:
        return self.student_id is not None


@dataclass(frozen=True)
class FaceResult:
    box: FaceBox
    match: MatchResult

    @property
    def student_id(self) -> Optional[str]:
        return self.match.student_id

    @property
    def score(self) -> float:
        return self.match.score

    @property
    def is_known(self) -> bool:
        return self.match.is_known


class FaceRecognizer:
    """Compares a query embedding with every stored embedding.

    A face is accepted only when
      1. its best cosine similarity is >= `threshold`, and
      2. it beats the second-best *student* by at least `margin`
         (prevents confusing two look-alike students).
    """

    def __init__(
        self,
        student_ids: Sequence[str],
        embeddings: np.ndarray,
        threshold: float | None = None,
        margin: float | None = None,
    ) -> None:
        self.student_ids = list(student_ids)
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        self.threshold = SETTINGS.similarity_threshold if threshold is None else threshold
        self.margin = SETTINGS.min_margin if margin is None else margin

    def identify(self, embedding: np.ndarray) -> MatchResult:
        if len(self.student_ids) == 0:
            return MatchResult(None, 0.0)

        sims = cosine_similarity(np.asarray(embedding).reshape(1, -1), self.embeddings)[0]

        # Keep each student's best similarity over all of their samples.
        best: Dict[str, float] = {}
        for sid, sim in zip(self.student_ids, sims):
            if sim > best.get(sid, -1.0):
                best[sid] = float(sim)

        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        top_id, top_score = ranked[0]
        runner_up = ranked[1][1] if len(ranked) > 1 else -1.0

        if top_score >= self.threshold and (top_score - runner_up) >= self.margin:
            return MatchResult(top_id, top_score)
        return MatchResult(None, top_score)


def recognize_frame(
    frame_bgr: np.ndarray,
    detector: FaceDetector,
    embedder,
    recognizer: FaceRecognizer,
    max_faces: int | None = None,
) -> List[FaceResult]:
    """Detect -> preprocess -> embed -> match for every face in one frame."""
    max_faces = max_faces or SETTINGS.max_faces_per_frame
    boxes = detector.detect(frame_bgr)[:max_faces]
    if not boxes:
        return []

    faces = [preprocess_face(frame_bgr, box) for box in boxes]
    embeddings = embedder.embed_batch(faces)
    return [FaceResult(box, recognizer.identify(emb)) for box, emb in zip(boxes, embeddings)]
