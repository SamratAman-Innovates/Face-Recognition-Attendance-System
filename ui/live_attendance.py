
"""Browser-based live attendance using Streamlit WebRTC."""
from __future__ import annotations

import threading
import time

import av
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from core import attendance, database as db
from core.attendance import ConfirmationTracker, MarkStatus
from core.config import SETTINGS
from core.recognition import FaceRecognizer, recognize_frame
from core.visualize import draw_results
from ui.resources import get_detector, get_embedder
from ui.styles import page_header


class AttendanceProcessor:
    """Processes browser webcam frames and marks recognised students."""

    def __init__(self, detector, embedder, recognizer, names):
        self.detector = detector
        self.embedder = embedder
        self.recognizer = recognizer
        self.names = names

        self.tracker = ConfirmationTracker(SETTINGS.confirm_frames)
        self.handled: set[str] = set()

        self.frame_no = 0
        self.results = []
        self.marked_now = 0
        self.unknown_events = 0
        self.message = "Waiting for camera..."
        self.error = None
        self.lock = threading.Lock()

    def recv(self, video_frame: av.VideoFrame) -> av.VideoFrame:
        """Receive a browser frame, recognise faces and return an annotated frame."""

        frame = video_frame.to_ndarray(format="bgr24")
        self.frame_no += 1

        try:
            if self.frame_no % max(1, SETTINGS.process_every_n_frames) == 0:
                results = recognize_frame(
                    frame,
                    self.detector,
                    self.embedder,
                    self.recognizer,
                )

                best = {
                    result.student_id: result.score
                    for result in results
                    if result.is_known
                }

                for student_id in self.tracker.update(best.keys()):
                    if student_id in self.handled:
                        continue

                    self.handled.add(student_id)

                    try:
                        outcome = attendance.mark_attendance(
                            student_id,
                            best[student_id],
                        )
                    except db.DatabaseError as exc:
                        with self.lock:
                            self.error = f"Database error: {exc}"
                        continue

                    if outcome.status is MarkStatus.MARKED:
                        with self.lock:
                            self.marked_now += 1
                            self.message = (
                                f"Marked present: {outcome.name} "
                                f"at {outcome.time}"
                            )

                    elif outcome.status is MarkStatus.ALREADY_MARKED:
                        with self.lock:
                            self.message = (
                                f"{outcome.name} was already marked today."
                            )

                with self.lock:
                    self.results = results

                    if any(not result.is_known for result in results):
                        self.unknown_events += 1

                    if not results:
                        self.message = "No face in view."
                    elif len(results) > 1:
                        self.message = (
                            f"{len(results)} faces in view. Checking all."
                        )
                    elif results[0].is_known:
                        name = self.names.get(
                            results[0].student_id, "Student"
                        )
                        self.message = f"Recognising {name}..."
                    else:
                        self.message = (
                            "Unknown face - student not registered "
                            "or match is uncertain."
                        )

        except Exception as exc:
            with self.lock:
                self.error = f"Recognition failed: {exc}"

        with self.lock:
            annotated = draw_results(frame, self.results, self.names)

        return av.VideoFrame.from_ndarray(
            annotated,
            format="bgr24",
        )

    def get_status(self):
        """Safely read the current processing status."""
        with self.lock:
            return {
                "message": self.message,
                "error": self.error,
                "marked": self.marked_now,
                "unknown": self.unknown_events,
            }


def _show_today(slot) -> None:
    """Display today's attendance records."""
    try:
        today = db.query_attendance(
            start_date=attendance.today_str(),
            end_date=attendance.today_str(),
        )
    except db.DatabaseError as exc:
        slot.error(str(exc))
        return

    if today.empty:
        slot.caption("Nobody marked yet today.")
    else:
        slot.dataframe(
            today[["Time", "Name", "Class"]],
            use_container_width=True,
            hide_index=True,
            height=260,
        )


def render() -> None:
    page_header(
        "Live Attendance",
        "Use your browser webcam to recognise students and mark attendance.",
    )

    embedder = get_embedder()
    if embedder is None:
        return

    detector = get_detector()

    try:
        ids, embeddings = db.load_gallery(embedder.model_name)
        students = db.list_students()
    except db.DatabaseError as exc:
        st.error(f"Database error: {exc}")
        return

    if len(ids) == 0:
        st.warning(
            "No registered faces yet. Register at least one student first."
        )
        return

    names = dict(zip(students["student_id"], students["name"]))

    threshold = st.slider(
        "Face match threshold",
        min_value=0.30,
        max_value=0.95,
        value=float(SETTINGS.similarity_threshold),
        step=0.01,
        help=(
            "Higher values are stricter. If wrong people are accepted, "
            "increase the threshold."
        ),
    )

    recognizer = FaceRecognizer(ids, embeddings, threshold)

    st.info(
        "Click START below and allow camera permission in your browser. "
        "Keep this page open while attendance is running."
    )

    processor = AttendanceProcessor(
        detector,
        embedder,
        recognizer,
        names,
    )

    webrtc_ctx = webrtc_streamer(
        key="faceattend-browser-camera",
        video_frame_callback=processor.recv,
        media_stream_constraints={
            "video": True,
            "audio": False,
        },
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]}
            ]
        },
        async_processing=True,
    )

    status = processor.get_status()

    if status["error"]:
        st.error(status["error"])

    if webrtc_ctx.state.playing:
        st.success(status["message"])
    else:
        st.info("Camera is off. Click START to begin.")

    col1, col2 = st.columns(2)
    col1.metric("Newly marked this session", status["marked"])
    col2.metric("Unknown sightings", status["unknown"])

    st.divider()

    st.subheader("Today's Attendance")

    if st.button("Refresh attendance table"):
        st.rerun()

    table_slot = st.empty()
    _show_today(table_slot)
