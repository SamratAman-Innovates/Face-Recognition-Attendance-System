
"""Student registration (webcam or photo upload) and student management."""
from __future__ import annotations

import hmac
import re
import shutil
import time
from typing import List, Tuple

import cv2
import numpy as np
import streamlit as st

from core import database as db
from core.camera import CameraError, open_camera, read_frame, release_camera
from core.config import SETTINGS
from core.preprocessing import is_sharp, preprocess_face
from core.recognition import FaceRecognizer
from core.visualize import GREEN, draw_banner, draw_label, RED
from ui.resources import get_detector, get_embedder
from ui.styles import page_header


ID_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,20}$")


class DuplicateFaceError(Exception):
    """Raised when a potential duplicate face registration is detected."""
    pass


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _validate(student_id: str, name: str, class_name: str) -> str | None:
    """Return an error message, or None if the form is valid."""

    if not ID_PATTERN.match(student_id):
        return "Student ID must be 1-20 characters: letters, digits, '-' or '_' (no spaces)."

    if not name.strip():
        return "Please enter the student's name."

    if not class_name.strip():
        return "Please enter the class."

    if db.student_exists(student_id):
        return f"Student ID '{student_id}' is already registered."

    return None


def _capture_from_webcam(
    num_samples: int,
    camera_index: int,
    preview,
    progress
) -> List[np.ndarray]:
    """Grab sharp, single-face crops from the webcam."""

    detector = get_detector()
    cap = open_camera(camera_index)

    samples: List[np.ndarray] = []
    started, last_capture = time.time(), 0.0

    try:
        while len(samples) < num_samples:

            if time.time() - started > SETTINGS.capture_timeout_sec:
                break

            frame = read_frame(cap)
            boxes = detector.detect(frame)

            display, message = frame.copy(), ""

            if not boxes:
                message = "No face detected - face the camera"

            elif len(boxes) > 1:
                message = "Multiple faces - only one person please"

                for b in boxes:
                    draw_label(
                        display,
                        b.x, b.y, b.w, b.h,
                        "Extra face",
                        RED
                    )

            else:
                box = boxes[0]

                draw_label(
                    display,
                    box.x, box.y, box.w, box.h,
                    f"Sample {len(samples)}/{num_samples}",
                    GREEN
                )

                if time.time() - last_capture >= SETTINGS.capture_interval_sec:

                    face = preprocess_face(frame, box)

                    if is_sharp(face):
                        samples.append(face)
                        last_capture = time.time()

                    else:
                        message = "Image too blurry - hold still"

            if message:
                draw_banner(display, message)

            preview.image(display, channels="BGR")

            progress.progress(
                min(len(samples) / num_samples, 1.0),
                text=f"{len(samples)} / {num_samples} samples"
            )

    finally:
        release_camera(cap)

    return samples


def _faces_from_uploads(files) -> Tuple[List[np.ndarray], List[str]]:
    """Extract one face per uploaded photo."""

    detector = get_detector()

    faces, warnings = [], []

    for file in files:

        image = cv2.imdecode(
            np.frombuffer(file.read(), np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            warnings.append(f"{file.name}: could not be read as an image.")
            continue

        longest = max(image.shape[:2])

        if longest > 1280:
            scale = 1280 / longest

            image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_AREA
            )

        boxes = detector.detect(image)

        if not boxes:
            warnings.append(f"{file.name}: no face found.")

        elif len(boxes) > 1:
            warnings.append(
                f"{file.name}: more than one face - skipped."
            )

        else:
            faces.append(preprocess_face(image, boxes[0]))

    return faces, warnings


# ----------------------------------------------------------------------------
# Duplicate Face Detection
# ----------------------------------------------------------------------------

def _check_duplicate_face(
    embeddings: np.ndarray,
    embedder
):
    """
    Compare new face embeddings with all existing students.

    Returns the best potential match or None.
    """

    ids, gallery = db.load_gallery(embedder.model_name)

    if not ids:
        return None

    recognizer = FaceRecognizer(
        ids,
        gallery,
        margin=0.0
    )

    best_match = None

    # Compare every new sample with the existing gallery.
    for embedding in embeddings:

        match = recognizer.identify(embedding)

        if match.is_known:

            if (
                best_match is None
                or match.score > best_match.score
            ):
                best_match = match

    return best_match


# ----------------------------------------------------------------------------
# Save Student
# ----------------------------------------------------------------------------

def _save_student(
    student_id: str,
    name: str,
    class_name: str,
    faces: List[np.ndarray],
    embedder,
    allow_similar: bool = False
) -> np.ndarray:

    """Generate embeddings, check duplicates and save student data."""

    embeddings = embedder.embed_batch(faces)

    # Check whether this face is already registered.
    match = _check_duplicate_face(embeddings, embedder)

    if match is not None:

        message = (
            f"Potential duplicate face detected! "
            f"Existing Student ID: {match.student_id}. "
            f"Similarity: {match.score:.3f}."
        )

        # Block registration unless manually verified.
        if not allow_similar:
            raise DuplicateFaceError(
                message
                + " Please verify the student's identity before registering."
            )

        st.warning(
            "Admin verification: " + message
        )

    # Save student face images.
    folder = SETTINGS.faces_dir / student_id

    folder.mkdir(parents=True, exist_ok=True)

    try:

        for i, face in enumerate(faces, start=1):

            ok, buffer = cv2.imencode(".jpg", face)

            if ok:
                buffer.tofile(
                    str(folder / f"{student_id}_{i:02d}.jpg")
                )

        # Save student and embeddings in SQLite.
        db.register_student(
            student_id,
            name.strip(),
            class_name.strip(),
            embeddings,
            embedder.model_name
        )

    except Exception:
        shutil.rmtree(folder, ignore_errors=True)
        raise

    return embeddings


# ----------------------------------------------------------------------------
# Page
# ----------------------------------------------------------------------------

def render() -> None:

    page_header(
        "Student Registration",
        "Add a student and capture their face samples."
    )

    tab_register, tab_manage = st.tabs(
        ["➕ Register student", "👥 Manage students"]
    )

    with tab_register:
        _register_tab()

    with tab_manage:
        _manage_tab()


# ----------------------------------------------------------------------------
# Registration Tab
# ----------------------------------------------------------------------------

def _register_tab() -> None:

    embedder = get_embedder()

    if embedder is None:
        return

    # Admin login is outside the registration form so the password
    # can be checked independently before a duplicate override is allowed.
    st.subheader("Admin verification")

    if st.session_state.get("admin_verified", False):
        st.success("Admin verified for this session.")

        if st.button("Lock admin access", key="lock_admin_access"):
            st.session_state["admin_verified"] = False
            st.rerun()

    else:
        with st.expander("Admin login — required only to approve a potential duplicate"):
            admin_password = st.text_input(
                "Admin password",
                type="password",
                key="admin_password_input"
            )

            if st.button("Verify admin", key="verify_admin_access"):
                try:
                    expected_password = st.secrets["admin"]["password"]
                except Exception:
                    st.error(
                        "Admin password is not configured. "
                        "Create .streamlit/secrets.toml as explained in the setup instructions."
                    )
                else:
                    if admin_password and hmac.compare_digest(
                        admin_password,
                        str(expected_password)
                    ):
                        st.session_state["admin_verified"] = True
                        st.success("Admin verification successful.")
                        st.rerun()
                    else:
                        st.error("Incorrect admin password.")

    admin_verified = st.session_state.get("admin_verified", False)

    method = st.radio(
        "Face capture method",
        ["📷 Webcam (automatic)", "🖼️ Upload photos"],
        horizontal=True
    )

    use_webcam = method.startswith("📷")

    with st.form("register_form", clear_on_submit=False):

        c1, c2, c3 = st.columns(3)

        student_id = c1.text_input(
            "Student ID",
            placeholder="e.g. S1001"
        ).strip()

        name = c2.text_input(
            "Full name",
            placeholder="e.g. Asha Verma"
        )

        class_name = c3.text_input(
            "Class",
            placeholder="e.g. 10-A"
        )

        files = None
        num_samples = SETTINGS.registration_samples
        camera_index = SETTINGS.camera_index

        if use_webcam:

            s1, s2 = st.columns(2)

            num_samples = s1.slider(
                "Face samples to capture",
                SETTINGS.min_samples,
                30,
                SETTINGS.registration_samples
            )

            camera_index = int(
                s2.number_input(
                    "Camera index",
                    0,
                    10,
                    SETTINGS.camera_index
                )
            )

            st.caption(
                "Look at the camera, then slowly turn your head "
                "a little left and right while samples are taken."
            )

        else:

            files = st.file_uploader(
                "Clear, front-facing photos (JPG/PNG), one person per photo",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True
            )

            st.caption(
                f"Upload at least {SETTINGS.min_samples} photos "
                "with different angles/lighting for best accuracy."
            )

        # Only a password-verified admin can enable this override.
        admin_override = st.checkbox(
            "I have independently verified that this is a different student "
            "despite the similar face warning.",
            disabled=not admin_verified
        )

        if not admin_verified:
            st.caption("Log in as admin above to enable duplicate-face override.")

        submitted = st.form_submit_button(
            "Register student",
            type="primary"
        )

    if not submitted:
        return

    try:

        # Validate student details.
        error = _validate(
            student_id,
            name,
            class_name
        )

        if error:
            st.error(error)
            return

        # Capture face samples.
        if use_webcam:

            preview = st.empty()

            progress = st.progress(
                0.0,
                text="Starting camera..."
            )

            faces = _capture_from_webcam(
                num_samples,
                camera_index,
                preview,
                progress
            )

            preview.empty()
            progress.empty()

        else:

            faces, warnings = _faces_from_uploads(
                files or []
            )

            for warning in warnings:
                st.warning(warning)

        # Check minimum samples.
        if len(faces) < SETTINGS.min_samples:

            st.error(
                f"Only {len(faces)} usable face sample(s) were captured; "
                f"at least {SETTINGS.min_samples} are needed. "
                "Improve the lighting, face the camera and try again."
            )

            return

        # Generate embeddings and save registration.
        with st.spinner(
            "Generating embeddings and checking duplicate faces..."
        ):

            _save_student(
                student_id,
                name,
                class_name,
                faces,
                embedder,
                allow_similar=(admin_verified and admin_override)
            )

        st.success(
            f"✅ Registered **{name.strip()}** "
            f"({student_id}) with {len(faces)} face samples."
        )

        st.toast(
            f"{name.strip()} registered",
            icon="✅"
        )

        st.image(
            faces[:8],
            width=90,
            channels="BGR"
        )

    except DuplicateFaceError as exc:

        st.error(str(exc))

    except CameraError as exc:

        st.error(f"📷 {exc}")

    except db.DatabaseError as exc:

        st.error(f"🗄️ {exc}")

    except Exception as exc:

        st.error(f"Registration failed: {exc}")


# ----------------------------------------------------------------------------
# Manage Students Tab
# ----------------------------------------------------------------------------

def _manage_tab() -> None:

    try:
        students = db.list_students()

    except db.DatabaseError as exc:
        st.error(f"🗄️ {exc}")
        return

    if students.empty:
        st.info("No students registered yet.")
        return

    st.dataframe(
        students.rename(
            columns={
                "student_id": "Student ID",
                "name": "Name",
                "class_name": "Class",
                "created_at": "Registered",
                "samples": "Face samples"
            }
        ),
        use_container_width=True,
        hide_index=True
    )

    with st.expander("🗑️ Delete a student"):

        options = {
            f"{r.student_id} - {r.name}": r.student_id
            for r in students.itertuples()
        }

        choice = st.selectbox(
            "Student",
            list(options)
        )

        confirm = st.checkbox(
            "I understand this also deletes their face data "
            "and attendance records."
        )

        if st.button(
            "Delete student",
            disabled=not confirm
        ):

            try:

                sid = options[choice]

                db.delete_student(sid)

                shutil.rmtree(
                    SETTINGS.faces_dir / sid,
                    ignore_errors=True
                )

                st.success(f"Deleted {choice}.")

                st.rerun()

            except db.DatabaseError as exc:
                st.error(f"🗄️ {exc}")
