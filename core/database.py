"""
SQLite database layer.

Tables
------
students   : one row per registered student
embeddings : one or more FaceNet vectors per student (stored as float32 bytes)
attendance : one row per student per day (UNIQUE constraint prevents duplicates)

Every public function opens its own short-lived connection, which keeps the
code safe inside Streamlit's multi-threaded environment.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.config import SETTINGS


class DatabaseError(RuntimeError):
    """Raised for any database problem, with a readable message."""


SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    student_id  TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    class_name  TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    model       TEXT NOT NULL,
    vector      BLOB NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS attendance (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL REFERENCES students(student_id) ON DELETE CASCADE,
    date        TEXT NOT NULL,      -- YYYY-MM-DD
    time        TEXT NOT NULL,      -- HH:MM:SS
    confidence  REAL,
    UNIQUE (student_id, date)       -- one record per student per day
);

CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date);
CREATE INDEX IF NOT EXISTS idx_embeddings_student ON embeddings(student_id);
"""


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Open a connection, commit on success, roll back on error, always close."""
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = sqlite3.connect(str(SETTINGS.database_path), timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
        conn.commit()
    except sqlite3.Error as exc:
        if conn is not None:
            conn.rollback()
        raise DatabaseError(f"Database error: {exc}") from exc
    finally:
        if conn is not None:
            conn.close()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _read_df(sql: str, params: Sequence = ()) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(sql, conn, params=list(params))


# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
def init_db() -> None:
    """Create the tables if they do not exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA)


# ----------------------------------------------------------------------------
# Students
# ----------------------------------------------------------------------------
def student_exists(student_id: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM students WHERE student_id = ?", (student_id,)).fetchone()
        return row is not None


def get_student(student_id: str) -> Optional[dict]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT student_id, name, class_name FROM students WHERE student_id = ?", (student_id,)
        ).fetchone()
        return dict(row) if row else None


def register_student(
    student_id: str,
    name: str,
    class_name: str,
    embeddings: np.ndarray,
    model_name: str,
) -> None:
    """Insert a student and all of their face embeddings in one transaction."""
    now = _now()
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO students (student_id, name, class_name, created_at) VALUES (?, ?, ?, ?)",
            (student_id, name, class_name, now),
        )
        conn.executemany(
            "INSERT INTO embeddings (student_id, model, vector, created_at) VALUES (?, ?, ?, ?)",
            [(student_id, model_name, np.asarray(v, dtype=np.float32).tobytes(), now) for v in embeddings],
        )


def list_students() -> pd.DataFrame:
    """All students with their number of stored face samples."""
    return _read_df(
        """
        SELECT s.student_id, s.name, s.class_name, s.created_at, COUNT(e.id) AS samples
        FROM students s LEFT JOIN embeddings e ON e.student_id = s.student_id
        GROUP BY s.student_id ORDER BY s.name
        """
    )


def list_class_names() -> List[str]:
    with get_connection() as conn:
        rows = conn.execute("SELECT DISTINCT class_name FROM students ORDER BY class_name").fetchall()
        return [r["class_name"] for r in rows]


def count_students() -> int:
    with get_connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM students").fetchone()[0])


def delete_student(student_id: str) -> None:
    """Delete a student, their embeddings and their attendance records (cascade)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))


# ----------------------------------------------------------------------------
# Embeddings
# ----------------------------------------------------------------------------
def load_gallery(model_name: str) -> Tuple[List[str], np.ndarray]:
    """Return (student_ids, matrix) of every stored embedding made with `model_name`."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT student_id, vector FROM embeddings WHERE model = ?", (model_name,)
        ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype=np.float32)
    ids = [r["student_id"] for r in rows]
    matrix = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
    return ids, matrix


# ----------------------------------------------------------------------------
# Attendance
# ----------------------------------------------------------------------------
def insert_attendance(student_id: str, date_str: str, time_str: str, confidence: float) -> bool:
    """Insert today's record. Returns False if the student already has one for that date."""
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO attendance (student_id, date, time, confidence) VALUES (?, ?, ?, ?)",
            (student_id, date_str, time_str, float(confidence)),
        )
        return cursor.rowcount == 1


def query_attendance(
    student_id: Optional[str] = None,
    class_name: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Attendance records joined with student info, newest first. All filters optional."""
    sql = """
        SELECT a.date AS Date, a.time AS Time, s.student_id AS "Student ID",
               s.name AS Name, s.class_name AS Class, ROUND(a.confidence, 3) AS Confidence
        FROM attendance a JOIN students s ON s.student_id = a.student_id
        WHERE 1 = 1
    """
    params: list = []
    if student_id:
        sql += " AND a.student_id = ?"
        params.append(student_id)
    if class_name:
        sql += " AND s.class_name = ?"
        params.append(class_name)
    if start_date:
        sql += " AND a.date >= ?"
        params.append(start_date)
    if end_date:
        sql += " AND a.date <= ?"
        params.append(end_date)
    sql += " ORDER BY a.date DESC, a.time DESC"
    return _read_df(sql, params)


def count_present(date_str: str) -> int:
    with get_connection() as conn:
        return int(conn.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (date_str,)).fetchone()[0])


def counts_by_date(start_date: str, end_date: str) -> pd.DataFrame:
    return _read_df(
        "SELECT date AS Date, COUNT(*) AS Present FROM attendance "
        "WHERE date BETWEEN ? AND ? GROUP BY date ORDER BY date",
        [start_date, end_date],
    )


def class_breakdown(date_str: str) -> pd.DataFrame:
    """Per class: total students and how many are present on `date_str`."""
    return _read_df(
        """
        SELECT s.class_name AS Class, COUNT(*) AS Total,
               SUM(CASE WHEN a.id IS NOT NULL THEN 1 ELSE 0 END) AS Present
        FROM students s LEFT JOIN attendance a ON a.student_id = s.student_id AND a.date = ?
        GROUP BY s.class_name ORDER BY s.class_name
        """,
        [date_str],
    )


def daily_sheet(date_str: str) -> pd.DataFrame:
    """Every student with Present/Absent status for one date."""
    return _read_df(
        """
        SELECT s.student_id AS "Student ID", s.name AS Name, s.class_name AS Class,
               CASE WHEN a.id IS NULL THEN 'Absent' ELSE 'Present' END AS Status,
               COALESCE(a.time, '') AS Time
        FROM students s LEFT JOIN attendance a ON a.student_id = s.student_id AND a.date = ?
        ORDER BY s.class_name, s.name
        """,
        [date_str],
    )


def student_summary(start_date: str, end_date: str) -> pd.DataFrame:
    """Days present per student in the range."""
    return _read_df(
        """
        SELECT s.student_id AS "Student ID", s.name AS Name, s.class_name AS Class,
               COUNT(a.id) AS "Days Present"
        FROM students s LEFT JOIN attendance a
             ON a.student_id = s.student_id AND a.date BETWEEN ? AND ?
        GROUP BY s.student_id ORDER BY s.class_name, s.name
        """,
        [start_date, end_date],
    )


def count_class_days(start_date: str, end_date: str) -> int:
    """Number of distinct dates in the range on which anyone was marked present."""
    with get_connection() as conn:
        row = conn.execute(
            "SELECT COUNT(DISTINCT date) FROM attendance WHERE date BETWEEN ? AND ?",
            (start_date, end_date),
        ).fetchone()
        return int(row[0])
