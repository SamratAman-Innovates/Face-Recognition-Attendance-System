"""Attendance business logic: marking, duplicate prevention, confirmation and statistics."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, Iterable, List, Optional

import pandas as pd

from core import database as db


class MarkStatus(Enum):
    MARKED = "marked"
    ALREADY_MARKED = "already_marked"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class MarkResult:
    status: MarkStatus
    student_id: str
    name: str = ""
    time: str = ""


def mark_attendance(student_id: str, confidence: float) -> MarkResult:
    """Mark a student present for today - at most once per day.

    Duplicate protection is enforced by the database itself (UNIQUE constraint),
    so it holds even if two sessions run at the same time.
    """
    student = db.get_student(student_id)
    if student is None:
        return MarkResult(MarkStatus.NOT_FOUND, student_id)

    now = datetime.now()
    inserted = db.insert_attendance(
        student_id, now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S"), confidence
    )
    status = MarkStatus.MARKED if inserted else MarkStatus.ALREADY_MARKED
    return MarkResult(status, student_id, student["name"], now.strftime("%H:%M:%S"))


class ConfirmationTracker:
    """Requires a student to be recognised N times in a row before attendance is marked.

    This avoids marking someone because of a single lucky/unlucky frame.
    """

    def __init__(self, required: int) -> None:
        self.required = max(1, int(required))
        self._counts: Dict[str, int] = {}

    def update(self, seen_ids: Iterable[str]) -> List[str]:
        """Feed the student IDs recognised in the latest cycle.

        Returns the IDs that just reached the required streak.
        """
        seen = set(seen_ids)
        for sid in list(self._counts):
            if sid not in seen:
                del self._counts[sid]          # streak broken

        confirmed = []
        for sid in seen:
            self._counts[sid] = self._counts.get(sid, 0) + 1
            if self._counts[sid] == self.required:
                confirmed.append(sid)
        return confirmed


# ----------------------------------------------------------------------------
# Statistics used by the dashboard and reports
# ----------------------------------------------------------------------------
def today_str() -> str:
    return date.today().isoformat()


def get_dashboard_stats(day: Optional[str] = None) -> dict:
    day = day or today_str()
    total = db.count_students()
    present = db.count_present(day)
    absent = max(total - present, 0)
    percentage = round(present / total * 100, 1) if total else 0.0
    return {"total": total, "present": present, "absent": absent, "percentage": percentage}


def weekly_trend(days: int = 7) -> pd.DataFrame:
    """Present count for each of the last `days` days (zero-filled)."""
    end = date.today()
    start = end - timedelta(days=days - 1)
    counts = db.counts_by_date(start.isoformat(), end.isoformat())
    full_range = pd.date_range(start, end, freq="D")
    series = pd.Series(0, index=full_range, name="Present")
    if not counts.empty:
        counts["Date"] = pd.to_datetime(counts["Date"])
        series.update(counts.set_index("Date")["Present"])
    return series.astype(int).to_frame()


def student_attendance_summary(start: str, end: str) -> pd.DataFrame:
    """Days present and attendance % per student (% = days present / days attendance was taken)."""
    summary = db.student_summary(start, end)
    class_days = db.count_class_days(start, end)
    summary["Class Days"] = class_days
    summary["Attendance %"] = (
        (summary["Days Present"] / class_days * 100).round(1) if class_days else 0.0
    )
    return summary
