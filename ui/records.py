"""Attendance records page with student / class / date filters and CSV download."""
from __future__ import annotations

from datetime import date, timedelta

import streamlit as st

from core import attendance, database as db
from ui.styles import page_header


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")   # BOM so Excel opens it correctly


def render() -> None:
    page_header("Attendance Records", "Search and filter every attendance entry.")

    try:
        students = db.list_students()
        classes = db.list_class_names()
    except db.DatabaseError as exc:
        st.error(f"🗄️ {exc}")
        return

    with st.container(border=True):
        c1, c2, c3 = st.columns(3)
        student_options = {"All students": None}
        student_options.update({f"{r.student_id} - {r.name}": r.student_id for r in students.itertuples()})
        student_id = student_options[c1.selectbox("Student", list(student_options))]
        class_choice = c2.selectbox("Class", ["All classes"] + classes)
        period = c3.selectbox("Period", ["Today", "Last 7 days", "Last 30 days", "Custom range", "All time"])

        today = date.today()
        start = end = None
        if period == "Today":
            start = end = today
        elif period == "Last 7 days":
            start, end = today - timedelta(days=6), today
        elif period == "Last 30 days":
            start, end = today - timedelta(days=29), today
        elif period == "Custom range":
            picked = st.date_input("Date range", value=(today - timedelta(days=7), today))
            if isinstance(picked, (tuple, list)):
                start = picked[0]
                end = picked[1] if len(picked) > 1 else picked[0]
            else:
                start = end = picked

    try:
        df = db.query_attendance(
            student_id=student_id,
            class_name=None if class_choice == "All classes" else class_choice,
            start_date=start.isoformat() if start else None,
            end_date=end.isoformat() if end else None,
        )
    except db.DatabaseError as exc:
        st.error(f"🗄️ {exc}")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Records", len(df))
    m2.metric("Unique students", df["Student ID"].nunique() if not df.empty else 0)
    m3.metric("Days covered", df["Date"].nunique() if not df.empty else 0)

    if df.empty:
        st.info("No attendance records match these filters.")
        return

    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download as CSV", _csv_bytes(df),
                       file_name=f"attendance_records_{attendance.today_str()}.csv",
                       mime="text/csv", type="primary")
