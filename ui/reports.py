"""Reports page: daily sheet, per-student summary and full export, all downloadable as CSV."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import streamlit as st

from core import attendance, database as db
from core.config import SETTINGS
from ui.styles import page_header


def _csv_bytes(df) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def _save_copy(df, prefix: str) -> None:
    """Also keep a timestamped copy in data/exports/."""
    path = SETTINGS.exports_dir / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    try:
        df.to_csv(path, index=False, encoding="utf-8-sig")
        st.success(f"Saved to `{path}`")
    except OSError as exc:
        st.error(f"Could not save the file: {exc}")


def render() -> None:
    page_header("Reports", "Summaries you can download as CSV.")
    tab_daily, tab_summary, tab_all = st.tabs(["📅 Daily sheet", "📈 Student summary", "🗂️ Full export"])

    try:
        with tab_daily:
            day = st.date_input("Date", value=date.today(), key="rep_day")
            sheet = db.daily_sheet(day.isoformat())
            if sheet.empty:
                st.info("No students registered yet.")
            else:
                present = int((sheet["Status"] == "Present").sum())
                a, b, c = st.columns(3)
                a.metric("Present", present)
                b.metric("Absent", len(sheet) - present)
                c.metric("Attendance", f"{present / len(sheet) * 100:.1f}%")
                st.dataframe(sheet, use_container_width=True, hide_index=True)
                d1, d2 = st.columns([1, 1])
                d1.download_button("⬇️ Download daily sheet", _csv_bytes(sheet),
                                   file_name=f"daily_sheet_{day.isoformat()}.csv", mime="text/csv", type="primary")
                if d2.button("💾 Also save a copy to data/exports", key="save_daily"):
                    _save_copy(sheet, "daily_sheet")

        with tab_summary:
            picked = st.date_input("Period", value=(date.today() - timedelta(days=29), date.today()), key="rep_range")
            if isinstance(picked, (tuple, list)) and len(picked) == 2:
                start, end = picked[0].isoformat(), picked[1].isoformat()
                summary = attendance.student_attendance_summary(start, end)
                if summary.empty:
                    st.info("No students registered yet.")
                else:
                    st.caption(f"Attendance % = days present ÷ days on which attendance was taken "
                               f"({int(summary['Class Days'].iloc[0])} in this period).")
                    st.dataframe(summary, use_container_width=True, hide_index=True)
                    st.bar_chart(summary.set_index("Name")["Attendance %"], color="#5B7CFA", height=260)
                    s1, s2 = st.columns([1, 1])
                    s1.download_button("⬇️ Download summary", _csv_bytes(summary),
                                       file_name=f"student_summary_{start}_to_{end}.csv", mime="text/csv",
                                       type="primary")
                    if s2.button("💾 Also save a copy to data/exports", key="save_summary"):
                        _save_copy(summary, "student_summary")
            else:
                st.info("Pick both a start and an end date.")

        with tab_all:
            everything = db.query_attendance()
            st.caption(f"{len(everything)} attendance records in total.")
            if everything.empty:
                st.info("No attendance has been recorded yet.")
            else:
                st.dataframe(everything.head(200), use_container_width=True, hide_index=True)
                if len(everything) > 200:
                    st.caption("Showing the newest 200 rows - the download contains all of them.")
                e1, e2 = st.columns([1, 1])
                e1.download_button("⬇️ Download all records", _csv_bytes(everything),
                                   file_name="attendance_all_records.csv", mime="text/csv", type="primary")
                if e2.button("💾 Also save a copy to data/exports", key="save_all"):
                    _save_copy(everything, "attendance_all")
    except db.DatabaseError as exc:
        st.error(f"🗄️ {exc}")
