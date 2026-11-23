
"""FaceAttend Admin Dashboard: statistics, trends and recent attendance."""

from __future__ import annotations

import streamlit as st

from core import attendance, database as db
from ui.styles import metric_card, page_header


def render() -> None:

    # Page heading
    page_header(
        "Admin Dashboard",
        "Monitor student attendance and system activity."
    )

    # Refresh button
    col1, col2 = st.columns([5, 1])

    with col2:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    # Load dashboard data
    try:
        stats = attendance.get_dashboard_stats()
        trend = attendance.weekly_trend(7)
        classes = db.class_breakdown(attendance.today_str())

        recent = db.query_attendance(
            start_date=attendance.today_str(),
            end_date=attendance.today_str()
        )

    except db.DatabaseError as exc:
        st.error(f"🗄️ Database Error: {exc}")
        return

    # --------------------------------------------------
    # ATTENDANCE STATISTICS
    # --------------------------------------------------

    st.subheader("Today's Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Total Students",
            stats["total"],
            "Registered students",
            "#5B7CFA"
        )

    with c2:
        metric_card(
            "Present Today",
            stats["present"],
            "Successfully marked",
            "#3DD6C6"
        )

    with c3:
        metric_card(
            "Absent Today",
            stats["absent"],
            "Not yet marked",
            "#FFB454"
        )

    with c4:
        metric_card(
            "Attendance Rate",
            f'{stats["percentage"]}%',
            "Today's attendance",
            "#C084FC"
        )

    # No students registered
    if stats["total"] == 0:

        st.info(
            "No students registered yet. "
            "Open Student Registration to add students."
        )

        return

    # --------------------------------------------------
    # ATTENDANCE VISUALIZATION
    # --------------------------------------------------

    st.write("")

    st.subheader("Attendance Analytics")

    left, right = st.columns([3, 2])

    # Weekly attendance chart
    with left:

        with st.container(border=True):

            st.markdown("### Weekly Attendance")

            st.caption("Number of students present during the last 7 days.")

            if trend.empty:
                st.info("No attendance data available yet.")

            else:
                st.bar_chart(
                    trend,
                    color="#3DD6C6",
                    height=300
                )

    # Class-wise attendance
    with right:

        with st.container(border=True):

            st.markdown("### Class-wise Attendance")

            st.caption("Present and absent students by class.")

            if classes.empty:

                st.info("No class data available.")

            else:

                classes = classes.copy()

                classes["Absent"] = (
                    classes["Total"] - classes["Present"]
                )

                st.bar_chart(
                    classes.set_index("Class")[["Present", "Absent"]],
                    color=["#3DD6C6", "#FFB454"],
                    height=300
                )

    # --------------------------------------------------
    # RECENT ATTENDANCE RECORDS
    # --------------------------------------------------

    st.write("")

    with st.container(border=True):

        st.markdown("### Recent Check-ins")

        st.caption(
            "Students who have been marked present today."
        )

        if recent.empty:

            st.info(
                "Nobody has checked in yet. "
                "Start the camera from Live Attendance."
            )

        else:

            st.dataframe(
                recent[
                    [
                        "Time",
                        "Student ID",
                        "Name",
                        "Class",
                        "Confidence"
                    ]
                ],
                use_container_width=True,
                hide_index=True
            )

            st.success(
                f"{len(recent)} attendance record(s) found today."
            )

    # --------------------------------------------------
    # FOOTER
    # --------------------------------------------------

    st.divider()

    st.caption(
        "FaceAttend | AI-Based Attendance Management System"
    )
