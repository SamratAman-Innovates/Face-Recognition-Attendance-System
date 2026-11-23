
"""
FaceAttend - AI-Based Attendance Management System
Run using: streamlit run app.py
"""


from __future__ import annotations


import hmac
import streamlit as st

st.set_page_config(
    page_title="FaceAttend | Admin Portal",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from core import database as db
from ui import dashboard, live_attendance, records, registration, reports
from ui.styles import apply_styles, sidebar_brand


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------

PAGES = {
    "📊  Dashboard": dashboard.render,
    "🧑‍🎓  Student Registration": registration.render,
    "📷  Live Attendance": live_attendance.render,
    "🗒️  Attendance Records": records.render,
    "📑  Reports": reports.render,
}


# --------------------------------------------------
# PREMIUM DARK THEME
# --------------------------------------------------

def load_custom_css():

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(ellipse at 10% 10%, rgba(47,91,190,0.18), transparent 35%),
            radial-gradient(ellipse at 90% 85%, rgba(25,180,205,0.10), transparent 35%),
            #080d1a;
        color: #f4f7ff;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #0c1425;
        border-right: 1px solid rgba(140,165,210,0.12);
    }

    [data-testid="stSidebar"] * {
        color: #dce6fa;
    }

    h1, h2, h3 {
        color: #f4f7ff !important;
    }

    p, label {
        color: #aebbd1;
    }

    /* BRAND */

    .brand-label {
        color: #8daeff;
        font-size: 13px;
        font-weight: 800;
        letter-spacing: 4px;
        text-transform: uppercase;
        margin-bottom: 20px;
    }

    /* HERO SECTION */

    .hero-heading {
        font-size: clamp(38px, 5vw, 62px);
        line-height: 1.12;
        font-weight: 800;
        letter-spacing: -2px;
        color: #f5f8ff;
        margin-bottom: 22px;
    }

    .hero-heading span {
        background: linear-gradient(90deg, #6e9aff, #43d9e9);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-description {
        color: #aab9d2;
        font-size: 15px;
        line-height: 1.9;
        max-width: 520px;
    }

    /* FEATURE PILLS */

    .feature-pill-container {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 28px;
    }

    .feature-pill {
        display: inline-block;
        padding: 10px 14px;
        border: 1px solid rgba(86,133,255,0.28);
        background: rgba(86,133,255,0.12);
        color: #d5e3ff;
        border-radius: 30px;
        font-size: 11px;
        font-weight: 600;
    }

    /* LOGIN CARD */

    .login-card {
        background: linear-gradient(
            145deg,
            rgba(26,40,67,0.96),
            rgba(14,23,41,0.97)
        );
        border: 1px solid rgba(137,165,220,0.19);
        border-radius: 22px;
        padding: 30px;
        box-shadow: 0 25px 80px rgba(0,0,0,0.38);
        margin-top: 12px;
        margin-bottom: 0;
    }

    .login-icon {
        width: 58px;
        height: 58px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 17px;
        font-size: 27px;
        background: linear-gradient(135deg, #477cff, #23bfd7);
        box-shadow: 0 8px 25px rgba(62,126,255,0.28);
        margin-bottom: 20px;
    }

    .login-title {
        color: #f4f7ff;
        font-size: 27px;
        font-weight: 750;
        margin-bottom: 8px;
    }

    .login-subtitle {
        color: #9baac3;
        font-size: 13px;
        line-height: 1.7;
    }

    .security-note {
        color: #8494af;
        font-size: 11px;
        text-align: center;
        margin-top: 20px;
    }

    .footer-text {
        text-align: center;
        color: #71819e;
        font-size: 11px;
        margin-top: 55px;
        line-height: 2;
    }

    /* INPUTS */

    .stTextInput input {
        background: #0c1629 !important;
        color: #f4f7ff !important;
        border: 1px solid #293b5c !important;
        border-radius: 10px !important;
        min-height: 46px;
    }

    .stTextInput input:focus {
        border-color: #5685ff !important;
        box-shadow: 0 0 0 1px #5685ff !important;
    }

    /* BUTTONS */

    .stButton button,
    .stFormSubmitButton button {
        background: linear-gradient(100deg, #477cff, #328fe8) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        min-height: 46px;
        font-weight: 650 !important;
        transition: 0.2s ease;
    }

    .stButton button:hover,
    .stFormSubmitButton button:hover {
        box-shadow: 0 5px 22px rgba(65,126,255,0.32);
        transform: translateY(-1px);
    }

    /* DASHBOARD */

    [data-testid="stMetric"] {
        background: #111c31;
        border: 1px solid rgba(137,165,220,0.15);
        padding: 18px;
        border-radius: 14px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid rgba(137,165,220,0.15);
        border-radius: 12px;
    }

    hr {
        border-color: rgba(137,165,220,0.14);
    }

    </style>
    """, unsafe_allow_html=True)


# --------------------------------------------------
# LOGIN PAGE
# --------------------------------------------------

def render_login():

    load_custom_css()

    st.html("""
    <div style="max-width:1050px; margin:5vh auto 0 auto;">
        <div class="brand-label">✦ FACEATTEND</div>
    </div>
    """)

    left, right = st.columns([1.15, 0.85], gap="large")

    # LEFT SIDE

    with left:

        st.html("""
        <div style="padding:45px 15px 20px 15px;">

            <div class="hero-heading">
                Smarter attendance.<br>
                <span>Powered by AI.</span>
            </div>

            <div class="hero-description">
                Experience intelligent attendance management
                powered by advanced facial recognition.
                Register students, track attendance in real time,
                and gain actionable insights through one
                seamless admin platform.
            </div>

            <div class="feature-pill-container">
                <span class="feature-pill">✦ AI Face Recognition</span>
                <span class="feature-pill">◈ Real-time Analytics</span>
                <span class="feature-pill">⌁ Smart Attendance</span>
                <span class="feature-pill">🔒 Secure Admin Portal</span>
            </div>

        </div>
        """)

    # RIGHT SIDE

    with right:

        st.html("""
        <div class="login-card">

            <div class="login-icon">🔐</div>

            <div class="login-title">
                Welcome back
            </div>

            <div class="login-subtitle">
                Sign in to access your FaceAttend
                admin portal.
            </div>

        </div>
        """)

        with st.form("admin_login_form"):

            password = st.text_input(
                "Admin password",
                type="password",
                placeholder="Enter your admin password",
            )

            submitted = st.form_submit_button(
                "Sign in to dashboard →",
                use_container_width=True,
            )

            if submitted:

                try:
                    correct_password = st.secrets["admin"]["password"]

                except (KeyError, FileNotFoundError):
                    st.error(
                        "Admin password is not configured. "
                        "Check .streamlit/secrets.toml."
                    )
                    return

                if password and hmac.compare_digest(
                    password,
                    str(correct_password),
                ):

                    st.session_state["authenticated"] = True
                    st.session_state["admin_verified"] = True

                    st.rerun()

                else:
                    st.error("Incorrect password. Please try again.")

        st.html("""
        <div class="security-note">
            🔒 Protected admin access · FaceAttend Security
        </div>
        """)

    # FOOTER

    st.html("""
    <div class="footer-text">
        FACEATTEND · AI-BASED ATTENDANCE MANAGEMENT SYSTEM
        <br>
        Designed for smarter attendance management
    </div>
    """)
   


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

def main():

    if not st.session_state.get("authenticated", False):

        st.session_state["admin_verified"] = False

        render_login()

        st.stop()

    # ADMIN APPLICATION

    apply_styles()
    load_custom_css()

    sidebar_brand()

    st.sidebar.divider()
    st.sidebar.caption("ADMIN CONTROL PANEL")

    if st.sidebar.button("↪ Logout", use_container_width=True):

        st.session_state["authenticated"] = False
        st.session_state["admin_verified"] = False

        st.rerun()

    # DATABASE

    try:
        db.init_db()

    except db.DatabaseError as exc:
        st.error(f"Database error: {exc}")
        st.stop()

    # NAVIGATION

    choice = st.sidebar.radio(
        "Navigation",
        list(PAGES),
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Face detection: OpenCV\n\n"
        "Embeddings: FaceNet\n\n"
        "Matching: Cosine similarity"
    )

    PAGES[choice]()


if __name__ == "__main__":
    main()
