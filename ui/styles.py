"""Dark theme CSS and small HTML helpers (cards, headers) for the Streamlit UI."""
from __future__ import annotations

import html

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, button, input, textarea, select {
    font-family: 'Manrope', 'Segoe UI', system-ui, -apple-system, sans-serif !important;
}
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1240px; }
#MainMenu, footer { visibility: hidden; }

/* ---------- sidebar ---------- */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111a30 0%, #0d1322 100%);
    border-right: 1px solid rgba(255,255,255,0.06);
}
.brand { display:flex; align-items:center; gap:.7rem; margin: .2rem 0 1.4rem 0; }
.brand-mark {
    width: 40px; height: 40px; border-radius: 12px; display:grid; place-items:center;
    background: linear-gradient(135deg,#3DD6C6,#5B7CFA); font-size: 1.3rem;
}
.brand-name { font-weight: 800; font-size: 1.05rem; line-height: 1.15; color:#F2F5FC; }
.brand-sub  { font-size: .78rem; color:#8E9BBF; }
[data-testid="stSidebar"] div[role="radiogroup"] { gap: .25rem; }
[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: .6rem .8rem; border-radius: 10px; transition: background .15s;
}
[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { background: rgba(255,255,255,0.05); }

/* ---------- page header ---------- */
.page-header { margin-bottom: 1.4rem; }
.page-header h1 { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em; color:#F2F5FC; }
.page-header p  { color:#8E9BBF; margin:.35rem 0 0 0; font-size: 1rem; }

/* ---------- metric cards ---------- */
.metric-card {
    position: relative; overflow: hidden; height: 100%;
    background: #141C30; border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.15rem 1.3rem;
}
.metric-card::before {
    content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background: var(--accent, #3DD6C6);
}
.metric-label { font-size: .88rem; color:#8E9BBF; font-weight: 600; }
.metric-value { font-size: 2.25rem; font-weight: 800; color:#F2F5FC; line-height: 1.15; margin-top:.2rem; }
.metric-caption { font-size: .8rem; color:#6F7C9E; margin-top:.15rem; }

/* ---------- status chips ---------- */
.chip { display:inline-block; padding:.18rem .65rem; border-radius:999px; font-size:.78rem; font-weight:700; }
.chip-ok  { background: rgba(61,214,198,.15); color:#3DD6C6; }
.chip-bad { background: rgba(255,107,107,.15); color:#FF8A8A; }

/* ---------- widgets ---------- */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 10px; font-weight: 700; padding: .55rem 1.1rem;
}
button[kind="primary"], [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-primaryFormSubmit"] {
    background: linear-gradient(135deg,#3DD6C6,#4F8BF5) !important; color:#0A1020 !important; border:none !important;
}
[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
.stTabs [data-baseweb="tab"] { font-weight: 700; }

@media (max-width: 768px) {
    .block-container { padding: 1rem .8rem 2rem .8rem; }
    .page-header h1 { font-size: 1.55rem; }
    .metric-value { font-size: 1.8rem; }
}
</style>
"""


def apply_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.sidebar.markdown(
        '<div class="brand"><div class="brand-mark">🎓</div>'
        '<div><div class="brand-name">FaceAttend</div>'
        '<div class="brand-sub">AI attendance system</div></div></div>',
        unsafe_allow_html=True,
    )


def page_header(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="page-header"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, caption: str = "", accent: str = "#3DD6C6") -> None:
    st.markdown(
        f'<div class="metric-card" style="--accent:{accent}">'
        f'<div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(str(value))}</div>'
        f'<div class="metric-caption">{html.escape(caption)}</div></div>',
        unsafe_allow_html=True,
    )


def chip(text: str, ok: bool = True) -> str:
    """Return an HTML status chip (render with unsafe_allow_html=True)."""
    return f'<span class="chip {"chip-ok" if ok else "chip-bad"}">{html.escape(text)}</span>'
