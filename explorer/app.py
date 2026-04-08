"""eBERlight Explorer — Main application entry point.

A Streamlit-based interactive portal over synchrotron-data-analysis-notes,
aligned with ANL/APS design standards.

Ref: PRD-001 — Product Requirements Document.
Ref: ADR-001 — Choose Streamlit.
Ref: DS-001 — Design system tokens.
"""

from pathlib import Path

import streamlit as st

from components.breadcrumb import render_breadcrumb
from components.footer import render_footer
from components.header import render_header

# --- Page Config ---
st.set_page_config(
    page_title="eBERlight Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# --- Load Custom CSS ---
_CSS_PATH = Path(__file__).parent / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)

# --- Header ---
render_header()

# --- Breadcrumb ---
render_breadcrumb([("Home", None)])

# --- Hero ---
st.markdown(
    """
    <div style="text-align: center; padding: 48px 0;">
        <h1 style="color: #0033A0; font-size: 36px; margin-bottom: 12px;">
            Hello, eBERlight
        </h1>
        <p style="color: #555555; font-size: 18px; max-width: 600px; margin: 0 auto;">
            Navigate synchrotron data analysis knowledge at
            Argonne's Advanced Photon Source
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Footer ---
render_footer()
