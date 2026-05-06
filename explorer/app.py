"""eBERlight Explorer — Main application entry point.

A Streamlit-based interactive portal over synchrotron-data-analysis-notes,
aligned with ANL/APS design standards.

Ref: PRD-001 — Product Requirements Document.
Ref: ADR-001 — Choose Streamlit.
Ref: DS-001 — Design system tokens.
Ref: FR-001 — Landing page with hero, search, 3 cluster cards.
"""

from pathlib import Path

import streamlit as st
from components.breadcrumb import render_breadcrumb
from components.footer import render_footer
from components.header import render_header
from lib.ia import CLUSTER_META

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

# --- Hero (FR-001 — hero + search bar + 3 cluster cards) ---
st.markdown(
    """
    <div style="text-align: center; padding: 48px 0 24px 0;">
        <h1 style="color: #0033A0; font-size: 36px; margin-bottom: 12px;">
            eBERlight Research Explorer
        </h1>
        <p style="color: #555555; font-size: 18px; max-width: 600px;
                  margin: 0 auto 24px auto;">
            Navigate synchrotron data analysis knowledge at
            Argonne's Advanced Photon Source
        </p>
        <form action="/Search" method="get" role="search"
              aria-label="Site search"
              style="display:flex;justify-content:center;gap:0;
                     max-width:540px;margin:0 auto;">
            <input type="search" name="q"
                   placeholder="Search 188 notes — modality, method, paper, tool…"
                   aria-label="Search query" autocomplete="off" spellcheck="false"
                   style="flex:1;padding:10px 14px;border:1px solid #C0C0C0;
                          border-radius:24px 0 0 24px;border-right:0;
                          font-size:15px;outline-offset:2px;background:#FFFFFF;">
            <button type="submit" aria-label="Search"
                    style="padding:10px 24px;border:1px solid #0033A0;
                           border-radius:0 24px 24px 0;background:#0033A0;
                           color:#FFFFFF;cursor:pointer;font-size:15px;
                           font-weight:600;">
                🔎 Search
            </button>
        </form>
        <p style="color:#888888;font-size:13px;margin:8px 0 0 0;">
            Tip: try <code>tomogan</code>, <code>vo 2018</code>,
            <code>ring artifact</code>, or <code>tomopy</code>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Cluster Cards ---
cols = st.columns(3)
cluster_order = ["discover", "explore", "build"]
cluster_pages = {"discover": "1_Discover", "explore": "2_Explore", "build": "3_Build"}

for col, cluster_id in zip(cols, cluster_order, strict=True):
    meta = CLUSTER_META[cluster_id]
    with col:
        st.markdown(
            f"""
            <div class="eberlight-card" style="border-top: 4px solid {meta["color"]}; min-height: 200px;">
                <h4 style="color: {meta["color"]}; margin: 0 0 12px 0;">{meta["name"]}</h4>
                <p style="font-size: 14px; color: #555555; margin: 0 0 16px 0;">{meta["description"]}</p>
                <span style="color: {meta["color"]}; font-weight: 600; font-size: 15px;">Enter →</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# --- Interactive Lab CTA (ADR-008) ---
_LAB_COLOR = CLUSTER_META["build"]["color"]
_DISCOVER_COLOR = CLUSTER_META["discover"]["color"]
st.markdown(
    f"""
    <div style="margin-top: 32px;display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        <div class="eberlight-card" style="border-left: 4px solid {_DISCOVER_COLOR};">
            <h4 style="color: {_DISCOVER_COLOR}; margin: 0 0 8px 0;">
                🧠 <a href="/Knowledge_Graph" target="_self"
                       style="color:{_DISCOVER_COLOR};text-decoration:none;">Knowledge Graph</a>
            </h4>
            <p style="font-size: 14px; color: #555555; margin: 0 0 8px 0;">
                Cross-reference network of every modality, AI/ML method, paper, tool,
                Interactive-Lab recipe, and noise/artifact in one interactive view.
                Hover for details, click to navigate.
            </p>
            <p style="font-size: 13px; color: #888888; margin: 0;">
                100+ entities · 120+ edges · auto-extracted from notes + recipe.yaml.
            </p>
        </div>
        <div class="eberlight-card" style="border-left: 4px solid {_LAB_COLOR};">
            <h4 style="color: {_LAB_COLOR}; margin: 0 0 8px 0;">
                🧪 <a href="/Experiment" target="_self"
                       style="color:{_LAB_COLOR};text-decoration:none;">Interactive Lab</a>
            </h4>
            <p style="font-size: 14px; color: #555555; margin: 0 0 8px 0;">
                Replay noise mitigation techniques from prior research on real bundled data —
                tune parameters, compare before/after, see PSNR/SSIM against a clean reference.
            </p>
            <p style="font-size: 13px; color: #888888; margin: 0;">
                3 recipes · 71 real samples · Vo 2018 / Munch 2009 / van Dokkum 2001.
            </p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Footer ---
render_footer()
