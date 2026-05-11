"""eBERlight Explorer — Main application entry point.

A Streamlit-based local portal over the author's synchrotron-data-analysis
study notes. **Personal research / learning project; not an official APS
or ANL property** — see ADR-005 / CLAUDE.md for the unaffiliated-personal
framing.

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
        <h1 style="color: var(--color-primary); font-size: 36px; margin-bottom: 12px;">
            eBERlight Research Explorer
        </h1>
        <p style="color: var(--color-text-secondary); font-size: 18px; max-width: 600px;
                  margin: 0 auto 24px auto;">
            Personal study notes on synchrotron data analysis
            <br>
            <span style="font-size:13px;color:var(--color-text-muted);">
                A self-directed learning workspace · ANL/APS-inspired ·
                unaffiliated personal research
            </span>
        </p>
        <form action="/Search" method="get" role="search"
              aria-label="Site search"
              style="display:flex;justify-content:center;gap:0;
                     max-width:540px;margin:0 auto;">
            <input type="search" name="q"
                   placeholder="Search 188 notes — modality, method, paper, tool…"
                   aria-label="Search query" autocomplete="off" spellcheck="false"
                   style="flex:1;padding:10px 14px;border:1px solid #C0C0C0;
                          border-radius:var(--radius-pill) 0 0 var(--radius-pill);
                          border-right:0;font-size:15px;outline-offset:2px;
                          background:var(--color-surface-alt);">
            <button type="submit" aria-label="Search"
                    style="padding:10px 24px;border:1px solid var(--color-primary);
                           border-radius:0 var(--radius-pill) var(--radius-pill) 0;
                           background:var(--color-primary);
                           color:var(--color-text-inverse);cursor:pointer;
                           font-size:15px;font-weight:600;">
                🔎 Search
            </button>
        </form>
        <p style="color:var(--color-text-muted);font-size:13px;margin:8px 0 0 0;">
            Tip: try <code>tomogan</code>, <code>vo 2018</code>,
            <code>ring artifact</code>, or <code>tomopy</code>
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Onboarding scenario picker (REL-E080) — surfaced above the cluster
#     cards so a first-time visitor can pick a start path instead of guessing
#     which of Discover / Explore / Build maps to their task. The three
#     scenarios cover the three high-leverage entry points (a fresh sample,
#     a weird-looking image, or a hands-on method tryout). ---
_SCENARIOS = [
    {
        "icon": "🔬",
        "title": "I have a sample to analyse",
        "body": (
            "Map sample → modality → method → tool. Best starting point when you "
            "know what you're imaging but not how."
        ),
        "href": "/Explore",
    },
    {
        "icon": "🩺",
        "title": "I see something weird in my data",
        "body": (
            "Symptom-driven troubleshooter walks the differential diagnoses for "
            "ring patterns, streaks, noise, blur, dead pixels, and more."
        ),
        "href": "/Troubleshooter",
    },
    {
        "icon": "🧪",
        "title": "I want to try a noise-mitigation method hands-on",
        "body": (
            "Replay 14 bundled recipes on real samples — slide parameters, watch "
            "PSNR/SSIM move, see the |Δ| diff panel."
        ),
        "href": "/Experiment",
    },
]

_scenario_cards = "".join(
    f"""
    <a class="scenario-card" href="{s["href"]}" target="_self">
        <div class="icon" aria-hidden="true">{s["icon"]}</div>
        <div class="title">{s["title"]}</div>
        <div class="body">{s["body"]}</div>
    </a>
    """
    for s in _SCENARIOS
)
st.markdown(
    f"""
    <section class="eberlight-onboarding" aria-label="Choose your scenario">
        <h3>New here? Pick your scenario</h3>
        <p class="sub">Three high-leverage entry points — or browse the
        three clusters below if you already know where you're heading.</p>
        <div class="scenarios">{_scenario_cards}</div>
    </section>
    """,
    unsafe_allow_html=True,
)

# --- Cluster Cards (R10 P0-2: each card is a real anchor so the whole tile
#     is clickable, not just inert "Enter →" text). ---
cols = st.columns(3)
cluster_order = ["discover", "explore", "build"]
cluster_paths = {"discover": "/Discover", "explore": "/Explore", "build": "/Build"}

for col, cluster_id in zip(cols, cluster_order, strict=True):
    meta = CLUSTER_META[cluster_id]
    href = cluster_paths[cluster_id]
    with col:
        st.markdown(
            f"""
            <a href="{href}" target="_self" class="eberlight-cluster-card-link"
               style="text-decoration:none;color:inherit;display:block;">
                <div class="eberlight-card eberlight-cluster-card"
                     style="border-top: 4px solid {meta["color"]}; min-height: 200px;
                            cursor:pointer;">
                    <h4 style="color: {meta["color"]}; margin: 0 0 12px 0;">
                        {meta["name"]}
                    </h4>
                    <p style="font-size: 14px; color: var(--color-text-secondary);
                              margin: 0 0 16px 0;">
                        {meta["description"]}
                    </p>
                    <span style="color: {meta["color"]}; font-weight: 600; font-size: 15px;">
                        Enter →
                    </span>
                </div>
            </a>
            """,
            unsafe_allow_html=True,
        )

# --- Feature CTA grid (mirrored on the static site for invariant #9) ---
_DISCOVER_COLOR = CLUSTER_META["discover"]["color"]
_EXPLORE_COLOR = CLUSTER_META["explore"]["color"]
_BUILD_COLOR = CLUSTER_META["build"]["color"]


def _cta_card(color: str, icon: str, title: str, href: str, summary: str, stat: str) -> str:
    return f"""
    <div class="eberlight-card" style="border-left: 4px solid {color};">
        <h4 style="color: {color}; margin: 0 0 8px 0;">
            {icon} <a href="{href}" target="_self"
                   style="color:{color};text-decoration:none;">{title}</a>
        </h4>
        <p style="font-size: 14px; color: var(--color-text-secondary); margin: 0 0 8px 0;">
            {summary}
        </p>
        <p style="font-size: 13px; color: var(--color-text-muted); margin: 0;">{stat}</p>
    </div>
    """


_FEATURE_CARDS = [
    _cta_card(
        _DISCOVER_COLOR,
        "🧠",
        "Knowledge Graph",
        "/Knowledge_Graph",
        "Cross-reference network of every modality, AI/ML method, paper, tool, "
        "Interactive-Lab recipe, and noise/artifact in one interactive view. "
        "Hover for details, click to navigate.",
        "100+ entities · 120+ edges · auto-extracted from notes + recipe.yaml.",
    ),
    _cta_card(
        _BUILD_COLOR,
        "🧪",
        "Interactive Lab",
        "/Experiment",
        "Replay noise mitigation techniques from prior research on real bundled "
        "data — tune parameters, compare before/after, see PSNR/SSIM against a "
        "clean reference.",
        "14 recipes · 90+ real samples · TomoGAN / Vo / Munch / Herraez phase-unwrap / "
        "TV / NLM / bilateral / wavelet / inpaint / beam-hardening / cosmic-ray.",
    ),
    _cta_card(
        _EXPLORE_COLOR,
        "🩺",
        "Troubleshooter",
        "/Troubleshooter",
        "Symptom-driven decision tree over the noise catalog. Pick what you see "
        "in the data; get differential diagnoses with severity, conditions, and "
        "a one-click jump to the matching Lab recipe.",
        "11 symptom categories · 35 differential cases · before/after comparisons.",
    ),
    _cta_card(
        _DISCOVER_COLOR,
        "🔎",
        "Search & Bibliography",
        "/Search",
        "Global full-text search across every note plus a filterable BibTeX "
        "bibliography. Title-boosted relevance, prefix matching, deep links.",
        "<10 ms typical query · TF-IDF approx · 19 + 20 BibTeX entries indexed.",
    ),
]

st.markdown(
    f"""
    <div style="margin-top: 32px;display:grid;grid-template-columns:1fr 1fr;gap:16px;">
        {"".join(_FEATURE_CARDS)}
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Footer ---
render_footer()
