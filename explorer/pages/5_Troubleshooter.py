"""Symptom-based noise/artifact troubleshooter — Phase R4.

Renders the 11-symptom decision tree from
``09_noise_catalog/troubleshooter.yaml`` as an interactive Streamlit
page. For each chosen symptom, the page shows:

- A short description and any quick-check Python snippets;
- All differential-diagnosis cases as cards, each with the
  ordered list of conditions, severity badge, link to the full
  guide, optional ▶ Run-experiment link into the Interactive Lab,
  and the bundled before/after image (when one exists);
- A modality filter at the top of the page;
- A jump-to-symptom selectbox + ``?symptom=<id>`` query-param deep
  linking so the Knowledge Graph and the noise catalog can both
  link directly to a specific symptom.

Ref: ADR-002 — notes (incl. troubleshooter.md / .yaml) are SoT.
Ref: ADR-008 — Interactive Lab recipes referenced via recipe_id.
Ref: FR-007, FR-009 — symptom-based discovery + filtering.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from components.footer import render_footer
from components.header import render_header
from lib.troubleshooter import (
    Diagnosis,
    Symptom,
    list_before_after_images,
    load_troubleshooter,
    severity_color,
)

st.set_page_config(page_title="Troubleshooter — eBERlight", page_icon="🩺", layout="wide")

_REPO_ROOT = _EXPLORER_DIR.parent
_CSS_PATH = _EXPLORER_DIR / "assets" / "styles.css"
if _CSS_PATH.exists():
    st.markdown(f"<style>{_CSS_PATH.read_text()}</style>", unsafe_allow_html=True)


@st.cache_resource
def _cached_troubleshooter():
    return load_troubleshooter(_REPO_ROOT)


@st.cache_resource
def _cached_image_index() -> dict[str, Path]:
    return list_before_after_images(_REPO_ROOT)


troubleshooter = _cached_troubleshooter()
image_index = _cached_image_index()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------


render_header(active_cluster=None)
st.markdown(
    '<h1 style="color:#C0392B;">🩺 Symptom-Based Troubleshooter</h1>'
    '<p style="color:#555;font-size:16px;margin-bottom:24px;">'
    "Start from what you <b>see</b> in your data, walk through the differential "
    "conditions, and land on the most likely diagnosis. Each diagnosis links "
    "to the full noise-catalog guide and — when a bundled mitigation recipe "
    "exists — to a one-click Interactive Lab run.</p>",
    unsafe_allow_html=True,
)

if not troubleshooter.symptoms:
    st.warning(
        "No symptoms loaded — `09_noise_catalog/troubleshooter.yaml` was "
        "missing or empty. Restore it and reload the page."
    )
    render_footer()
    st.stop()


# ---------------------------------------------------------------------------
# Symptom picker — supports ?symptom=<id> deep links
# ---------------------------------------------------------------------------


def _query_param(name: str) -> str | None:
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        return unquote(raw[0]) if raw else None
    return unquote(str(raw))


symptom_ids = [s.id for s in troubleshooter.symptoms]
deep_link_id = _query_param("symptom")
default_idx = symptom_ids.index(deep_link_id) if deep_link_id in symptom_ids else 0

# R10 P0-4: pickers were in the sidebar (collapsed by default + invisible
# on mobile); moved to the main column. R10 P1-7: numbered stepper.

st.markdown("#### 1️⃣ Pick the symptom you observe")
symptom_idx = st.selectbox(
    "Symptom",
    options=list(range(len(troubleshooter.symptoms))),
    index=default_idx,
    format_func=lambda i: troubleshooter.symptoms[i].title,
    label_visibility="collapsed",
    key="ts_symptom_picker",
)

st.markdown("#### 2️⃣ Narrow down (optional)")
modalities = sorted(
    {
        d.guide.split("/", 1)[0] if "/" in d.guide else "(uncategorised)"
        for d in troubleshooter.all_diagnoses()
        if d.guide
    }
)
filt_cols = st.columns(2)
with filt_cols[0]:
    chosen_modalities = st.multiselect(
        "Modality",
        options=modalities,
        default=modalities,
        key="ts_mod_filter",
        help="Show only diagnoses whose guide lives in these modality folders.",
    )
with filt_cols[1]:
    severity_filter = st.multiselect(
        "Severity",
        options=["critical", "major", "minor"],
        default=["critical", "major", "minor"],
        key="ts_sev_filter",
        help="Filter to differential cases of these severities only.",
    )

symptom: Symptom = troubleshooter.symptoms[symptom_idx]


# ---------------------------------------------------------------------------
# Symptom header + summary
# ---------------------------------------------------------------------------


st.markdown(f"## {symptom.title}")
if symptom.summary:
    st.markdown(symptom.summary)

if symptom.quick_checks:
    with st.expander(f"Quick diagnostic checks ({len(symptom.quick_checks)})", expanded=False):
        for qc in symptom.quick_checks:
            st.markdown(f"**{qc.title}**")
            st.code(qc.code, language=qc.language)


# ---------------------------------------------------------------------------
# Differential cases
# ---------------------------------------------------------------------------


def _diagnosis_severity_badge(d: Diagnosis) -> str:
    color = severity_color(d.severity)
    return (
        f'<span style="background:{color};color:white;padding:3px 10px;'
        f"border-radius:10px;font-size:11px;font-weight:600;"
        f'text-transform:uppercase;letter-spacing:0.5px;">'
        f"{d.severity}</span>"
    )


def _guide_url(guide_path: str) -> str:
    """Map ``tomography/ring_artifact.md`` → ``?note=09_noise_catalog/...``."""
    full = f"09_noise_catalog/{guide_path}" if guide_path else ""
    return f"?note={quote(full, safe='/')}" if full else "#"


def _matches_filter(d: Diagnosis) -> bool:
    if d.severity not in severity_filter:
        return False
    if not d.guide or "/" not in d.guide:
        # uncategorised diagnoses always show when "(uncategorised)" is on
        return "(uncategorised)" in chosen_modalities
    modality = d.guide.split("/", 1)[0]
    return modality in chosen_modalities


visible_cases = [c for c in symptom.cases if _matches_filter(c.diagnosis)]
st.markdown(
    f"#### 3️⃣ Read the differential diagnoses "
    f"<span style='color:#888;font-weight:400;font-size:14px;'>"
    f"({len(visible_cases)} of {len(symptom.cases)})</span>",
    unsafe_allow_html=True,
)

if not visible_cases:
    st.info("No cases match the current modality / severity filters.")

for case in visible_cases:
    diag = case.diagnosis
    cols = st.columns([3, 1])
    with cols[0]:
        st.markdown(
            f'<div class="eberlight-card" '
            f'style="border-left:4px solid {severity_color(diag.severity)};">'
            f'<div style="display:flex;align-items:center;gap:10px;'
            f'justify-content:space-between;">'
            f'<h4 style="margin:0;color:#1A1A1A;">{diag.name}</h4>'
            f"{_diagnosis_severity_badge(diag)}"
            "</div>"
            '<p style="font-size:12px;color:#888;margin:4px 0 8px 0;'
            'text-transform:uppercase;letter-spacing:0.5px;">If you see…</p>'
            '<ul style="font-size:14px;color:#333;margin:0 0 12px 0;padding-left:20px;">'
            + "".join(f"<li>{c}</li>" for c in case.conditions)
            + "</ul>"
            f'<p style="margin:0;font-size:14px;">'
            f'<a href="{_guide_url(diag.guide)}" target="_self" '
            f'style="color:#0033A0;font-weight:600;text-decoration:none;">'
            f"📖 Full guide → {diag.guide}</a>",
            unsafe_allow_html=True,
        )
        if diag.recipe:
            st.markdown(
                f'<a href="/Experiment?recipe={quote(diag.recipe)}" target="_self" '
                f'style="color:#E8515D;font-weight:600;text-decoration:none;'
                f'margin-left:16px;">▶ Run experiment ({diag.recipe})</a>',
                unsafe_allow_html=True,
            )
        st.markdown("</p></div>", unsafe_allow_html=True)

    with cols[1]:
        if diag.image:
            img_path = _REPO_ROOT / "09_noise_catalog" / "images" / diag.image
            if img_path.exists():
                st.image(
                    str(img_path),
                    caption=f"{diag.name} — before / after",
                    width="stretch",
                )
            else:
                st.caption(f"📷 image declared but missing: `{diag.image}`")


# ---------------------------------------------------------------------------
# Footer — link to the prose troubleshooter + summary table
# ---------------------------------------------------------------------------


st.markdown("---")
st.markdown(
    f"""
**Looking for the original prose tree?**
The full markdown version with the original ASCII decision branches is at
[`09_noise_catalog/troubleshooter.md`]({_guide_url("../troubleshooter.md")}).

**Looking for the master matrix?**
[`09_noise_catalog/summary_table.md`]({_guide_url("../summary_table.md")})
lists every catalogued noise/artifact across all 8 modalities with detection
methods, traditional fixes, and AI/ML alternatives.
""".strip()
)

st.caption(
    f"Loaded {len(troubleshooter.symptoms)} symptoms · "
    f"{len(troubleshooter.all_diagnoses())} differential cases · "
    f"{len(image_index)} before/after images available."
)

render_footer()
