"""Footer component for eBERlight Explorer.

Renders the personal-research disclaimer block that must appear on every
page, plus reference links to the upstream APS / eBERlight sites where
the real research lives, and a last-updated timestamp.

Ref: DS-001 (design_system.md) — Footer component spec.
Ref: FR-010 — personal-research disclaimer footer on every page.
Ref: NFR-001 — unaffiliated / personal-archive framing.
"""

import subprocess
from datetime import datetime

import streamlit as st


def _resolve_last_updated() -> str:
    """Resolve last-updated date once at module import.

    R13 Rec #2 — was previously called from ``render_footer`` on every
    page render, which forks a ``git log`` process per Streamlit
    re-run. Slider drags on the Lab triggered 30+ git forks per second.
    Now computed once and cached as a module-level constant.

    Returns:
        ISO date string (YYYY-MM-DD).
    """
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%ci"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()[:10]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return datetime.now().strftime("%Y-%m-%d")


_LAST_UPDATED = _resolve_last_updated()


def render_footer() -> None:
    """Render the personal-archive disclaimer footer.

    Produces the footer with:
    - Personal-research disclaimer (unaffiliated with ANL / APS / DOE)
    - Reference pointers to the official APS / eBERlight sites
    - Link to the source repository
    - Last-updated timestamp from git
    """
    last_updated = _LAST_UPDATED

    footer_html = f"""
    <div class="eberlight-footer">
        <p>
            <b>Personal eBERlight archive — not an official site.</b>
            This portal is a personal study / learning project that
            collects the author's own notes on synchrotron data
            analysis around the eBERlight program as a reference
            topic. It is <b>not affiliated with or endorsed by</b>
            ANL, APS, DOE, or the eBERlight program.
        </p>
        <p>
            For the actual research, programs, beamtime calls, and
            authoritative documentation, please refer to the official
            sites linked below. Any opinions, summaries, or mistakes
            in these notes are the author's own.
        </p>
        <div class="eberlight-footer-links">
            <a href="https://www.aps.anl.gov/" target="_blank" rel="noopener">APS (official — actual research here)</a>
            <a href="https://eberlight.aps.anl.gov/" target="_blank" rel="noopener">eBERlight (official — actual research here)</a>
            <a href="https://github.com/Denny-Hwang/synchrotron-data-analysis-notes" target="_blank" rel="noopener">Repository</a>
        </div>
        <div class="eberlight-footer-updated">Last updated: {last_updated}</div>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)
