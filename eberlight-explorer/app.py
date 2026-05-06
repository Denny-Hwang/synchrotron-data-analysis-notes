"""DEPRECATED — see ADR-009.

This is the legacy first-generation Streamlit app. It has been
superseded by the new explorer at ``../explorer/`` (per ADR-001,
ADR-002, ADR-005, ADR-008). Running this file no longer launches
the old portal — it only renders a redirect notice. The directory
is scheduled for removal at ``notes-v1.0.0``.

If you want the **Interactive Lab** (real noise-mitigation experiments
on bundled research data), the **Experiment** page, the **3-cluster
information architecture**, the **GitHub Pages mirror**, or the
**recipe gallery**, run the new app instead:

    streamlit run explorer/app.py

See README.md (root) and docs/02_design/decisions/ADR-009.md.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="DEPRECATED — eBERlight Explorer",
    page_icon="⚠️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        .block-container { padding-top: 2.5rem; }
        .deprecated-banner {
            background: #FDECEA;
            border-left: 6px solid #C0392B;
            padding: 24px 32px;
            border-radius: 8px;
            margin-bottom: 24px;
        }
        .deprecated-banner h1 {
            color: #C0392B;
            margin: 0 0 8px 0;
            font-size: 28px;
        }
        .deprecated-banner code {
            background: #2B2B2B;
            color: #FFE9A8;
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 14px;
        }
        .deprecated-banner ul { margin-top: 12px; }
        .deprecated-banner li { font-size: 15px; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="deprecated-banner">
        <h1>⚠️ This is the deprecated legacy app.</h1>
        <p style="margin: 0; font-size: 16px;">
            <b>You are running</b> <code>eberlight-explorer/app.py</code> —
            the first-generation Streamlit portal that was superseded in
            <b>notes-v0.10.0</b>. It is missing the
            <b>Interactive Lab</b>, the <b>Experiment</b> page, the
            ANL/APS-aligned design system, and the recipe gallery.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### How to run the current app")

st.code("streamlit run explorer/app.py", language="bash")

st.markdown(
    """
The current app (`explorer/`) ships:

- **4 pages** — *Discover*, *Explore*, *Build*, **Experiment** — mapped to
  the 3-cluster information architecture (ADR-004).
- **Interactive Lab** (the *Experiment* page) — replays prior-research
  noise-mitigation algorithms on real bundled data with parameter sliders,
  before/after side-by-side, and PSNR/SSIM metrics.
- **Recipe gallery** on the Build cluster page — auto-discovers every
  `recipe.yaml` under `experiments/`.
- **GitHub Pages mirror** — read-only static site, regenerated on every push.
- **CI test suite** with recipe-contract drift protection.

### Why is this app still here?

ADR-009 explains the rationale: the legacy directory remains on `main`
through the end of `notes-v0.x` because ADR-001 / ADR-002 / ADR-005 cite
it as their before-state. It will be deleted in the `notes-v1.0.0` PR.

### Where to read more

- [Root README](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes#readme)
- [ADR-009 — Deprecate the legacy `eberlight-explorer/` directory](../docs/02_design/decisions/ADR-009.md)
- [Interactive Lab data inventory](../10_interactive_lab/README.md)
- [Recipe schema](../experiments/README.md)
    """
)

# Refuse to render anything else from this app — the legacy pages
# (Knowledge Graph, Modalities, etc.) still ship in `pages/` and
# Streamlit auto-discovers them, but each page also redirects.
_HERE = Path(__file__).parent
_PAGES_DIR = _HERE / "pages"
if _PAGES_DIR.is_dir():
    st.caption(
        "Note: the legacy `pages/` directory is still on disk for ADR-archival "
        "purposes; clicking those entries in the sidebar will show this same "
        "redirect notice. Stop the server and switch to "
        "`streamlit run explorer/app.py` to see the current product."
    )

st.stop()
