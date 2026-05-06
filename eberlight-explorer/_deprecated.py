"""Shared deprecation banner for the legacy eberlight-explorer pages.

Each legacy page imports ``render_deprecation_and_stop`` and calls it
as the very first action after ``st.set_page_config``. The banner is
intentionally noisy and ends with ``st.stop()`` so the legacy page
content (which is no longer maintained) never renders.

This module is part of the ADR-009 deprecation plan — it will be
removed together with the rest of ``eberlight-explorer/`` at
``notes-v1.0.0``.
"""

from __future__ import annotations

import streamlit as st


def render_deprecation_and_stop() -> None:
    """Render a prominent deprecation notice and halt the page."""
    st.markdown(
        """
        <style>
            .legacy-stop-banner {
                background: #FDECEA;
                border-left: 6px solid #C0392B;
                padding: 20px 28px;
                border-radius: 8px;
                margin: 16px 0 24px 0;
            }
            .legacy-stop-banner h2 {
                color: #C0392B;
                margin: 0 0 8px 0;
                font-size: 22px;
            }
            .legacy-stop-banner code {
                background: #2B2B2B;
                color: #FFE9A8;
                padding: 3px 8px;
                border-radius: 4px;
                font-size: 13px;
            }
        </style>
        <div class="legacy-stop-banner">
            <h2>⚠️ This page is part of the deprecated legacy app.</h2>
            <p style="margin: 0; font-size: 15px;">
                The current product lives at <code>explorer/</code>.
                Run it with:
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.code("streamlit run explorer/app.py", language="bash")
    st.markdown(
        "The new app includes the **Interactive Lab** (the *Experiment* page) "
        "with real bundled noise-mitigation samples — see the "
        "[root README](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes#readme) "
        "and [ADR-009](../docs/02_design/decisions/ADR-009.md)."
    )
    st.stop()
