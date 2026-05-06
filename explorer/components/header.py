"""Header component for eBERlight Explorer.

Renders the site header with logo placeholder, title, and top
navigation. Cluster links route to the auto-discovered Streamlit
multi-page entries under ``explorer/pages/``.

Ref: DS-001 (design_system.md) — Header component spec.
Ref: FR-011 — Header with site title and top navigation.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

# Streamlit's auto-discovered page slug is derived from the filename
# minus the leading "<digit>_". e.g. ``pages/1_Discover.py`` →
# slug "Discover" → URL ``/Discover``.
_CLUSTER_SLUGS = {
    "discover": "Discover",
    "explore": "Explore",
    "build": "Build",
}


def _nav_link(label: str, slug: str, *, active: bool) -> str:
    """Render one header nav link as plain HTML.

    We avoid ``st.page_link`` here because the header lives outside
    Streamlit's main column layout; HTML anchors with ``target=_self``
    let Streamlit's router pick up the navigation while keeping the
    flexbox header intact.
    """
    style = (
        "color:#FFFFFF;text-decoration:none;padding:6px 12px;"
        "border-radius:4px;font-weight:500;margin-left:6px;"
    )
    if active:
        style += "background:rgba(255,255,255,0.18);"
    return f'<a href="/{slug}" target="_self" style="{style}">{label}</a>'


def render_header(active_cluster: str | None = None) -> None:
    """Render the site header with logo, title, and clickable nav.

    Args:
        active_cluster: Cluster id (``"discover" | "explore" | "build"``)
            to highlight as the current page; ``None`` on the landing
            page or a free-standing page (e.g. Experiment).
    """
    home_link_open = (
        '<a href="/" target="_self" '
        'style="color:#FFFFFF;text-decoration:none;display:flex;'
        'align-items:center;gap:8px;">'
    )
    nav_html = "".join(
        _nav_link(label, slug, active=(active_cluster == cluster))
        for cluster, label, slug in [
            ("discover", "Discover", _CLUSTER_SLUGS["discover"]),
            ("explore", "Explore", _CLUSTER_SLUGS["explore"]),
            ("build", "Build", _CLUSTER_SLUGS["build"]),
        ]
    )
    # The Experiment page lives outside the 3-cluster IA but is reachable
    # from the header so users always have a single click to the lab.
    nav_html += _nav_link("🧪 Experiment", "Experiment", active=False)

    header_html = f"""
    <div class="eberlight-header">
        <div class="eberlight-header-brand">
            {home_link_open}
                <div class="eberlight-header-logo">eB</div>
                <span class="eberlight-header-title">eBERlight Explorer</span>
            </a>
        </div>
        <nav class="eberlight-header-nav" aria-label="Main navigation">
            {nav_html}
        </nav>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)


def repo_root_from_explorer() -> Path:
    """Return the repository root, assuming ``explorer/`` is a sibling."""
    return Path(__file__).resolve().parent.parent.parent
