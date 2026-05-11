"""Header component for eBERlight Explorer.

Renders the site header with logo placeholder, title, top
navigation, and a global search box. Cluster links route to the
auto-discovered Streamlit multi-page entries under
``explorer/pages/``; the search box submits to ``/Search?q=…`` so
users can drop into search from any page (FR-009).

R13 Rec #4 — also emits the skip-to-main-content link as the first
focusable element and the corresponding ``#main-content`` anchor
right after the header, so every page that calls ``render_header()``
gets WCAG 2.4.1 (Bypass Blocks) for free.

Ref: DS-001 (design_system.md) — Header component spec.
Ref: FR-001 — Landing page hero + search bar + cluster cards.
Ref: FR-009 — Global search bar on every page.
Ref: FR-011 — Header with site title and top navigation.
Ref: FR-015 — Keyboard navigation; WCAG 2.4.1.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import streamlit as st
from lib.a11y import main_anchor_html, skip_link_html

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

    R11 I4 — previously emitted inline ``color:#FFFFFF`` which made the
    nav links invisible on the white header background (the legacy
    design assumed a navy bar). All visual styling now comes from
    ``.eberlight-header-nav a`` in styles.css; here we only flip the
    ``active`` class so the current cluster gets highlighted.
    """
    cls = "active" if active else ""
    return f'<a href="/{slug}" target="_self" class="{cls}">{label}</a>'


def _search_form_html(initial_query: str = "") -> str:
    """Return the global search form. Submits GET to ``/Search?q=…``.

    The form uses a plain HTML ``<form action="/Search">`` so it works
    on every page without any Streamlit session-state wiring; whatever
    the user types lands in ``?q=…`` on the Search page where
    ``query_param("q")`` (from ``lib.routing``) already drives the results.

    Returned as a single line with no leading / trailing newline. The
    CommonMark renderer Streamlit uses treats any whitespace-only line
    inside an HTML block as a *blank line*, which closes the block
    early and lets the trailing ``</div>`` and ``#main-content`` anchor
    leak into the page as raw text. Keeping the form on one logical
    line means the entire ``render_header`` HTML stays a single,
    unambiguous block.
    """
    safe = quote(initial_query, safe="")
    return (
        '<form class="eberlight-header-search" action="/Search" method="get"'
        ' role="search" aria-label="Site search">'
        '<input type="search" name="q" placeholder="Search notes…"'
        f' value="{safe}" aria-label="Search query"'
        ' autocomplete="off" spellcheck="false">'
        '<button type="submit" aria-label="Search">🔎</button>'
        "</form>"
    )


def render_header(
    active_cluster: str | None = None,
    *,
    show_search: bool = True,
    initial_query: str = "",
) -> None:
    """Render the site header with logo, title, clickable nav, and search.

    Args:
        active_cluster: Cluster id (``"discover" | "explore" | "build"``)
            to highlight as the current page; ``None`` on the landing
            page or a free-standing page (e.g. Experiment).
        show_search: When ``True`` the global search input renders next
            to the nav. The Search page itself sets this to ``False``
            because it already has a focused query box in the body.
        initial_query: Pre-fills the search input. Useful when the
            current page already received a query via ``?q=…`` so
            the user sees what they searched for.
    """
    # R11 I4 — brand link uses the title token color directly so it
    # reads on the white header instead of the legacy navy-bar white.
    home_link_open = (
        '<a href="/" target="_self" class="eberlight-header-brand-link" '
        'style="text-decoration:none;display:flex;align-items:center;'
        'gap:8px;color:#0033A0;">'
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

    search_html = _search_form_html(initial_query) if show_search else ""

    # R13 Rec #4 — emit the WCAG-mandated skip link as the very first
    # focusable element on the page, then the header, then the
    # ``#main-content`` anchor that the skip link jumps to.
    #
    # Concatenated as one logical line (no embedded newlines). Streamlit's
    # CommonMark renderer interprets whitespace-only lines as blank lines
    # that close an HTML block, so a previous "pretty-printed" version of
    # this string leaked ``</div>`` and the ``#main-content`` anchor as
    # visible text in the header.
    header_html = (
        f"{skip_link_html()}"
        '<div class="eberlight-header">'
        '<div class="eberlight-header-brand">'
        f"{home_link_open}"
        '<div class="eberlight-header-logo">eB</div>'
        '<span class="eberlight-header-title">eBERlight Explorer</span>'
        "</a>"
        "</div>"
        '<nav class="eberlight-header-nav" aria-label="Main navigation">'
        f"{nav_html}"
        "</nav>"
        f"{search_html}"
        "</div>"
        f"{main_anchor_html()}"
    )
    st.markdown(header_html, unsafe_allow_html=True)


def repo_root_from_explorer() -> Path:
    """Return the repository root, assuming ``explorer/`` is a sibling."""
    return Path(__file__).resolve().parent.parent.parent
