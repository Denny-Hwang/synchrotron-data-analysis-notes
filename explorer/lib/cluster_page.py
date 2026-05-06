"""Shared rendering for the three cluster landing pages.

Every cluster page (``1_Discover.py``, ``2_Explore.py``, ``3_Build.py``)
delegates to :func:`render_cluster_page` so the routing logic for
``?note=...`` deep links and ``?tag=...`` filters lives in exactly one
place.

Three modes are handled here:

1. ``?note=<url_id>`` is set → render the note-detail view via
   :func:`components.note_view.render_note_view` and return.
2. ``?tag=<tag>`` is set → render the cluster's card grid filtered to
   notes whose ``tags`` list contains that tag, plus a small "filtering
   by …" banner with a clear-filter link.
3. Neither set → render the default card grid grouped by folder.

Ref: ADR-004 — 3-cluster IA mapping.
Ref: FR-003 — Card-based cluster pages.
Ref: FR-004 — Clicking a card opens the note-detail view.
Ref: FR-007 — Clickable tag filtering.
"""

from __future__ import annotations

from itertools import groupby
from pathlib import Path
from urllib.parse import unquote

import streamlit as st
from components.breadcrumb import render_breadcrumb
from components.card import render_note_card
from components.footer import render_footer
from components.header import render_header
from components.note_view import render_note_view

from lib.ia import CLUSTER_META, get_folders_for_cluster
from lib.notes import Note, find_note_by_url_id, load_notes


def _folder_label(folder: str) -> str:
    """Render '03_ai_ml_methods' → 'Ai Ml Methods'; verbatim otherwise."""
    return (
        folder.split("_", 1)[1].replace("_", " ").title()
        if "_" in folder and folder.split("_", 1)[0].isdigit()
        else folder
    )


def _query_param(name: str) -> str | None:
    """Read a single query-param value, robust to Streamlit's API shape."""
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        return unquote(raw[0]) if raw else None
    return unquote(str(raw))


def _render_filter_banner(tag: str, cluster_id: str) -> None:
    """Banner shown above the card grid when a ``?tag=`` filter is active."""
    st.markdown(
        f"""
        <div style="background:#E8EEF6;border-left:4px solid #0033A0;
                    padding:12px 16px;border-radius:4px;margin-bottom:16px;">
            <span style="font-size:14px;color:#333;">
                Filtering by tag: <b><code>{tag}</code></b>
            </span>
            <a href="/{cluster_id.title()}" target="_self"
               style="margin-left:12px;color:#0033A0;font-size:13px;">
               ✕ clear filter
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_card_grid(
    notes: list[Note],
    repo_root: Path,
    *,
    group_by_folder: bool,
) -> None:
    if not notes:
        st.info("No notes match this view.")
        return

    if group_by_folder:
        for folder, folder_notes_iter in groupby(notes, key=lambda n: n.folder):
            folder_notes = list(folder_notes_iter)
            st.markdown(f"### {_folder_label(folder)}")
            for note in folder_notes:
                render_note_card(note, repo_root)
    else:
        for note in notes:
            render_note_card(note, repo_root)


def _render_note_detail(note: Note, cluster_id: str, level: str = "L2") -> None:
    """Render the note-detail view for a single note at the chosen level."""
    from lib import detail_level as _dl

    cluster_meta = CLUSTER_META[cluster_id]
    body_for_level = _dl.render(level, note.body)
    render_note_view(
        title=note.title,
        body=body_for_level,
        cluster_name=cluster_meta["name"],
        cluster_url=f"/{cluster_id.title()}",
        tags=note.tags,
        modality=note.modality,
        beamline=note.beamline,
        related_publications=note.related_publications,
        related_tools=note.related_tools,
    )


def _render_level_selector(current: str, note_url_id: str) -> None:
    """Sidebar radio that switches the ``?level=`` query parameter.

    Streamlit can't easily read & write the same query param in the
    same render, so we render a row of HTML anchor links instead. Each
    link preserves ``?note=…`` and just swaps ``?level=…``.
    """
    from urllib.parse import quote as _q

    from lib import detail_level as _dl

    pills = []
    for lvl in _dl.LEVELS:
        is_active = lvl == current
        bg = "#0033A0" if is_active else "#E8EEF6"
        fg = "#FFFFFF" if is_active else "#0033A0"
        href = f"?note={_q(note_url_id, safe='/')}&level={lvl}"
        pills.append(
            f'<a href="{href}" target="_self" '
            f'style="text-decoration:none;background:{bg};color:{fg};'
            f"padding:6px 14px;border-radius:14px;font-size:13px;"
            f"font-weight:600;margin-right:6px;display:inline-block;"
            f'margin-bottom:6px;">{_dl.LEVEL_LABELS[lvl]}</a>'
        )
    st.markdown(
        '<div style="margin:8px 0 16px 0;">'
        '<div style="font-size:11px;color:#888;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:6px;">Detail level</div>'
        + "".join(pills)
        + f'<div style="font-size:12px;color:#666;margin-top:6px;">'
        f"{_dl.LEVEL_HELP[current]}</div></div>",
        unsafe_allow_html=True,
    )


def render_cluster_page(
    cluster_id: str,
    *,
    group_by_folder: bool = True,
) -> None:
    """Top-level entry called by every cluster page file.

    Args:
        cluster_id: One of ``"discover"``, ``"explore"``, ``"build"``.
        group_by_folder: When showing the card grid, group by folder
            with a folder-name heading.
    """
    from lib import detail_level as _dl

    explorer_dir = Path(__file__).resolve().parent.parent
    repo_root = explorer_dir.parent

    css_path = explorer_dir / "assets" / "styles.css"
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

    meta = CLUSTER_META[cluster_id]

    all_notes = load_notes(repo_root)
    cluster_folders = set(get_folders_for_cluster(cluster_id))
    cluster_notes = [n for n in all_notes if n.folder in cluster_folders]

    note_url_id = _query_param("note")
    tag_filter = _query_param("tag")
    level = _dl.normalise_level(_query_param("level"))

    # Mode 1 — note-detail deep link.
    if note_url_id:
        target = find_note_by_url_id(all_notes, repo_root, note_url_id)
        if target is not None:
            render_header(active_cluster=target.cluster)
            _render_level_selector(level, note_url_id)
            _render_note_detail(target, target.cluster, level=level)
            render_footer()
            return
        # Fall through with a warning — show the cluster grid instead.
        st.warning(f"Note not found: `{note_url_id}` — showing the cluster overview instead.")

    # Mode 2 / 3 — cluster grid (with optional tag filter).
    render_header(active_cluster=cluster_id)
    render_breadcrumb([("Home", "/"), (meta["name"], None)])
    st.markdown(
        f'<h1 style="color:{meta["color"]};">{meta["name"]}</h1>'
        f'<p style="color:#555;font-size:16px;margin-bottom:24px;">{meta["description"]}</p>',
        unsafe_allow_html=True,
    )

    visible_notes = cluster_notes
    if tag_filter:
        visible_notes = [n for n in cluster_notes if tag_filter in (n.tags or [])]
        _render_filter_banner(tag_filter, cluster_id)

    _render_card_grid(visible_notes, repo_root, group_by_folder=group_by_folder)
    render_footer()
