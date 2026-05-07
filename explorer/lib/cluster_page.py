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

from collections import defaultdict
from pathlib import Path
from urllib.parse import quote, unquote

import streamlit as st
from components.breadcrumb import render_breadcrumb
from components.footer import render_footer
from components.header import render_header
from components.note_view import render_note_view

from lib.ia import CLUSTER_META, get_folders_for_cluster
from lib.notes import (
    Note,
    discover_sibling_notebooks,
    find_note_by_basename,
    find_note_by_url_id,
    load_notes,
    neighbor_notes,
    resolve_publication_ref,
    resolve_tool_ref,
)

# Folders the legacy ``?doc=`` parameter targeted; we use them as the
# folder-hint for basename resolution when the cluster matches.
_CLUSTER_DOC_FOLDER_HINTS: dict[str, list[str]] = {
    "discover": ["01_program_overview", "08_references"],
    "explore": [
        "02_xray_modalities",
        "03_ai_ml_methods",
        "04_publications",
        "09_noise_catalog",
    ],
    "build": [
        "05_tools_and_code",
        "06_data_structures",
        "07_data_pipeline",
        "10_interactive_lab",
    ],
}

# GitHub blob URL prefix for sibling-notebook links rendered on note-detail.
_GITHUB_BLOB_PREFIX = "https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/blob/main/"


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


_RECENT_KEY = "_eberlight_recently_viewed"
_RECENT_LIMIT = 8


def _track_recently_viewed(note: Note, repo_root: Path) -> None:
    """Push the just-viewed note onto the session-state recents list."""
    url_id = note.url_id(repo_root)
    history: list[dict[str, str]] = st.session_state.get(_RECENT_KEY, [])
    history = [item for item in history if item.get("url_id") != url_id]
    history.insert(
        0,
        {
            "url_id": url_id,
            "title": note.title,
            "cluster": note.cluster,
        },
    )
    st.session_state[_RECENT_KEY] = history[:_RECENT_LIMIT]


def _render_recently_viewed_sidebar(
    all_notes: list[Note], repo_root: Path, *, current: Note | None
) -> None:
    """Render a sidebar block listing recently viewed notes.

    The sidebar lives outside the main two-column layout so it doesn't
    compete with the note body. The current note is highlighted but
    not deduped (so a refresh keeps the page in the list).
    """
    history: list[dict[str, str]] = st.session_state.get(_RECENT_KEY, [])
    if not history:
        return
    current_url_id = current.url_id(repo_root) if current is not None else None
    valid_url_ids = {n.url_id(repo_root) for n in all_notes}

    with st.sidebar:
        st.markdown("#### 🕘 Recently viewed")
        for item in history:
            uid = item.get("url_id", "")
            if uid not in valid_url_ids:
                continue
            cluster = item.get("cluster") or "discover"
            title = item.get("title") or uid
            href = f"/{cluster.title()}?note={quote(uid, safe='/')}"
            mark = " · *current*" if uid == current_url_id else ""
            st.markdown(
                f'<div style="font-size:13px;margin:2px 0;">'
                f'<a href="{href}" target="_self" '
                f'style="color:#0033A0;text-decoration:none;">{title}</a>'
                f'<span style="color:#888;">{mark}</span></div>',
                unsafe_allow_html=True,
            )


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


def _render_folder_filter_chips(
    notes: list[Note],
    cluster_id: str,
    *,
    active_folder: str | None,
    tag: str | None,
) -> None:
    """Render folder-filter chips above the cluster table (R11 I3).

    Replaces the old ``?view=cards|table`` toggle. The cluster page now
    has a single dense compare-table view; instead of switching layouts,
    the user narrows the view by folder via a chip row. "All" resets
    to the full cluster.
    """
    folder_counts: dict[str, int] = defaultdict(int)
    for n in notes:
        folder_counts[n.folder] += 1
    folders = sorted(folder_counts.keys())

    base = f"/{cluster_id.title()}"
    tag_extra = f"&tag={quote(tag, safe='')}" if tag else ""

    chips: list[str] = []
    is_all_active = active_folder is None
    chips.append(
        f'<a href="{base}?{("tag=" + quote(tag, safe="")) if tag else ""}" '
        f'target="_self" '
        f'class="eberlight-chip {"active" if is_all_active else ""}">'
        f'All <span class="eberlight-chip-count">({len(notes)})</span></a>'
    )
    for folder in folders:
        count = folder_counts[folder]
        is_active = folder == active_folder
        href = f"{base}?folder={quote(folder, safe='')}{tag_extra}"
        chips.append(
            f'<a href="{href}" target="_self" '
            f'class="eberlight-chip {"active" if is_active else ""}">'
            f"{_folder_label(folder)} "
            f'<span class="eberlight-chip-count">({count})</span></a>'
        )
    st.markdown(
        '<div class="eberlight-chip-row" role="tablist" aria-label="Filter by folder">'
        + "".join(chips)
        + "</div>",
        unsafe_allow_html=True,
    )


def _render_compare_table(notes: list[Note], repo_root: Path, cluster_id: str) -> None:
    """Render the cluster's notes as a sortable dataframe.

    Columns surface the same frontmatter the metadata panel uses on
    note detail — title, folder, modality, beamlines, tags, related
    publications/tools count, last_reviewed if available — so the
    comparison view is the legacy L0/L1 power-user surface.
    """
    if not notes:
        st.info("No notes match this view.")
        return

    import pandas as pd

    # R10 P1-2: previously emitted ``<a href="...">title</a>`` into the cell
    # which ``st.dataframe`` showed as raw HTML. Switch to a parallel
    # ``Open`` column rendered via ``column_config.LinkColumn`` so the
    # title cell stays plain text (sortable + searchable) and the link
    # is a proper clickable column.
    rows: list[dict[str, object]] = []
    for n in notes:
        href = f"/{cluster_id.title()}?note={quote(n.url_id(repo_root), safe='/')}"
        rows.append(
            {
                "Title": n.title,
                "Folder": _folder_label(n.folder),
                "Modality": n.modality or "—",
                "Beamlines": ", ".join(n.beamline) if n.beamline else "—",
                "Tags": ", ".join(n.tags[:6]) if n.tags else "—",
                "Pubs": len(n.related_publications) or None,
                "Tools": len(n.related_tools) or None,
                "Description": n.description or "",
                "Open": href,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width="stretch",
        height=min(560, 60 + 35 * len(df)),
        hide_index=True,
        column_config={
            "Title": st.column_config.Column("Title", width="medium"),
            "Open": st.column_config.LinkColumn(
                "Open",
                help="Click to open the note detail.",
                display_text="Open →",
                width="small",
            ),
            "Pubs": st.column_config.NumberColumn(
                "Pubs",
                help="Number of related publication references.",
                format="%d",
            ),
            "Tools": st.column_config.NumberColumn(
                "Tools",
                help="Number of related tool references.",
                format="%d",
            ),
        },
    )
    st.caption(
        f"Compare table — {len(df)} notes · sort any column · use the **Open** "
        "column to jump to a note."
    )


def _build_metric_pairs(note: Note) -> list[tuple[str, str]]:
    """Translate optional frontmatter into ``(label, value)`` metric pairs.

    Only fields the author actually declared are surfaced; a note with
    no rich metadata returns an empty list and the metric row hides.
    """
    pairs: list[tuple[str, str]] = []
    if note.modality:
        pairs.append(("Modality", note.modality))
    if note.resolution:
        pairs.append(("Resolution", note.resolution))
    if note.beamline:
        pairs.append(("Beamlines", str(len(note.beamline))))
    if note.year is not None:
        pairs.append(("Year", str(note.year)))
    if note.maturity:
        pairs.append(("Maturity", note.maturity))
    if note.language:
        pairs.append(("Language", note.language))
    if note.gpu is not None:
        pairs.append(("GPU", "Yes" if note.gpu else "No"))
    if note.priority:
        pairs.append(("Priority", note.priority))
    if note.pipeline_stage:
        pairs.append(("Pipeline stage", note.pipeline_stage.title()))
    return pairs[:6]  # Cap at 6 — beyond that the row wraps awkwardly.


def _render_note_detail(
    note: Note,
    cluster_id: str,
    *,
    level: str = "L2",
    view_mode: str = "default",
    all_notes: list[Note] | None = None,
    repo_root: Path | None = None,
) -> None:
    """Render the note-detail view for a single note at the chosen level.

    Resolves ``related_publications`` / ``related_tools`` strings into
    real ``?note=…`` deep links, computes the prev/next-in-folder
    neighbours, surfaces sibling Jupyter notebooks, and provides a
    permalink + table-of-contents — all derived at runtime from the
    repo so ADR-002 stays intact.
    """
    from lib import detail_level as _dl

    cluster_meta = CLUSTER_META[cluster_id]
    body_for_level = _dl.render(level, note.body)

    publication_links: list[tuple[str, str | None]] | None = None
    tool_links: list[tuple[str, str | None]] | None = None
    notebooks: list[tuple[str, str]] | None = None
    prev_link: tuple[str, str] | None = None
    next_link: tuple[str, str] | None = None

    if all_notes is not None and repo_root is not None:
        publication_links = []
        for ref in note.related_publications:
            target = resolve_publication_ref(all_notes, ref, repo_root)
            if target is not None:
                href = f"/{target.cluster.title()}?note={quote(target.url_id(repo_root), safe='/')}"
                publication_links.append((target.title, href))
            else:
                publication_links.append((ref, None))

        tool_links = []
        for ref in note.related_tools:
            target = resolve_tool_ref(all_notes, ref, repo_root)
            if target is not None:
                href = f"/{target.cluster.title()}?note={quote(target.url_id(repo_root), safe='/')}"
                tool_links.append((target.title, href))
            else:
                tool_links.append((ref, None))

        # Sibling notebooks for the Tools / Data-Structures pages.
        ipynbs = discover_sibling_notebooks(note, repo_root)
        if ipynbs:
            notebooks = [
                (
                    p.name,
                    _GITHUB_BLOB_PREFIX + p.relative_to(repo_root).as_posix(),
                )
                for p in ipynbs
            ]

        prev_n, next_n = neighbor_notes(all_notes, note)
        if prev_n is not None:
            prev_link = (
                prev_n.title,
                f"/{prev_n.cluster.title()}?note={quote(prev_n.url_id(repo_root), safe='/')}",
            )
        if next_n is not None:
            next_link = (
                next_n.title,
                f"/{next_n.cluster.title()}?note={quote(next_n.url_id(repo_root), safe='/')}",
            )

    permalink = None
    if repo_root is not None:
        # The same query the user is on, written with the explicit
        # cluster route so it works from any tab.
        url_id = note.url_id(repo_root)
        permalink = f"/{cluster_id.title()}?note={quote(url_id, safe='/')}&level={level}"

    toc = _dl.extract_toc(note.body) if level == "L2" else None
    metrics = _build_metric_pairs(note) if level == "L2" else None

    section_tabs: list[tuple[str, str]] | None = None
    if view_mode == "tabs" and level == "L2":
        # Pass the original body (NOT body_for_level) so section split
        # happens against the canonical markdown — and force the body
        # parameter below to empty so we don't render the body twice.
        sections = _dl.split_into_sections(note.body, level=2)
        if sections:
            section_tabs = sections
            body_for_level = ""

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
        publication_links=publication_links,
        tool_links=tool_links,
        notebooks=notebooks,
        prev_note=prev_link,
        next_note=next_link,
        permalink=permalink,
        toc_items=toc,
        metrics=metrics,
        section_tabs=section_tabs,
        last_reviewed=note.last_reviewed,
    )


def _render_level_selector(current: str, note_url_id: str, *, view_mode: str = "default") -> None:
    """Pill row that switches ``?level=…`` and toggles ``?view=tabs``.

    The L0..L3 pills work as before; an extra **📑 Tabs** pill toggles
    section-tab rendering of the body. Tabs are only meaningful at L2
    (the only level that renders the full body) — at other levels the
    pill greys out by including an info caption rather than disabling
    the link.
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

    # Section-tabs toggle pill — switches `?view=tabs` on/off.
    is_tabs = view_mode == "tabs"
    bg = "#0033A0" if is_tabs else "#E8EEF6"
    fg = "#FFFFFF" if is_tabs else "#0033A0"
    target_view = "default" if is_tabs else "tabs"
    href = f"?note={_q(note_url_id, safe='/')}&level={current}"
    if target_view == "tabs":
        href += "&view=tabs"
    pills.append(
        f'<a href="{href}" target="_self" '
        f'style="text-decoration:none;background:{bg};color:{fg};'
        f"padding:6px 14px;border-radius:14px;font-size:13px;"
        f"font-weight:600;margin-left:12px;display:inline-block;"
        f'margin-bottom:6px;">📑 Tabs</a>'
    )

    helper_msg = _dl.LEVEL_HELP[current]
    if is_tabs:
        helper_msg += " · Body is split into tabs by H2 heading."
    elif current == "L2":
        helper_msg += " · Click 📑 Tabs to split the body into H2-section tabs."

    st.markdown(
        '<div style="margin:8px 0 16px 0;">'
        '<div style="font-size:11px;color:#888;text-transform:uppercase;'
        'letter-spacing:0.5px;margin-bottom:6px;">Detail level</div>'
        + "".join(pills)
        + f'<div style="font-size:12px;color:#666;margin-top:6px;">'
        f"{helper_msg}</div></div>",
        unsafe_allow_html=True,
    )


def render_cluster_page(
    cluster_id: str,
    *,
    group_by_folder: bool = True,  # deprecated, kept for backward compat
) -> None:
    """Top-level entry called by every cluster page file.

    Args:
        cluster_id: One of ``"discover"``, ``"explore"``, ``"build"``.
        group_by_folder: Deprecated since R11 — the page now uses a
            single dense compare-table view with folder-filter chips.
            Argument retained so unmodified callers don't fail.
    """
    del group_by_folder  # silence linters; argument retained for compat.
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
    doc_param = _query_param("doc")
    tag_filter = _query_param("tag")
    folder_filter = _query_param("folder")
    level = _dl.normalise_level(_query_param("level"))
    # R11 I3 — `?view=cards|table` toggle is gone; only `?view=tabs`
    # (the note-detail section-tabs mode) is honoured.
    raw_view = (_query_param("view") or "").lower()
    detail_view_mode = "tabs" if raw_view == "tabs" else "default"

    # ``?doc=<basename>`` is the legacy URL shape. Resolve it to a real
    # ``url_id`` so the rest of the router (Mode 1) just works. Folder
    # hints are scoped to the current cluster so e.g. ``?doc=ring_artifact``
    # on /Explore lands on ``09_noise_catalog/tomography/ring_artifact.md``.
    if doc_param and not note_url_id:
        hint_folders = _CLUSTER_DOC_FOLDER_HINTS.get(cluster_id, [])
        for folder in [*hint_folders, None]:
            target = find_note_by_basename(all_notes, doc_param, folder_hint=folder)
            if target is not None:
                note_url_id = target.url_id(repo_root)
                break

    # Mode 1 — note-detail deep link.
    if note_url_id:
        target = find_note_by_url_id(all_notes, repo_root, note_url_id)
        if target is not None:
            render_header(active_cluster=target.cluster)
            _render_level_selector(level, note_url_id, view_mode=detail_view_mode)
            _track_recently_viewed(target, repo_root)
            _render_note_detail(
                target,
                target.cluster,
                level=level,
                view_mode=detail_view_mode,
                all_notes=all_notes,
                repo_root=repo_root,
            )
            _render_recently_viewed_sidebar(all_notes, repo_root, current=target)
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

    # R11 I3 — folder-filter chips replace the Cards/Table toggle. The
    # cluster page now has one dense table view; the chip row narrows
    # by folder. ``?folder=`` is preserved across tag filtering.
    active_folder = (
        folder_filter
        if folder_filter and any(n.folder == folder_filter for n in cluster_notes)
        else None
    )
    _render_folder_filter_chips(
        cluster_notes, cluster_id, active_folder=active_folder, tag=tag_filter
    )
    if active_folder:
        visible_notes = [n for n in visible_notes if n.folder == active_folder]

    _render_compare_table(visible_notes, repo_root, cluster_id)
    render_footer()
