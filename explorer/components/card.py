"""Card component for eBERlight Explorer.

Renders a navigation card for cluster pages, showing title, summary,
and tags. Used on landing and cluster landing pages.

Cards link to the same Streamlit page with ``?note=<url_id>`` so the
cluster page can detect the deep-link and switch to the note-detail
view. Tags are clickable: each emits ``?tag=<tag>`` to filter the
cluster page to matching notes (FR-007).

Ref: DS-001 (design_system.md) — Card component spec.
Ref: FR-003 — Cluster pages list notes as cards.
Ref: FR-004 — Note detail view.
Ref: FR-007 — Tags as clickable filters.
"""

from __future__ import annotations

from urllib.parse import quote

import streamlit as st


def _tag_pill(tag: str) -> str:
    """Render one tag as a clickable pill that sets ``?tag=<tag>``."""
    safe = quote(tag, safe="")
    return (
        f'<a class="eberlight-tag" href="?tag={safe}" '
        'style="text-decoration:none;cursor:pointer;" '
        f'target="_self">{tag}</a>'
    )


def render_card(
    title: str,
    summary: str,
    tags: list[str],
    href: str = "#",
) -> None:
    """Render a content card with title, summary, and tags.

    Args:
        title: Card title (H4).
        summary: Brief description (max ~2 lines).
        tags: List of tag labels to display as chips.
        href: Link URL for the card title — typically
            ``?note=<note.url_id(repo_root)>`` so the cluster page can
            switch to the note-detail view on click.
    """
    tags_html = "".join(_tag_pill(t) for t in tags[:5])

    card_html = f"""
    <div class="eberlight-card" style="margin-bottom: 16px;">
        <h4 style="margin: 0 0 8px 0;">
            <a href="{href}" style="color: #1A1A1A; text-decoration: none;"
               target="_self">{title}</a>
        </h4>
        <p style="font-size: 14px; color: #555555; margin: 0 0 12px 0;
                  display: -webkit-box; -webkit-line-clamp: 2;
                  -webkit-box-orient: vertical; overflow: hidden;">
            {summary}
        </p>
        <div>{tags_html}</div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_note_card(note, repo_root) -> None:  # type: ignore[no-untyped-def]
    """Convenience wrapper: render a card for a parsed Note.

    Auto-fills the title, summary (description or first body chars),
    tags, and ``href`` from ``note.url_id(repo_root)``.
    """
    summary = note.description or note.body[:150].strip().replace("\n", " ")
    href = f"?note={quote(note.url_id(repo_root), safe='/')}"
    render_card(title=note.title, summary=summary, tags=note.tags, href=href)
