"""Note detail view component for eBERlight Explorer.

Renders a single note: breadcrumb, title, markdown body, and a
right-side metadata panel with tags, modality, and beamline info.
``‌```mermaid`` fenced code blocks in the body are rendered
as live diagrams via the Streamlit components iframe (loading
``mermaid.min.js`` from a public CDN), keeping the rest of the
markdown in normal Streamlit flow so internal links / cross-page
deep links still work.

Ref: DS-001 (design_system.md) — MetadataPanel, CodeBlock specs.
Ref: FR-004 — Note detail view renders markdown.
Ref: FR-006 — Metadata panel with beamline/modality tags.
Ref: FR-013 — Code blocks with syntax highlighting.
Ref: ADR-002 — Notes are the single source of truth (Mermaid blocks
              live inside the note markdown, not in page-side dicts).
"""

from __future__ import annotations

import re

import markdown
import streamlit as st
import streamlit.components.v1 as components
from pygments.formatters import HtmlFormatter

from .breadcrumb import render_breadcrumb

# A ``‌```mermaid`` fenced code block. We accept either a leading
# space or none after the language tag and any line endings before
# the closing fence — markdown is picky about this in the wild.
_MERMAID_BLOCK = re.compile(
    r"```mermaid[ \t]*\n(?P<code>.*?)\n```",
    re.DOTALL,
)


def _md_to_html(body: str) -> str:
    """Render markdown to HTML using the same extensions as the static site."""
    return markdown.markdown(
        body,
        extensions=["fenced_code", "tables", "toc", "codehilite"],
        extension_configs={"codehilite": {"css_class": "highlight", "linenums": False}},
    )


def _render_mermaid_iframe(code: str, height: int = 480) -> None:
    """Render one Mermaid diagram via :mod:`streamlit.components.v1` html.

    Each block lives in its own iframe; that's the only way Streamlit
    will execute the ``mermaid.initialize(...)`` script. The iframe
    background is set to match the surrounding page so the diagram
    visually merges with the rest of the note.
    """
    # Defensive — break any unintended ``</script>`` inside the user's
    # mermaid source. Mermaid syntax does not legitimately need it.
    safe = code.replace("</", r"<\/")
    html = f"""<!doctype html>
<html><head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #FAFBFC; }}
  .mermaid-wrap {{ padding: 12px; }}
  .mermaid {{ display: flex; justify-content: center; }}
</style>
</head><body>
<div class="mermaid-wrap"><div class="mermaid">{safe}</div></div>
<script>
  mermaid.initialize({{
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: {{ htmlLabels: true, curve: 'basis' }}
  }});
</script>
</body></html>"""
    components.html(html, height=height, scrolling=True)


def _render_body_with_mermaid(body: str, highlight_css: str) -> None:
    """Render markdown, splitting out ``‌```mermaid`` blocks into iframes.

    Walks through the body emitting alternating segments: a normal
    markdown segment via ``st.markdown(...)``, then a Mermaid
    component, then the next markdown segment, etc. The fast path
    (no Mermaid blocks) is a single ``st.markdown`` call.
    """
    matches = list(_MERMAID_BLOCK.finditer(body))
    if not matches:
        html = _md_to_html(body)
        st.markdown(f"<style>{highlight_css}</style>{html}", unsafe_allow_html=True)
        return

    last_end = 0
    for match in matches:
        if match.start() > last_end:
            seg = body[last_end : match.start()]
            html = _md_to_html(seg)
            st.markdown(f"<style>{highlight_css}</style>{html}", unsafe_allow_html=True)
        _render_mermaid_iframe(match.group("code").strip())
        last_end = match.end()

    if last_end < len(body):
        seg = body[last_end:]
        html = _md_to_html(seg)
        st.markdown(f"<style>{highlight_css}</style>{html}", unsafe_allow_html=True)


def render_note_view(
    title: str,
    body: str,
    cluster_name: str,
    tags: list[str],
    modality: str | None,
    beamline: list[str],
    related_publications: list[str],
    related_tools: list[str],
    cluster_url: str = "#",
    *,
    publication_links: list[tuple[str, str | None]] | None = None,
    tool_links: list[tuple[str, str | None]] | None = None,
    notebooks: list[tuple[str, str]] | None = None,
    prev_note: tuple[str, str] | None = None,
    next_note: tuple[str, str] | None = None,
    permalink: str | None = None,
    toc_items: list[tuple[int, str, str]] | None = None,
    metrics: list[tuple[str, str]] | None = None,
    section_tabs: list[tuple[str, str]] | None = None,
) -> None:
    """Render a full note detail view with metadata panel.

    Args:
        title: Note title.
        body: Markdown body content. ``‌```mermaid`` fenced code
            blocks are rendered as live diagrams.
        cluster_name: Human-readable cluster name for breadcrumb.
        tags: List of tag labels.
        modality: Modality value or None.
        beamline: List of beamline identifiers.
        related_publications: List of related publication filenames.
        related_tools: List of related tool names.
        cluster_url: URL the breadcrumb's cluster crumb links back to.
        publication_links: Optional ``(label, href)`` tuples — when
            present, overrides ``related_publications`` to render
            clickable links instead of plain text.
        tool_links: Same idea for ``related_tools``.
        notebooks: Optional ``(filename, github_url)`` tuples for
            sibling ``*.ipynb`` files; rendered as a "Notebooks"
            section under the body.
        prev_note: Optional ``(title, href)`` to render as "← prev".
        next_note: Optional ``(title, href)`` to render as "next →".
        permalink: Optional shareable URL — when set, renders a
            "Copy permalink" button that copies it to the clipboard.
        toc_items: Optional ``(depth, anchor, heading)`` triples —
            when present, renders an in-page table of contents
            sidebar above the metadata panel.
    """
    # Breadcrumb
    render_breadcrumb(
        [
            ("Home", "/"),
            (cluster_name, cluster_url),
            (title, None),
        ]
    )

    # Two-column layout: main content + metadata panel
    col_main, col_meta = st.columns([3, 1])

    with col_main:
        st.markdown(f"# {title}")
        if permalink:
            _render_permalink_button(permalink)

        if metrics:
            _render_metric_row(metrics)

        # Render markdown with code highlighting + inline Mermaid blocks.
        formatter = HtmlFormatter(style="monokai", noclasses=True)
        highlight_css = formatter.get_style_defs(".highlight")
        if section_tabs:
            _render_section_tabs(section_tabs, highlight_css)
        else:
            _render_body_with_mermaid(body, highlight_css)

        if notebooks:
            _render_notebooks_section(notebooks)

        if prev_note or next_note:
            _render_prev_next_nav(prev_note, next_note)

    with col_meta:
        if toc_items:
            _render_toc(toc_items)
        _render_metadata_panel(
            tags,
            modality,
            beamline,
            related_publications,
            related_tools,
            publication_links=publication_links,
            tool_links=tool_links,
        )


def _render_metric_row(metrics: list[tuple[str, str]]) -> None:
    """Render a row of ``st.metric`` cards above the body.

    The legacy Modalities / Tools / Publications pages each had a row
    of 3-4 ``st.metric`` widgets right under the title (resolution,
    GPU, maturity, year, …). We surface the same control here driven
    purely by frontmatter, so the legacy "L2 with metrics" UX returns
    without re-introducing YAML catalogs.
    """
    if not metrics:
        return
    cols = st.columns(min(4, len(metrics)))
    for col, (label, value) in zip(cols, metrics, strict=False):
        col.metric(label=label, value=value)


def _render_section_tabs(sections: list[tuple[str, str]], highlight_css: str) -> None:
    """Render an H2-section-per-tab view of the body.

    Reproduces the legacy Publications page L2 UX where Background,
    Method, Key Results, … each got their own tab. We do the H2 split
    on the markdown body at runtime; sections without a body fall back
    to a brief "(empty section)" hint to avoid a silent gap.
    """
    if not sections:
        return
    labels = [label for label, _ in sections]
    tabs = st.tabs(labels)
    for tab, (_, content) in zip(tabs, sections, strict=False):
        with tab:
            text = content.strip()
            if not text:
                st.caption("(empty section)")
                continue
            html = _md_to_html(text)
            st.markdown(f"<style>{highlight_css}</style>{html}", unsafe_allow_html=True)


def _render_permalink_button(url: str) -> None:
    """Small "Copy permalink" button that uses the navigator clipboard API."""
    safe = url.replace("'", "&#39;").replace('"', "&quot;")
    st.markdown(
        f"""
        <div style="margin:-12px 0 16px 0;">
            <button onclick="navigator.clipboard.writeText('{safe}');
                             this.innerText='✓ Copied'"
                    style="background:#E8EEF6;border:1px solid #0033A0;
                           color:#0033A0;border-radius:14px;
                           padding:4px 12px;font-size:12px;cursor:pointer;
                           font-weight:600;">
                🔗 Copy permalink
            </button>
            <span style="font-size:12px;color:#888;margin-left:8px;">
                Share this exact view
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_toc(items: list[tuple[int, str, str]]) -> None:
    """Render the in-page table of contents above the metadata panel."""
    if not items:
        return
    rows: list[str] = []
    for depth, anchor, heading in items:
        if depth == 1:
            continue  # the page already prints the H1 title above
        indent = "  " * max(depth - 2, 0)
        safe_h = heading.replace("<", "&lt;").replace(">", "&gt;")
        rows.append(
            f'<div style="font-size:13px;margin:2px 0;padding-left:{len(indent) * 8}px;">'
            f'<a href="#{anchor}" style="color:#0033A0;text-decoration:none;">'
            f"{safe_h}</a></div>"
        )
    if not rows:
        return
    st.markdown(
        f"""
        <aside aria-label="On this page" style="background:#FFFFFF;
               border:1px solid #E0E0E0;border-radius:8px;
               padding:16px;margin-bottom:16px;">
            <div style="font-size:11px;font-weight:700;text-transform:uppercase;
                        letter-spacing:0.5px;color:#555;margin-bottom:8px;">
                On this page
            </div>
            {"".join(rows)}
        </aside>
        """,
        unsafe_allow_html=True,
    )


def _render_notebooks_section(notebooks: list[tuple[str, str]]) -> None:
    """Render a 'Notebooks' section listing each .ipynb sibling.

    Args:
        notebooks: List of ``(filename, github_blob_url)`` tuples.
            The GitHub URL is the canonical home; we also synthesise
            an nbviewer link so the user can preview without GitHub.
    """
    items: list[str] = []
    for filename, github_url in notebooks:
        nbviewer = github_url.replace(
            "https://github.com/",
            "https://nbviewer.org/github/",
            1,
        ).replace("/blob/", "/blob/", 1)
        items.append(
            f'<li style="margin-bottom:6px;">'
            f"<code>{filename}</code> &nbsp;·&nbsp; "
            f'<a href="{github_url}" target="_blank" rel="noopener" '
            f'style="color:#0033A0;">GitHub ↗</a> &nbsp;·&nbsp; '
            f'<a href="{nbviewer}" target="_blank" rel="noopener" '
            f'style="color:#0033A0;">nbviewer ↗</a></li>'
        )
    st.markdown(
        f"""
        <section style="margin-top:32px;padding:20px;background:#FAFBFC;
                        border-left:4px solid #D86510;border-radius:8px;">
            <h3 style="margin:0 0 8px 0;font-size:18px;color:#D86510;">
                📓 Notebooks
            </h3>
            <p style="font-size:13px;color:#555;margin:0 0 12px 0;">
                Sibling Jupyter notebooks in this note's folder.
                Open in your browser via nbviewer (read-only) or fork via GitHub.
            </p>
            <ul style="margin:0;padding-left:20px;">{"".join(items)}</ul>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_prev_next_nav(
    prev_note: tuple[str, str] | None,
    next_note: tuple[str, str] | None,
) -> None:
    """Render bottom-of-page prev/next-in-folder navigation."""
    prev_html = (
        f'<a href="{prev_note[1]}" target="_self" '
        f'style="color:#0033A0;text-decoration:none;font-size:14px;">'
        f"← {prev_note[0]}</a>"
        if prev_note
        else "<span></span>"
    )
    next_html = (
        f'<a href="{next_note[1]}" target="_self" '
        f'style="color:#0033A0;text-decoration:none;font-size:14px;">'
        f"{next_note[0]} →</a>"
        if next_note
        else "<span></span>"
    )
    st.markdown(
        f"""
        <nav aria-label="Folder navigation" style="display:flex;
             justify-content:space-between;margin-top:32px;padding:16px 0;
             border-top:1px solid #E0E0E0;">
            <div style="text-align:left;max-width:45%;">
                <div style="font-size:11px;color:#888;text-transform:uppercase;
                            letter-spacing:0.5px;">Previous</div>
                {prev_html}
            </div>
            <div style="text-align:right;max-width:45%;">
                <div style="font-size:11px;color:#888;text-transform:uppercase;
                            letter-spacing:0.5px;">Next</div>
                {next_html}
            </div>
        </nav>
        """,
        unsafe_allow_html=True,
    )


def _format_link_list(
    raw: list[str],
    overrides: list[tuple[str, str | None]] | None,
) -> str:
    """Render a list of metadata-panel entries.

    If ``overrides`` is provided it takes precedence and entries with
    a non-None href become anchor links; entries with ``None`` href
    fall back to plain text (e.g. unresolved references). Without
    overrides the raw strings are rendered as plain text — the legacy
    behaviour preserved for backwards-compatible callers.
    """
    rows = overrides if overrides is not None else [(label, None) for label in raw]
    pieces: list[str] = []
    for label, href in rows:
        if href:
            pieces.append(
                f'<div style="font-size:14px;margin-bottom:4px;">'
                f'<a href="{href}" target="_self" '
                f'style="color:#0033A0;text-decoration:none;">{label}</a></div>'
            )
        else:
            pieces.append(
                f'<div style="font-size:14px;color:#555;margin-bottom:4px;" '
                f'title="No matching note found">{label}</div>'
            )
    return "".join(pieces)


def _render_metadata_panel(
    tags: list[str],
    modality: str | None,
    beamline: list[str],
    related_publications: list[str],
    related_tools: list[str],
    *,
    publication_links: list[tuple[str, str | None]] | None = None,
    tool_links: list[tuple[str, str | None]] | None = None,
) -> None:
    """Render the right-side metadata panel."""
    sections: list[str] = []

    if beamline:
        badges = " ".join(
            f'<span style="background:#0033A0;color:white;padding:4px 12px;'
            f'border-radius:12px;font-size:12px;font-weight:600;">{bl}</span>'
            for bl in beamline
        )
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Beamlines</div>'
            f'<div style="display:flex;gap:6px;flex-wrap:wrap;">{badges}</div></div>'
        )

    if modality:
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Modality</div>'
            f'<span class="eberlight-tag">{modality}</span></div>'
        )

    if tags:
        tags_html = " ".join(f'<span class="eberlight-tag">{t}</span>' for t in tags)
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Tags</div>'
            f"<div>{tags_html}</div></div>"
        )

    if related_publications or publication_links:
        links_html = _format_link_list(related_publications, publication_links)
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Publications</div>'
            f"{links_html}</div>"
        )

    if related_tools or tool_links:
        links_html = _format_link_list(related_tools, tool_links)
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Related Tools</div>'
            f"{links_html}</div>"
        )

    if sections:
        panel_html = (
            '<aside aria-label="Note metadata" style="background:#FFFFFF;'
            'border:1px solid #E0E0E0;border-radius:8px;padding:24px;">'
            + "".join(sections)
            + "</aside>"
        )
        st.markdown(panel_html, unsafe_allow_html=True)
