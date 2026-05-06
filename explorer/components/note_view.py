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
            Defaults to ``"#"`` for backward compatibility; the
            cluster-page router passes a real path like ``"/Build"``
            so the user can navigate up one level.
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

        # Render markdown with code highlighting + inline Mermaid blocks.
        formatter = HtmlFormatter(style="monokai", noclasses=True)
        highlight_css = formatter.get_style_defs(".highlight")
        _render_body_with_mermaid(body, highlight_css)

    with col_meta:
        _render_metadata_panel(tags, modality, beamline, related_publications, related_tools)


def _render_metadata_panel(
    tags: list[str],
    modality: str | None,
    beamline: list[str],
    related_publications: list[str],
    related_tools: list[str],
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

    if related_publications:
        links = "".join(
            f'<div style="font-size:14px;color:#0033A0;margin-bottom:4px;">{p}</div>'
            for p in related_publications
        )
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Publications</div>'
            f"{links}</div>"
        )

    if related_tools:
        links = "".join(
            f'<div style="font-size:14px;color:#0033A0;margin-bottom:4px;">{t}</div>'
            for t in related_tools
        )
        sections.append(
            f'<div style="margin-bottom:20px;">'
            f'<div style="font-size:12px;font-weight:600;text-transform:uppercase;'
            f'letter-spacing:0.5px;color:#555;margin-bottom:8px;">Related Tools</div>'
            f"{links}</div>"
        )

    if sections:
        panel_html = (
            '<aside aria-label="Note metadata" style="background:#FFFFFF;'
            'border:1px solid #E0E0E0;border-radius:8px;padding:24px;">'
            + "".join(sections)
            + "</aside>"
        )
        st.markdown(panel_html, unsafe_allow_html=True)
