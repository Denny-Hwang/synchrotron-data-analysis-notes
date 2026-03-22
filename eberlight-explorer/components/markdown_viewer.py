"""Enhanced markdown viewer component."""

import os
import re
import streamlit as st
from utils.content_parser import read_local_file, extract_title
from components.mermaid_diagram import render_mermaid

# Pattern to split markdown on ```mermaid ... ``` blocks
_MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)

# Pattern for relative markdown links (not URLs, not anchors)
_RELATIVE_LINK_RE = re.compile(
    r'\[([^\]]+)\]\((?!https?://|#|mailto:)([^)]+)\)'
)

# Map repo directory prefixes to explorer page paths
_PAGE_ROUTES = {
    "02_xray_modalities": "/Modalities",
    "03_ai_ml_methods": "/AI_ML_Methods",
    "04_publications": "/Publications",
    "05_tools_and_code": "/Tools",
    "06_data_structures": "/Data_Structures",
    "09_noise_catalog": "/Noise_Catalog",
}


def _transform_relative_links(text: str, source_dir: str = "") -> str:
    """Transform relative .md links into explorer deep-links or plain text."""
    def _replace_link(m: re.Match) -> str:
        label = m.group(1)
        target = m.group(2).split("#")[0]  # strip anchor
        if not target:
            return m.group(0)
        # Skip non-markdown links (images, notebooks, etc.)
        if not target.endswith(".md"):
            return f"**{label}**"
        # Resolve relative path
        if source_dir:
            resolved = os.path.normpath(os.path.join(source_dir, target))
        else:
            resolved = target
        # Find matching explorer page
        basename = os.path.splitext(os.path.basename(resolved))[0]
        for prefix, route in _PAGE_ROUTES.items():
            if resolved.startswith(prefix):
                return f"**{label}** (`{route}?doc={basename}`)"
        return f"**{label}**"

    return _RELATIVE_LINK_RE.sub(_replace_link, text)


def render_content(content: str, source_path: str = ""):
    """Render markdown content, extracting mermaid blocks for proper rendering."""
    source_dir = os.path.dirname(source_path) if source_path else ""
    parts = _MERMAID_RE.split(content)
    # parts alternates: [text, mermaid_code, text, mermaid_code, ...]
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular markdown text
            text = part.strip()
            if text:
                text = _transform_relative_links(text, source_dir)
                st.markdown(text, unsafe_allow_html=False)
        else:
            # Mermaid diagram code
            lines = part.strip().split("\n")
            # Use higher multiplier for complex diagrams (subgraphs, sequences)
            has_subgraph = any("subgraph" in l for l in lines)
            has_sequence = any("sequenceDiagram" in l or "participant" in l for l in lines)
            if has_subgraph:
                multiplier = 55
            elif has_sequence:
                multiplier = 45
            else:
                multiplier = 40
            height = max(350, len(lines) * multiplier + 150)
            render_mermaid(part, height=height)


def render_markdown(file_path: str, show_title: bool = True):
    """Render a markdown file from the repo."""
    content = read_local_file(file_path)
    if content is None:
        st.warning(f"File not found: `{file_path}`")
        return

    if show_title:
        title = extract_title(content)
        st.subheader(title)

    render_content(content, source_path=file_path)


def render_markdown_card(file_path: str, title: str | None = None, expanded: bool = False):
    """Render markdown content inside an expandable card."""
    content = read_local_file(file_path)
    if content is None:
        return

    display_title = title or extract_title(content)
    with st.expander(display_title, expanded=expanded):
        render_content(content, source_path=file_path)
