"""Enhanced markdown viewer component."""

import re
import streamlit as st
from utils.content_parser import read_local_file, extract_title
from components.mermaid_diagram import render_mermaid

# Pattern to split markdown on ```mermaid ... ``` blocks
_MERMAID_RE = re.compile(r"```mermaid\s*\n(.*?)```", re.DOTALL)


def render_content(content: str):
    """Render markdown content, extracting mermaid blocks for proper rendering."""
    parts = _MERMAID_RE.split(content)
    # parts alternates: [text, mermaid_code, text, mermaid_code, ...]
    for i, part in enumerate(parts):
        if i % 2 == 0:
            # Regular markdown text
            text = part.strip()
            if text:
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

    render_content(content)


def render_markdown_card(file_path: str, title: str | None = None, expanded: bool = False):
    """Render markdown content inside an expandable card."""
    content = read_local_file(file_path)
    if content is None:
        return

    display_title = title or extract_title(content)
    with st.expander(display_title, expanded=expanded):
        render_content(content)
