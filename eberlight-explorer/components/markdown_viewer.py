"""Enhanced markdown viewer component."""

import streamlit as st
from utils.content_parser import read_local_file, extract_title


def render_markdown(file_path: str, show_title: bool = True):
    """Render a markdown file from the repo."""
    content = read_local_file(file_path)
    if content is None:
        st.warning(f"File not found: `{file_path}`")
        return

    if show_title:
        title = extract_title(content)
        st.subheader(title)

    st.markdown(content, unsafe_allow_html=False)


def render_markdown_card(file_path: str, title: str | None = None, expanded: bool = False):
    """Render markdown content inside an expandable card."""
    content = read_local_file(file_path)
    if content is None:
        return

    display_title = title or extract_title(content)
    with st.expander(display_title, expanded=expanded):
        st.markdown(content, unsafe_allow_html=False)
