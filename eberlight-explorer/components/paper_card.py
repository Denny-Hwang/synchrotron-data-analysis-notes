"""Publication review card component."""

import streamlit as st
from utils.content_parser import read_local_file, extract_tldr, extract_section


def render_paper_card(paper: dict, show_detail: bool = False):
    """Render a paper review card.

    Args:
        paper: dict from publication_catalog.yaml
        show_detail: if True, show full review content
    """
    priority_colors = {
        "High": "🔴",
        "Medium-High": "🟠",
        "Medium": "🟡",
        "Low": "🟢",
    }
    priority_icon = priority_colors.get(paper.get("priority", ""), "⚪")

    with st.container(border=True):
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{paper['title']}**")
            st.caption(f"{paper.get('authors', 'N/A')} | {paper['journal']} ({paper['year']})")
        with col2:
            st.markdown(f"{priority_icon} **{paper.get('priority', 'N/A')}**")

        # Tags
        tag_str = " ".join(f"`{t}`" for t in paper.get("tags", [])[:5])
        if tag_str:
            st.markdown(tag_str)

        if show_detail:
            content = read_local_file(paper["file"])
            if content:
                tldr = extract_tldr(content)
                if tldr:
                    st.info(tldr)

                with st.expander("Full Review", expanded=False):
                    st.markdown(content, unsafe_allow_html=False)
