"""Level selector component for progressive disclosure (L0-L3)."""

import streamlit as st

LEVELS = {
    "L0 Bird's Eye": "Program-wide overview, key stats at a glance",
    "L1 Section View": "Section-level summaries and comparison tables",
    "L2 Detail View": "Individual method, tool, or paper details",
    "L3 Source View": "Original markdown source and raw content",
}


def render_level_selector(key: str = "level") -> str:
    """Render a level selector in the sidebar and return selected level."""
    with st.sidebar:
        st.markdown("### Depth Level")
        level = st.radio(
            "Choose detail level",
            options=list(LEVELS.keys()),
            index=1,
            key=key,
            help="Control how much detail is shown",
            label_visibility="collapsed",
        )
        st.caption(LEVELS[level])
    return level.split(" ")[0]  # returns "L0", "L1", etc.
