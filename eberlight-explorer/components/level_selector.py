"""Level selector component for progressive disclosure (L0-L3)."""

import streamlit as st

LEVELS = {
    "🌍 Overview": ("L0", "High-level stats and program summary"),
    "📋 Sections": ("L1", "Section summaries, comparison tables"),
    "🔎 Details": ("L2", "Individual method, tool, or paper deep-dive"),
    "📄 Source": ("L3", "Raw markdown source and code"),
}


def render_level_selector(key: str = "level") -> str:
    """Render a level selector in the sidebar and return selected level."""
    with st.sidebar:
        st.markdown("#### 🔎 Detail Level")
        level_name = st.radio(
            "Detail level",
            options=list(LEVELS.keys()),
            index=1,
            key=key,
            label_visibility="collapsed",
        )
        code, desc = LEVELS[level_name]
        st.caption(f"_{desc}_")
        st.markdown("---")
    return code
