"""Tool comparison card component."""

import streamlit as st


def render_tool_card(tool: dict, show_detail: bool = False):
    """Render a tool info card."""
    with st.container(border=True):
        st.markdown(f"{tool['icon']} **{tool['name']}**")
        st.caption(f"{tool['category']} | {tool['language']} | {'GPU' if tool.get('gpu') else 'CPU'}")

        cols = st.columns(3)
        with cols[0]:
            st.metric("Maturity", tool.get("maturity", "N/A"))
        with cols[1]:
            stage = tool.get("pipeline_stage", "N/A") or "N/A"
            st.metric("Stage", stage.title())
        with cols[2]:
            mod_count = len(tool.get("modalities", []))
            st.metric("Modalities", mod_count)

        if show_detail:
            tag_str = " ".join(f"`{t}`" for t in tool.get("tags", []))
            if tag_str:
                st.markdown(tag_str)

            modalities = ", ".join(tool.get("modalities", []))
            if modalities:
                st.markdown(f"**Supported modalities:** {modalities}")
