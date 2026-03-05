"""Interactive pipeline diagram component."""

import streamlit as st
from utils.content_parser import load_yaml


def render_pipeline_diagram():
    """Render data pipeline as a visual flow."""
    cross_refs = load_yaml("cross_references.yaml")
    stages = cross_refs.get("pipeline_stages", [])

    cols = st.columns(len(stages))
    for i, stage in enumerate(stages):
        with cols[i]:
            st.markdown(
                f"<div style='text-align:center; padding:10px; "
                f"background:#E8EEF6; border-radius:10px; min-height:120px;'>"
                f"<h3>{stage['icon']}</h3>"
                f"<b>{stage['name']}</b><br>"
                f"<small>{'<br>'.join(stage.get('tools', []) or ['—'])}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )
        if i < len(stages) - 1:
            pass  # arrow between stages is handled by column layout

    # Show arrows as text
    arrow_row = " → ".join(f"{s['icon']} {s['name']}" for s in stages)
    st.markdown(f"**Flow:** {arrow_row}")
