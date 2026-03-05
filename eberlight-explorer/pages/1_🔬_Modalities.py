"""X-ray Modalities Explorer (F2)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file, extract_section
from components.level_selector import render_level_selector
from components.markdown_viewer import render_markdown, render_markdown_card

st.set_page_config(page_title="X-ray Modalities", page_icon="🔬", layout="wide")

level = render_level_selector(key="mod_level")
modalities = load_yaml("modality_metadata.yaml")["modalities"]

st.title("🔬 X-ray Modalities")
st.markdown("Six X-ray measurement techniques used at APS beamlines for biological and environmental science.")

if level == "L0":
    # High-level comparison
    import pandas as pd
    data = []
    for m in modalities:
        data.append({
            "": m["icon"],
            "Modality": m["short_name"],
            "Interaction": m["interaction"],
            "Measures": m["measures"],
            "Resolution": m["resolution"],
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

elif level == "L1":
    # Comparison table + per-modality summary
    import pandas as pd
    data = []
    for m in modalities:
        data.append({
            "": m["icon"],
            "Modality": m["short_name"],
            "Interaction": m["interaction"],
            "Measures": m["measures"],
            "Resolution": m["resolution"],
            "Beamlines": ", ".join(m["beamlines"]),
            "AI Methods": ", ".join(m["ai_methods"]) if m["ai_methods"] else "—",
        })
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("---")
    for m in modalities:
        with st.expander(f"{m['icon']} {m['name']}", expanded=False):
            cols = st.columns([2, 1])
            with cols[0]:
                content = read_local_file(m["files"]["readme"])
                if content:
                    # Show first few paragraphs
                    paragraphs = content.split("\n\n")
                    preview = "\n\n".join(paragraphs[:5])
                    st.markdown(preview)
            with cols[1]:
                st.markdown("**Key Info**")
                st.markdown(f"- **Resolution:** {m['resolution']}")
                st.markdown(f"- **Beamlines:** {', '.join(m['beamlines'])}")
                st.markdown(f"- **Tags:** {', '.join(m['tags'])}")
                if m["ai_methods"]:
                    st.markdown(f"- **AI Methods:** {', '.join(m['ai_methods'])}")

elif level == "L2":
    # Individual modality detail pages
    selected = st.selectbox(
        "Select Modality",
        options=[m["short_name"] for m in modalities],
        format_func=lambda x: next(f"{m['icon']} {m['name']}" for m in modalities if m["short_name"] == x),
    )
    mod = next(m for m in modalities if m["short_name"] == selected)

    st.subheader(f"{mod['icon']} {mod['name']}")

    info_cols = st.columns(4)
    with info_cols[0]:
        st.metric("Resolution", mod["resolution"])
    with info_cols[1]:
        st.metric("Beamlines", len(mod["beamlines"]))
    with info_cols[2]:
        st.metric("AI Methods", len(mod["ai_methods"]))
    with info_cols[3]:
        st.metric("Interaction", mod["interaction"])

    tabs = st.tabs(["Principles", "Data Format", "AI/ML Methods"] +
                   [e["title"] for e in mod["files"].get("extra", [])])

    with tabs[0]:
        render_markdown(mod["files"]["readme"], show_title=False)

    with tabs[1]:
        if "data_format" in mod["files"]:
            render_markdown(mod["files"]["data_format"], show_title=False)

    with tabs[2]:
        if "ai_ml" in mod["files"]:
            render_markdown(mod["files"]["ai_ml"], show_title=False)

    for i, extra in enumerate(mod["files"].get("extra", [])):
        with tabs[3 + i]:
            render_markdown(extra["path"], show_title=False)

elif level == "L3":
    # Source view
    selected = st.selectbox(
        "Select Modality",
        options=[m["short_name"] for m in modalities],
    )
    mod = next(m for m in modalities if m["short_name"] == selected)

    all_files = [mod["files"]["readme"], mod["files"].get("data_format"), mod["files"].get("ai_ml")]
    all_files += [e["path"] for e in mod["files"].get("extra", [])]
    all_files = [f for f in all_files if f]

    selected_file = st.selectbox("Select file", options=all_files)
    content = read_local_file(selected_file)
    if content:
        st.code(content, language="markdown")
