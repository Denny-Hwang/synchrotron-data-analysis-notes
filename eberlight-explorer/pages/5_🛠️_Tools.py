"""Tools & Code Ecosystem (F5)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.tool_card import render_tool_card
from components.markdown_viewer import render_markdown

st.set_page_config(page_title="Tools & Code", page_icon="🛠️", layout="wide")


# Hard redirect to the current app — see ADR-009. The legacy page body
# below is no longer maintained and is preserved only for ADR archival.
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from _deprecated import render_deprecation_and_stop  # noqa: E402

render_deprecation_and_stop()

inject_styles()

level = render_level_selector(key="tool_level")
tools = load_yaml("tool_catalog.yaml")["tools"]
# Filter out catalog entry for card display
display_tools = [t for t in tools if t["id"] != "aps_github_repos"]

st.title("🛠️ Tools & Code Ecosystem")
st.markdown("Open-source tools and software for synchrotron data processing and analysis.")

if level == "L0":
    # Tool landscape
    cols = st.columns(3)
    stages = {"acquisition": [], "processing": [], "analysis": []}
    for t in display_tools:
        stage = t.get("pipeline_stage")
        if stage and stage in stages:
            stages[stage].append(t)

    stage_info = [
        ("📡 Acquisition", stages["acquisition"]),
        ("⚙️ Processing", stages["processing"]),
        ("🧠 Analysis", stages["analysis"]),
    ]
    for col, (title, stage_tools) in zip(cols, stage_info):
        with col:
            with st.container(border=True):
                st.markdown(f"### {title}")
                for t in stage_tools:
                    st.markdown(f"- {t['icon']} **{t['name']}** ({t['language']})")

elif level == "L1":
    # Comparison table
    import pandas as pd
    data = []
    for t in display_tools:
        data.append({
            "": t["icon"],
            "Tool": t["name"],
            "Category": t["category"],
            "Language": t["language"],
            "GPU": "✅" if t.get("gpu") else "❌",
            "Maturity": t.get("maturity", "N/A"),
            "Stage": (t.get("pipeline_stage") or "N/A").title(),
            "Modalities": len(t.get("modalities", [])),
        })
    df = pd.DataFrame(data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.markdown("---")
    for t in display_tools:
        render_tool_card(t, show_detail=True)

elif level == "L2":
    # Individual tool detail
    tool_names = [f"{t['icon']} {t['name']}" for t in display_tools]
    selected_idx = st.selectbox(
        "Select Tool",
        options=range(len(display_tools)),
        format_func=lambda i: tool_names[i],
    )
    tool = display_tools[selected_idx]

    st.subheader(f"{tool['icon']} {tool['name']}")
    render_tool_card(tool, show_detail=True)

    # Show available files as tabs
    files = tool.get("files", {})
    tab_names = []
    tab_files = []
    for key, path in files.items():
        if key == "readme":
            continue
        tab_names.append(key.replace("_", " ").title())
        tab_files.append(path)

    # README first
    st.markdown("---")
    render_markdown(files.get("readme", ""), show_title=False)

    if tab_names:
        tabs = st.tabs(tab_names)
        for tab, fpath in zip(tabs, tab_files):
            with tab:
                render_markdown(fpath, show_title=False)

    # Notebooks
    notebooks = tool.get("notebooks", [])
    if notebooks:
        st.markdown("---")
        st.subheader("📓 Notebooks")
        for nb_path in notebooks:
            st.markdown(f"- `{nb_path}`")

elif level == "L3":
    tool_names = [f"{t['icon']} {t['name']}" for t in display_tools]
    selected_idx = st.selectbox(
        "Select Tool",
        options=range(len(display_tools)),
        format_func=lambda i: tool_names[i],
    )
    tool = display_tools[selected_idx]

    all_files = list(tool.get("files", {}).values())
    selected_file = st.selectbox("Select file", options=all_files)
    content = read_local_file(selected_file)
    if content:
        st.code(content, language="markdown")
