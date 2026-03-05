"""
eBERlight Research Explorer
Interactive Streamlit app for exploring synchrotron data analysis research.
"""

import streamlit as st

st.set_page_config(
    page_title="eBERlight Research Explorer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    h1 { color: #0A1628; }
    .stMetric > div { background: #E8EEF6; padding: 10px; border-radius: 8px; }
    div[data-testid="stExpander"] { border: 1px solid #E8EEF6; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.content_parser import load_yaml
from components.level_selector import render_level_selector
from components.pipeline_diagram import render_pipeline_diagram

# Sidebar
st.sidebar.title("🔬 eBERlight Explorer")
st.sidebar.markdown("---")
level = render_level_selector()

# Load data
index = load_yaml("content_index.yaml")
modalities = load_yaml("modality_metadata.yaml")["modalities"]
methods = load_yaml("method_taxonomy.yaml")["categories"]
papers = load_yaml("publication_catalog.yaml")["publications"]
tools_data = load_yaml("tool_catalog.yaml")["tools"]

# ──────────────────────────────────────────────
# HOME PAGE
# ──────────────────────────────────────────────
st.title("🔬 eBERlight Research Explorer")
st.markdown(
    "Interactive exploration of the **Synchrotron Data Analysis Notes** — "
    "covering APS beamlines, X-ray modalities, AI/ML methods, publications, and tools."
)

# Stat Cards
st.markdown("---")
cols = st.columns(5)
stats = [
    ("🔬", "Modalities", len(modalities)),
    ("🤖", "AI/ML Categories", len(methods)),
    ("📚", "Paper Reviews", len(papers)),
    ("🛠️", "Tools", len([t for t in tools_data if t["id"] != "aps_github_repos"])),
    ("📊", "Repo Sections", len(index["sections"])),
]
for col, (icon, label, value) in zip(cols, stats):
    with col:
        st.metric(label=f"{icon} {label}", value=value)

if level in ("L0", "L1"):
    # Research Domain Quick Guide
    st.markdown("---")
    st.subheader("🧭 Quick Navigation Guide")
    guide_cols = st.columns(3)
    guides = [
        ("🧪 New to synchrotrons?", "Start with **Program Overview** → then explore **X-ray Modalities**"),
        ("🤖 Want to apply AI/ML?", "Jump to **AI/ML Methods** → browse **Publications** for reviews"),
        ("📊 Need to understand data?", "Check **Data Structures** → then **Data Pipeline**"),
    ]
    for col, (title, desc) in zip(guide_cols, guides):
        with col:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.markdown(desc)

    # Section Overview Cards
    st.markdown("---")
    st.subheader("📂 Repository Sections")
    section_cols = st.columns(4)
    for i, section in enumerate(index["sections"]):
        with section_cols[i % 4]:
            with st.container(border=True):
                st.markdown(f"### {section['icon']} {section['title']}")
                st.caption(section["description"])
                st.markdown(f"`{len(section['files'])} files`")

if level in ("L1", "L2"):
    # Modality comparison table
    st.markdown("---")
    st.subheader("🔬 X-ray Modalities at a Glance")
    import pandas as pd
    mod_data = []
    for m in modalities:
        mod_data.append({
            "Modality": f"{m['icon']} {m['short_name']}",
            "Interaction": m["interaction"],
            "Measures": m["measures"],
            "Resolution": m["resolution"],
            "Beamlines": ", ".join(m["beamlines"]),
            "AI Methods": len(m["ai_methods"]),
        })
    df = pd.DataFrame(mod_data)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # AI/ML Method overview
    st.markdown("---")
    st.subheader("🤖 AI/ML Method Categories")
    method_cols = st.columns(len(methods))
    for col, cat in zip(method_cols, methods):
        with col:
            with st.container(border=True):
                st.markdown(f"### {cat['icon']}")
                st.markdown(f"**{cat['name']}**")
                st.caption(f"{len(cat['methods'])} methods")
                for m in cat["methods"]:
                    st.markdown(f"- {m['name']}")

    # Pipeline overview
    st.markdown("---")
    st.subheader("🔄 Data Pipeline Overview")
    render_pipeline_diagram()

st.markdown("---")
st.caption(
    "Built from [synchrotron-data-analysis-notes](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes) | "
    "eBERlight Research Explorer v0.1.0"
)
