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

# Custom CSS — larger fonts, better spacing, bullet styling
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    h1 { color: #0A1628; font-size: 2.4rem !important; }
    h2 { font-size: 1.8rem !important; }
    h3 { font-size: 1.4rem !important; }

    /* Body text larger */
    .stMarkdown p, .stMarkdown li { font-size: 1.08rem; line-height: 1.7; }

    /* Metric cards */
    .stMetric > div { background: #E8EEF6; padding: 12px; border-radius: 8px; }

    /* Expanders */
    div[data-testid="stExpander"] { border: 1px solid #E8EEF6; border-radius: 8px; }
    div[data-testid="stExpander"] summary { font-size: 1.15rem; font-weight: 600; }

    /* Bullet list styling */
    .stMarkdown ul { padding-left: 1.4em; }
    .stMarkdown ul li { margin-bottom: 0.35em; }
    .stMarkdown ul li::marker { color: #00D4AA; font-size: 1.1em; }

    /* Tables */
    .stDataFrame { font-size: 1.02rem; }

    /* Sidebar */
    section[data-testid="stSidebar"] > div { padding-top: 1rem; }
    section[data-testid="stSidebar"] .stMarkdown p { font-size: 1.0rem; }
    section[data-testid="stSidebar"] h4 { font-size: 1.1rem !important; margin-bottom: 0.3rem; }
    section[data-testid="stSidebar"] .stRadio label { font-size: 1.05rem; }

    /* Captions a bit bigger */
    .stCaption, .stMarkdown small { font-size: 0.92rem !important; }

    /* Info boxes */
    .stAlert p { font-size: 1.05rem; }

    /* Tabs */
    button[data-baseweb="tab"] { font-size: 1.05rem; }

    /* Container borders */
    div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from utils.content_parser import load_yaml
from components.level_selector import render_level_selector
from components.pipeline_diagram import render_pipeline_diagram

# ── Sidebar ───────────────────────────────────────────
st.sidebar.markdown("## 🔬 eBERlight Explorer")
st.sidebar.caption("Synchrotron Data Analysis Research")
st.sidebar.markdown("---")

st.sidebar.markdown("#### 📂 Navigate")
st.sidebar.markdown(
    "Use the **pages** in the left menu to explore:\n"
    "- 🧠 **Knowledge Graph** — Visual research map\n"
    "- 🔬 **Modalities** — X-ray techniques\n"
    "- 🤖 **AI/ML** — Methods & algorithms\n"
    "- 📚 **Papers** — Reviewed publications\n"
    "- 🛠️ **Tools** — Software ecosystem\n"
    "- 🔄 **Pipeline** — Data flow\n"
    "- 📊 **Data** — Schemas & EDA"
)
st.sidebar.markdown("---")
level = render_level_selector()

# ── Load data ─────────────────────────────────────────
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
    # Quick Navigation Guide
    st.markdown("---")
    st.subheader("🧭 Quick Navigation Guide")
    guide_cols = st.columns(3)
    guides = [
        ("🧪 New to synchrotrons?",
         "- Start with **Program Overview**\n- Then explore **X-ray Modalities**\n- Check **Glossary** for terms"),
        ("🤖 Want to apply AI/ML?",
         "- Jump to **AI/ML Methods**\n- Browse **Publications** for reviews\n- See **Tools** for implementations"),
        ("📊 Need to understand data?",
         "- Check **Data Structures** for schemas\n- Explore **Data Pipeline** flow\n- Run **EDA notebooks** interactively"),
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
                st.markdown(section["description"])
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
    "eBERlight Research Explorer v1.0.0"
)
