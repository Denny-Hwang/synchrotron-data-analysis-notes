"""AI/ML Methods Taxonomy Map (F3)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import load_yaml, read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.markdown_viewer import render_markdown

st.set_page_config(page_title="AI/ML Methods", page_icon="🤖", layout="wide")

inject_styles()

level = render_level_selector(key="aiml_level")
categories = load_yaml("method_taxonomy.yaml")["categories"]
modalities = load_yaml("modality_metadata.yaml")["modalities"]

st.title("🤖 AI/ML Methods")
st.markdown("Machine learning and AI methods applied to synchrotron data analysis.")

# Taxonomy diagram (shown at L0 and L1)
TAXONOMY_DIAGRAM = """
```mermaid
graph TD
    ROOT["🤖 AI/ML for<br>Synchrotron Data"] --> SEG["🎯 Segmentation"]
    ROOT --> DEN["🧹 Denoising"]
    ROOT --> REC["🏗️ Reconstruction"]
    ROOT --> AUT["🤖 Autonomous<br>Experiment"]
    ROOT --> MUL["🔗 Multimodal<br>Integration"]

    SEG --> S1["U-Net Variants"]
    SEG --> S2["Tomo Segmentation"]
    SEG --> S3["XRF Cell Segmentation"]

    DEN --> D1["TomoGAN"]
    DEN --> D2["Deep Residual XRF"]
    DEN --> D3["Noise2Noise"]

    REC --> R1["PtychoNet"]
    REC --> R2["TomocuPy"]
    REC --> R3["INR Dynamic"]

    AUT --> A1["AI-NERD"]
    AUT --> A2["ROI-Finder"]
    AUT --> A3["Bayesian Opt."]

    MUL --> M1["XRF + Ptychography"]
    MUL --> M2["CT-XAS Correlation"]
    MUL --> M3["Optical-X-ray"]

    style SEG fill:#00D4AA22,stroke:#00D4AA
    style DEN fill:#FFB80022,stroke:#FFB800
    style REC fill:#1B3A5C22,stroke:#1B3A5C
    style AUT fill:#E8515D22,stroke:#E8515D
    style MUL fill:#9B59B622,stroke:#9B59B6
```
"""

if level == "L0":
    st.markdown(TAXONOMY_DIAGRAM)

    st.markdown("---")
    cols = st.columns(len(categories))
    for col, cat in zip(cols, categories):
        with col:
            with st.container(border=True):
                st.markdown(f"### {cat['icon']}")
                st.markdown(f"**{cat['name']}**")
                st.caption(cat["description"])
                st.metric("Methods", len(cat["methods"]))

elif level == "L1":
    st.markdown(TAXONOMY_DIAGRAM)

    # Category overview
    st.markdown("---")
    st.subheader("Method Categories")
    for cat in categories:
        with st.expander(f"{cat['icon']} {cat['name']} ({len(cat['methods'])} methods)", expanded=True):
            st.markdown(cat["description"])
            st.markdown("**Methods:**")
            for m in cat["methods"]:
                st.markdown(f"- **{m['name']}**")
            applicable = ", ".join(cat["modalities"])
            st.caption(f"Applicable modalities: {applicable}")

    # Heatmap
    st.markdown("---")
    st.subheader("Modality × Method Matrix")
    import pandas as pd

    mod_names = [f"{m['icon']} {m['short_name']}" for m in modalities]
    matrix = []
    for m in modalities:
        row = {}
        for c in categories:
            row[c["name"]] = "✅" if m["id"] in c["modalities"] else "—"
        matrix.append(row)
    df = pd.DataFrame(matrix, index=mod_names)
    st.dataframe(df, use_container_width=True)

elif level == "L2":
    # Method detail cards
    cat_names = [f"{c['icon']} {c['name']}" for c in categories]
    selected_cat_name = st.selectbox("Select Category", options=cat_names)
    cat_idx = cat_names.index(selected_cat_name)
    cat = categories[cat_idx]

    st.subheader(f"{cat['icon']} {cat['name']}")
    st.markdown(cat["description"])

    method_names = [m["name"] for m in cat["methods"]]
    selected_method = st.selectbox("Select Method", options=method_names)
    method = next(m for m in cat["methods"] if m["name"] == selected_method)

    st.markdown("---")
    render_markdown(method["file"], show_title=True)

elif level == "L3":
    # Source view
    all_files = []
    for cat in categories:
        for m in cat["methods"]:
            all_files.append((f"{cat['icon']} {cat['name']} / {m['name']}", m["file"]))

    selected = st.selectbox("Select file", options=all_files, format_func=lambda x: x[0])
    content = read_local_file(selected[1])
    if content:
        st.code(content, language="markdown")
