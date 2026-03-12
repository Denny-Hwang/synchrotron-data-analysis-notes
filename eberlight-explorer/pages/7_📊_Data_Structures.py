"""Data Structures & EDA Playground (F7)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.markdown_viewer import render_markdown

st.set_page_config(page_title="Data Structures", page_icon="📊", layout="wide")

inject_styles()

level = render_level_selector(key="data_level")

st.title("📊 Data Structures")
st.markdown("HDF5 schemas, data formats, data scale analysis, and exploratory data analysis for synchrotron datasets.")

HDF5_SCHEMAS = {
    "Tomography": "06_data_structures/hdf5_structure/tomo_hdf5_schema.md",
    "XRF Microscopy": "06_data_structures/hdf5_structure/xrf_hdf5_schema.md",
    "Ptychography": "06_data_structures/hdf5_structure/ptychography_hdf5_schema.md",
}

DEEP_DIVE_FILES = {
    "HDF5 Deep Dive (SWMR, Parallel I/O, Limitations)": "06_data_structures/hdf5_deep_dive.md",
    "Data Formats Comparison (HDF5 vs Zarr vs TIFF)": "06_data_structures/data_formats_comparison.md",
    "APS-U Data Challenges (100+ TB/Day)": "06_data_structures/data_challenges_apsu.md",
    "Data Scale Analysis (Pre vs Post APS-U)": "06_data_structures/data_scale_analysis.md",
}

EDA_FILES = {
    "XRF EDA": "06_data_structures/eda/xrf_eda.md",
    "Tomography EDA": "06_data_structures/eda/tomo_eda.md",
    "Spectroscopy EDA": "06_data_structures/eda/spectroscopy_eda.md",
}

NOTEBOOKS = [
    "06_data_structures/hdf5_structure/notebooks/01_hdf5_exploration.ipynb",
    "06_data_structures/hdf5_structure/notebooks/02_data_visualization.ipynb",
    "06_data_structures/eda/notebooks/01_xrf_eda.ipynb",
    "06_data_structures/eda/notebooks/02_tomo_eda.ipynb",
    "06_data_structures/eda/notebooks/03_spectral_eda.ipynb",
]

if level == "L0":
    cols = st.columns(3)
    with cols[0]:
        with st.container(border=True):
            st.markdown("### 📁 HDF5 Schemas")
            st.metric("Modalities", len(HDF5_SCHEMAS))
            for name in HDF5_SCHEMAS:
                st.markdown(f"- {name}")
    with cols[1]:
        with st.container(border=True):
            st.markdown("### 📈 EDA Reports")
            st.metric("Reports", len(EDA_FILES))
            for name in EDA_FILES:
                st.markdown(f"- {name}")
    with cols[2]:
        with st.container(border=True):
            st.markdown("### 📓 Notebooks")
            st.metric("Notebooks", len(NOTEBOOKS))
            for nb in NOTEBOOKS:
                st.markdown(f"- `{os.path.basename(nb)}`")

    # Data scale overview
    st.markdown("---")
    st.subheader("Data Scale Overview")
    render_markdown("06_data_structures/data_scale_analysis.md", show_title=False)

elif level == "L1":
    render_markdown("06_data_structures/README.md", show_title=False)

    st.markdown("---")
    st.subheader("Data Scale Analysis")
    render_markdown("06_data_structures/data_scale_analysis.md", show_title=False)

    st.markdown("---")
    st.subheader("APS-U Data Challenges")
    render_markdown("06_data_structures/data_challenges_apsu.md", show_title=False)

elif level == "L2":
    tab_names = ["HDF5 & Data Formats", "HDF5 Schemas", "EDA Reports", "Notebooks"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        selected_topic = st.selectbox(
            "Select Topic", options=list(DEEP_DIVE_FILES.keys())
        )
        render_markdown(DEEP_DIVE_FILES[selected_topic], show_title=True)

    with tabs[1]:
        selected_schema = st.selectbox("Select Schema", options=list(HDF5_SCHEMAS.keys()))
        render_markdown(HDF5_SCHEMAS[selected_schema], show_title=True)

    with tabs[2]:
        selected_eda = st.selectbox("Select EDA", options=list(EDA_FILES.keys()))
        render_markdown(EDA_FILES[selected_eda], show_title=True)

    with tabs[3]:
        st.subheader("Available Notebooks")
        for nb in NOTEBOOKS:
            st.markdown(f"- `{nb}`")
        st.info("Notebook rendering requires nbconvert. Files are available in the repository.")

elif level == "L3":
    all_files = (
        list(HDF5_SCHEMAS.values()) +
        list(DEEP_DIVE_FILES.values()) +
        list(EDA_FILES.values()) +
        ["06_data_structures/README.md"]
    )
    selected = st.selectbox("Select file", options=all_files)
    content = read_local_file(selected)
    if content:
        st.code(content, language="markdown")
