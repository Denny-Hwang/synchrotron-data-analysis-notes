"""Data Structures & EDA Playground (F7)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.markdown_viewer import render_markdown

st.set_page_config(page_title="Data Structures", page_icon="📊", layout="wide")


# Hard redirect to the current app — see ADR-009. The legacy page body
# below is no longer maintained and is preserved only for ADR archival.
import sys as _sys
import os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
from _deprecated import render_deprecation_and_stop  # noqa: E402

render_deprecation_and_stop()

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

# ── Deep-linking via query params ──────────────────────
# Supports URLs like ?doc=tomo_eda or ?doc=06_data_structures/eda/tomo_eda.md
_ALL_DOCS = {**HDF5_SCHEMAS, **DEEP_DIVE_FILES, **EDA_FILES}
_DOC_BY_BASENAME: dict[str, tuple[str, str, str]] = {}  # basename -> (display_name, path, category)
for _name, _path in HDF5_SCHEMAS.items():
    _DOC_BY_BASENAME[os.path.splitext(os.path.basename(_path))[0]] = (_name, _path, "schema")
for _name, _path in DEEP_DIVE_FILES.items():
    _DOC_BY_BASENAME[os.path.splitext(os.path.basename(_path))[0]] = (_name, _path, "deep_dive")
for _name, _path in EDA_FILES.items():
    _DOC_BY_BASENAME[os.path.splitext(os.path.basename(_path))[0]] = (_name, _path, "eda")

_doc_param = st.query_params.get("doc", None)
_deep_link_target = None
if _doc_param:
    # Try matching by basename (e.g., "tomo_eda") or full path
    key = os.path.splitext(os.path.basename(_doc_param))[0]
    if key in _DOC_BY_BASENAME:
        _deep_link_target = _DOC_BY_BASENAME[key]
        level = "L2"  # Force detail view for deep links

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
    # If deep-linked to a specific document, show it directly
    if _deep_link_target:
        dl_name, dl_path, dl_cat = _deep_link_target
        st.info(f"Showing: **{dl_name}**")
        render_markdown(dl_path, show_title=True)
        st.markdown("---")
        st.caption("Browse all documents below:")

    tab_names = ["HDF5 & Data Formats", "HDF5 Schemas", "EDA Reports", "Notebooks"]
    # Default to the tab matching the deep-link category
    default_tab = 0
    if _deep_link_target:
        cat = _deep_link_target[2]
        default_tab = {"deep_dive": 0, "schema": 1, "eda": 2}.get(cat, 0)
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _dd_keys = list(DEEP_DIVE_FILES.keys())
        _dd_default = 0
        if _deep_link_target and _deep_link_target[2] == "deep_dive":
            _dd_default = _dd_keys.index(_deep_link_target[0]) if _deep_link_target[0] in _dd_keys else 0
        selected_topic = st.selectbox("Select Topic", options=_dd_keys, index=_dd_default)
        render_markdown(DEEP_DIVE_FILES[selected_topic], show_title=True)

    with tabs[1]:
        _sc_keys = list(HDF5_SCHEMAS.keys())
        _sc_default = 0
        if _deep_link_target and _deep_link_target[2] == "schema":
            _sc_default = _sc_keys.index(_deep_link_target[0]) if _deep_link_target[0] in _sc_keys else 0
        selected_schema = st.selectbox("Select Schema", options=_sc_keys, index=_sc_default)
        render_markdown(HDF5_SCHEMAS[selected_schema], show_title=True)

    with tabs[2]:
        _eda_keys = list(EDA_FILES.keys())
        _eda_default = 0
        if _deep_link_target and _deep_link_target[2] == "eda":
            _eda_default = _eda_keys.index(_deep_link_target[0]) if _deep_link_target[0] in _eda_keys else 0
        selected_eda = st.selectbox("Select EDA", options=_eda_keys, index=_eda_default)
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
