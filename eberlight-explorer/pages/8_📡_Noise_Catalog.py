"""Noise & Artifact Catalog Explorer (F8)."""

import streamlit as st
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.content_parser import read_local_file
from components.level_selector import render_level_selector
from components.common_styles import inject_styles
from components.markdown_viewer import render_markdown

st.set_page_config(page_title="Noise & Artifact Catalog", page_icon="📡", layout="wide")

inject_styles()

level = render_level_selector(key="noise_level")

st.title("📡 Noise & Artifact Catalog")
st.markdown("29 noise and artifact types across synchrotron modalities — detection, correction, and troubleshooting.")

MODALITIES = {
    "Tomography": {
        "icon": "🔬",
        "files": [
            ("Ring Artifact", "09_noise_catalog/tomography/ring_artifact.md"),
            ("Zinger", "09_noise_catalog/tomography/zinger.md"),
            ("Streak Artifact", "09_noise_catalog/tomography/streak_artifact.md"),
            ("Low-Dose Noise", "09_noise_catalog/tomography/low_dose_noise.md"),
            ("Sparse-Angle Artifact", "09_noise_catalog/tomography/sparse_angle_artifact.md"),
            ("Motion Artifact", "09_noise_catalog/tomography/motion_artifact.md"),
            ("Flat-Field Issues", "09_noise_catalog/tomography/flatfield_issues.md"),
            ("Rotation Center Error", "09_noise_catalog/tomography/rotation_center_error.md"),
            ("Beam Intensity Drop", "09_noise_catalog/tomography/beam_intensity_drop.md"),
        ],
    },
    "XRF Microscopy": {
        "icon": "🔍",
        "files": [
            ("Photon Counting Noise", "09_noise_catalog/xrf_microscopy/photon_counting_noise.md"),
            ("Dead/Hot Pixel", "09_noise_catalog/xrf_microscopy/dead_hot_pixel.md"),
            ("Peak Overlap", "09_noise_catalog/xrf_microscopy/peak_overlap.md"),
            ("Self-Absorption", "09_noise_catalog/xrf_microscopy/self_absorption.md"),
            ("Dead-Time Saturation", "09_noise_catalog/xrf_microscopy/dead_time_saturation.md"),
            ("I0 Normalization", "09_noise_catalog/xrf_microscopy/i0_normalization.md"),
            ("Probe Blurring", "09_noise_catalog/xrf_microscopy/probe_blurring.md"),
            ("Scan Stripe", "09_noise_catalog/xrf_microscopy/scan_stripe.md"),
        ],
    },
    "Spectroscopy": {
        "icon": "📊",
        "files": [
            ("Statistical Noise (EXAFS)", "09_noise_catalog/spectroscopy/statistical_noise_exafs.md"),
            ("Energy Calibration Drift", "09_noise_catalog/spectroscopy/energy_calibration_drift.md"),
            ("Harmonics Contamination", "09_noise_catalog/spectroscopy/harmonics_contamination.md"),
            ("Self-Absorption (XAS)", "09_noise_catalog/spectroscopy/self_absorption_xas.md"),
            ("Radiation Damage", "09_noise_catalog/spectroscopy/radiation_damage.md"),
            ("Outlier Spectra", "09_noise_catalog/spectroscopy/outlier_spectra.md"),
        ],
    },
    "Ptychography": {
        "icon": "🔬",
        "files": [
            ("Partial Coherence", "09_noise_catalog/ptychography/partial_coherence.md"),
            ("Position Error", "09_noise_catalog/ptychography/position_error.md"),
            ("Stitching Artifact", "09_noise_catalog/ptychography/stitching_artifact.md"),
        ],
    },
    "Cross-Cutting": {
        "icon": "🔗",
        "files": [
            ("Detector Common Issues", "09_noise_catalog/cross_cutting/detector_common_issues.md"),
            ("DL Hallucination", "09_noise_catalog/cross_cutting/dl_hallucination.md"),
            ("Rechunking Data Integrity", "09_noise_catalog/cross_cutting/rechunking_data_integrity.md"),
        ],
    },
}

if level == "L0":
    # Overview cards per modality
    cols = st.columns(3)
    for i, (mod_name, mod_data) in enumerate(MODALITIES.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {mod_data['icon']} {mod_name}")
                st.metric("Noise Types", len(mod_data["files"]))
                for name, _ in mod_data["files"]:
                    st.markdown(f"- {name}")

    st.markdown("---")
    st.subheader("Quick Access")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### 📋 Summary Table")
            st.markdown("Complete matrix of all 29 noise/artifact types.")
    with c2:
        with st.container(border=True):
            st.markdown("### 🔍 Troubleshooter")
            st.markdown("Symptom-based decision tree for diagnosis.")

elif level == "L1":
    render_markdown("09_noise_catalog/README.md", show_title=False)

    st.markdown("---")
    st.subheader("Summary Table")
    render_markdown("09_noise_catalog/summary_table.md", show_title=False)

elif level == "L2":
    tab_names = ["By Modality", "Summary Table", "Troubleshooter"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        selected_mod = st.selectbox(
            "Select Modality",
            options=list(MODALITIES.keys()),
            format_func=lambda x: f"{MODALITIES[x]['icon']} {x}",
        )
        mod_data = MODALITIES[selected_mod]

        selected_noise = st.selectbox(
            "Select Noise/Artifact",
            options=[name for name, _ in mod_data["files"]],
        )
        noise_path = next(p for n, p in mod_data["files"] if n == selected_noise)
        render_markdown(noise_path, show_title=True)

    with tabs[1]:
        render_markdown("09_noise_catalog/summary_table.md", show_title=True)

    with tabs[2]:
        render_markdown("09_noise_catalog/troubleshooter.md", show_title=True)

elif level == "L3":
    # Source view — all noise catalog files
    all_files = ["09_noise_catalog/README.md", "09_noise_catalog/summary_table.md",
                 "09_noise_catalog/troubleshooter.md"]
    for mod_data in MODALITIES.values():
        all_files.extend(p for _, p in mod_data["files"])

    selected = st.selectbox("Select file", options=all_files)
    content = read_local_file(selected)
    if content:
        st.code(content, language="markdown")
