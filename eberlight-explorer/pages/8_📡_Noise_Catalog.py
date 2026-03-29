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

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

level = render_level_selector(key="noise_level")

st.title("📡 Noise & Artifact Catalog")
st.markdown("47 noise and artifact types across synchrotron and related imaging domains — detection, correction, and troubleshooting.")

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
            ("Cosmic Ray / Outlier", "09_noise_catalog/cross_cutting/cosmic_ray_outlier.md"),
            ("Noise Estimation Methods", "09_noise_catalog/cross_cutting/noise_estimation_methods.md"),
            ("Afterglow / Persistence", "09_noise_catalog/cross_cutting/afterglow_persistence.md"),
        ],
    },
    "Medical Imaging": {
        "icon": "🏥",
        "files": [
            ("Beam Hardening", "09_noise_catalog/medical_imaging/beam_hardening.md"),
            ("Truncation Artifact", "09_noise_catalog/medical_imaging/truncation_artifact.md"),
            ("Partial Volume Effect", "09_noise_catalog/medical_imaging/partial_volume_effect.md"),
            ("Scatter Artifact", "09_noise_catalog/medical_imaging/scatter_artifact.md"),
            ("Gibbs Ringing", "09_noise_catalog/medical_imaging/gibbs_ringing.md"),
            ("Bias Field", "09_noise_catalog/medical_imaging/bias_field.md"),
            ("Metal Artifact", "09_noise_catalog/medical_imaging/metal_artifact.md"),
        ],
    },
    "Electron Microscopy": {
        "icon": "🔬",
        "files": [
            ("Shot Noise (Low-Dose)", "09_noise_catalog/electron_microscopy/shot_noise_low_dose.md"),
            ("Charging Artifact", "09_noise_catalog/electron_microscopy/charging_artifact.md"),
            ("Drift & Vibration", "09_noise_catalog/electron_microscopy/drift_vibration.md"),
            ("CTF Artifact", "09_noise_catalog/electron_microscopy/ctf_artifact.md"),
            ("Contamination Buildup", "09_noise_catalog/electron_microscopy/contamination_buildup.md"),
        ],
    },
    "Scattering & Diffraction": {
        "icon": "💎",
        "files": [
            ("Parasitic Scattering", "09_noise_catalog/scattering_diffraction/parasitic_scattering.md"),
            ("Ice Rings", "09_noise_catalog/scattering_diffraction/ice_rings.md"),
            ("Detector Gaps & Parallax", "09_noise_catalog/scattering_diffraction/detector_gaps_parallax.md"),
            ("Phase Wrapping", "09_noise_catalog/scattering_diffraction/phase_wrapping.md"),
            ("Radiation Damage (MX)", "09_noise_catalog/scattering_diffraction/radiation_damage_crystallography.md"),
        ],
    },
}

# Before/After image mapping: noise name -> image file (relative to repo root)
BEFORE_AFTER_IMAGES = {
    "Ring Artifact": "09_noise_catalog/images/ring_artifact_before_after.png",
    "Zinger": "09_noise_catalog/images/zinger_before_after.png",
    "Low-Dose Noise": "09_noise_catalog/images/low_dose_noise_before_after.png",
    "Sparse-Angle Artifact": "09_noise_catalog/images/sparse_angle_before_after.png",
    "Flat-Field Issues": "09_noise_catalog/images/flatfield_before_after.png",
    "Rotation Center Error": "09_noise_catalog/images/rotation_center_error_before_after.png",
    "Dead/Hot Pixel": "09_noise_catalog/images/dead_hot_pixel_before_after.png",
    "I0 Normalization": "09_noise_catalog/images/i0_drop_before_after.png",
    # Cross-domain benchmarked entries
    "Beam Hardening": "09_noise_catalog/images/beam_hardening_before_after.png",
    "Truncation Artifact": "09_noise_catalog/images/truncation_artifact_before_after.png",
    "Scatter Artifact": "09_noise_catalog/images/scatter_artifact_before_after.png",
    "Gibbs Ringing": "09_noise_catalog/images/gibbs_ringing_before_after.png",
    "Metal Artifact": "09_noise_catalog/images/metal_artifact_before_after.png",
    "Shot Noise (Low-Dose)": "09_noise_catalog/images/shot_noise_low_dose_before_after.png",
    "CTF Artifact": "09_noise_catalog/images/ctf_artifact_before_after.png",
    "Drift & Vibration": "09_noise_catalog/images/drift_vibration_before_after.png",
    "Parasitic Scattering": "09_noise_catalog/images/parasitic_scattering_before_after.png",
    "Ice Rings": "09_noise_catalog/images/ice_rings_before_after.png",
    "Phase Wrapping": "09_noise_catalog/images/phase_wrapping_before_after.png",
    "Cosmic Ray / Outlier": "09_noise_catalog/images/cosmic_ray_before_after.png",
    "Afterglow / Persistence": "09_noise_catalog/images/afterglow_before_after.png",
}

# All noise items for lookup (name -> doc path)
ALL_NOISE_DOCS = {}
for mod_data in MODALITIES.values():
    for name, path in mod_data["files"]:
        ALL_NOISE_DOCS[name] = path

# ── Troubleshooter Decision Tree Data ─────────────────────
TROUBLESHOOTER_TREE = {
    "Circular/ring patterns": {
        "description": "Concentric rings or circular features in reconstructed image",
        "branches": [
            {
                "question": "Rings centered on rotation axis?",
                "yes": [
                    {"condition": "Sharp, well-defined rings", "diagnosis": "Ring Artifact", "severity": "Critical"},
                    {"condition": "Broad/blurry or partial arcs", "diagnosis": "Flat-Field Issues", "severity": "Major"},
                ],
                "no": [
                    {"condition": "Object edges appear doubled", "diagnosis": "Rotation Center Error", "severity": "Critical"},
                    {"condition": "Rings around dense inclusions", "diagnosis": "Streak Artifact", "severity": "Critical"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
sinogram = data[data.shape[0]//2]  # middle slice sinogram
col_std = np.std(sinogram, axis=0)
suspect_cols = np.where(col_std < np.median(col_std) * 0.1)[0]
print(f"Dead columns (ring sources): {suspect_cols}")
```""",
    },
    "Isolated bright/dark spots": {
        "description": "Random bright or dark spots that don't correspond to real features",
        "branches": [
            {
                "question": "Where do you see the spots?",
                "options": [
                    {"label": "Raw projections (single-frame only)", "diagnosis": "Zinger", "severity": "Major"},
                    {"label": "Raw projections (same pixel every frame)", "diagnosis": "Detector Common Issues", "severity": "Major"},
                    {"label": "XRF maps (abnormally bright/dark)", "diagnosis": "Dead/Hot Pixel", "severity": "Major"},
                    {"label": "Reconstructed CT (from zingers)", "diagnosis": "Zinger", "severity": "Major"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
diff = np.abs(np.diff(projections, axis=0))
threshold = np.median(diff) + 10 * np.std(diff)
zingers = np.where(diff > threshold)
print(f"Potential zingers found: {len(zingers[0])}")
```""",
    },
    "Streak/stripe patterns": {
        "description": "Linear streaks, stripes, or banding in the data",
        "branches": [
            {
                "question": "What direction and context?",
                "options": [
                    {"label": "Bright streaks from dense objects (CT)", "diagnosis": "Streak Artifact", "severity": "Critical"},
                    {"label": "Star-like streaks throughout recon", "diagnosis": "Sparse-Angle Artifact", "severity": "Major"},
                    {"label": "Horizontal stripes in sinogram (with I0 drops)", "diagnosis": "Beam Intensity Drop", "severity": "Major"},
                    {"label": "Vertical stripes in sinogram", "diagnosis": "Ring Artifact", "severity": "Critical"},
                    {"label": "Stripes in XRF maps (scan direction)", "diagnosis": "Scan Stripe", "severity": "Major"},
                    {"label": "Stripes in XRF maps (I0 correlation)", "diagnosis": "I0 Normalization", "severity": "Major"},
                    {"label": "Stripe at tile boundaries (ptychography)", "diagnosis": "Stitching Artifact", "severity": "Minor"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
n_projections = projections.shape[0]
detector_width = projections.shape[-1]
nyquist = int(np.pi/2 * detector_width)
print(f"Projections: {n_projections}, Nyquist minimum: {nyquist}")
print(f"{'UNDERSAMPLED' if n_projections < nyquist else 'OK'}")
```""",
    },
    "Overall graininess/noise": {
        "description": "Image appears grainy, speckled, or noisy throughout",
        "branches": [
            {
                "question": "What type of data?",
                "options": [
                    {"label": "CT reconstruction (uniform noise)", "diagnosis": "Low-Dose Noise", "severity": "Major"},
                    {"label": "CT reconstruction (worse in dense regions)", "diagnosis": "Streak Artifact", "severity": "Major"},
                    {"label": "CT reconstruction (worse at edges)", "diagnosis": "Flat-Field Issues", "severity": "Major"},
                    {"label": "XRF map (low-conc. elements noisier)", "diagnosis": "Photon Counting Noise", "severity": "Major"},
                    {"label": "XRF map (all elements noisy)", "diagnosis": "Dead-Time Saturation", "severity": "Major"},
                    {"label": "EXAFS (noisy at high k)", "diagnosis": "Statistical Noise (EXAFS)", "severity": "Major"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
roi_signal = recon[100:150, 100:150]  # region inside sample
roi_bg = recon[10:30, 10:30]          # background region
snr = np.mean(roi_signal) / np.std(roi_bg)
print(f"SNR estimate: {snr:.1f} (< 5 is very noisy, > 20 is good)")
```""",
    },
    "Blurring / loss of detail": {
        "description": "Features appear blurred, smeared, or larger than expected",
        "branches": [
            {
                "question": "What type of data?",
                "options": [
                    {"label": "CT — directional blurring", "diagnosis": "Motion Artifact", "severity": "Critical"},
                    {"label": "CT — uniform softness (few projections?)", "diagnosis": "Sparse-Angle Artifact", "severity": "Major"},
                    {"label": "CT — doubled edges", "diagnosis": "Rotation Center Error", "severity": "Critical"},
                    {"label": "XRF — features larger than expected", "diagnosis": "Probe Blurring", "severity": "Minor"},
                    {"label": "XRF — sharp in some elements, blurred in others", "diagnosis": "Peak Overlap", "severity": "Major"},
                    {"label": "Ptychography — uniform contrast loss", "diagnosis": "Partial Coherence", "severity": "Major"},
                    {"label": "Ptychography — varies across FOV", "diagnosis": "Position Error", "severity": "Critical"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
from scipy.signal import correlate
shift_series = []
for i in range(projections.shape[0]-1):
    cc = np.correlate(projections[i].mean(0), projections[i+1].mean(0), 'full')
    shift_series.append(np.argmax(cc) - projections.shape[-1] + 1)
print(f"Max inter-projection shift: {max(np.abs(shift_series))} pixels")
```""",
    },
    "Intensity/value anomalies": {
        "description": "Unexpected intensity variations, plateaus, or jumps",
        "branches": [
            {
                "question": "What do you observe?",
                "options": [
                    {"label": "Sudden intensity jumps in sinogram", "diagnosis": "Beam Intensity Drop", "severity": "Major"},
                    {"label": "XRF concentrations plateau (ICR/OCR off)", "diagnosis": "Dead-Time Saturation", "severity": "Critical"},
                    {"label": "XRF concentrations plateau (ICR/OCR OK)", "diagnosis": "Self-Absorption", "severity": "Major"},
                    {"label": "Systematic gradient in XRF scan", "diagnosis": "I0 Normalization", "severity": "Major"},
                    {"label": "Dampened fluorescence (concentrated sample)", "diagnosis": "Self-Absorption (XAS)", "severity": "Major"},
                    {"label": "Dampened fluorescence (dilute sample)", "diagnosis": "Harmonics Contamination", "severity": "Major"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
i0 = np.array(i0_values)
drops = np.where(i0 < 0.8 * np.median(i0))[0]
print(f"I0 drops (>20%): {len(drops)} events at indices {drops[:5]}")
```""",
    },
    "Spectral abnormalities": {
        "description": "XAS/XANES spectrum looks wrong, features distorted or shifting",
        "branches": [
            {
                "question": "What's wrong with the spectrum?",
                "options": [
                    {"label": "Edge shifts monotonically between scans", "diagnosis": "Radiation Damage", "severity": "Critical"},
                    {"label": "Edge shifts randomly between scans", "diagnosis": "Energy Calibration Drift", "severity": "Critical"},
                    {"label": "White-line flattened (fluorescence, concentrated)", "diagnosis": "Self-Absorption (XAS)", "severity": "Major"},
                    {"label": "XANES features damped/distorted", "diagnosis": "Harmonics Contamination", "severity": "Major"},
                    {"label": "Individual scans differ from average", "diagnosis": "Outlier Spectra", "severity": "Minor"},
                    {"label": "EXAFS amplitudes decrease with scans", "diagnosis": "Radiation Damage", "severity": "Critical"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
e0_values = []
for scan in scans:
    deriv = np.gradient(scan['mu'], scan['energy'])
    e0_values.append(scan['energy'][np.argmax(deriv)])
drift = max(e0_values) - min(e0_values)
print(f"Edge drift: {drift:.2f} eV across {len(scans)} scans (> 0.5 eV is problematic)")
```""",
    },
    "Boundary/stitching artifacts": {
        "description": "Discontinuities at boundaries, missing or shifted features",
        "branches": [
            {
                "question": "Where is the discontinuity?",
                "options": [
                    {"label": "Tile boundaries in ptychography (phase jumps)", "diagnosis": "Stitching Artifact", "severity": "Minor"},
                    {"label": "Features shifted in ptychography", "diagnosis": "Position Error", "severity": "Critical"},
                    {"label": "Missing/corrupted data after HDF5 rechunking", "diagnosis": "Rechunking Data Integrity", "severity": "Major"},
                    {"label": "Missing/corrupted data after format conversion", "diagnosis": "Rechunking Data Integrity", "severity": "Major"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
boundary_col = tile_width
left = recon[:, boundary_col-5:boundary_col]
right = recon[:, boundary_col:boundary_col+5]
jump = np.mean(np.abs(left.mean(1) - right.mean(1)))
print(f"Mean boundary jump: {jump:.4f} (should be close to 0)")
```""",
    },
    "Suspicious 'too-good' features": {
        "description": "Results from DL/AI processing look too clean or contain unexpected detail",
        "branches": [
            {
                "question": "Did you apply a neural network?",
                "options": [
                    {"label": "Yes — high-freq details in low-SNR regions", "diagnosis": "DL Hallucination", "severity": "Critical"},
                    {"label": "Yes — periodic/repetitive patterns appeared", "diagnosis": "DL Hallucination", "severity": "Critical"},
                    {"label": "Yes — but results look identical to input", "diagnosis": None, "severity": None, "note": "Network may not be trained for this data distribution."},
                    {"label": "No neural network was used", "diagnosis": None, "severity": None, "note": "Re-examine other symptom categories."},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
residual = dl_output - conventional_recon
residual_std = np.std(residual)
signal_std = np.std(conventional_recon)
print(f"Residual/signal ratio: {residual_std/signal_std:.3f}")
print(f"If > 0.1, DL is adding significant content - verify carefully")
```""",
    },
    "Phase map discontinuities": {
        "description": "Abrupt jumps or 'cliffs' in retrieved phase images",
        "branches": [
            {
                "question": "What type of discontinuity?",
                "options": [
                    {"label": "Sharp lines of ±2π discontinuity", "diagnosis": "Phase Wrapping", "severity": "Critical"},
                    {"label": "Oscillating bands parallel to edges", "diagnosis": "Gibbs Ringing", "severity": "Moderate"},
                    {"label": "Contrast reversals at different defocus", "diagnosis": "CTF Artifact", "severity": "Critical"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
dy = np.diff(phase_map, axis=0)
dx = np.diff(phase_map, axis=1)
wraps = (np.abs(dy) > 5).sum() + (np.abs(dx) > 5).sum()
print(f"Phase wraps detected: {wraps} ({wraps/phase_map.size:.2%} of pixels)")
```""",
    },
    "Ghost/residual from previous exposure": {
        "description": "Faint image of previous sample or bright region persists",
        "branches": [
            {
                "question": "How does the residual behave?",
                "options": [
                    {"label": "Fades over seconds to minutes", "diagnosis": "Afterglow / Persistence", "severity": "Major"},
                    {"label": "Persists indefinitely at same pixels", "diagnosis": "Detector Common Issues", "severity": "Major"},
                    {"label": "Only at low-q in scattering pattern", "diagnosis": "Parasitic Scattering", "severity": "Critical"},
                    {"label": "Signal grows over time under beam", "diagnosis": "Contamination Buildup", "severity": "Moderate"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
mean_signals = [frame.mean() for frame in dark_frames_after_bright]
print(f"Signal decay: {mean_signals[:5]}")
if all(a > b for a, b in zip(mean_signals, mean_signals[1:])):
    print("Monotonic decay detected — likely afterglow/persistence")
```""",
    },
    "Sample/beam damage effects": {
        "description": "Progressive data quality degradation during measurement",
        "branches": [
            {
                "question": "What modality?",
                "options": [
                    {"label": "XAS — edge position shifts between scans", "diagnosis": "Radiation Damage", "severity": "Critical"},
                    {"label": "Crystallography — B-factor increases, diffraction fades", "diagnosis": "Radiation Damage (MX)", "severity": "Critical"},
                    {"label": "EM — carbon deposit builds up under beam", "diagnosis": "Contamination Buildup", "severity": "Moderate"},
                    {"label": "CT — same material shows different density over time", "diagnosis": "Radiation Damage", "severity": "Critical"},
                ],
            },
        ],
        "quick_check": """```python
import numpy as np
# Track signal vs accumulated dose
for i, scan in enumerate(scans):
    print(f"Scan {i}: edge_position={e0[i]:.2f} eV, amplitude={amp[i]:.4f}")
```""",
    },
}

SEVERITY_COLORS = {
    "Critical": "red",
    "Major": "orange",
    "Minor": "blue",
}


def _render_before_after_viewer(noise_name: str):
    """Show a Before/After image viewer if available for this noise type."""
    img_path = BEFORE_AFTER_IMAGES.get(noise_name)
    if not img_path:
        return False
    full_path = os.path.join(REPO_ROOT, img_path)
    if not os.path.exists(full_path):
        return False
    st.image(full_path, caption=f"{noise_name} — Before / After", use_container_width=True)
    return True


def _render_summary_table_interactive():
    """Render the summary table with clickable Before/After images."""
    import pandas as pd

    st.subheader("Summary Table")
    st.markdown("Complete matrix of all 47 noise/artifact types. Click **View** to see before/after comparisons.")

    rows = []
    for mod_name, mod_data in MODALITIES.items():
        for name, path in mod_data["files"]:
            has_ba = name in BEFORE_AFTER_IMAGES
            rows.append({
                "Modality": f"{mod_data['icon']} {mod_name}",
                "Noise/Artifact": name,
                "Before/After": "Yes" if has_ba else "No",
                "_path": path,
                "_has_ba": has_ba,
            })

    # Filter by modality
    mod_options = ["All"] + list(MODALITIES.keys())
    selected_filter = st.selectbox("Filter by Modality", mod_options, key="summary_filter")
    if selected_filter != "All":
        rows = [r for r in rows if selected_filter in r["Modality"]]

    # Display table
    df = pd.DataFrame(rows)
    display_df = df[["Modality", "Noise/Artifact", "Before/After"]]
    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Before/After viewer section
    ba_items = [r for r in rows if r["_has_ba"]]
    if ba_items:
        st.markdown("---")
        st.subheader("Before/After Comparisons")
        selected_ba = st.selectbox(
            "Select noise type to view Before/After",
            options=[r["Noise/Artifact"] for r in ba_items],
            key="ba_viewer",
        )
        if selected_ba:
            _render_before_after_viewer(selected_ba)
            with st.expander("View full documentation"):
                doc_path = ALL_NOISE_DOCS.get(selected_ba)
                if doc_path:
                    render_markdown(doc_path, show_title=True)


def _render_interactive_troubleshooter():
    """Interactive troubleshooter built into the explorer."""
    st.subheader("Symptom-Based Troubleshooter")
    st.markdown("Select the symptom you observe and follow the guided diagnosis.")

    # Step 1: Select symptom category
    symptom_names = list(TROUBLESHOOTER_TREE.keys())
    selected_symptom = st.selectbox(
        "Step 1: What symptom do you see?",
        options=symptom_names,
        key="troubleshoot_symptom",
    )

    symptom = TROUBLESHOOTER_TREE[selected_symptom]
    st.info(f"**{selected_symptom}:** {symptom['description']}")

    # Step 2: Follow branches
    st.markdown("---")
    st.markdown("**Step 2: Narrow down the diagnosis**")

    for branch in symptom["branches"]:
        question = branch["question"]
        st.markdown(f"*{question}*")

        if "yes" in branch and "no" in branch:
            # Yes/No branch
            answer = st.radio(
                question,
                options=["Yes", "No"],
                key=f"branch_{selected_symptom}_{question}",
                horizontal=True,
                label_visibility="collapsed",
            )
            options = branch["yes"] if answer == "Yes" else branch["no"]
        elif "options" in branch:
            options = branch["options"]
        else:
            options = []

        if options:
            option_labels = [o.get("label") or o.get("condition", "") for o in options]
            selected_option = st.radio(
                "Select the best match:",
                options=option_labels,
                key=f"option_{selected_symptom}_{question}",
            )

            # Find matched option
            matched = next(o for o in options
                           if (o.get("label") or o.get("condition")) == selected_option)

            diagnosis = matched.get("diagnosis")
            severity = matched.get("severity")
            note = matched.get("note")

            st.markdown("---")
            st.markdown("**Step 3: Diagnosis**")

            if diagnosis:
                sev_color = SEVERITY_COLORS.get(severity, "gray")
                st.markdown(
                    f"### Diagnosis: {diagnosis}\n"
                    f"**Severity:** :{sev_color}[{severity}]"
                )

                # Show Before/After if available
                if diagnosis in BEFORE_AFTER_IMAGES:
                    with st.expander("View Before/After comparison", expanded=True):
                        _render_before_after_viewer(diagnosis)

                # Link to full guide
                doc_path = ALL_NOISE_DOCS.get(diagnosis)
                if doc_path:
                    with st.expander("View full guide", expanded=False):
                        render_markdown(doc_path, show_title=True)
                else:
                    st.warning(f"No detailed guide found for '{diagnosis}'.")
                    _render_no_solution_actions(diagnosis)
            else:
                if note:
                    st.info(note)
                else:
                    st.info("Could not determine a specific diagnosis. Try another symptom category.")
                _render_no_solution_actions(selected_symptom)

    # Quick check code
    st.markdown("---")
    with st.expander("Quick diagnostic code snippet"):
        st.markdown(symptom["quick_check"])


def _render_no_solution_actions(topic: str):
    """Show actions when no solution is available — PR or email."""
    st.markdown("---")
    st.markdown("**No solution found?** Help us improve the catalog:")
    c1, c2 = st.columns(2)
    with c1:
        pr_title = f"[Noise Catalog] Add solution for: {topic}"
        pr_url = (
            "https://github.com/Denny-Hwang/synchrotron-data-analysis-notes"
            f"/issues/new?title={pr_title.replace(' ', '+')}"
            "&labels=noise-catalog,enhancement"
            f"&body=Please+add+solution+or+documentation+for:+{topic.replace(' ', '+')}"
        )
        st.link_button("Open GitHub Issue / PR", pr_url)
    with c2:
        email_subject = f"Noise Catalog - {topic}"
        email_url = f"mailto:dhwang@anl.gov?subject={email_subject.replace(' ', '%20')}"
        st.link_button("Contact Repository Maintainer", email_url)


# ── Deep-linking via query params ──────────────────────
# Supports URLs like ?doc=ring_artifact or ?doc=09_noise_catalog/tomography/ring_artifact.md
_DOC_BY_BASENAME: dict[str, tuple[str, str, str]] = {}  # basename -> (name, path, modality)
for _mod_name, _mod_data in MODALITIES.items():
    for _name, _path in _mod_data["files"]:
        _DOC_BY_BASENAME[os.path.splitext(os.path.basename(_path))[0]] = (_name, _path, _mod_name)

_doc_param = st.query_params.get("doc", None)
_deep_link_target = None
if _doc_param:
    key = os.path.splitext(os.path.basename(_doc_param))[0]
    if key in _DOC_BY_BASENAME:
        _deep_link_target = _DOC_BY_BASENAME[key]
        level = "L2"

# ══════════════════════════════════════════════
# Main Page Rendering
# ══════════════════════════════════════════════

if level == "L0":
    # Overview cards per modality
    cols = st.columns(3)
    for i, (mod_name, mod_data) in enumerate(MODALITIES.items()):
        with cols[i % 3]:
            with st.container(border=True):
                st.markdown(f"### {mod_data['icon']} {mod_name}")
                st.metric("Noise Types", len(mod_data["files"]))
                for name, _ in mod_data["files"]:
                    ba_tag = " **[B/A]**" if name in BEFORE_AFTER_IMAGES else ""
                    st.markdown(f"- {name}{ba_tag}")

    st.markdown("---")
    st.subheader("Quick Access")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### Summary Table")
            st.markdown("Complete matrix of all 47 noise/artifact types with Before/After comparisons.")
    with c2:
        with st.container(border=True):
            st.markdown("### Troubleshooter")
            st.markdown("Interactive symptom-based diagnosis — find the problem from what you see.")

elif level == "L1":
    render_markdown("09_noise_catalog/README.md", show_title=False)

    st.markdown("---")
    _render_summary_table_interactive()

elif level == "L2":
    # If deep-linked to a specific document, show it directly
    if _deep_link_target:
        dl_name, dl_path, dl_mod = _deep_link_target
        st.info(f"Showing: **{dl_name}** ({dl_mod})")
        if dl_name in BEFORE_AFTER_IMAGES:
            with st.expander("Before/After Comparison", expanded=True):
                _render_before_after_viewer(dl_name)
        render_markdown(dl_path, show_title=True)
        st.markdown("---")
        st.caption("Browse all documents below:")

    tab_names = ["By Modality", "Summary Table", "Troubleshooter"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        _mod_keys = list(MODALITIES.keys())
        _mod_default = 0
        if _deep_link_target:
            _mod_default = _mod_keys.index(_deep_link_target[2]) if _deep_link_target[2] in _mod_keys else 0
        selected_mod = st.selectbox(
            "Select Modality",
            options=_mod_keys,
            index=_mod_default,
            format_func=lambda x: f"{MODALITIES[x]['icon']} {x}",
        )
        mod_data = MODALITIES[selected_mod]

        _noise_names = [name for name, _ in mod_data["files"]]
        _noise_default = 0
        if _deep_link_target and _deep_link_target[2] == selected_mod:
            _noise_default = _noise_names.index(_deep_link_target[0]) if _deep_link_target[0] in _noise_names else 0
        selected_noise = st.selectbox(
            "Select Noise/Artifact",
            options=_noise_names,
            index=_noise_default,
        )
        noise_path = next(p for n, p in mod_data["files"] if n == selected_noise)

        # Show Before/After if available
        if selected_noise in BEFORE_AFTER_IMAGES:
            with st.expander("Before/After Comparison", expanded=True):
                _render_before_after_viewer(selected_noise)

        render_markdown(noise_path, show_title=True)

    with tabs[1]:
        _render_summary_table_interactive()

    with tabs[2]:
        _render_interactive_troubleshooter()

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
