# Noise & Artifact Catalog

A practical catalog of **47 noise and artifact types** encountered in synchrotron X-ray data analysis and related imaging domains. Each entry includes detection code, correction methods, before/after examples, and links to deeper resources in this repository.

> **New in v2:** 18 entries added from cross-domain benchmarking — Medical Imaging (CT/MRI), Electron Microscopy (SEM/TEM/Cryo-EM), Scattering & Diffraction (SAXS/Crystallography), and cross-cutting methods from Astronomy. These entries include facility benchmarks from ESRF, Diamond, DESY, SPring-8, and more.

## How to Use This Catalog

This catalog provides **two access modes** depending on your situation:

### Mode 1: Classification-Based Browsing

**"I know what modality I'm working with."**

Browse the subdirectories below by technique. Each file documents one specific noise type with detection code, correction recipes, and visual examples.

### Mode 2: Symptom-Based Troubleshooting

**"I see something wrong but don't know what it is."**

Use the **[Symptom-Based Troubleshooter](troubleshooter.md)** — an interactive decision tree that starts from what the problem *looks like* and guides you to the correct diagnosis.

## 3-Axis Classification

```
Synchrotron Noise & Artifact Catalog
│
├── By Modality (directories)
│   ├── Tomography (9 types)
│   ├── XRF Microscopy (8 types)
│   ├── Spectroscopy (6 types)
│   ├── Ptychography (3 types)
│   ├── Cross-cutting (6 types)
│   ├── Medical Imaging (6 types) ← NEW
│   ├── Electron Microscopy (5 types) ← NEW
│   └── Scattering & Diffraction (5 types) ← NEW
│
├── By Noise Type (attribute in each doc)
│   ├── Statistical — photon counting, Poisson, thermal
│   ├── Systematic — self-absorption, beam drift, calibration
│   ├── Instrumental — detector defects, dead-time, optics
│   └── Computational — reconstruction artifacts, DL hallucination
│
└── By Symptom (troubleshooter.md)
    ├── "I see rings/circles"
    ├── "I see bright spots"
    ├── "I see stripes/streaks"
    ├── "Image is grainy/noisy"
    ├── "Features are blurred/shifted"
    ├── "Spectrum looks wrong"
    ├── "Data values seem off"
    ├── "Phase map has discontinuities" ← NEW
    └── "Ghost/residual from previous exposure" ← NEW
```

## Quick Reference Table

| # | Noise/Artifact | Modality | Type | Severity | Freq | 1-Line Solution |
|---|---------------|----------|------|----------|------|-----------------|
| 1 | [Ring artifact](tomography/ring_artifact.md) | Tomo | Instrumental | Critical | Common | Fourier stripe removal in sinogram |
| 2 | [Zinger](tomography/zinger.md) | Tomo | Instrumental | Major | Occasional | Median filter on projections |
| 3 | [Streak artifact](tomography/streak_artifact.md) | Tomo | Systematic | Critical | Common | Metal artifact reduction (MAR) |
| 4 | [Low-dose noise](tomography/low_dose_noise.md) | Tomo | Statistical | Major | Common | TomoGAN / BM3D denoising |
| 5 | [Sparse-angle artifact](tomography/sparse_angle_artifact.md) | Tomo | Computational | Major | Occasional | Iterative reconstruction (TV, SIRT) |
| 6 | [Motion artifact](tomography/motion_artifact.md) | Tomo | Systematic | Critical | Occasional | Motion-compensated reconstruction |
| 7 | [Flat-field issues](tomography/flatfield_issues.md) | Tomo | Instrumental | Major | Common | Proper flat/dark acquisition & normalization |
| 8 | [Rotation center error](tomography/rotation_center_error.md) | Tomo | Systematic | Critical | Common | Automated center-finding algorithm |
| 9 | [Beam intensity drop](tomography/beam_intensity_drop.md) | Tomo | Instrumental | Major | Occasional | I0 normalization per projection |
| 10 | [Photon counting noise](xrf_microscopy/photon_counting_noise.md) | XRF | Statistical | Major | Always | Longer dwell time or repeat scans |
| 11 | [Dead/hot pixel](xrf_microscopy/dead_hot_pixel.md) | XRF | Instrumental | Major | Common | Median filter outlier replacement |
| 12 | [Peak overlap](xrf_microscopy/peak_overlap.md) | XRF | Systematic | Major | Common | Spectral fitting with deconvolution |
| 13 | [Self-absorption](xrf_microscopy/self_absorption.md) | XRF | Systematic | Major | Common | Absorption correction model |
| 14 | [Dead-time saturation](xrf_microscopy/dead_time_saturation.md) | XRF | Instrumental | Critical | Common | Dead-time correction via ICR/OCR |
| 15 | [I0 normalization](xrf_microscopy/i0_normalization.md) | XRF | Systematic | Major | Common | Normalize elemental maps by I0 |
| 16 | [Probe blurring](xrf_microscopy/probe_blurring.md) | XRF | Instrumental | Minor | Always | Deconvolution / super-resolution DL |
| 17 | [Scan stripe](xrf_microscopy/scan_stripe.md) | XRF | Systematic | Major | Occasional | Row-by-row normalization |
| 18 | [Statistical noise (EXAFS)](spectroscopy/statistical_noise_exafs.md) | Spec | Statistical | Major | Always | Averaging multiple scans |
| 19 | [Energy calibration drift](spectroscopy/energy_calibration_drift.md) | Spec | Systematic | Critical | Occasional | Align edge position per scan |
| 20 | [Harmonics contamination](spectroscopy/harmonics_contamination.md) | Spec | Instrumental | Major | Occasional | Detuning monochromator or harmonic rejection mirror |
| 21 | [Self-absorption (XAS)](spectroscopy/self_absorption_xas.md) | Spec | Systematic | Major | Common | Sample dilution or correction algorithm |
| 22 | [Radiation damage](spectroscopy/radiation_damage.md) | Spec | Systematic | Critical | Occasional | Reduce dose rate, quick scans, cryogenic cooling |
| 23 | [Outlier spectra](spectroscopy/outlier_spectra.md) | Spec | Statistical | Minor | Occasional | PCA-based outlier detection and removal |
| 24 | [Position error](ptychography/position_error.md) | Ptycho | Systematic | Critical | Common | Joint position refinement in reconstruction |
| 25 | [Partial coherence](ptychography/partial_coherence.md) | Ptycho | Instrumental | Major | Common | Mixed-state reconstruction |
| 26 | [Stitching artifact](ptychography/stitching_artifact.md) | Ptycho | Computational | Minor | Occasional | Overlap-weighted blending |
| 27 | [DL hallucination](cross_cutting/dl_hallucination.md) | Cross | Computational | Critical | Occasional | Uncertainty quantification, residual analysis |
| 28 | [Rechunking data integrity](cross_cutting/rechunking_data_integrity.md) | Cross | Computational | Major | Occasional | Checksum verification after rechunking |
| 29 | [Detector common issues](cross_cutting/detector_common_issues.md) | Cross | Instrumental | Major | Common | Regular detector calibration |
| | **Medical Imaging (Benchmarked)** | | | | | |
| 30 | [Beam hardening](medical_imaging/beam_hardening.md) | Med/Tomo | Systematic | Major | Common | Polynomial correction or dual-energy |
| 31 | [Truncation artifact](medical_imaging/truncation_artifact.md) | Med/Tomo | Systematic | Major | Occasional | Sinogram extrapolation / padding |
| 32 | [Partial volume effect](medical_imaging/partial_volume_effect.md) | Med/Micro | Systematic | Major | Always | Higher resolution or sub-voxel modeling |
| 33 | [Scatter artifact](medical_imaging/scatter_artifact.md) | Med/Tomo | Systematic | Major | Common | Anti-scatter grid or kernel correction |
| 34 | [Gibbs ringing](medical_imaging/gibbs_ringing.md) | Med/CDI | Computational | Moderate | Common | Apodization (Hamming window) |
| 35 | [Bias field](medical_imaging/bias_field.md) | Med/Micro | Instrumental | Major | Common | N4ITK or homomorphic filtering |
| 36 | [Metal artifact](medical_imaging/metal_artifact.md) | Med/Tomo | Systematic | Critical | Occasional | MAR sinogram inpainting |
| | **Electron Microscopy (Benchmarked)** | | | | | |
| 37 | [Shot noise (low-dose)](electron_microscopy/shot_noise_low_dose.md) | EM | Statistical | Critical | Always | Class averaging, dose weighting |
| 38 | [Charging artifact](electron_microscopy/charging_artifact.md) | SEM | Instrumental | Major | Common | Conductive coating or low-voltage SEM |
| 39 | [Drift & vibration](electron_microscopy/drift_vibration.md) | EM/Nano | Systematic | Major | Common | Cross-correlation alignment |
| 40 | [CTF artifact](electron_microscopy/ctf_artifact.md) | TEM | Instrumental | Critical | Always | CTF estimation + Wiener correction |
| 41 | [Contamination buildup](electron_microscopy/contamination_buildup.md) | EM | Systematic | Moderate | Common | Plasma cleaning, beam shower |
| | **Scattering & Diffraction (Benchmarked)** | | | | | |
| 42 | [Parasitic scattering](scattering_diffraction/parasitic_scattering.md) | SAXS/WAXS | Instrumental | Critical | Always | Guard slits + buffer subtraction |
| 43 | [Ice rings](scattering_diffraction/ice_rings.md) | MX | Systematic | Major | Common | Cryoprotection + resolution exclusion |
| 44 | [Detector gaps & parallax](scattering_diffraction/detector_gaps_parallax.md) | Scatter/Diff | Instrumental | Major | Common | Multi-position merge + geometry correction |
| 45 | [Phase wrapping](scattering_diffraction/phase_wrapping.md) | Phase/CDI | Computational | Critical | Common | Quality-guided phase unwrapping |
| 46 | [Radiation damage (MX)](scattering_diffraction/radiation_damage_crystallography.md) | MX | Systematic | Critical | Common | Zero-dose extrapolation, multi-crystal |
| | **Cross-cutting (Benchmarked)** | | | | | |
| 47 | [Cosmic ray / outlier](cross_cutting/cosmic_ray_outlier.md) | Cross | Statistical | Major | Common | L.A.Cosmic Laplacian detection |
| 48 | [Noise estimation methods](cross_cutting/noise_estimation_methods.md) | Cross | Reference | N/A | N/A | NPS, DQE, MAD, Noise2Void (methodology) |
| 49 | [Afterglow / persistence](cross_cutting/afterglow_persistence.md) | Cross | Instrumental | Major | Common | Decay modeling + flush frames |

## Directory Contents

```
09_noise_catalog/
├── README.md                          # This file
├── troubleshooter.md                  # Symptom-based decision trees
├── summary_table.md                   # Full noise × detection × solution matrix
├── tomography/
│   ├── ring_artifact.md
│   ├── zinger.md
│   ├── streak_artifact.md
│   ├── low_dose_noise.md
│   ├── sparse_angle_artifact.md
│   ├── motion_artifact.md
│   ├── flatfield_issues.md
│   ├── rotation_center_error.md
│   └── beam_intensity_drop.md
├── xrf_microscopy/
│   ├── photon_counting_noise.md
│   ├── dead_hot_pixel.md
│   ├── peak_overlap.md
│   ├── self_absorption.md
│   ├── dead_time_saturation.md
│   ├── i0_normalization.md
│   ├── probe_blurring.md
│   └── scan_stripe.md
├── spectroscopy/
│   ├── statistical_noise_exafs.md
│   ├── energy_calibration_drift.md
│   ├── harmonics_contamination.md
│   ├── self_absorption_xas.md
│   ├── radiation_damage.md
│   └── outlier_spectra.md
├── ptychography/
│   ├── position_error.md
│   ├── partial_coherence.md
│   └── stitching_artifact.md
├── cross_cutting/
│   ├── dl_hallucination.md
│   ├── rechunking_data_integrity.md
│   ├── detector_common_issues.md
│   ├── cosmic_ray_outlier.md              # NEW — astronomy benchmarked
│   ├── noise_estimation_methods.md        # NEW — cross-domain methodology
│   └── afterglow_persistence.md           # NEW — astronomy/medical benchmarked
├── medical_imaging/                       # NEW — benchmarked from CT/MRI/PET
│   ├── beam_hardening.md
│   ├── truncation_artifact.md
│   ├── partial_volume_effect.md
│   ├── scatter_artifact.md
│   ├── gibbs_ringing.md
│   ├── bias_field.md
│   └── metal_artifact.md
├── electron_microscopy/                   # NEW — benchmarked from SEM/TEM/Cryo-EM
│   ├── shot_noise_low_dose.md
│   ├── charging_artifact.md
│   ├── drift_vibration.md
│   ├── ctf_artifact.md
│   └── contamination_buildup.md
├── scattering_diffraction/                # NEW — SAXS/WAXS/MX from global facilities
│   ├── parasitic_scattering.md
│   ├── ice_rings.md
│   ├── detector_gaps_parallax.md
│   ├── phase_wrapping.md
│   └── radiation_damage_crystallography.md
└── images/
    ├── README.md                      # Image attribution and license info
    └── generate_examples.py           # Script to generate synthetic examples
```

## Related Sections

- [AI/ML Methods — Denoising](../03_ai_ml_methods/denoising/) — TomoGAN, Noise2Noise, deep residual XRF
- [EDA Notebooks](../06_data_structures/eda/) — Exploratory data analysis with noise detection code
- [TomoPy Tools](../05_tools_and_code/tomopy/) — Stripe removal and preprocessing functions
- [X-ray Modalities](../02_xray_modalities/) — Technique-specific data formats and challenges
