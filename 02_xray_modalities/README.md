# X-ray Modalities

This section documents the six primary X-ray modalities used in the eBERlight program.
Each modality exploits different X-ray–matter interactions to reveal complementary
aspects of sample structure and composition.

## Modality Overview

| Modality | Interaction | What It Measures | Spatial Resolution | Key Beamlines |
|----------|-------------|------------------|-------------------|---------------|
| [Crystallography](crystallography/) | Diffraction | Atomic structure | Å (0.1 nm) | 21-ID-D/F/G |
| [Tomography](tomography/) | Absorption/Phase | 3D microstructure | 50 nm – 1 µm | 2-BM-A, 7-BM-B, 32-ID |
| [XRF Microscopy](xrf_microscopy/) | Fluorescence | Elemental distribution | 30 nm – 20 µm | 2-ID-D/E, 8-BM-B |
| [Spectroscopy](spectroscopy/) | Absorption | Chemical speciation | 1 µm – 1 mm | 9-BM, 20-BM, 25-ID |
| [Ptychography](ptychography/) | Coherent scattering | Phase & amplitude | 5–20 nm | 2-ID-E, 33-ID-C |
| [Scattering](scattering/) | Small/wide-angle | Nanostructure, dynamics | Statistical (nm) | 12-ID-B/E |

## X-ray–Matter Interactions

```
Incident X-ray beam
        │
        ▼
   ┌─────────┐
   │  Sample  │──→ Transmitted beam (absorption → tomography)
   │          │──→ Fluorescence X-rays (elemental composition → XRF)
   │          │──→ Diffracted beam (crystal structure → crystallography)
   │          │──→ Scattered beam (nanostructure → SAXS/WAXS/XPCS)
   └─────────┘
        │
        ▼
   Absorption spectrum (chemical state → XAS/XANES/EXAFS)
```

## Data Scale Comparison

| Modality | Typical Scan Time | Raw Data Size | Post-Processing Size |
|----------|------------------|---------------|---------------------|
| MX Dataset | 5–30 min | 10–100 GB | 1–10 GB (structure) |
| µCT Scan | 1–60 min | 10–500 GB | 1–50 GB (volume) |
| XRF Map | 30 min – 24 hr | 1–100 GB | 0.1–10 GB (maps) |
| XAS Spectrum | 10–60 min | 10–100 MB | 1–10 MB |
| Ptychography Scan | 10–60 min | 10–500 GB | 1–50 GB (images) |
| SAXS/WAXS | 1–60 sec/frame | 0.1–10 GB | 10–100 MB (profiles) |

## Common Data Formats

- **HDF5**: Primary format for most modalities (hierarchical, self-describing, parallel I/O)
- **TIFF**: Legacy image format, still used for projections and reconstructed slices
- **CBF**: Crystallographic Binary File (legacy diffraction format)
- **NeXus**: HDF5-based standard with defined schemas for synchrotron data

## Directory Contents

| Subdirectory | Modality | Files |
|-------------|----------|-------|
| [crystallography/](crystallography/) | MX, SSX | README, data_format, ai_ml_methods |
| [tomography/](tomography/) | µCT, nano-CT | README, data_format, reconstruction, ai_ml_methods |
| [xrf_microscopy/](xrf_microscopy/) | XRF mapping | README, data_format, analysis_pipeline, ai_ml_methods |
| [spectroscopy/](spectroscopy/) | XANES, EXAFS | README, data_format, ai_ml_methods |
| [ptychography/](ptychography/) | Coherent imaging | README, data_format, ai_ml_methods |
| [scattering/](scattering/) | SAXS, WAXS, XPCS | README, data_format, ai_ml_methods |
