# XRF Microscopy (X-ray Fluorescence Microscopy)

## Principles

**X-ray Fluorescence (XRF) Microscopy** maps the elemental composition of a sample by
detecting the characteristic fluorescence X-rays emitted when inner-shell electrons are
excited by the incident X-ray beam. By raster-scanning a focused beam across the sample,
2D (or 3D) elemental distribution maps are generated.

### Physical Basis
1. Incident X-ray photon ejects an inner-shell electron (photoelectric absorption)
2. Outer-shell electron fills the vacancy, emitting a characteristic fluorescence X-ray
3. Fluorescence energy is element-specific (Moseley's Law): E ∝ (Z-1)²
4. Fluorescence intensity is proportional to element concentration

### What XRF Measures
- **Qualitative**: Which elements are present (multi-element capability)
- **Quantitative**: Concentration of each element (ppm–wt% sensitivity)
- **Spatial**: 2D elemental distribution maps at sub-µm resolution
- **Simultaneous**: All elements above detection threshold measured at once

### Key Advantages
- Non-destructive and non-contact
- Multi-element detection in a single scan
- Trace-level sensitivity (sub-ppm for some elements)
- Compatible with hydrated, frozen, or ambient samples

## Experimental Setup

```
Focused X-ray beam (zone plate or KB mirrors)
    │
    ├── Beam size: 30 nm – 20 µm depending on optics
    │
    ▼
Sample (on XY scanning stage)
    │
    ├──→ XRF detector (90° geometry)     → Elemental maps
    │         (Vortex ME-4, SDD)
    │
    ├──→ Transmitted beam detector       → Absorption map
    │
    └──→ Diffraction detector (optional) → Ptychography/XRD
```

### eBERlight Beamlines

| Beamline | Optics | Beam Size | Energy Range | Specialty |
|----------|--------|-----------|-------------|-----------|
| **2-ID-D** | Zone plate nanoprobe | 30–200 nm | 5–30 keV | Highest resolution, trace elements |
| **2-ID-E** | Zone plate / KB | 0.5–5 µm | 5–30 keV | Combined XRF + ptychography |
| **8-BM-B** | KB mirrors | 2–20 µm | 4.5–20 keV | Large area mapping, XANES imaging |

## Data Collection Process

1. **Sample preparation**: Thin section, freeze-dried, or cryogenic
2. **Survey scan**: Coarse resolution (5–20 µm step) over large area
3. **ROI selection**: Identify regions of interest for detailed scanning
4. **Fine scan**: High resolution (30 nm – 1 µm step) of selected ROIs
5. **Spectral processing**: Fit fluorescence spectra to extract elemental concentrations

### Typical Parameters

| Parameter | Nanoprobe (2-ID-D) | Microprobe (2-ID-E) | Mapping (8-BM-B) |
|-----------|-------------------|---------------------|------------------|
| **Beam size** | 30–200 nm | 0.5–5 µm | 2–20 µm |
| **Step size** | 50–500 nm | 1–10 µm | 5–50 µm |
| **Dwell time** | 50–500 ms/pixel | 10–100 ms/pixel | 10–50 ms/pixel |
| **Map size** | 50×50 – 500×500 | 100×100 – 1000×1000 | 200×200 – 2000×2000 |
| **Scan time** | 30 min – 24 hr | 10 min – 4 hr | 30 min – 8 hr |
| **Elements** | Z ≥ 14 (Si) | Z ≥ 14 (Si) | Z ≥ 15 (P) |

## Data Scale

| Component | Typical Size |
|-----------|-------------|
| Raw spectrum per pixel | ~8 KB (2048 channels × 4 bytes) |
| Full spectral dataset | 0.5–50 GB |
| Fitted elemental maps | 10–500 MB |
| Multi-ROI experiment | 1–100 GB total |

## eBERlight Applications

- **Microbiology**: Single-cell elemental analysis, trace metal quantification in bacteria
- **Plant science**: Nutrient distribution in root cross-sections, seed micronutrient mapping
- **Soil science**: Element association with mineral phases, organic-mineral complexes
- **Environmental**: Contaminant distribution (As, Pb, Cr, U) in environmental samples
- **Climate**: Aerosol particle composition, ice core particulate analysis

## Key Software

- **MAPS**: Primary XRF analysis software at APS (spectral fitting, quantification)
- **PyXRF**: Python-based XRF analysis (NSLS-II developed)
- **ROI-Finder**: ML-guided ROI selection for XRF experiments
