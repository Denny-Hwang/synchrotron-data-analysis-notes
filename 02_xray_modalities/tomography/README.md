# Tomography (µCT / Nano-CT)

## Principles

**X-ray Computed Tomography (CT)** reconstructs the 3D internal structure of a sample
from a series of 2D projection images acquired at different rotation angles. The
technique is non-destructive and can image samples in their natural (wet, frozen, or
in-situ) state.

### Physical Basis
- X-rays are attenuated as they pass through a sample (Beer-Lambert Law)
- Attenuation depends on material density, composition, and X-ray energy
- By collecting projections at many angles (typically 180° or 360°), the 3D attenuation
  coefficient distribution can be mathematically reconstructed (Radon transform / inverse)

### Contrast Mechanisms

| Mechanism | Basis | Best For | APS-U Impact |
|-----------|-------|----------|-------------|
| **Absorption** | Mass attenuation | Dense/high-Z materials | Standard |
| **Phase contrast** | Refractive index | Soft tissue, low-Z | Dramatically enhanced |
| **Edge enhancement** | Free-space propagation | Interfaces, boundaries | Enhanced |
| **Grating-based** | Talbot interferometry | Quantitative phase | New capabilities |

### Variants at APS

| Variant | Resolution | FOV | Scan Time | Application |
|---------|-----------|-----|-----------|-------------|
| **µCT** | 0.5–5 µm | 1–10 mm | 1–60 min | Soil aggregates, roots, rocks |
| **Nano-CT (TXM)** | 50–100 nm | 50–200 µm | 30–120 min | Cell ultrastructure |
| **Fast CT** | 1–5 µm | 1–5 mm | 0.1–10 s | Dynamic processes |
| **4D CT** | 1–5 µm | 1–5 mm | Time series | In-situ reactions |

## Experimental Setup

```
X-ray source (bending magnet or undulator)
    │
    ▼
Monochromator or pink beam (for speed)
    │
    ▼
Sample on rotation stage ──→ 0° to 180° rotation
    │
    ▼
Scintillator (converts X-rays to visible light)
    │
    ▼
Optical magnification (objective lens)
    │
    ▼
Camera (sCMOS: PCO Edge 5.5, Grasshopper, Oryx)
```

### APS BER Beamlines
- **2-BM-A**: Workhorse µCT, fast tomography, environmental cells, 10–40 keV
- **7-BM-B**: Broad energy range (6–80 keV), dual-energy capabilities
- **32-ID-B/C**: Transmission X-ray microscope (TXM), zone-plate optics, nano-CT

## Data Collection Process

1. **Sample mounting**: Sample in tube/capillary/environmental cell on rotation stage
2. **Flat field collection**: Images without sample (I₀) for normalization
3. **Dark field collection**: Images with beam off (detector background)
4. **Projection acquisition**: Rotate sample, collect images at each angle
5. **Reconstruction**: Compute 3D volume from projections

### Typical Parameters

| Parameter | µCT (2-BM) | Nano-CT (32-ID) |
|-----------|-----------|----------------|
| **Energy** | 20–40 keV | 8–12 keV |
| **Projections** | 900–3600 | 720–1500 |
| **Image size** | 2048×2048 or 4096×4096 | 1024×1024 to 2048×2048 |
| **Exposure/frame** | 1–100 ms | 0.5–5 s |
| **Total scan time** | 2–60 min | 30–120 min |
| **Pixel size** | 0.65–6.5 µm | 50–100 nm |

## Data Scale

| Component | Size |
|-----------|------|
| Single projection | 8–32 MB |
| Flat/dark fields | 100 MB – 1 GB |
| Full projection set | 10–100 GB |
| Reconstructed volume (16-bit) | 8–128 GB (2048³ to 4096³ voxels) |
| Time series (4D) | 100 GB – 10 TB |

## APS BER Applications

- **Soil science**: 3D pore network analysis, aggregate structure, root-soil interface
- **Plant science**: Root system architecture in soil, vasculature structure
- **Geochemistry**: Rock microstructure, fluid inclusion analysis, mineral textures
- **Environmental**: Biofilm structure, contaminant distribution in porous media
- **In-situ studies**: Wetting/drying cycles, mineral dissolution/precipitation
