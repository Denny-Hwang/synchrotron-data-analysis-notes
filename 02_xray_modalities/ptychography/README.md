# Ptychography (Coherent X-ray Imaging)

## Principles

**Ptychography** is a coherent imaging technique that reconstructs both the amplitude
and phase of an object from a series of overlapping diffraction patterns. It achieves
spatial resolution beyond the lens limit and provides quantitative phase information.

### Physical Basis
1. A coherent X-ray beam is focused to a spot on the sample
2. The exit wave is the product of the illumination (probe) and the sample (object)
3. A far-field diffraction pattern is recorded at each scan position
4. Overlapping illuminated regions provide redundant information for phase retrieval
5. Iterative algorithms reconstruct both probe and object complex transmission functions

### Key Concept: Phase Problem Solution

Unlike other X-ray techniques, ptychography solves the **phase problem** directly:
- Detectors record intensity |ψ|² (losing phase information)
- Overlap constraint between adjacent scan positions provides additional equations
- Iterative algorithms converge to unique solution for both amplitude and phase

### What Ptychography Measures

| Quantity | Physical Meaning | Resolution |
|----------|-----------------|------------|
| **Amplitude** | X-ray absorption | 5–20 nm |
| **Phase** | Electron density (refractive index) | 5–20 nm |
| **Probe** | Illumination function | Self-calibrating |

Phase contrast sensitivity is ~1000× better than absorption, making it ideal for
low-Z biological and environmental samples.

## Experimental Setup

```
Coherent X-ray beam (undulator + monochromator)
    │
    ▼
Focusing optics (zone plate or KB mirrors)
    │
    ▼
Sample on piezo scanning stage
    │    ├── Step size: 50–500 nm (with ~60-80% overlap)
    │    └── Scan pattern: raster, spiral, or Fermat spiral
    │
    ▼
Pixelated area detector (in far field)
    │    ├── EIGER 500K, Lambda
    │    └── 256×256 to 512×512 pixels per pattern
    │
    ▼
Diffraction pattern per scan position
```

### eBERlight Beamlines

| Beamline | Source | Resolution | Energy | Specialty |
|----------|--------|-----------|--------|-----------|
| **2-ID-E** | Undulator | 10–50 nm | 5–30 keV | Combined XRF + ptychography |
| **33-ID-C** | Undulator | 5–20 nm | 6–25 keV | APS-U flagship coherent imaging |

### APS-U Impact
The APS-U dramatically improves ptychography by increasing coherent flux by ~100×:
- Higher throughput (faster scans)
- Better signal-to-noise at each scan position
- Enables ptychographic tomography in practical time

## Variants

| Variant | Description | Dimensionality |
|---------|-------------|---------------|
| **2D Ptychography** | Standard forward-scattering | 2D (x, y) |
| **Ptycho-tomography** | Ptychography at multiple rotations | 3D (x, y, z) |
| **Bragg ptychography** | Diffraction from crystal planes | 3D strain mapping |
| **Multi-slice ptychography** | Thick sample modeling | Extended depth of field |
| **Spectro-ptychography** | Energy-resolved | 2D + chemical speciation |

## Data Collection Process

1. **Alignment**: Center sample, set up detector geometry
2. **Scan definition**: Define scan area, step size, pattern type
3. **Data acquisition**: Record diffraction pattern at each position (100–10,000 positions)
4. **Background collection**: Measure detector background and dark current
5. **Reconstruction**: Iterative phase retrieval algorithm

### Typical Parameters

| Parameter | Standard | High-Speed | Ptycho-Tomo |
|-----------|---------|-----------|-------------|
| **Scan positions** | 500–5,000 | 100–500 | 500–2,000 × 180 angles |
| **Exposure/position** | 10–100 ms | 0.5–5 ms | 10–50 ms |
| **Detector pixels** | 256×256 or 512×512 | 256×256 | 256×256 |
| **Overlap** | 60–80% | 50–70% | 60–70% |
| **Scan area** | 10–100 µm | 5–50 µm | 10–50 µm |
| **Total scan time** | 5–60 min | 10–60 s | 1–24 hr |
| **Achieved resolution** | 5–20 nm | 20–50 nm | 10–30 nm |

## Data Scale

| Component | Size |
|-----------|------|
| Single diffraction pattern | 0.5–2 MB |
| Full 2D scan (2,000 positions) | 1–4 GB |
| Ptycho-tomography (180 angles) | 100 GB – 1 TB |
| Reconstructed 2D image | 10–100 MB |
| Reconstructed 3D volume | 1–50 GB |

## Reconstruction Algorithms

### Iterative Methods

| Algorithm | Type | Key Feature |
|-----------|------|-------------|
| **PIE** | Projective | Original ptychographic algorithm |
| **ePIE** | Extended PIE | Simultaneously reconstructs probe + object |
| **DM** | Difference Map | More robust convergence |
| **LSQ-ML** | Maximum Likelihood | Handles Poisson noise optimally |
| **rPIE** | Regularized PIE | Improved convergence, relaxation parameter |

### Reconstruction Pipeline
```
Diffraction patterns + scan positions
    │
    ├─→ Preprocessing (background subtraction, hot pixel removal)
    │
    ├─→ Initial guess (probe from simulation, object = uniform)
    │
    ├─→ Iterative reconstruction (100-1000 iterations)
    │       ├── Forward: probe × object → exit wave → propagate → predicted pattern
    │       ├── Constraint: replace predicted amplitude with measured √intensity
    │       └── Update: back-propagate → update probe and object estimates
    │
    └─→ Reconstructed complex images
            ├── Amplitude image (absorption)
            └── Phase image (electron density / refractive index)
```

## eBERlight Applications

- **Microbiology**: Unstained cell ultrastructure at nanometer resolution
- **Plant science**: Cell wall nanostructure, organelle imaging
- **Soil science**: Mineral-organic interfaces at the nanoscale
- **Environmental**: Nanoparticle characterization, aerosol internal structure
- **Materials in context**: Biomineralization processes, bone microstructure
