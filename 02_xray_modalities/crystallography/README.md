# Crystallography (MX / SSX)

## Principles

**Macromolecular Crystallography (MX)** determines the 3D atomic structure of biological
macromolecules (proteins, nucleic acids, complexes) by analyzing the diffraction pattern
produced when X-rays interact with a crystal lattice.

### Physical Basis
- X-rays scatter from electron clouds of atoms in the crystal
- Constructive interference at specific angles produces a diffraction pattern (Bragg's Law: nλ = 2d sin θ)
- Each diffraction spot intensity encodes structural information
- Phase problem: intensities are measured but phases must be determined computationally

### Variants at APS

| Variant | Description | Crystal Size | Data Collection |
|---------|-------------|-------------|-----------------|
| **MX** | Standard single-crystal rotation | 20–500 µm | Single crystal, 180° rotation |
| **SSX** | Serial synchrotron crystallography | 1–20 µm | Thousands of random crystals |
| **Microcrystallography** | Small/challenging crystals | 5–50 µm | Focused microbeam |
| **MAD/SAD** | Anomalous dispersion phasing | Any | Multiple wavelengths |

## Experimental Setup

```
X-ray source (undulator)
    │
    ▼
Monochromator (select energy/wavelength)
    │
    ▼
Focusing optics (KB mirrors)
    │
    ▼
Goniometer + Crystal ──→ Diffraction pattern ──→ Pixel Array Detector
    │                                                  (EIGER/PILATUS)
    ▼
Cryo-stream (100 K)
```

### APS BER Beamlines
- **21-ID-D**: Tunable energy (6.5–20 keV), EIGER 16M, automated sample changer
- **21-ID-F**: Fixed energy (12.7 keV), high-throughput screening
- **21-ID-G**: Tunable, serial crystallography capabilities

## Data Collection Process

1. **Crystal mounting**: Loop/mesh mount, flash-cooled to 100 K (or room temp for SSX)
2. **Centering**: Align crystal in X-ray beam (automated or ML-assisted)
3. **Strategy determination**: Collect a few test images, determine optimal rotation range and exposure
4. **Data collection**: Rotate crystal through angular range while recording diffraction frames
5. **Data processing**: Index → integrate → scale → merge → phase → build model → refine

### Typical Parameters
- **Energy**: 12.66 keV (Se-edge for SAD) or 12.0 keV (standard)
- **Rotation range**: 180° (MX) or 0.1–0.5° per frame
- **Exposure**: 0.01–1 s per frame
- **Total frames**: 360–3600 per dataset
- **Total time**: 5–30 minutes per dataset

## Data Size

| Component | Typical Size |
|-----------|-------------|
| Single frame | 8–32 MB (EIGER HDF5) |
| Complete dataset | 1–100 GB |
| Processed structure factors | 10–100 MB |
| Final PDB model | 1–10 MB |

## APS BER Applications

- **Structural enzymology**: Enzyme structures relevant to carbon/nitrogen cycling in soils
- **Metalloprotein structures**: Iron-sulfur clusters, manganese centers, zinc fingers
- **Membrane proteins**: Nutrient transporters, efflux pumps
- **Time-resolved**: Enzyme intermediates captured by SSX at room temperature
