# X-ray Scattering (SAXS / WAXS / XPCS)

## Principles

**X-ray scattering** techniques probe the structure and dynamics of materials by analyzing
the angular distribution of X-rays scattered by the sample. Unlike crystallography (which
requires long-range order), scattering methods work on disordered, partially ordered, and
amorphous systems.

### Physical Basis
- X-rays scatter from electron density fluctuations in the sample
- Scattering angle relates to structural length scale: d = 2π/q
- Scattering vector: q = (4π/λ)sin(θ), where 2θ is the scattering angle
- Small angles → large structures (nm–µm); wide angles → atomic/molecular (Å–nm)

### Technique Variants

| Technique | q Range (Å⁻¹) | Length Scale | What It Probes |
|-----------|---------------|-------------|----------------|
| **USAXS** | 10⁻⁴ – 0.1 | 0.06–60 µm | Large-scale heterogeneities, voids, particles |
| **SAXS** | 0.001 – 0.5 | 1–600 nm | Nanostructure, protein shape, aggregation |
| **WAXS** | 0.5 – 10 | 0.06–1 nm | Crystal structure, molecular packing |
| **XPCS** | 0.001 – 0.1 | 6–600 nm | Dynamics (diffusion, relaxation, aging) |

### SAXS vs. WAXS

```
                     SAXS              WAXS
                  (small angle)    (wide angle)
                       ↓                ↓
X-ray beam → ┌────────┬────────────────────┐
              │ Sample │→ 2θ < 5°   │ 2θ > 5°  │
              └────────┘    ↓        │    ↓      │
                       2D pattern   │ 2D pattern │
                       (ring-like)  │ (Debye rings)
                            ↓        │     ↓      │
                       I(q) profile │ I(q) profile │
                       Nanostructure│ Atomic/molecular
```

## XPCS: X-ray Photon Correlation Spectroscopy

XPCS is the X-ray analog of dynamic light scattering (DLS):

### Principle
- Coherent X-ray beam produces **speckle pattern** (interference from individual scatterers)
- Speckle pattern fluctuates as scatterers move
- **Temporal autocorrelation** of speckle intensity → dynamics information

```
g₂(q, τ) = <I(q,t) × I(q,t+τ)> / <I(q,t)>²

g₂ = 1 + β|f(q,τ)|²

where f(q,τ) = intermediate scattering function
      β = speckle contrast (coherence factor)
      τ = time delay
```

### APS-U Impact on XPCS
- Coherent flux increases by ~100× → XPCS becomes dramatically more powerful
- Access faster dynamics (microsecond regime)
- Higher q-resolution for smaller structural features
- Enable multispeckle XPCS (2D correlation instead of single-point)

## Experimental Setup

### SAXS/WAXS Setup (12-ID-B)

```
Monochromatic X-ray beam
    │
    ▼
Sample (capillary, flow cell, or thin film)
    │
    ├──→ WAXS detector (close, ~200 mm) → wide-angle pattern
    │
    └──→ SAXS detector (far, 2–10 m) → small-angle pattern
              │
              └── Beamstop (absorbs direct beam)
```

### XPCS Setup (12-ID-E)

```
Coherent X-ray beam (small source, high brightness)
    │
    ▼
Collimation (define coherence volume)
    │
    ▼
Sample
    │
    └──→ Fast area detector (EIGER 500K, 10 kHz+)
              │
              └── Records speckle patterns as function of time
```

### eBERlight Beamlines

| Beamline | Technique | Energy | Detector | Specialty |
|----------|-----------|--------|----------|-----------|
| **12-ID-B** | SAXS/WAXS, GISAXS | 7.9–14 keV | PILATUS 2M/300K | Solution scattering, thin films |
| **12-ID-E** | USAXS/SAXS/WAXS | 10–28 keV | Bonse-Hart + area | Hierarchical structures |
| **12-BM** | XAS, SAXS/WAXS | 4.5–40 keV | Multiple | Combined spectroscopy + scattering |

## Data Collection

### SAXS/WAXS
1. **Calibration**: Measure standard (AgBeh, glassy carbon) for q-calibration
2. **Background**: Measure empty capillary / solvent
3. **Sample measurement**: 0.1 s – 60 s exposure depending on concentration
4. **Concentration series**: Multiple concentrations for protein SAXS
5. **Temperature/time series**: For kinetic or phase behavior studies

### XPCS
1. **Alignment**: Optimize coherence and speckle contrast
2. **Time series**: Continuous acquisition at high frame rate (100–10,000 Hz)
3. **Multiple q-values**: Detector captures dynamics at many q simultaneously
4. **Temperature/field scans**: Probe dynamics as function of external parameters

### Typical Parameters

| Parameter | SAXS (12-ID-B) | XPCS (12-ID-E post APS-U) |
|-----------|----------------|---------------------------|
| **Energy** | 12 keV | 8–12 keV |
| **Beam size** | 120×80 µm | 1–20 µm |
| **Exposure** | 0.1–60 s | 0.1–10 ms per frame |
| **Frames** | 10–100 | 10,000–1,000,000 |
| **q range** | 0.002–10 Å⁻¹ | 0.001–0.1 Å⁻¹ |
| **Sample-det distance** | 0.2–10 m | 2–10 m |

## Data Scale

| Component | Size |
|-----------|------|
| Single SAXS frame (2M detector) | ~4 MB |
| SAXS dataset (100 frames) | ~400 MB |
| XPCS time series (100K frames) | 10–100 GB |
| Processed 1D profiles | ~100 KB each |
| Correlation functions | ~1 MB |

## eBERlight Applications

- **Structural biology**: Protein solution structure and conformational states (SAXS)
- **Soil science**: Soil aggregate nanostructure (USAXS/SAXS)
- **Environmental**: Nanoparticle size distribution and stability in suspension (SAXS)
- **Geochemistry**: Mineral nucleation and growth kinetics (SAXS time-resolved)
- **Microbiology**: Biofilm ultrastructure, bacterial membrane organization
- **Materials dynamics**: Colloidal gel aging, nanoparticle diffusion (XPCS)
