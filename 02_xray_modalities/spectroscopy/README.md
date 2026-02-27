# X-ray Absorption Spectroscopy (XAS / XANES / EXAFS)

## Principles

**X-ray Absorption Spectroscopy (XAS)** probes the local atomic environment and chemical
state of specific elements by measuring how X-ray absorption varies as a function of
photon energy near an element's absorption edge.

### Physical Basis
1. When X-ray energy matches an inner-shell electron binding energy → **absorption edge**
2. The ejected photoelectron wave scatters off neighboring atoms
3. Interference between outgoing and backscattered waves modulates absorption
4. The resulting oscillations encode information about:
   - **Oxidation state** (edge position)
   - **Coordination geometry** (near-edge fine structure)
   - **Bond distances** (extended oscillations)
   - **Coordination numbers** (oscillation amplitudes)

### XAS Regions

```
                 ┌── XANES ──┐┌──── EXAFS ────────────────┐
                 │            ││                            │
Absorption │    ╱╲          │╱╲ ╱╲ ╱╲                      │
           │   ╱  ╲         ╱  ╲╱  ╲╱  ╲ ...              │
           │  ╱    ╲───────╱                               │
           │ ╱                                             │
           │╱                                              │
           └──────────────────────────────────────────────→
                     Energy (eV) →
              Edge    -50 to +50 eV    +50 to +1000 eV
```

| Region | Energy Range | Information | Sensitivity |
|--------|-------------|-------------|-------------|
| **Pre-edge** | -20 to 0 eV | Electronic transitions, symmetry | Oxidation state |
| **XANES** | -10 to +50 eV | Oxidation state, coordination geometry | Chemical species |
| **EXAFS** | +50 to +1000 eV | Bond distances, coordination numbers | Local structure |

### Variants

| Variant | Description | Spatial Resolution |
|---------|-------------|-------------------|
| **Bulk XAS** | Transmission/fluorescence, mm beam | Bulk average |
| **µ-XANES** | Focused beam, spatial mapping | 1–20 µm |
| **µ-EXAFS** | Focused beam, full EXAFS | 1–20 µm |
| **XANES imaging** | Energy stack at each pixel | 1–20 µm |
| **RIXS** | Resonant inelastic X-ray scattering | 1–20 µm |

## Experimental Setup

```
Monochromatic X-ray beam (tunable energy)
    │
    ├─→ I₀ (incident flux monitor, ion chamber)
    │
    ▼
Sample
    │
    ├─→ I₁ (transmitted flux, ion chamber) → Transmission mode
    │
    └─→ Fluorescence detector (SDD/Vortex) → Fluorescence mode
            (for dilute samples, preferred for environmental)
```

### eBERlight Beamlines

| Beamline | Energy Range | Mode | Specialty |
|----------|-------------|------|-----------|
| **9-BM** | 2.1–23 keV | Transmission, fluorescence | Tender X-ray (P, S, Cl, K, Ca edges) |
| **20-BM** | 2.7–32 keV | Transmission, fluorescence | Heavy metals (As, Pb, Hg, U, etc.) |
| **25-ID** | 5–28 keV | RIXS, µ-XANES | High-resolution, microbeam |

## Data Collection Process

1. **Energy calibration**: Measure known reference compound at element edge
2. **Pre-edge scan**: Coarse energy steps below the edge (background)
3. **Edge scan**: Fine energy steps through the edge region (XANES)
4. **Post-edge scan**: Progressively coarser steps in k-space (EXAFS)
5. **Multiple scans**: Average 3-20 scans for improved signal-to-noise

### Typical Parameters (Bulk XAS)

| Parameter | XANES Only | Full EXAFS |
|-----------|-----------|------------|
| **Energy range** | Edge ± 100 eV | Edge - 200 to + 1000 eV |
| **Energy steps** | 0.25–0.5 eV (edge), 5 eV (pre/post) | + 0.05 Å⁻¹ in k-space |
| **Points** | 200–500 | 500–1000 |
| **Time per point** | 1–5 s | 1–10 s |
| **Total scans** | 3–5 | 5–20 |
| **Total time** | 30–60 min | 1–6 hr |

## Data Scale

| Component | Typical Size |
|-----------|-------------|
| Single XAS spectrum | 10–100 KB |
| Averaged spectrum + metadata | 1 MB |
| µ-XANES imaging stack | 0.5–10 GB |
| Full experiment dataset | 0.1–10 GB |

## eBERlight Applications

- **Soil science**: Fe/Mn redox speciation, P bonding environment, S species identification
- **Environmental chemistry**: As(III)/As(V) speciation, Cr(III)/Cr(VI), U speciation in waste
- **Plant science**: Metal tolerance mechanisms, nutrient speciation in root nodules
- **Geochemistry**: Mineral phase identification, element oxidation state mapping
- **Microbiology**: Metal uptake and storage mechanisms in bacteria
