# APS Facility Overview & APS-U Upgrade

## Advanced Photon Source (APS)

The **Advanced Photon Source (APS)** at Argonne National Laboratory is one of the world's
most productive synchrotron X-ray light sources. Located in Lemont, Illinois, the APS
serves thousands of researchers annually across diverse scientific disciplines.

### Facility Specifications

| Parameter | Pre-Upgrade (3rd Gen) | Post-Upgrade APS-U (4th Gen) |
|-----------|----------------------|------------------------------|
| **Ring Energy** | 7.0 GeV | 6.0 GeV |
| **Circumference** | 1,104 m | 1,104 m |
| **Lattice Type** | Double-bend achromat (DBA) | Multi-bend achromat (MBA) |
| **Emittance** | 3.1 nm·rad | 42 pm·rad |
| **Brightness Increase** | Baseline | Up to 500× |
| **Injection** | Top-up | Multi-bunch swap-out injection |
| **Beamlines** | 68 | 68+ (with enhanced capabilities) |
| **Current** | 100 mA | 200 mA (design) |

## APS-U: The Upgrade

### Overview

The **APS Upgrade (APS-U)** is an **$815 million** project that replaced the original
storage ring with a state-of-the-art **multi-bend achromat (MBA) lattice**. This upgrade
transformed APS from a 3rd-generation to a **4th-generation synchrotron**, delivering
coherent X-ray beams up to **500× brighter** than the original source.

### Key Technical Improvements

#### 1. Multi-Bend Achromat (MBA) Lattice
- Replaces the original double-bend achromat design
- Uses 7 bends per sector (vs. 2 previously) to reduce horizontal emittance
- Achieves **42 pm·rad emittance** — approaching the diffraction limit
- Results in dramatically increased coherent flux

#### 2. Multi-Bunch Swap-Out Injection
- Novel injection scheme replacing traditional top-up injection
- Entire bunch trains are swapped rather than individual bunches "topped up"
- Enables more stable, uniform beam delivery
- Supports timing-mode experiments with flexible bunch patterns

#### 3. Enhanced Coherence
- Coherent fraction increases from ~0.1% to ~50% at hard X-ray energies
- Enables coherent imaging techniques (ptychography, CDI) as routine methods
- Dramatically improves phase-contrast imaging capabilities

### Upgrade Timeline

| Date | Event |
|------|-------|
| 2019 | APS-U project receives CD-3 approval (start of construction) |
| 2023 Apr | APS ceases user operations for installation |
| 2023–2024 | Storage ring replacement and installation |
| 2024 Q2 | First stored beam in new ring |
| 2024 Q3 | Beam commissioning and first light at beamlines |
| 2024 Q4 | Early user operations resume |
| 2025 | Full user operations with enhanced capabilities |

## Post-Upgrade Data Challenges

The APS-U creates unprecedented data challenges due to dramatically increased data rates:

### Data Volume Increase

| Modality | Pre-Upgrade | Post-Upgrade | Factor |
|----------|-------------|--------------|--------|
| **Tomography** | 10-50 GB/scan | 100-500 GB/scan | 10× |
| **Ptychography** | 5-50 GB/scan | 50-500 GB/scan | 10× |
| **XRF Microscopy** | 1-10 GB/scan | 10-100 GB/scan | 10× |
| **Crystallography** | 10-100 GB/dataset | 100 GB-1 TB/dataset | 10× |
| **Daily facility total** | ~10 TB/day | ~100+ TB/day | 10× |

### Key Data Challenges

1. **Real-time processing**: Data arrives faster than traditional reconstruction methods can process
2. **Storage**: Annual data volume projected to exceed 100 PB
3. **Transfer**: Need for high-throughput data transfer to computing facilities (ALCF)
4. **AI/ML necessity**: Manual analysis becomes impossible at these data rates
5. **Streaming analysis**: Shift from post-hoc to real-time, streaming analysis paradigms

### Computing Infrastructure

- **On-site**: APS computing cluster for immediate processing
- **ALCF** (Argonne Leadership Computing Facility): Aurora exascale supercomputer
  - Located on same ANL campus
  - Low-latency connection to APS
  - Enables real-time AI/ML inference during experiments
- **Globus**: High-speed data transfer service connecting APS to remote computing resources
- **Edge computing**: FPGA and GPU-based edge devices at beamlines for real-time analysis

## Significance for eBERlight

The APS-U directly benefits eBERlight science through:

- **Higher spatial resolution**: Resolve finer structures in biological/environmental samples
- **Faster data collection**: Enable time-resolved studies of dynamic processes
- **Dose reduction**: Achieve same signal quality with lower radiation dose (critical for biological samples)
- **Coherent imaging**: Routine ptychographic imaging for nanoscale biological structures
- **Multimodal capability**: Sufficient flux to combine multiple X-ray techniques simultaneously
