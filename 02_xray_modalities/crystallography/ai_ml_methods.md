# AI/ML Methods for Crystallography

## Overview

AI/ML is transforming every step of the macromolecular crystallography pipeline, from
crystal detection to structure validation. The most impactful development has been
AlphaFold's revolution of structure prediction, but significant advances are also being
made in experimental data processing.

## ML Problem Types

| Problem | Type | Input | Output |
|---------|------|-------|--------|
| Crystal centering | Object detection | Optical images | Crystal coordinates |
| Indexing | Classification + regression | Diffraction pattern | Space group + cell |
| Resolution estimation | Regression | Diffraction image | Resolution (Å) |
| Phase retrieval | Generative / regression | Structure factors | Electron density |
| Model building | Structure prediction | Sequence + density | Atomic coordinates |
| Ligand identification | Classification | Electron density blob | Ligand identity |

## Methods

### 1. AlphaFold Integration

**AlphaFold** (DeepMind, 2021) has fundamentally changed crystallography by providing
accurate initial models for molecular replacement phasing.

**Workflow**:
```
Protein sequence → AlphaFold prediction → Initial model
                                              │
Experimental data (MTZ) ─────────────────→ Molecular Replacement (PHASER)
                                              │
                                         Phased electron density
                                              │
                                         Refinement → Final structure
```

**Impact**:
- Previously unsolvable structures (no homolog for MR) can now be phased
- Reduces time from data to structure from days/weeks to hours
- ~85% of new PDB depositions now use AlphaFold models

**Limitations**:
- AlphaFold models may have incorrect local conformations
- Multimeric complexes and flexible regions remain challenging
- Model bias: refined structure may retain AlphaFold errors if not carefully validated

### 2. Automated Crystal Centering

ML-based crystal detection from optical microscope images at the beamline.

**Approaches**:
- **CNN-based detection**: Classify loop regions as crystal/no-crystal
- **Segmentation**: U-Net for crystal boundary detection
- **Diffraction-based**: Use X-ray raster scan + CNN to identify best diffraction spots

**Architecture**: Typically ResNet or EfficientNet backbone + detection head
**Training data**: Thousands of annotated crystal images from beamline cameras
**Speed**: <1 second per sample (real-time compatible)

### 3. Auto-Indexing and Processing

**DIALS (Diffraction Integration for Advanced Light Sources)**:
- Spot finding with adaptive thresholding
- ML-enhanced indexing for challenging cases (multi-lattice, weak diffraction)
- Bayesian approaches to space group determination

**xia2**: Automated processing pipeline that selects optimal parameters
- Decision-tree logic for choosing processing strategy
- Anomaly detection for radiation damage and ice rings

### 4. Serial Crystallography Data Processing

SSX presents unique ML challenges due to the large number of partial observations:

- **Hit finding**: CNN classifies diffraction images as crystal/blank (filter 10-90% blanks)
- **Indexing rate optimization**: ML predicts optimal indexing parameters per image
- **Merging**: Expectation-maximization algorithms for optimal intensity merging
- **Isomorphism classification**: Cluster crystals by unit cell to handle polymorphism

### 5. Electron Density Interpretation

- **Automated model building**: ARP/wARP, Buccaneer use statistical methods to trace chains
- **Ligand fitting**: ML classifies electron density blobs into known ligand libraries
- **Water placement**: CNN-based approaches for identifying ordered water molecules
- **Validation**: MolProbity with ML-enhanced outlier detection

## Benchmark Datasets

| Dataset | Description | Source |
|---------|-------------|--------|
| PDB | 200K+ experimental structures | [wwPDB](https://www.wwpdb.org/) |
| AlphaFold DB | 200M+ predicted structures | [alphafold.ebi.ac.uk](https://alphafold.ebi.ac.uk/) |
| SBGrid datasets | Raw diffraction data | [data.sbgrid.org](https://data.sbgrid.org/) |

## Current Limitations & Opportunities

### Limitations
- AlphaFold model bias in molecular replacement needs careful handling
- Limited labeled data for beamline-specific ML (crystal centering, hit finding)
- SSX data processing still computationally expensive for real-time analysis
- Flexible regions and intrinsically disordered proteins remain challenging

### Opportunities
- **Real-time processing**: ML-guided data collection decisions during beam time
- **Microcrystal detection**: Improved detection of sub-10 µm crystals with deep learning
- **Joint refinement**: Combine experimental data with ML predictions in refinement
- **Dynamic structures**: ML methods for extracting conformational ensembles from diffraction data
- **De novo phasing**: Direct phase prediction from diffraction intensities (active research area)
