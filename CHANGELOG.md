# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-27

### Added

#### Phase 1: Repository Scaffolding
- Directory structure for 8 main sections
- Root README with navigation guide
- MIT License

#### Phase 2: Program Overview (01_program_overview/)
- eBERlight mission, history, and research domains
- APS facility specs and APS-U upgrade details
- 15 beamline profiles organized by technique
- Partner facility descriptions (EMSL, JGI, NEON, HFIR, ALCF, CNM)
- 7 research domain mappings with X-ray technique connections

#### Phase 3: X-ray Modalities (02_xray_modalities/)
- 6 modality directories: crystallography, tomography, XRF microscopy, spectroscopy, ptychography, scattering
- Data format specifications with HDF5 schema details for each modality
- AI/ML method summaries per modality
- 21 documentation files total

#### Phase 4: AI/ML Methods Taxonomy (03_ai_ml_methods/)
- Image segmentation: U-Net variants, XRF cell segmentation, tomography 3D segmentation
- Denoising: TomoGAN, Noise2Noise, deep residual XRF enhancement
- Reconstruction: TomocuPy GPU acceleration, PtychoNet CNN phase retrieval, INR for dynamic data
- Autonomous experiments: ROI-Finder, Bayesian optimization, AI-NERD
- Multimodal integration: XRF+ptychography, CT+XAS correlation, optical-X-ray registration

#### Phase 5: Publication Archive (04_publications/)
- Paper review template
- eBERlight publications overview
- 10 detailed paper reviews covering ROI-Finder, TomoGAN, XRF GMM, AI-NERD, PtychoNet, AI@Edge ptychography, deep residual XRF, full-stack DL tomo, real-time µCT HPC, AI@ALS workshop

#### Phase 6: Tools & Code Reverse Engineering (05_tools_and_code/)
- ROI-Finder: reverse engineering, pros/cons, reproduction guide, 4 Jupyter notebooks
- TomocuPy: architecture analysis, GPU kernel details, benchmarks vs TomoPy
- TomoPy: module structure, reconstruction algorithms
- MAPS software: workflow analysis
- MLExchange: microservice architecture, pros/cons
- APS GitHub repos: catalog of key repositories
- Bluesky/EPICS: architecture overview, RunEngine, ophyd, document model

#### Phase 7: Data Structures & EDA (06_data_structures/)
- HDF5 schemas: XRF (MAPS format), tomography (Data Exchange), ptychography (CXI)
- HDF5 exploration and visualization notebooks
- Data scale analysis: pre- vs post-APS-U projections
- EDA guides: XRF, tomography, spectroscopy with code examples
- 3 EDA Jupyter notebooks
- Sample data directory with links to TomoBank, CXIDB, PDB

#### Phase 8: Data Pipeline Architecture (07_data_pipeline/)
- Acquisition layer: detector specs, EPICS IOC integration
- Streaming: ZMQ, PV Access, Globus transfer
- Processing: preprocessing → reconstruction → denoising → segmentation
- Analysis: ML inference, visualization (Jupyter, Streamlit, Napari)
- Storage: 3-tier architecture (GPFS/Petrel/HPSS), NeXus compliance
- Architecture diagrams (5 Mermaid flowcharts)

#### Phase 9: References & Utilities (08_references/)
- BibTeX bibliography with 20+ entries
- Glossary of synchrotron science terms (A-Z)
- Useful links: eBERlight, APS, partner facilities, tools, datasets, tutorials
