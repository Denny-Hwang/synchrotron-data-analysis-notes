# Changelog

All notable changes to this project will be documented in this file.

This project uses two independent SemVer streams per ADR-006:
- `notes-vX.Y.Z` — content in the note folders
- `explorer-vX.Y.Z` — the explorer application

## [explorer-0.4.0] - 2026-05-05

### Added
- `explorer/pages/4_Experiment.py` — Interactive Lab page with auto-generated parameter widgets, side-by-side before/after display, and PSNR/SSIM metrics vs clean reference (ADR-008)
- `explorer/lib/experiments.py` — Recipe loader, sample loader (TIFF/NumPy/FITS, multi-extension FITS), pure-function pipeline dispatch, metrics
- `explorer/lib/model_zoo.py` — Lazy-download infrastructure consuming `10_interactive_lab/models/lazy_download_recipes.yaml`. `pooch`-based hash-verified fetch + Hugging Face `snapshot_download`. License is shown to the user before any download.
- `experiments/` directory with recipe schema (`experiments/README.md`) and two bundled recipes:
  - `experiments/tomography/ring_artifact/` — Vo et al. (2018) sorting-based stripe removal
  - `experiments/cross_cutting/cosmic_ray_lacosmic/` — van Dokkum (2001) L.A.Cosmic via `astroscrappy.detect_cosmics`
- Landing-page CTA card pointing to the Interactive Lab (`app.py`)
- `10_interactive_lab` added to `FOLDER_TO_CLUSTER` mapping → Build cluster (`explorer/lib/ia.py`)
- `explorer/tests/test_experiments.py` — 20 tests covering recipe parsing, function resolution, pipeline dispatch with type coercion, metrics, end-to-end run on bundled samples, **and CI-quality recipe-contract validation** (every `recipe.yaml` parses, every `manifest_path` exists, every `function` resolves, every metric is known, every `noise_catalog_ref` exists)
- `explorer/tests/test_model_zoo.py` — 8 offline tests; caught a real omission: `topaz_denoise_unet_3d` and `cryodrgn` had no `license_warning` despite GPL licenses (now fixed)
- `numpy`, `scipy`, `scikit-image`, `tifffile`, `astropy`, `astroscrappy`, `pooch` pinned in `explorer/requirements.txt`
- `docs/05_release/release_notes/explorer-v0.4.0.md`

### Changed
- `explorer/lib/ia.py` Build cluster description mentions the Interactive Lab
- `explorer/tests/test_ia.py` updated to expect 10 note folders
- `10_interactive_lab/models/lazy_download_recipes.yaml` — added missing `license_warning` to GPL entries (Topaz 3D, CryoDRGN)
- The Pages mirror picks up `10_interactive_lab/` automatically (no generator change needed); interactive pipelines remain Streamlit-only per ADR-007 / invariant #9

## [notes-0.10.0] - 2026-05-05

### Added
- New section `10_interactive_lab/` — real sample data for hands-on noise mitigation experiments (ADR-008)
- 71 sample files (~135 MB) bundled across 6 modalities: tomography (ring artifact, neutron CT, flatfield), XRF spectra and PyXRF configs, spectroscopy (EXAFS + FEFF + Athena), scattering/diffraction calibrants, cross-cutting (cosmic ray FITS)
- 8 `ATTRIBUTION.md` files with YAML frontmatter declaring upstream URL, pinned commit SHA, authors, license, and required citation
- 8 verbatim upstream LICENSE files in `10_interactive_lab/LICENSES/` (Apache-2.0, BSD-Argonne, MIT, LGPL-2.1+, BSD-3)
- `10_interactive_lab/manifest.yaml` — machine-readable inventory consumed by the planned Streamlit Lab page
- `10_interactive_lab/CITATIONS.bib` — 19 BibTeX entries for bundled and external datasets
- `10_interactive_lab/docs/external_data_sources.md` — curated atlas of bigger / lazy-load datasets (TomoBank, EMPIAR, CXIDB, AAPM, etc.) with download recipes, license rules, citation strings, and a research-ethics reminder
- `10_interactive_lab/models/README.md` and `lazy_download_recipes.yaml` — registry for native synchrotron models (TomoGAN, Topaz, CryoDRGN, edgePtychoNN) and Hugging Face Hub baselines (NAFNet, SwinIR, Swin2SR)
- `docs/02_design/decisions/ADR-008.md` — accepts a tenth note folder, extends the Build cluster (ADR-004), and documents free-tier-only constraints (no Git LFS, no file > 100 MB, lazy-download for heavy weights/data)
- `docs/05_release/release_notes/notes-v0.10.0.md`

### Notes
- `09_noise_catalog/` content is unchanged; the Lab consumes its taxonomy
- No pretrained weights are bundled; all model downloads are deferred to runtime via `pooch.retrieve(...)` with hash verification
- Pages mirror (ADR-007, invariant #9) renders the Lab's markdown but interactive parameter tuning remains Streamlit-only

## [explorer-0.3.0] - 2026-04-21

### Added
- Static HTML mirror of the Streamlit explorer, deployed to GitHub Pages (ADR-007)
- `scripts/build_static_site.py` — generator that reuses `explorer/lib/ia.py`, `explorer/lib/notes.py`, and `explorer/assets/styles.css` to mirror the Streamlit app 1:1
- `.github/workflows/pages.yml` — rebuilds and deploys the site on every push to `main` touching notes, `explorer/**`, the generator, wireframes, or the workflow
- `docs/03_implementation/github_pages_sync.md` (IMPL-002) — sync contract between Streamlit and Pages
- `CLAUDE.md` Invariant #9 — Pages must mirror the Streamlit explorer
- Note-folder images (e.g. `09_noise_catalog/images/*`) are mirrored so markdown image references resolve on the static site
- `.nojekyll`, `404.html`, and a regenerated `wireframes/index.html` on the published site

### Changed
- `.github/workflows/pages.yml` previously published only the 3 design wireframes; now builds and deploys the full explorer mirror plus wireframes
- `CLAUDE.md` directory map updated to include `scripts/`

### Notes
- The generated `site/` directory is git-ignored and must not be committed or hand-edited

## [explorer-0.2.0] - 2026-04-08

### Added
- 3-cluster information architecture mapping (9 folders → 3 clusters) per ADR-004
- Note loader with YAML frontmatter parsing and graceful degradation (ADR-002, ADR-003)
- Controlled vocabulary validation for cluster, modality, beamline (DC-001)
- Cluster landing pages: Discover, Explore, Build
- Card and note view components per DS-001
- Landing page updated with hero + 3 cluster cards (FR-001)
- 12 new tests (IA mapping + note parser), total 16 passing

## [explorer-0.1.0] - 2026-04-08

### Added
- Streamlit theme with ANL-aligned design tokens (ADR-005)
- Header component with logo placeholder and top nav stubs
- Breadcrumb component per IA-001 navigation rules
- DOE acknowledgment footer with Contract No. DE-AC02-06CH11357
- Custom CSS for header, breadcrumb, footer, card, and tag components
- Placeholder landing page ("Hello, eBERlight")
- Smoke tests for all 3 components
- Full documentation suite: CLAUDE.md, product layer, design layer,
  6 ADRs, implementation docs, test plan, glossary, contributing guide

## [1.1.0] - 2026-03-21

### Added

#### Phase 10: Noise Handling Catalog (09_noise_catalog/)
- Dual-mode noise catalog: classification-based browsing + symptom-based troubleshooter
- 29 noise/artifact type documents across 5 categories (tomography, XRF, spectroscopy, ptychography, cross-cutting)
- Symptom-based troubleshooter with ASCII decision trees for all 29 noise types
- Synthetic before/after example image generator (Shepp-Logan phantom based)
- Visual example references for 9+ open-source image sources
- Full summary matrix (summary_table.md) with detection methods and solutions
- Image attribution and regeneration guide

## [1.0.0] - 2026-02-27

### Added

#### Phase 1: Repository Scaffolding
- Directory structure for 8 main sections
- Root README with navigation guide
- MIT License

#### Phase 2: Program Overview (01_program_overview/)
- BER program mission, history, and research domains
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
- BER program publications overview
- 14 detailed paper reviews covering ROI-Finder, TomoGAN, XRF GMM, AI-NERD, PtychoNet, AI@Edge ptychography, deep residual XRF, full-stack DL tomo, real-time µCT HPC, AI@ALS workshop, AI-driven XANES, AlphaFold

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
- Useful links: APS BER program, partner facilities, tools, datasets, tutorials
