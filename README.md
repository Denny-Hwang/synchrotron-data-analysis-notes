# Synchrotron Data Analysis Notes

Personal study notes on synchrotron X-ray data analysis and AI/ML methods at Argonne National Laboratory's Advanced Photon Source (APS).

## About

This repository documents the DOE BER (Biological and Environmental Research) program's integrated X-ray capabilities at the upgraded APS facility, which delivers **500x brighter X-rays** since the APS-U completion in 2024. It covers:

- **6 X-ray modalities** — Tomography, XRF Microscopy, Ptychography, Spectroscopy, Crystallography, Scattering
- **14 AI/ML methods** — organized across 5 categories (Segmentation, Denoising, Reconstruction, Autonomous Experiments, Multimodal Integration)
- **14 paper reviews** — detailed analyses of key publications in synchrotron AI/ML
- **7 open-source tools** — reverse-engineered architectures for ROI-Finder, TomocuPy, TomoPy, MAPS, MLExchange, Bluesky/EPICS
- **HDF5 data schemas** — with EDA notebooks and sample data links
- **End-to-end data pipeline** — from acquisition to storage with architecture diagrams
- **29 noise/artifact types** — cataloged with detection code, before/after examples, and symptom-based troubleshooter

## Documentation

![notes-v0.1.0](https://img.shields.io/badge/notes-v0.1.0-blue)
![explorer-v0.2.0](https://img.shields.io/badge/explorer-v0.2.0-green)

Full project documentation is available in [`docs/README.md`](docs/README.md), including:

- **Product layer**: Vision, personas, roadmap, PRD, user stories, NFRs
- **Design layer**: Information architecture, design system, wireframes
- **Architecture decisions**: 6 ADRs covering framework, IA, schema, tokens, versioning
- **Implementation**: Setup guide, coding standards, data contracts
- **Testing**: Test plan, accessibility audit checklist

### Wireframe Preview

Static HTML wireframe mockups are published via GitHub Pages:
- [Landing page](docs/02_design/wireframes/html/landing_v0.1.html)
- [Section page](docs/02_design/wireframes/html/section_v0.1.html)
- [Tool detail page](docs/02_design/wireframes/html/tool_v0.1.html)

## Explorer (Redesigned)

The **eBERlight Explorer** (`explorer/`) is a redesigned Streamlit portal with ANL-aligned design, 3-cluster IA, and runtime note rendering.

### Running the Explorer

```bash
# 1. Clone the repository
git clone https://github.com/Denny-Hwang/synchrotron-data-analysis-notes.git
cd synchrotron-data-analysis-notes

# 2. Install dependencies
pip install -r explorer/requirements.txt

# 3. Launch the app
streamlit run explorer/app.py
```

The app opens at `http://localhost:8501` in your browser.

### Explorer Pages

| Page | Description |
|------|-------------|
| **Home** | Overview dashboard with statistics and quick navigation guides |
| **Knowledge Graph** | Interactive network visualization of relationships between modalities, methods, tools, and papers |
| **Modalities** | Explore 6 X-ray techniques with specs, beamlines, and related AI methods |
| **AI/ML Methods** | Browse 14 methods organized by category with detailed documentation |
| **Publications** | Archive of 14 paper reviews with TL;DR summaries and workflow diagrams |
| **Tools** | Catalog of 7 open-source tools with architecture analysis and pros/cons |
| **Pipeline** | Visual walkthrough of the end-to-end data pipeline (acquisition → storage) |
| **Data Structures** | HDF5 schemas, EDA guides, and data scale analysis for pre/post APS-U |

### Difficulty Levels

The Explorer provides three levels of detail on each page:

- **L0 (Overview)** — High-level summaries for quick orientation
- **L1 (Intermediate)** — Detailed content with tables and comparisons
- **L2 (Deep Dive)** — Full technical details, code examples, and architecture diagrams

## Repository Structure

```
synchrotron-data-analysis-notes/
├── 01_program_overview/     # BER program mission, APS facility, 15 beamlines, partners
├── 02_xray_modalities/      # 6 X-ray techniques: principles, data formats, AI/ML applications
├── 03_ai_ml_methods/        # AI/ML taxonomy: segmentation, denoising, reconstruction, autonomous
├── 04_publications/         # 14 paper reviews with detailed analysis and key findings
├── 05_tools_and_code/       # Tool analysis: ROI-Finder, TomocuPy, TomoPy, Bluesky, etc.
├── 06_data_structures/      # HDF5 schemas, EDA notebooks, sample data links
├── 07_data_pipeline/        # End-to-end pipeline: acquisition → streaming → processing → storage
├── 08_references/           # Bibliography (BibTeX), glossary (A-Z), useful links
├── 09_noise_catalog/        # Noise/artifact catalog: detection, examples, troubleshooter
└── eberlight-explorer/      # Streamlit web app for interactive exploration
```

## Quick Start

### New to synchrotron science?

1. Start with [`01_program_overview/`](01_program_overview/) to understand the BER program and APS facility
2. Explore [`02_xray_modalities/`](02_xray_modalities/) to learn about X-ray measurement techniques
3. Check [`08_references/glossary.md`](08_references/glossary.md) for terminology

### Want to apply AI/ML?

1. Browse [`03_ai_ml_methods/`](03_ai_ml_methods/) for the method taxonomy
2. Read [`04_publications/`](04_publications/) for detailed paper reviews
3. See [`05_tools_and_code/`](05_tools_and_code/) for tool implementations

### Need hands-on code?

1. Run the Jupyter notebooks in [`05_tools_and_code/roi_finder/notebooks/`](05_tools_and_code/roi_finder/notebooks/)
2. Explore HDF5 data with [`06_data_structures/hdf5_structure/notebooks/`](06_data_structures/hdf5_structure/notebooks/)
3. Try the EDA notebooks in [`06_data_structures/eda/notebooks/`](06_data_structures/eda/notebooks/)

### Dealing with noisy data?

1. **Know the modality?** Browse [`09_noise_catalog/`](09_noise_catalog/) by technique
2. **See something wrong but unsure?** Use the [Symptom-Based Troubleshooter](09_noise_catalog/troubleshooter.md)
3. **Quick reference**: Check the [Summary Table](09_noise_catalog/summary_table.md) for all 29 types at a glance

### Just want to browse?

Launch the [eBERlight Explorer](#eberlight-explorer-interactive-web-app) for a visual, interactive experience.

## Key Resources

| Resource | Link |
|----------|------|
| APS BER Program | [eberlight.aps.anl.gov](https://eberlight.aps.anl.gov) |
| APS Facility | [aps.anl.gov](https://www.aps.anl.gov) |
| APS GitHub Organization | [github.com/AdvancedPhotonSource](https://github.com/AdvancedPhotonSource) |
| ROI-Finder | [github.com/arshadzahangirchowdhury/ROI-Finder](https://github.com/arshadzahangirchowdhury/ROI-Finder) |
| TomoPy | [tomopy.readthedocs.io](https://tomopy.readthedocs.io) |
| TomocuPy | [github.com/nikitinvv/tomocupy](https://github.com/nikitinvv/tomocupy) |
| Bluesky Project | [blueskyproject.io](https://blueskyproject.io) |
| MLExchange | [mlexchange.als.lbl.gov](https://mlexchange.als.lbl.gov) |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
