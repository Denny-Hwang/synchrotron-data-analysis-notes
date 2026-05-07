# Synchrotron Data Analysis Notes & eBERlight Explorer

Personal study notes plus an interactive portal over synchrotron X-ray data
analysis and AI/ML methods at Argonne National Laboratory's Advanced Photon
Source (APS).

![notes-v0.10.0](https://img.shields.io/badge/notes-v0.10.0-blue)
![explorer-v0.6.0](https://img.shields.io/badge/explorer-v0.6.0-green)
[![tests](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/test.yml/badge.svg)](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/test.yml)
[![pages](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/pages.yml/badge.svg)](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/pages.yml)
![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

## What's in this repository

Two independent versioned artifacts (per [ADR-006](docs/02_design/decisions/ADR-006.md)):

- **Notes** (10 folders, `01_program_overview` … `10_interactive_lab`) — markdown-only knowledge base on the BER program at APS, X-ray modalities, AI/ML methods, publications, tools, data structures, the end-to-end pipeline, references, the noise-and-artifact catalog, and **real bundled sample data for hands-on experimentation**.
- **Explorer** (`explorer/`) — a Streamlit web app that reads the notes at runtime and lets users navigate them through a 3-cluster information architecture, plus replay noise-mitigation algorithms on the bundled samples interactively.

The notes ship as `notes-vX.Y.Z` and the app ships as `explorer-vX.Y.Z` — content velocity differs from app velocity. See [CHANGELOG.md](CHANGELOG.md).

## At a glance

| Artifact | Coverage |
|---|---|
| **Notes** | 6 X-ray modalities · 14 AI/ML methods · 14 paper reviews · 7 reverse-engineered tools · HDF5 schemas + EDA · end-to-end data pipeline · **47 noise/artifact types** with symptom-based troubleshooter · 71 real sample files (~135 MB) |
| **Explorer** | 7 Streamlit pages · 3-cluster IA · GitHub Pages static mirror · **Knowledge Graph** (100+ entities, 120+ edges) · **Interactive Lab** with 3 noise-mitigation recipes · **Troubleshooter** (11 symptoms × 35 cases) · **Search** + bibliography · L0/L1/L2/L3 progressive disclosure · Mermaid diagrams · WCAG 2.1 AA palette |
| **CI** | pytest on Python 3.11 + 3.12 · recipe-contract drift protection · static-site rebuild on every push |

## Repository layout

```
synchrotron-data-analysis-notes/
├── 01_program_overview/        # BER mission, APS facility, beamlines, partners
├── 02_xray_modalities/         # 6 X-ray techniques: principles, formats, AI/ML uses
├── 03_ai_ml_methods/           # 14 methods across 5 categories
├── 04_publications/            # 14 paper reviews + template
├── 05_tools_and_code/          # 7 tools: architecture, pros/cons, reproduction
├── 06_data_structures/         # HDF5 schemas + EDA notebooks + scale analysis
├── 07_data_pipeline/           # acquisition → streaming → processing → storage
├── 08_references/              # bibliography (BibTeX), glossary, useful links
├── 09_noise_catalog/           # 47 noise/artifact types + symptom troubleshooter
├── 10_interactive_lab/         # 71 real sample files + ATTRIBUTION + LICENSES + lazy-download recipes (ADR-008)
│
├── explorer/                   # Streamlit app (ADR-001)
│   ├── app.py                  # Landing (3 cluster cards + 4 feature CTAs)
│   ├── pages/                  # 0_Knowledge_Graph, 1_Discover, 2_Explore,
│   │                           # 3_Build, 4_Experiment, 5_Troubleshooter, 6_Search
│   ├── lib/                    # ia, notes, experiments, model_zoo, cross_refs,
│   │                           # troubleshooter, detail_level, search,
│   │                           # bibliography, a11y, cluster_page
│   ├── components/             # header, footer, breadcrumb, card, note_view
│   └── tests/                  # pytest suite (220 tests)
│
├── experiments/                # Pure-function noise-mitigation recipes
│   └── <modality>/<case>/      # recipe.yaml + pipeline.py
│
├── scripts/
│   ├── build_static_site.py    # GitHub Pages mirror generator (ADR-007)
│   └── requirements.txt
│
├── docs/                       # Product, design, ADRs, release notes
└── .github/workflows/          # test.yml, pages.yml
```

> ⚠️ **Don't run `eberlight-explorer/` by mistake** — that directory is the **legacy** first-generation app and ships a redirect notice. The current product is `explorer/`. See [ADR-009](docs/02_design/decisions/ADR-009.md) for the deprecation plan.

## Quick start

### Run the Explorer locally

```bash
git clone https://github.com/Denny-Hwang/synchrotron-data-analysis-notes.git
cd synchrotron-data-analysis-notes

python -m venv .venv && source .venv/bin/activate          # Linux/macOS
# .venv\Scripts\activate                                    # Windows

pip install -r explorer/requirements.txt
streamlit run explorer/app.py    # ← THIS one. NOT eberlight-explorer/app.py.
```

The app opens at `http://localhost:8501`. The seven pages mirror the
information architecture (ADR-004) and the parity-restored feature set
(`explorer-v0.5.0`):

| Page | What it does |
|---|---|
| **Home** (`/`) | Hero + 3 cluster cards + 4 feature CTAs (KG / Lab / Troubleshooter / Search) |
| **Knowledge Graph** | Plotly + NetworkX cross-reference network — modality / method / paper / tool / recipe / noise (100+ entities, 120+ edges) |
| **Discover the Program** | Notes from `01_program_overview/` + `08_references/` |
| **Explore the Science** | Notes from `02_xray_modalities/`, `03_ai_ml_methods/`, `04_publications/`, `09_noise_catalog/` |
| **Build and Compute** | Notes from `05_tools_and_code/`, `06_data_structures/`, `07_data_pipeline/`, `10_interactive_lab/` + recipe gallery |
| **Experiment** | Pick a recipe → choose a real sample → tune parameters → see before/after with PSNR/SSIM |
| **Troubleshooter** | 11 symptom categories × 35 differential cases with severity, conditions, before/after images, ▶ Run-experiment links |
| **Search** | Global full-text search + filterable BibTeX bibliography with title-boosted relevance |

Every note page also exposes a **Detail Level** pill row (L0 Overview · L1
Sections · L2 Details · L3 Source) and renders Mermaid diagrams inline.

### Try the Interactive Lab

The Experiment page exposes 3 noise-mitigation recipes from prior research:

- **Ring artifact — sorting-based filter** (Vo et al. 2018, *Optics Express*) on Sarepy sinograms
- **Ring artifact — wavelet-FFT** (Munch et al. 2009, *Optics Express*) on the same sinograms — pedagogical "two algorithms, one input"
- **Cosmic ray / zinger — L.A.Cosmic** (van Dokkum 2001, *PASP*) via `astroscrappy` on a real GMOS CCD frame

All inputs are **real published research data** bundled under permissive licenses (Apache-2.0, BSD-3, MIT, LGPL). See [`10_interactive_lab/README.md`](10_interactive_lab/README.md) for the full inventory and [`10_interactive_lab/docs/external_data_sources.md`](10_interactive_lab/docs/external_data_sources.md) for larger datasets you can plug in via `pooch.retrieve(...)`.

### Browse without Python (GitHub Pages)

A read-only static mirror of the Explorer is published to GitHub Pages on every push to `main`. The Build cluster page on the mirror includes a recipe gallery; the four interactive surfaces (Knowledge Graph / Interactive Lab / Troubleshooter / Search) ship as stub pages that point readers to `streamlit run explorer/app.py`. See [ADR-007](docs/02_design/decisions/ADR-007.md) and [`docs/03_implementation/github_pages_sync.md`](docs/03_implementation/github_pages_sync.md).

### Read the notes directly

You can browse the markdown straight from the GitHub UI — no Python needed.
The most useful entry points:

| If you want to … | Start here |
|---|---|
| Understand the BER program at APS | [`01_program_overview/`](01_program_overview/) |
| Pick an X-ray technique for a sample | [`02_xray_modalities/`](02_xray_modalities/) |
| Apply AI/ML to your data | [`03_ai_ml_methods/`](03_ai_ml_methods/) |
| Read the paper reviews | [`04_publications/`](04_publications/) |
| Explore tool internals | [`05_tools_and_code/`](05_tools_and_code/) |
| Diagnose a weird-looking image | [`09_noise_catalog/troubleshooter.md`](09_noise_catalog/troubleshooter.md) |
| Run a noise-mitigation experiment | [`10_interactive_lab/README.md`](10_interactive_lab/README.md) |
| Look up an unfamiliar term | [`08_references/glossary.md`](08_references/glossary.md) |

## Project documentation

Full project documentation lives in [`docs/`](docs/) — see [`docs/README.md`](docs/README.md) for the index. Key entry points:

- **Product**: [vision](docs/00_product/vision.md) · [personas](docs/00_product/personas.md) · [roadmap](docs/00_product/roadmap.md)
- **Requirements**: [PRD](docs/01_requirements/PRD.md) (FR-001 … FR-022) · [user stories](docs/01_requirements/user_stories.md) (US-001 … US-016) · [NFRs](docs/01_requirements/non_functional.md) · [release notes](docs/05_release/release_notes/) (REL-E050 = parity restoration R1 → R7)
- **Architecture decisions**: 9 ADRs at [`docs/02_design/decisions/`](docs/02_design/decisions/) — Streamlit choice (ADR-001), notes-as-SoT (ADR-002), frontmatter (ADR-003), 3-cluster IA (ADR-004), design tokens (ADR-005), dual SemVer (ADR-006), Pages mirror (ADR-007), Interactive Lab (ADR-008), legacy deprecation (ADR-009)
- **Implementation**: [coding standards](docs/03_implementation/coding_standards.md) · [data contracts](docs/03_implementation/data_contracts.md) · [Pages sync contract](docs/03_implementation/github_pages_sync.md)
- **Release notes**: [`docs/05_release/release_notes/`](docs/05_release/release_notes/) — per-version
- **Glossary & contributing**: [`docs/06_meta/`](docs/06_meta/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch-naming convention, ADR
process, and PR checklist. Bug reports and feature ideas: [open an issue](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/issues).

For security issues, see [SECURITY.md](SECURITY.md) — please **do not** open
public issues for vulnerabilities.

## Key external resources

| Resource | Link |
|---|---|
| APS BER Program | [eberlight.aps.anl.gov](https://eberlight.aps.anl.gov) |
| APS Facility | [aps.anl.gov](https://www.aps.anl.gov) |
| APS GitHub | [github.com/AdvancedPhotonSource](https://github.com/AdvancedPhotonSource) |
| TomoBank | [tomobank.readthedocs.io](https://tomobank.readthedocs.io) |
| EMPIAR (cryo-EM data) | [ebi.ac.uk/empiar](https://www.ebi.ac.uk/empiar/) |
| CXIDB (CDI / ptychography data) | [cxidb.org](https://cxidb.org) |
| TomoPy | [tomopy.readthedocs.io](https://tomopy.readthedocs.io) |
| Bluesky Project | [blueskyproject.io](https://blueskyproject.io) |

## License

This project is MIT-licensed — see [LICENSE](LICENSE).

Bundled sample data in [`10_interactive_lab/datasets/`](10_interactive_lab/datasets/) is redistributed under the upstream licenses preserved verbatim in [`10_interactive_lab/LICENSES/`](10_interactive_lab/LICENSES/) (Apache-2.0, BSD-3, MIT, LGPL-2.1+). Each dataset folder ships an `ATTRIBUTION.md` with the original author, citation, and the exact upstream commit we mirrored.

This work acknowledges support from the U.S. Department of Energy, Office of Science, Office of Biological and Environmental Research under Contract No. DE-AC02-06CH11357.
