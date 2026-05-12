# Synchrotron Data Analysis Notes & eBERlight Explorer

> **Personal research project — not an official APS / ANL property.**
> This repository is a self-directed learning and study workspace for one
> researcher. It is **not** affiliated with, endorsed by, or operated on
> behalf of Argonne National Laboratory, the Advanced Photon Source, or
> the DOE. The visual style is inspired by APS materials for personal
> familiarity only; no official branding is claimed. The bundled sample
> data is redistributed under its original permissive licenses (see
> `10_interactive_lab/LICENSES/`), but **the original data owners have
> not been consulted regarding any public deployment of this app**, so
> the project is intended for local / private use only — please do not
> host it publicly without first contacting the upstream data sources.

Personal study notes plus a local interactive portal over synchrotron
X-ray data analysis and AI/ML methods, organised around the eBERlight
program at the APS as a learning topic. Written so the same notes can
serve later review as a beamline scientist, a new BER user, or a
computational researcher might each need from one corpus.

![notes-v0.10.0](https://img.shields.io/badge/notes-v0.10.0-blue)
![explorer-v0.8.1](https://img.shields.io/badge/explorer-v0.8.1-green)
[![tests](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/test.yml/badge.svg)](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/test.yml)
[![pages](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/pages.yml/badge.svg)](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/actions/workflows/pages.yml)
![license: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

## Why this exists

Synchrotron data analysis sits at an awkward intersection: domain knowledge
(beamlines, detectors, modalities) collides with fast-moving AI/ML research
(GANs for denoising, INRs for reconstruction, foundation models for
autonomous experiments). Useful work usually requires reading a dozen
papers, three GitHub README files, and asking someone in the lab. This
repo is the result of that reading, plus a deliberate attempt to make the
knowledge **navigable**, **interactive**, and **runnable**:

- **Navigable** — a 3-cluster information architecture (Discover the
  program · Explore the science · Build and compute) with a Knowledge
  Graph that wires every modality, AI/ML method, paper, tool, and
  Interactive-Lab recipe into one draggable network.
- **Interactive** — an **Interactive Lab** where you load real bundled
  research data, pick a noise-mitigation algorithm, drag a slider, and
  watch the before/after with PSNR/SSIM plus a difference map.
- **Runnable** — every recipe is a pure function in `experiments/` that
  CI verifies end-to-end against bundled samples. No vendor magic, no
  hidden state.

## What's in this repository

Two independent versioned artifacts (per [ADR-006](docs/02_design/decisions/ADR-006.md)):

- **Notes** (10 folders, `01_program_overview` … `10_interactive_lab`) —
  markdown-only knowledge base on the BER program at APS, X-ray
  modalities, AI/ML methods, publications, tools, data structures, the
  end-to-end pipeline, references, the noise-and-artifact catalog, and
  **real bundled sample data for hands-on experimentation**.
- **Explorer** (`explorer/`) — a Streamlit web app that reads the notes
  at runtime (ADR-002 — notes are the single source of truth) and lets
  users navigate them through a 3-cluster IA, plus replay
  noise-mitigation algorithms on the bundled samples interactively.

The notes ship as `notes-vX.Y.Z` and the app ships as `explorer-vX.Y.Z` —
content velocity differs from app velocity. See
[CHANGELOG.md](CHANGELOG.md) for the full release history (currently at
notes-v0.10.0 / explorer-v0.8.1).

## At a glance

| Artifact | Coverage |
|---|---|
| **Notes** | 6 X-ray modalities · 14 AI/ML methods · 14 paper reviews · 7 reverse-engineered tools · HDF5 schemas + EDA · end-to-end data pipeline · **47 noise/artifact types** with symptom-based troubleshooter · 71 real sample files (~135 MB) · 35 inline Mermaid architecture diagrams |
| **Explorer** | 7 Streamlit pages · 3-cluster IA · GitHub Pages static mirror · **draggable vis.js Knowledge Graph** (100+ entities, 120+ edges, 3 layout modes) · **Interactive Lab** with **5 noise-mitigation recipes** + 3-panel before/after/Δ + 🎯 impact card · **Troubleshooter** (11 symptoms × 35 cases) · **Search** + bibliography (LaTeX accents decoded) · L0/L1/L2/L3 progressive disclosure · WCAG 2.1 AA palette |
| **CI** | pytest on Python 3.11 + 3.12 (264 tests) · ruff lint + format · recipe-contract drift protection · static-site rebuild on every push · Mermaid-migration drift catcher |

## Repository layout

```
synchrotron-data-analysis-notes/
├── 01_program_overview/        # BER mission, APS facility, beamlines, partners
├── 02_xray_modalities/         # 6 X-ray techniques: principles, formats, AI/ML uses
├── 03_ai_ml_methods/           # 14 methods across 5 categories (Mermaid diagrams inline)
├── 04_publications/            # 14 paper reviews + Mermaid pipeline diagrams per paper
├── 05_tools_and_code/          # 7 tools: architecture, pros/cons, reproduction
├── 06_data_structures/         # HDF5 schemas + EDA notebooks + scale analysis
├── 07_data_pipeline/           # acquisition → streaming → processing → storage
├── 08_references/              # bibliography (BibTeX), glossary, useful links
├── 09_noise_catalog/           # 47 noise/artifact types + symptom troubleshooter
├── 10_interactive_lab/         # 71 real sample files + ATTRIBUTION + LICENSES + lazy-download recipes (ADR-008)
│
├── explorer/                   # Streamlit app (ADR-001)
│   ├── app.py                  # Landing (3 clickable cluster cards + 4 feature CTAs + hero search)
│   ├── pages/                  # 0_Knowledge_Graph, 1_Discover, 2_Explore,
│   │                           # 3_Build, 4_Experiment, 5_Troubleshooter, 6_Search
│   ├── lib/                    # ia, notes, experiments, model_zoo, cross_refs,
│   │                           # troubleshooter, detail_level, search,
│   │                           # bibliography (with LaTeX accent decoder),
│   │                           # a11y, cluster_page
│   ├── components/             # header, footer, breadcrumb, card, note_view,
│   │                           # visjs_graph (R11 — replaces Plotly+NetworkX)
│   └── tests/                  # pytest suite (264 tests, runs on 3.11 + 3.12)
│
├── experiments/                # Pure-function noise-mitigation recipes (5 bundled)
│   ├── tomography/
│   │   ├── ring_artifact/                # Vo et al. 2018 — sorting filter
│   │   ├── ring_artifact_wavelet/        # Munch et al. 2009 — wavelet-FFT
│   │   └── flatfield_correction/         # I0 normalisation (R11 — most dramatic)
│   └── cross_cutting/
│       ├── cosmic_ray_lacosmic/          # van Dokkum 2001 — L.A.Cosmic
│       └── gaussian_denoise/             # Gaussian / median baseline (R11)
│
├── scripts/
│   ├── build_static_site.py    # GitHub Pages mirror generator (ADR-007)
│   ├── migrate_legacy_mermaid.py  # one-shot migration of 35 diagrams (R9)
│   └── requirements.txt
│
├── docs/                       # Product, design, ADRs, release notes
└── .github/workflows/          # test.yml, pages.yml, lint.yml
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
information architecture (ADR-004) and have been refined across phases
R1 → R15 (currently `explorer-v0.8.0`):

| Page | What it does |
|---|---|
| **Home** (`/`) | Hero + global search bar + 3 **clickable** cluster cards + 4 feature CTAs (KG · Lab · Troubleshooter · Search) |
| **Knowledge Graph** | **Draggable** vis.js network — modality / method / paper / tool / recipe / noise (100+ entities, 120+ edges). Three layout modes (force-directed · hierarchical · freeze), click-to-highlight, double-click to open |
| **Discover the Program** | Notes from `01_program_overview/` + `08_references/` — single dense compare-table with folder-filter chips |
| **Explore the Science** | Notes from `02_xray_modalities/`, `03_ai_ml_methods/`, `04_publications/`, `09_noise_catalog/` — same compare-table view |
| **Build and Compute** | Notes from `05_tools_and_code/`, `06_data_structures/`, `07_data_pipeline/`, `10_interactive_lab/` + recipe gallery |
| **Experiment** | Stepper UI: ① pick recipe → ② pick sample → ③ tune parameters → ④ before / after / **\|Δ\|** + 🎯 Impact card with PSNR/SSIM win-loss banner |
| **Troubleshooter** | 11 symptom categories × 35 differential cases with severity, conditions, before/after images, ▶ Run-experiment links (one click into the Lab with the matching recipe pre-selected) |
| **Search** | Global full-text search (TF-IDF, title boost ×2, prefix match, "did you mean" suggestions) + filterable BibTeX bibliography with LaTeX accents decoded to Unicode |

Every note page also exposes a **Detail Level** pill row (L0 Overview · L1
Sections · L2 Details · L3 Source), a **📑 Tabs** toggle that auto-splits
the body at H2 headings, an in-page **TOC**, **prev / next** navigation,
a **🔗 Copy permalink** button, and an automatic **Notebooks** section
when a folder ships ``*.ipynb`` files. Architecture diagrams in note
markdown render live as **Mermaid** flowcharts.

### Try the Interactive Lab

The Experiment page exposes 5 noise-mitigation recipes from prior
research. Each recipe ships with a **3-card narrative** above the
parameter sliders (⚠️ what was wrong · 🛠️ how we fix it · 👀 what to
observe) and a **3-panel before/after/Δ** below them, so the impact
story is visible without flicking your eyes back and forth:

| Recipe | Method | Sample data |
|---|---|---|
| **Ring artifact — sorting filter** | Vo et al. 2018, *Optics Express* | Sarepy sinograms (real µCT, multiple stripe types) |
| **Ring artifact — wavelet-FFT** | Munch et al. 2009, *Optics Express* | Same Sarepy sinograms — pedagogical "two algorithms, one input" |
| **Flat-field correction** | The textbook I0 normalisation step | `flatfield_correction` dataset — most dramatic |Δ\| panel |
| **Cosmic ray / zinger — L.A.Cosmic** | van Dokkum 2001, *PASP* | Real Gemini Multi-Object Spectrograph CCD frame |
| **Classical denoise (Gaussian / median)** | Baseline every paper compares to | Sarepy sinograms + GMOS frame — same UI shows both |

All inputs are **real published research data** bundled under
permissive licenses (Apache-2.0, BSD-3, MIT, LGPL). See
[`10_interactive_lab/README.md`](10_interactive_lab/README.md) for the
full inventory and
[`10_interactive_lab/docs/external_data_sources.md`](10_interactive_lab/docs/external_data_sources.md)
for larger datasets you can plug in via `pooch.retrieve(...)`.

### Browse without Python (GitHub Pages)

A read-only static mirror of the Explorer is published to GitHub Pages
on every push to `main`. The Build cluster page on the mirror includes
a recipe gallery; the four interactive surfaces (Knowledge Graph /
Interactive Lab / Troubleshooter / Search) ship as stub pages that
point readers to `streamlit run explorer/app.py`. See
[ADR-007](docs/02_design/decisions/ADR-007.md) and
[`docs/03_implementation/github_pages_sync.md`](docs/03_implementation/github_pages_sync.md).

### Read the notes directly

You can browse the markdown straight from the GitHub UI — no Python
needed. The most useful entry points:

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

Full project documentation lives in [`docs/`](docs/) — see
[`docs/README.md`](docs/README.md) for the index. Key entry points:

- **Product**: [vision](docs/00_product/vision.md) ·
  [personas](docs/00_product/personas.md) ·
  [roadmap](docs/00_product/roadmap.md)
- **Requirements**: [PRD](docs/01_requirements/PRD.md) (FR-001 … FR-022) ·
  [user stories](docs/01_requirements/user_stories.md)
  (US-001 … US-016) · [NFRs](docs/01_requirements/non_functional.md)
- **Architecture decisions**: 9 ADRs at
  [`docs/02_design/decisions/`](docs/02_design/decisions/) — Streamlit
  choice (ADR-001), notes-as-SoT (ADR-002), frontmatter (ADR-003),
  3-cluster IA (ADR-004), design tokens (ADR-005), dual SemVer (ADR-006),
  Pages mirror (ADR-007), Interactive Lab (ADR-008), legacy
  deprecation (ADR-009)
- **Implementation**: [coding standards](docs/03_implementation/coding_standards.md) ·
  [data contracts](docs/03_implementation/data_contracts.md) ·
  [Pages sync contract](docs/03_implementation/github_pages_sync.md)
- **Release notes** (per-version) at
  [`docs/05_release/release_notes/`](docs/05_release/release_notes/):
  - `notes-v0.10.0` — Section 10 Interactive Lab
  - `explorer-v0.5.0` (REL-E050) — parity restoration R1 → R7
  - `explorer-v0.6.0` (REL-E060) — final feature parity (R9)
  - `explorer-v0.6.1` (REL-E061) — first-impression UX polish (R10)
  - `explorer-v0.7.0` (REL-E070) — vis.js KG + Lab impact + 5 recipes (R11)
  - `explorer-v0.7.1` (REL-E071) — bug fixes from user review (R12)
  - `explorer-v0.8.0` (REL-E080) — senior-review polish: tone reframing,
    routing/CSS-token unification, onboarding + cluster orientation,
    glossary auto-link, smoke tests (R15)
  - `explorer-v0.8.1` (REL-E081) — re-review follow-ups: glossary
    cross-segment fix, regex caching, keyboard focus, layout-toggle,
    related-views, Lab replay-only banner, tablet + dark-mode tokens (R15.1)
- **Glossary & contributing**: [`docs/06_meta/`](docs/06_meta/)

## Engineering principles

- **Notes are the single source of truth** ([ADR-002](docs/02_design/decisions/ADR-002.md)).
  The explorer parses note frontmatter + folder structure at runtime.
  No YAML catalogs, no duplicated content. Architecture diagrams live
  inside the note markdown as ` ```mermaid ` blocks.
- **Two SemVer streams** ([ADR-006](docs/02_design/decisions/ADR-006.md)).
  `notes-vX.Y.Z` and `explorer-vX.Y.Z` advance independently because
  content velocity differs from application velocity.
- **Every feature PR carries a release-note entry**
  ([CLAUDE.md invariant #3](CLAUDE.md)). Drift between code and docs
  is caught at PR review.
- **Pages mirror is part of the contract**
  ([ADR-007](docs/02_design/decisions/ADR-007.md), invariant #9). Any
  change to `explorer/` is reflected in the static-site generator in
  the same PR.
- **Recipes are pure functions**. CI runs every bundled recipe end-to-end
  against the bundled samples; the recipe-contract test asserts every
  `recipe.yaml` parses cleanly, every function path resolves, every
  metric name is known, and every `noise_catalog_ref` exists.
- **License safety** ([ADR-008](docs/02_design/decisions/ADR-008.md)).
  No pretrained model weights are bundled; all model downloads are
  deferred to runtime via `pooch.retrieve(...)` with hash verification,
  and the user sees the upstream license string before any download
  begins.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branch-naming convention,
ADR process, and PR checklist. Bug reports and feature ideas:
[open an issue](https://github.com/Denny-Hwang/synchrotron-data-analysis-notes/issues).

For security issues, see [SECURITY.md](SECURITY.md) — please **do not**
open public issues for vulnerabilities.

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

Bundled sample data in
[`10_interactive_lab/datasets/`](10_interactive_lab/datasets/) is
redistributed under the upstream licenses preserved verbatim in
[`10_interactive_lab/LICENSES/`](10_interactive_lab/LICENSES/)
(Apache-2.0, BSD-3, MIT, LGPL-2.1+). Each dataset folder ships an
`ATTRIBUTION.md` with the original author, citation, and the exact
upstream commit we mirrored.

This work acknowledges support from the U.S. Department of Energy,
Office of Science, Office of Biological and Environmental Research
under Contract No. DE-AC02-06CH11357.
