# CLAUDE.md — eBERlight Explorer

## Project Identity

eBERlight Explorer — a **personal research / learning project**: an interactive portal over the author's own synchrotron-data-analysis study notes. The visual style is **ANL/APS-inspired for personal familiarity**, not an official ANL or APS property; this repository is **unaffiliated with and unendorsed by** ANL, APS, or DOE.

**Do not deploy publicly.** Original data owners (Sarepy, TomoBank, GMOS, etc.) have *not* been consulted regarding any public hosting of this app. Bundled samples are redistributed under their upstream permissive licenses for **local / private use only**. Any commit, design, or feature work assistant performs must respect this scope — never frame the project as an official portal, never wire up production deploy/marketing surfaces, never add language suggesting institutional endorsement.

This repository contains two versioned artifacts:

1. **Notes** (folders `01_program_overview` through `10_interactive_lab`) — the single source of truth for synchrotron data analysis knowledge plus the bundled real sample data for hands-on experimentation.
2. **Explorer** (`explorer/`) — a Streamlit application that reads and presents the notes interactively. The accompanying `experiments/` directory holds noise-mitigation recipes wired to the lab samples.

## Invariants (MUST Rules)

1. **Notes are the single source of truth.** The 10 note folders (`01_program_overview` through `10_interactive_lab`) are canonical content. `explorer/` reads them at runtime and MUST NOT duplicate or mutate note content. `experiments/` references samples in `10_interactive_lab/datasets/` by relative path; it does not copy them.

2. **Every code change must reference a PRD section, user story, or ADR.** Traceability is mandatory — cite the relevant doc ID (e.g., FR-003, US-007, ADR-002) in commit messages or PR descriptions.

3. **Every feature PR must update `docs/05_release/release_notes/` and `CHANGELOG.md`.** No feature ships without a release note entry. Release-note doc_ids follow the scheme `REL-N<MMM>` for `notes-vM.M.M` (e.g. `REL-N100` → `notes-v1.0.0`) and `REL-E<MMM>` for `explorer-vM.M.M` (e.g. `REL-E040` → `explorer-v0.4.0`).

4. **IA, design\_system, and data\_contracts changes require an ADR status update.** If you change information architecture, the design system, or data contracts, create or update the corresponding ADR and set its status accordingly.

5. **Wireframe changes are versioned by filename suffix** (`_v0.1`, `_v0.2`, …). Never overwrite an existing wireframe version; create a new suffixed file instead.

6. **Two independent SemVer streams:** `notes-vX.Y.Z` and `explorer-vX.Y.Z`. Notes and explorer are versioned separately because content velocity differs from application velocity.

7. **All documents carry YAML frontmatter** with these fields:
   ```yaml
   doc_id: <unique identifier, e.g., PRD-001, ADR-003>
   title: <human-readable title>
   status: <draft | proposed | accepted | superseded>
   version: <SemVer, e.g., 0.1.0>
   last_updated: <YYYY-MM-DD>
   supersedes: <doc_id or null>
   related: <list of related doc_ids>
   ```

8. **Status vocabulary:** `draft` | `proposed` | `accepted` | `superseded`. No other values are permitted.

9. **GitHub Pages MUST mirror the Streamlit explorer.** Any change to `explorer/` (pages, components, assets, IA mapping, or note parser) MUST be reflected in the static site generator `scripts/build_static_site.py` in the **same PR**. The generated `site/` directory is built by CI on every push to `main` (see `.github/workflows/pages.yml`) and MUST NOT be hand-edited or committed. See `docs/03_implementation/github_pages_sync.md` for the full sync contract and ADR-007 for rationale.

## Directory Map

```
synchrotron-data-analysis-notes/
├── 01_program_overview/        # Note folder (source of truth)
├── 02_xray_modalities/         # Note folder
├── 03_ai_ml_methods/           # Note folder
├── 04_publications/            # Note folder
├── 05_tools_and_code/          # Note folder
├── 06_data_structures/         # Note folder
├── 07_data_pipeline/           # Note folder
├── 08_references/              # Note folder
├── 09_noise_catalog/           # Note folder
├── 10_interactive_lab/         # Note folder — real sample data for hands-on experiments (ADR-008)
│   ├── datasets/               # Bundled samples grouped by modality
│   ├── LICENSES/               # Verbatim upstream LICENSE files
│   ├── docs/                   # external_data_sources.md
│   ├── models/                 # lazy_download_recipes.yaml (no weights bundled)
│   ├── manifest.yaml           # Machine-readable inventory
│   └── CITATIONS.bib           # BibTeX for bundled + external datasets
├── docs/
│   ├── 00_product/             # Vision, personas, roadmap
│   ├── 01_requirements/        # PRD, user stories, NFRs
│   ├── 02_design/
│   │   ├── wireframes/         # Versioned wireframe files
│   │   │   └── html/           # Static HTML mockups
│   │   └── decisions/          # Architecture Decision Records
│   ├── 03_implementation/      # Setup, coding standards, data contracts
│   ├── 04_testing/             # Test plan, accessibility audit
│   ├── 05_release/
│   │   └── release_notes/      # Per-version release notes
│   ├── 06_meta/                # Glossary, contributing guide
│   └── README.md               # Documentation map
├── explorer/
│   ├── .streamlit/             # Streamlit configuration
│   ├── pages/                  # Multi-page Streamlit pages
│   ├── components/             # Reusable UI components
│   ├── assets/                 # CSS, images
│   ├── lib/                    # Core logic (notes loader, IA mapping)
│   ├── tests/                  # Unit and integration tests
│   └── app.py                  # Streamlit entry point
├── scripts/
│   ├── build_static_site.py    # Generator for GitHub Pages (mirrors explorer/)
│   └── requirements.txt        # Generator deps (kept in sync with explorer)
├── CLAUDE.md                   # This file
├── CHANGELOG.md                # Combined changelog
└── README.md                   # Repository overview
```

## Coding Standards

See `docs/03_implementation/coding_standards.md` for full Python style guide, linting configuration, commit conventions, and PR checklist.

## Design Tokens

See `docs/02_design/design_system.md` for color tokens, typography, spacing, and component specifications. The visual style is **ANL/APS-inspired**, not officially ANL-branded — see ADR-005 + ADR-010 for the personal-research framing and the WCAG-driven secondary-color darkening.

## How to Run the Explorer Locally

```bash
# 1. Clone the repository
git clone https://github.com/Denny-Hwang/synchrotron-data-analysis-notes.git
cd synchrotron-data-analysis-notes

# 2. Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r explorer/requirements.txt

# 4. Run the app
streamlit run explorer/app.py

# The app opens at http://localhost:8501
```
