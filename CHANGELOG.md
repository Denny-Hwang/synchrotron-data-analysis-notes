# Changelog

All notable changes to this project will be documented in this file.

This project uses two independent SemVer streams per ADR-006:
- `notes-vX.Y.Z` — content in the note folders
- `explorer-vX.Y.Z` — the explorer application

## [Unreleased] — legacy hard-redirect

### Fixed
- **`eberlight-explorer/app.py`** is now a hard-redirect page: it shows a prominent "⚠️ This is the deprecated legacy app" banner with the launch command for the current app (`streamlit run explorer/app.py`), then calls `st.stop()` so the old portal never renders. Resolves user confusion where someone running `streamlit run eberlight-explorer/app.py` could not find the **Interactive Lab** / **Experiment** page (those live in `explorer/`, not the legacy app).
- **All 8 legacy pages** (`eberlight-explorer/pages/0_*`, `2_*`, `3_*`, …, `8_*`) now import a shared `_deprecated.render_deprecation_and_stop()` helper right after `st.set_page_config` so any sidebar click also lands on the redirect notice instead of the old content.
- **`README.md`** quick-start has a `# ← THIS one. NOT eberlight-explorer/app.py.` inline comment on the canonical launch command, plus a stronger "⚠️ Don't run `eberlight-explorer/`" caution under the repository layout.

### Added
- **`eberlight-explorer/_deprecated.py`** — small helper module that emits the shared redirect banner. Will be removed alongside the rest of `eberlight-explorer/` at the `notes-v1.0.0` cut (per ADR-009).

## [Unreleased] — review pass 2

### Added
- **Pre-commit + linting** (`.pre-commit-config.yaml`, `pyproject.toml`, `.github/workflows/lint.yml`). Runs `ruff check` (E/W/F/I/B/UP/SIM/RET/PTH/PIE/T20/RUF), `ruff format --check`, and `mypy` (informational). The same suite runs locally via `pre-commit install` and in CI via the new `Lint` workflow on Python 3.12. Ruff configuration is per-file: tests get `T20`/`B`/`E402` leeway, Streamlit pages get `E402` leeway (legitimate `sys.path` setup), the static-site generator gets `T20`/`E402` leeway. Closes P1-9.
- **`explorer/pages/4_Experiment.py`** now offers a **download button** for the processed array — `.npy` (raw float32 for downstream analysis) and `.tiff` (lossless, ImageJ-compatible). Helps users plug Lab outputs into their own pipelines (P2-6).
- **`explorer/tests/test_build_static_site.py`** — 17 new tests for `scripts/build_static_site.py`. Covers `_rel`, `_md_link_rewrite`, `_folder_label`, `_card_html` (incl. HTML-escaping of titles), `_recipe_card_html`, `_recipe_gallery_html` (verifies FR-022), and a full `build()` end-to-end run that asserts every cluster page, the recipe gallery banner, the 404 page, and the `.nojekyll` guard. Closes P1-10.
- 30+ ruff-driven cleanups across `explorer/` and `experiments/` — sorted imports, list spread instead of `+`, explicit `strict=` on every `zip(...)`, raw strings on regex `match=` patterns, dropped unused `noqa`, `pyupgrade` syntactic fixes.

### Changed
- All Python source under `explorer/`, `experiments/`, `scripts/` is now `ruff format`ed (line-length 100, double quotes). Future PRs will keep this clean via the pre-commit hook + Lint workflow.
- `pyproject.toml` introduced as the single tool-config source — `[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`. There is intentionally no `[project]` table — the repo is not a packaged distribution.

## [Unreleased] — review pass 1

### Fixed
- **README.md** rewritten end-to-end to match the current state: 10 note folders (was 8), `explorer/` (was `eberlight-explorer/`), 4 Streamlit pages including the Interactive Lab (was 7 hypothetical pages), 47 noise/artifact types (was 29), version badges `notes-v0.10.0` / `explorer-v0.4.0` (were v0.1.0 / v0.3.0), MIT license disclosure for bundled data, ADR / FR / US cross-links.
- **CLAUDE.md** invariant #1 now correctly states "10 note folders"; "Project Identity" section updated. Resolves the previous 8-vs-10 self-contradiction. Release-note doc_id scheme (`REL-N<MMM>` / `REL-E<MMM>`) documented in invariant #3.
- **`docs/README.md`** — ADR table now lists ADR-008 and ADR-009 (both had been missing despite being accepted).
- **`compute_metrics`** is now NaN/inf safe (`_normalize` coerces non-finite values to 0). Inputs with NaN no longer silently propagate to the metric panel (P1-2).
- **`model_zoo.py` exception handling** — bare `except Exception` replaced with `except (OSError, ValueError, RuntimeError)` so `KeyboardInterrupt`, `SystemExit`, and `MemoryError` propagate. Users can now Ctrl+C a hung download (P1-3).
- **`_parse_parameter`** validates `default ∈ [min, max]`, rejects `min > max`, requires `options` for `select`, and rejects unknown `type`. Bad recipes fail fast at parse time instead of crashing the Streamlit page later (P1-6, P1-7).
- **`resolve_function`** has an explicit `Callable[..., np.ndarray]` return type annotation (P1-8).
- **`requirements.txt`** — added upper bounds to every dependency (e.g. `streamlit>=1.30,<2.0`, `numpy>=1.24,<3.0`). Protects against future major-version breakage (P0-6).
- **GitHub Actions** — pinned action minor versions (`@v4.1.7`, `@v5.1.1`, `@v4.3.6`, `@v5.0.0`, `@v3.0.1`, `@v4.0.5`); switched to `--only-binary=astroscrappy,PyWavelets,scipy,scikit-image` to skip native compilation (drops the apt build-essential step); set `cache-dependency-path` on the Pages workflow for deterministic caching (P0-6, P1-11, P1-4).
- **`.gitignore`** — added `.venv/`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `*.egg-info/`, `.coverage*`, editor / OS noise files (P2-3).

### Added
- **ADR-009** — `Deprecate the legacy eberlight-explorer/ directory`. Documents the deprecation policy and the deletion plan (at `notes-v1.0.0`); the legacy directory now ships a DEPRECATED notice (`eberlight-explorer/README.md`).
- **`explorer/tests/test_lab_integrity.py`** — 21 new CI tests acting as drift protection for `10_interactive_lab/`: every `manifest.yaml` sample path resolves; every `ATTRIBUTION.md` carries the required YAML frontmatter and references a license + citation; `LICENSES/` holds the verbatim upstream files; the lazy-download YAML loads cleanly; `CITATIONS.bib` has BibTeX entries. Closes ADR-008 follow-up #3 (P0-5).
- **`explorer/tests/test_experiments.py`** — 8 new tests for parameter parse-time validation (default-in-range, min-not-greater-than-max, select-needs-options, etc.) and `compute_metrics` NaN/inf handling.
- **`CONTRIBUTING.md`** at the repository root, pointing to the canonical `docs/06_meta/contributing.md` and adding sections for adding recipes / data with research-ethics requirements.
- **`SECURITY.md`** — vulnerability reporting policy with explicit in-scope / out-of-scope items.
- **`CODE_OF_CONDUCT.md`** — Contributor Covenant v2.1.
- **`.github/ISSUE_TEMPLATE/{bug_report,feature_request,config}.yml`** — structured issue templates with surface dropdown, doc-id requirement, and a security-issue redirect.
- **`.github/PULL_REQUEST_TEMPLATE.md`** — checklist enforcing CLAUDE.md invariants #2/#3/#4/#5/#7 and the free-tier constraints.

## [explorer-0.4.0] - 2026-05-05

### Added
- `explorer/pages/4_Experiment.py` — Interactive Lab page with auto-generated parameter widgets, side-by-side before/after display, and PSNR/SSIM metrics vs clean reference (ADR-008)
- `explorer/lib/experiments.py` — Recipe loader, sample loader (TIFF/NumPy/FITS, multi-extension FITS), pure-function pipeline dispatch, metrics
- `explorer/lib/model_zoo.py` — Lazy-download infrastructure consuming `10_interactive_lab/models/lazy_download_recipes.yaml`. `pooch`-based hash-verified fetch + Hugging Face `snapshot_download`. License is shown to the user before any download.
- `experiments/` directory with recipe schema (`experiments/README.md`) and **three** bundled recipes:
  - `experiments/tomography/ring_artifact/` — Vo et al. (2018) sorting-based stripe removal
  - `experiments/tomography/ring_artifact_wavelet/` — Munch et al. (2009) combined wavelet-Fourier filter (different algorithm on the same Sarepy sinograms — pedagogical comparison)
  - `experiments/cross_cutting/cosmic_ray_lacosmic/` — van Dokkum (2001) L.A.Cosmic via `astroscrappy.detect_cosmics`
- Landing-page CTA card pointing to the Interactive Lab (`app.py`)
- `10_interactive_lab` added to `FOLDER_TO_CLUSTER` mapping → Build cluster (`explorer/lib/ia.py`)
- `explorer/tests/test_experiments.py` — 20 tests covering recipe parsing, function resolution, pipeline dispatch with type coercion, metrics, end-to-end run on bundled samples, **and CI-quality recipe-contract validation** (every `recipe.yaml` parses, every `manifest_path` exists, every `function` resolves, every metric is known, every `noise_catalog_ref` exists)
- `explorer/tests/test_model_zoo.py` — 8 offline tests; caught a real omission: `topaz_denoise_unet_3d` and `cryodrgn` had no `license_warning` despite GPL licenses (now fixed)
- `.github/workflows/test.yml` — runs `pytest explorer/tests/` on Python 3.11 + 3.12 for every push and PR touching `explorer/`, `experiments/`, `10_interactive_lab/`, or the workflow itself
- `scripts/build_static_site.py` — renders an "Interactive Lab — Recipes" gallery on the Build cluster page with title, modality badge, sample/parameter counts, and primary citation per recipe; explicitly states pipelines run only in the Streamlit Explorer (FR-022)
- `numpy`, `scipy`, `scikit-image`, `tifffile`, `astropy`, `astroscrappy`, `pooch`, `PyWavelets` pinned in `explorer/requirements.txt`
- `docs/05_release/release_notes/explorer-v0.4.0.md`

### Changed
- `explorer/lib/ia.py` Build cluster description mentions the Interactive Lab
- `explorer/tests/test_ia.py` updated to expect 10 note folders
- `10_interactive_lab/models/lazy_download_recipes.yaml` — added missing `license_warning` to GPL entries (Topaz 3D, CryoDRGN)
- `.github/workflows/pages.yml` — adds `10_interactive_lab/**` and `experiments/**` to trigger paths so Pages rebuilds when lab content or recipes change
- `docs/01_requirements/PRD.md` — bumped to v0.2.0; adds FR-017–FR-022 covering the Interactive Lab; scope/dependencies amended
- `docs/01_requirements/user_stories.md` — adds US-013–US-016 for the Interactive Lab personas
- The Pages mirror picks up `10_interactive_lab/` automatically (no generator change needed); interactive pipelines remain Streamlit-only per ADR-007 / invariant #9

### Fixed
- `compute_metrics` now centre-crops to the common minimum shape when the reference and candidate differ by ≤ 2 pixels in either dim (controlled by `align_tolerance` kwarg). Sarepy ships the clean reference at (1801, 2560) and noisy variants at (1800, 2560); without alignment, **PSNR/SSIM were silently skipped on every ring-artifact sample**, defeating the metric panel. The previously-skipped `test_ring_artifact_pipeline_reduces_stripes` now runs and asserts a meaningful PSNR improvement (uses `all_stripe_types_sample1.tif`, the sample with the cleanest reference relationship).
- `4_Experiment.py` now special-cases `role: false_positive_trap` samples with an info banner explaining why metrics are skipped (the sample is a different scene from the clean reference). Other shape-mismatched cases show a centre-crop caption so the user knows alignment was applied.
- Ring-artifact recipe descriptions now include a "note on metrics" caveat explaining that Sarepy's `sinogram_normal.tif` is a visual reference, not a paired ground truth — directing users to `all_stripe_types_sample1` for unambiguous metric comparisons.

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
