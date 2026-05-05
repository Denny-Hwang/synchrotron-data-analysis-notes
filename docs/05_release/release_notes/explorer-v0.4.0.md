---
doc_id: REL-E040
title: "Release Notes — explorer-v0.4.0"
status: draft
version: 0.4.0
last_updated: 2026-05-05
supersedes: null
related: [ADR-008, REL-N100]
---

# Release Notes — explorer-v0.4.0

**Interactive Lab Page**

## Summary

Adds the **Experiment** page to the Streamlit Explorer (`explorer/pages/4_Experiment.py`), the
interactive surface for the data bundled in `notes-v0.10.0`. Users pick a recipe, choose a
bundled sample, tune algorithm parameters via auto-generated widgets, and see the original /
processed sinograms side-by-side with PSNR/SSIM against a clean reference when shapes match.

This is the first release where `10_interactive_lab/` is consumable end-to-end.

## What's New

### `experiments/` directory and recipe schema

```
experiments/
├── README.md                          schema documentation
├── __init__.py
└── tomography/
    └── ring_artifact/
        ├── __init__.py
        ├── recipe.yaml                first bundled recipe
        └── pipeline.py                pure-function implementation
```

Recipe schema (full reference in `experiments/README.md`):

- **`recipe_id`, `title`, `modality`, `function`** — required.
- **`samples[]`** — list of bundled `manifest_path`s relative to `10_interactive_lab/`.
- **`clean_reference`** — optional sample that triggers PSNR/SSIM computation when shape matches.
- **`parameters[]`** — `int | float | select` with min/max/step/options/help; auto-rendered as Streamlit widgets.
- **`metrics[]`** — `psnr`, `ssim` (silently skipped if unknown).
- **`references[]`** — paper citations rendered in an expander.

Pipeline functions are **pure** (`np.ndarray + kwargs → np.ndarray`), no Streamlit imports,
no I/O, no global state. This makes them trivially unit-testable and `@st.cache_data`-friendly.

### Bundled recipes

**1. Ring Artifact — Sorting-Based Filter** (`experiments/tomography/ring_artifact/`)

- **Algorithm.** Sorting-based stripe removal from Vo, Atwood & Drakopoulos (2018), *Optics Express*. For each detector column, sort across angles, apply a 1-D median filter, revert ordering.
- **Samples.** All 7 ring-artifact sinograms bundled in `10_interactive_lab/datasets/tomography/ring_artifact/`, including the false-positive trap `valid_stripes.tif`.
- **Clean reference.** `sinogram_normal.tif` (PSNR/SSIM auto-skipped for samples with mismatched shape).
- **Parameter.** `size` (median window length, 3 ≤ odd ≤ 101).
- **Citation.** DOI: 10.1364/OE.26.028396 (also in `10_interactive_lab/CITATIONS.bib`).

**2. Ring Artifact — Wavelet-FFT** (`experiments/tomography/ring_artifact_wavelet/`)

- **Algorithm.** Combined wavelet-Fourier filter from Munch, Trtik, Marone & Stampanoni (2009), *Optics Express*. Multi-level Daubechies DWT, Gaussian-damped FFT along the column direction of each level's vertical-detail band, inverse DWT.
- **Samples.** Same 6 noisy Sarepy sinograms as recipe 1 — pedagogical "two algorithms, one input" comparison (US-014).
- **Clean reference.** Same `sinogram_normal.tif`.
- **Parameters.** `level` (DWT depth, 1-7), `sigma` (FFT damping width), `wname` (`haar | db2 | db5 | db10 | sym4 | coif5`).
- **Citation.** DOI: 10.1364/OE.17.008567.

**3. Cosmic Ray / Zinger — L.A.Cosmic** (`experiments/cross_cutting/cosmic_ray_lacosmic/`)

- **Algorithm.** Laplacian-edge cosmic ray detection from van Dokkum (2001), *PASP*. Wraps `astroscrappy.detect_cosmics`.
- **Sample.** Real GMOS CCD frame `gmos.fits` (150 × 200) — proxy for tomography zingers, XRF dead/hot pixels, and outlier spectra.
- **Parameters.** `sigclip` (detection threshold), `objlim` (object protection), `niter` (iterations), `readnoise`.
- **No clean reference** — metrics omitted; users compare visually.
- **Citation.** DOI: 10.1086/323894.

### `explorer/lib/experiments.py`

New module — single import path for:

- `parse_recipe(path) -> Recipe` and `load_recipes(experiments_root) -> list[Recipe]`.
- `load_sample(repo_root, manifest_path) -> np.ndarray` (TIFF / NumPy / FITS).
- `resolve_function(dotted_path)` and `run_pipeline(recipe, arr, params)` — light type coercion of widget values to declared kwargs.
- `compute_metrics(reference, candidate, metrics)` — PSNR / SSIM via `skimage.metrics`, with min-max normalisation so dtype changes are not penalised.

### `explorer/pages/4_Experiment.py`

- Reads recipes once via `@st.cache_resource`.
- Sample loading and pipeline runs are `@st.cache_data` keyed on `(recipe_id, manifest_path, params_tuple)` — re-running the same configuration is instant.
- Auto-generates parameter widgets from `recipe.yaml`.
- Side-by-side display via `st.image(..., clamp=True)` after min-max normalisation.
- PSNR/SSIM metrics shown via `st.metric(...)` with delta-vs-raw, but only when shapes match. Otherwise an info banner explains the skip.
- Pipeline failures surface via `st.error` + `st.exception` rather than crashing the page.

### Landing-page CTA

`app.py` adds a fourth call-to-action card under the 3 cluster cards, surfacing the new Experiment page with a one-line description.

### IA mapping

`explorer/lib/ia.py` adds `"10_interactive_lab": "build"` (per ADR-008). The Build cluster page now lists 12 new entries from the lab — top-level README, `docs/external_data_sources.md`, `models/README.md`, and 7 `ATTRIBUTION.md` files for the bundled dataset folders. Test `test_ia.py::test_all_note_folders_mapped` updated to expect 10 folders.

### Static site mirror (ADR-007 / invariant #9)

`scripts/build_static_site.py` is extended in two complementary ways:

1. **Auto-discovery** of `10_interactive_lab/`. The generator already iterates over `FOLDER_TO_CLUSTER`, so all 12 lab markdown files render in the Build cluster page with no further work. Binary samples (TIFF, FITS, NumPy) are intentionally not served on Pages.
2. **Recipe gallery** (FR-022). A new "Interactive Lab — Recipes" section is rendered above the per-folder note grid on the Build cluster page. Each recipe.yaml becomes a card with the recipe title, a modality badge, sample / parameter counts, and the primary citation as a clickable DOI link. The gallery banner explicitly states that pipelines run only in the Streamlit Explorer.

The `4_Experiment.py` page itself remains **Streamlit-only** — running pipelines requires a Python runtime. Users wanting interactivity run the Streamlit app locally per `10_interactive_lab/README.md`.

`.github/workflows/pages.yml` adds `10_interactive_lab/**` and `experiments/**` to the trigger paths so the Pages site rebuilds when lab content or recipes change.

### Continuous integration

New `.github/workflows/test.yml` runs `pytest explorer/tests/` on Python 3.11 + 3.12 for every push and PR touching `explorer/`, `experiments/`, `10_interactive_lab/`, or the workflow itself. Coverage XML is uploaded as an artifact. The recipe-contract tests in `test_experiments.py` now provide drift protection at CI time, not just locally — any future PR that renames a sample, moves a function, or breaks a `noise_catalog_ref` will fail CI.

### PRD revision (v0.2.0)

`docs/01_requirements/PRD.md` is bumped to v0.2.0 and adds **FR-017 through FR-022** covering the Interactive Lab end-to-end: recipe discovery (FR-017), auto-generated parameter widgets (FR-018), before/after with metrics (FR-019), automated recipe-contract validation (FR-020), license-gated lazy downloads (FR-021), and the static-site recipe gallery (FR-022). Scope text now mentions 10 note folders and the Lab.

`docs/01_requirements/user_stories.md` adds **US-013 through US-016** spanning all three personas: replay with parameter tuning (Computational Scientist), compare two algorithms on the same input (Beamline Scientist), spot zingers safely (Beamline Scientist), and license-aware lazy fetch (Computational Scientist).

### Test coverage

`explorer/tests/test_experiments.py` (20 tests):

- Recipe parsing — minimal valid case, missing-required-fields error, invalid-recipe skip.
- Function resolution — dotted-path import, invalid path raises.
- Pipeline dispatch — kwargs coercion (str → int), default substitution.
- Metrics — perfect match → upper bounds, shape mismatch → ValueError, unknown metric → silent skip.
- Integration — run the bundled ring-artifact pipeline on the bundled `sinogram_dead_stripe.tif`, assert shape and dtype preserved.
- **CI-quality recipe contract validation** — for every `recipe.yaml` in `experiments/`:
  - parses successfully,
  - `recipe_id` is unique across recipes,
  - `function` dotted path resolves to a callable,
  - every `manifest_path` (samples + clean_reference) resolves to a real file in `10_interactive_lab/`,
  - parameter `type` is one of `int | float | select`, with required `min/max/default` for numeric and `options` for select,
  - declared `metrics` are known (`psnr`, `ssim`),
  - `noise_catalog_ref` points to an existing markdown file.

`explorer/tests/test_model_zoo.py` (8 tests, **offline-only**):

- Parses three top-level sections (`native_synchrotron_models`, `huggingface_baselines`, `external_datasets`) and skips doc-only entries.
- Classifies entries by URL vs Hugging Face ID.
- Raises `DownloadError` for entries without a URL or HF ID.
- Validates the bundled `lazy_download_recipes.yaml` parses cleanly.
- **Catches missing `license_warning` for CC-BY-NC and GPL entries** — caught a real omission for the Topaz 3D and CryoDRGN entries, now fixed.

`explorer/tests/test_ia.py` updated for 10-folder IA.

Total test count: **41 + 1 skipped** (the skip is the PSNR-improvement integration test on samples whose shape doesn't match the clean reference — Sarepy's `sinogram_dead_stripe.tif` is `(1800, 2560)` while `sinogram_normal.tif` is `(1801, 2560)`).

### Lazy-download infrastructure (`explorer/lib/model_zoo.py`)

New module that consumes `10_interactive_lab/models/lazy_download_recipes.yaml`
and exposes a thin API for the Streamlit page to fetch model weights or
external datasets on demand:

- `load_zoo(yaml_path) → list[ZooEntry]` — parses the three sections (`native_synchrotron_models`, `huggingface_baselines`, `external_datasets`).
- `find_entry(entries, name)` — name-based lookup.
- `is_cached(entry)` — best-effort cache hit check.
- `fetch(entry, progressbar=False) → Path` — `pooch.retrieve(...)` with hash verification.
- `fetch_huggingface(entry) → Path` — `huggingface_hub.snapshot_download(...)`.
- `DownloadError` raised on missing URL, network failure, or hash mismatch.

**License safety.** Each `ZooEntry` carries a `license_warning` string; the
Streamlit page MUST surface it before initiating any download. Bundled YAML
already pins warnings for CC-BY-NC (TomoGAN) and GPL (Topaz, CryoDRGN)
entries — verified by `test_bundled_zoo_no_unbalanced_warnings`.

No actual downloads happen in CI. End-to-end fetch is deferred to a future
PR that adds the first DL recipe (e.g., Noise2Void on a TomoBank sample).

### Dependencies

`explorer/requirements.txt` adds:

- `numpy>=1.24` — already a transitive dependency
- `scipy>=1.11` — `scipy.ndimage.median_filter` for ring-artifact pipeline
- `scikit-image>=0.22` — PSNR / SSIM metrics
- `tifffile>=2023.7.10` — read Sarepy `.tif` sinograms
- `astropy>=5.3` — read `gmos.fits` cosmic-ray sample (lazy import, only loaded if user picks a FITS sample)
- `astroscrappy>=1.1` — L.A.Cosmic implementation for the cosmic-ray recipe
- `pooch>=1.7` — lazy-download with hash verification for `model_zoo`

These add ~250 MB to a fresh install; on Streamlit Community Cloud they install on first cold-start.

## Compatibility / Migration

- No breaking changes to existing pages (`app.py`, `1_Discover.py`, `2_Explore.py`, `3_Build.py`).
- The Build cluster page (`3_Build.py`) automatically picks up the lab notes via the IA mapping — no edits.
- Pages mirror builds with no generator changes; run `python scripts/build_static_site.py --out site` to verify.

## Known Limitations

- **PSNR/SSIM are skipped** when the chosen sample's shape doesn't match the clean reference. This is an artefact of how Sarepy bundled their dataset (different angular sampling per artifact). The Lab page surfaces a clear info banner; users can still visually compare before/after.
- **Streamlit Cloud free-tier RAM** (~1 GB) might struggle with the largest sinogram (`sinogram_partial_stripe.tif`, ~25 MB on disk → ~500 MB after float32 expansion in scipy). If this becomes an issue, the recipe will be amended to crop or downsample before processing.
- **No DL pipeline yet.** This release ships only the traditional Vo et al. sorting-based filter. TomoGAN / Noise2Void / Topaz-Denoise recipes are scheduled for a follow-up release once `pooch` lazy-download of weights is wired up (`models/lazy_download_recipes.yaml` already lists the URLs).

## What's Next

- **First DL recipe** wiring `model_zoo.fetch(...)` to load TomoGAN or Topaz-Denoise weights on demand inside `4_Experiment.py`. Probably starts with Noise2Void since its weights are small and BSD-3.
- **GitHub Actions workflow** running `pytest explorer/tests/` on every push — the recipe-contract tests then act as drift protection at CI time, not just locally.
- **PRD revision** to formalise the Lab as a first-class feature (FR-XXX) and cross-link from `09_noise_catalog/troubleshooter.md`.
- **Recipe gallery on the static site** — render each `recipe.yaml` as a card under the Build cluster page even though pipelines themselves are Streamlit-only.

## References

- ADR-008 — Section 10 — Interactive Lab as a tenth note folder
- REL-N100 — `notes-v0.10.0` (data bundle this release consumes)
- CLAUDE.md invariants #1, #3, #6, #9
