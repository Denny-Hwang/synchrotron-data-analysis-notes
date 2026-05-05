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

### First bundled recipe — Ring Artifact (Sorting-Based Filter)

`experiments/tomography/ring_artifact/`:

- **Algorithm.** Sorting-based stripe removal from Vo, Atwood & Drakopoulos (2018), *Optics Express*. For each detector column, sort across angles, apply a 1-D median filter, revert ordering.
- **Samples.** All 7 ring-artifact sinograms bundled in `10_interactive_lab/datasets/tomography/ring_artifact/`, including the false-positive trap `valid_stripes.tif`.
- **Clean reference.** `sinogram_normal.tif` (PSNR/SSIM auto-skipped for samples with mismatched shape).
- **Parameter.** `size` (median window length, 3 ≤ odd ≤ 101).
- **Citation.** DOI: 10.1364/OE.26.028396 (also in `10_interactive_lab/CITATIONS.bib`).

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

`scripts/build_static_site.py` requires no changes — it already iterates over `FOLDER_TO_CLUSTER`, so it auto-discovers the new folder. Verified locally: all 12 lab markdown files render in the Build cluster page; binary samples (TIFF, FITS, NumPy) are intentionally not served on Pages.

The `4_Experiment.py` page itself is **Streamlit-only** — running pipelines requires a Python runtime. The Pages mirror shows a Build-cluster card pointing to the lab markdown; users wanting interactivity run the Streamlit app locally per `10_interactive_lab/README.md`.

### Test coverage

`explorer/tests/test_experiments.py` (new, 14 tests):

- Recipe parsing — minimal valid case, missing-required-fields error, invalid-recipe skip.
- Function resolution — dotted-path import, invalid path raises.
- Pipeline dispatch — kwargs coercion (str → int), default substitution.
- Metrics — perfect match → upper bounds, shape mismatch → ValueError, unknown metric → silent skip.
- Integration — run the bundled ring-artifact pipeline on the bundled `sinogram_dead_stripe.tif`, assert shape and dtype preserved.
- Bundled-recipe sanity — every `recipe.yaml` under `experiments/` parses successfully.

`explorer/tests/test_ia.py` updated for 10-folder IA.

Total test count: **27 + 1 skipped** (the skip is the PSNR-improvement integration test on samples whose shape doesn't match the clean reference — Sarepy's `sinogram_dead_stripe.tif` is `(1800, 2560)` while `sinogram_normal.tif` is `(1801, 2560)`).

### Dependencies

`explorer/requirements.txt` adds:

- `numpy>=1.24` — already a transitive dependency
- `scipy>=1.11` — `scipy.ndimage.median_filter` for ring-artifact pipeline
- `scikit-image>=0.22` — PSNR / SSIM metrics
- `tifffile>=2023.7.10` — read Sarepy `.tif` sinograms
- `astropy>=5.3` — read `gmos.fits` cosmic-ray sample (lazy import, only loaded if user picks a FITS sample)

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

- **`pooch`-based weight fetcher** (`explorer/lib/model_zoo.py`): consume `lazy_download_recipes.yaml`, show the license to the user before download, cache by SHA-256.
- **Second recipe** (next PR): low-dose denoising via Noise2Void on a TomoBank sample (lazy-fetched).
- **Recipe-validation CI step**: assert every `recipe.yaml` parses, every `manifest_path` exists, every `function` imports.
- **PRD revision** to formalise the Lab as a first-class feature (FR-XXX).

## References

- ADR-008 — Section 10 — Interactive Lab as a tenth note folder
- REL-N100 — `notes-v0.10.0` (data bundle this release consumes)
- CLAUDE.md invariants #1, #3, #6, #9
