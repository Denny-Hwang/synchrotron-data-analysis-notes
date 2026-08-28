---
doc_id: REL-E090
title: "Release Notes — explorer-v0.9.0"
status: draft
version: 0.9.0
last_updated: 2026-08-28
supersedes: null
related: [ADR-007, ADR-008, REL-E083, REL-N110]
---

# Release Notes — explorer-v0.9.0

**Phase R16 — comprehensive service + experiments review.** Minor
release: navigation-breaking bug fixes, a Lab metric-correctness fix
that required a recipe-schema extension (ADR-008 v0.2.0), CI hardening,
and static-site parity repairs.

## Fixed — user-facing (P0)

- **Search results are clickable again.** Result links emitted a bare
  `?note=…` href that resolved against `/Search` (which has no `note`
  handler) and silently cleared the query. They now route through the
  owning cluster page via the new `lib/routing.note_url()` helper
  (`/Explore?note=…` etc.) — the same fix R12 B3 applied to the
  Troubleshooter.
- **Knowledge Graph navigation works.** `cross_refs.entity_url()` had
  the identical bare-`?note=` bug, so double-click-to-open and the
  entity navigator's "Open →" link did nothing. Both now deep-link
  through the owning cluster. `test_cross_refs.py` previously pinned
  the broken behaviour; the assertion now pins the fix.
- **Experiment page catalog link fixed** — `../09_noise_catalog/…`
  404'd; now routes through `/Explore?note=…`.
- **Lab crash on 3-D samples fixed.** Selecting the flat-field recipe's
  `sinogram.npy` (shape `(45, 1, 91)`) crashed `st.image` with
  `StreamlitAPIException: Channel can only be 1, 3, or 4`. The display
  path now squeezes singleton axes and falls back to the central slice
  for genuine volumes (`_to_2d`).
- **Lab metrics scored against the wrong ground truth.** The
  phase-unwrap recipe's four samples come from four different synthetic
  scenes, but all were scored against the Gaussian scene's reference —
  the near-perfect two-bump unwrap reported ~10 dB (real score:
  ~164 dB). The recipe schema now supports a per-sample
  `clean_reference` override (ADR-008 v0.2.0) and the phase recipe
  wires each scene to its own bundled ground truth.
- **`identity_check` sample role introduced.** Three "trap" samples
  (beam-hardening, low-dose, gaussian-baseline) were byte-identical to
  their clean references but labelled `false_positive_trap`, so the UI
  claimed they were "a different scene" (false) and the gaussian one
  produced `PSNR = inf` with a divide-by-zero warning. Identity checks
  now get honest copy and score only output-vs-reference.
- **Bibliography was missing more than half the corpus.** The BibTeX
  parser's entry regex ended an entry only when its closing brace was
  directly followed by the next `@`, so every entry preceding a
  `% === section ===` banner was silently dropped (31 of 67 entries
  parsed). Comment lines are now stripped first; all 67 entries parse.
- **Stale hardcoded counts replaced with computed ones** — the landing
  page claimed "188 notes / 35 differential cases / 19 + 20 BibTeX
  entries" (actual: 192 / 42 / 67). Counts now derive from the corpus
  at render/build time on both the Streamlit and static sides.

## Fixed — correctness / performance

- `_schema` cache-key parameter renamed to `schema` on the Experiment
  page: Streamlit excludes underscore-prefixed parameters from cache
  keys, so the REL-E081 M1 auto-invalidation mechanism was a no-op.
- Download buttons on the Lab are gated behind a "Prepare download
  files" toggle — `st.download_button(data=…)` serialises eagerly, so
  up to ~37 MB was re-encoded on every slider tick of the large ring
  TIFFs.
- Knowledge Graph: L3 no longer shows an "L1 —" caption; the layer
  checkboxes render only at L2 (the only level they affect).
- Folder-filter chips no longer discard the active Cards layout.
- "Note not found" warning renders below the header instead of above
  the site chrome.
- TOC restricted to L2 (at L1 every TOC link pointed at a non-existent
  anchor).
- Metric-pair cap aligned to the 4 columns the note view renders
  (pairs 5–6 were silently dropped).
- Footer git-date lookup anchored to the repo root (`cwd`).
- `notes.py` accepts governance-frontmatter `last_updated` as the
  `last_reviewed` fallback, and the compare table drops metadata
  columns that are empty for the whole corpus.
- Deprecated `group_by_folder` compat argument removed from
  `render_cluster_page` and its three callers.

## Static-site parity (invariant #9)

- Static header now emits the 🧪 Experiment nav link and the WCAG 2.4.1
  skip link + `#main-content` anchor (both present in the Streamlit
  header since R13).
- Static note aside now shows "Last reviewed" like the Streamlit panel.
- The static cluster toggle's first pill is labelled "📁 By folder"
  (it renders folder-grouped cards, not the Streamlit compare table it
  was labelled after).
- Stale "Plotly" references purged (vis-network since R11).

## CI / infra

- `scripts/requirements.txt` gains `numpy` — the Pages build imported
  `lib.experiments` (→ numpy) unconditionally and crashed with
  `ModuleNotFoundError` in CI.
- `pages.yml` now builds (without deploying) on pull requests, so
  invariant-#9 drift is caught pre-merge; `test.yml` triggers on
  `scripts/**` and `09_noise_catalog/**`.
- `lint.yml` mypy step drops the redundant `|| true`.
- `.devcontainer` launches `explorer/app.py` (was the deprecated
  `eberlight-explorer/` app) and installs `explorer/requirements.txt`.
- Dependency floors corrected: `streamlit>=1.50` (`width="stretch"`),
  `scikit-image>=0.25` (`unwrap_phase(rng=…)`).

## Tests

- New `test_r16_review.py` (9 tests): note_url routing for every
  folder, no bare `?note=` hrefs in pages, `_CLUSTER_TAGLINE` parity
  between Streamlit and the static builder, requirements-pin sync,
  recipe↔manifest drift (every sample + reference on disk **and** in
  the manifest), identity-check semantics, per-sample reference shape
  compatibility.
- Suite: 331 → **341 passing**.

## Known follow-ups (out of scope for this pass)

- `components/card.py` and `lib/model_zoo.py` remain staged-not-live
  (model zoo now documented as such in `SECURITY.md` +
  `models/README.md`).
- Metric normalisation scores reference and candidate independently,
  which penalises value-distribution-changing corrections (beam
  hardening) — metric redesign deferred.
- Static note pages still lack the TOC aside and prev/next nav.
- DC-001 rich frontmatter (tags/modality/beamlines) is still authored
  nowhere; the `?tag=` filter stays dormant until notes adopt it.

Traceability: FR-003 (search), FR-021 (knowledge graph), ADR-007
(static mirror), ADR-008 v0.2.0 (Lab data contract), invariant #9.
