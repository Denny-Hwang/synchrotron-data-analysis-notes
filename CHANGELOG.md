# Changelog

All notable changes to this project will be documented in this file.

This project uses two independent SemVer streams per ADR-006:
- `notes-vX.Y.Z` — content in the note folders
- `explorer-vX.Y.Z` — the explorer application

## [explorer-0.6.1] - 2026-05-07

**Phase R10 — first-impression UX polish.** Fixes 4 P0 + 8 P1 issues
identified in the post-R9 user-perspective review. No new feature
surface; existing surfaces just behave correctly now. Release notes:
[REL-E061](docs/05_release/release_notes/explorer-v0.6.1.md).

### Fixed
- **P0-1** Header cluster nav links re-enabled (CSS rule was dimming
  + disabling them in the Streamlit shell).
- **P0-2** Landing 3-cluster cards are real ``<a href>`` anchors with
  hover lift + focus-visible outline.
- **P0-3** Permalink button uses ``streamlit.components.v1.html`` so
  the clipboard JS actually runs; ``st.code`` fallback exposes the
  raw URL.
- **P0-4** Lab + Troubleshooter selectors moved out of the sidebar
  into the main column — mobile users can finally change them.
- **P1-1** Mobile media queries for header / cluster grid / CTA grid.
- **P1-2** Compare-table title column uses ``LinkColumn`` instead of
  emitting raw ``<a>`` HTML into a dataframe cell.
- **P1-3** TOC slug now matches ``markdown.extensions.toc.slugify``
  exactly so anchor jumps actually land; adds ``scroll-margin-top``
  + ``scroll-behavior:smooth``.
- **P1-4** Section-tab view renders Mermaid blocks via
  ``_render_body_with_mermaid`` (was ``_md_to_html`` direct, dropping
  the diagram split).
- **P1-5** Empty search results offer "Did you mean…", popular
  queries, and a Knowledge-Graph fallback link.
- **P1-6** Knowledge Graph default view: 4 layers
  (modality+method+recipe+noise) instead of all 6 — less overwhelming.
- **P1-7** Numbered 1️⃣2️⃣3️⃣ stepper headings on Lab + Troubleshooter.
- **P1-8** ``last_reviewed`` (DC-001 optional frontmatter) surfaced
  in the note-detail metadata panel.

### Added
- ``Index.suggest(query, *, limit)`` — empty-search "did you mean"
  helper. Pure function, no new deps.
- 5 new tests: anchor-slug drift protection, suggest helper (3),
  ``last_reviewed`` parse.

### Notes
- Total tests: **256 passed** (251 → 256).
- ``ruff check / format --check explorer/ experiments/ scripts/`` clean.

## [explorer-0.6.0] - 2026-05-07

**Phase R9 — final feature parity.** Closes the last five gaps identified in the
legacy-vs-new audit. After R9 the new explorer matches the legacy `eberlight-explorer/`
on every measured surface and exceeds it on governance, accessibility, search,
Interactive Lab, static-site mirror, and power-user UX. Release notes:
[REL-E060](docs/05_release/release_notes/explorer-v0.6.0.md).

### Added
- **Mermaid library restoration (35 diagrams).** `scripts/migrate_legacy_mermaid.py`
  lifts the legacy ``CATEGORY_DIAGRAMS`` / ``METHOD_DIAGRAMS`` / ``PAPER_DIAGRAMS``
  page-side dictionaries into the matching note markdown so every category README,
  method note, and paper review opens with an inline architecture diagram. Idempotent
  (R3's two existing diagrams are preserved).
- **`?doc=<basename>` deep links** restored on every cluster page.
  `lib/notes.py::find_note_by_basename` resolves bare filenames with optional folder
  hints; cluster pages prefer matches in their own scope.
- **Frontmatter-driven metric row** on note-detail. New optional fields (`resolution`,
  `maturity`, `language`, `gpu`, `year`, `journal`, `authors`, `doi`, `priority`,
  `pipeline_stage`) appear as `st.metric()` cards above the body when declared. Notes
  without rich frontmatter render unchanged.
- **`?view=tabs` auto-section-tabs view.** `lib/detail_level.py::split_into_sections`
  splits the body at every H2 heading and the note-view renders each chunk in its own
  `st.tabs(...)` panel. The level pill row gains a 📑 Tabs toggle. In-fence-aware (a
  ``## comment`` inside a Python code fence does NOT start a section).
- **Knowledge Graph hierarchical layout toggle.** Plotly + NetworkX via
  `nx.multipartite_layout` — modality → noise → method → recipe → paper → tool columns.
  Pill row above the graph switches between spring (default) and hierarchical.

### Changed
- `lib/notes.py::Note` dataclass extended with 10 optional fields and parsers for
  string / int / bool frontmatter values.
- `lib/cluster_page.py::render_cluster_page` accepts the legacy `?doc=` query
  parameter in addition to `?note=`, `?tag=`, `?level=`, `?view=`.
- `pyproject.toml` per-file ruff ignores: `scripts/migrate_legacy_mermaid.py` exempt
  from `T20` (one-shot script with progress prints).

### Notes
- 16 new tests (251 passed total). 5 drift-protection tests in
  `test_legacy_mermaid_migration.py` lock the migration in CI: every diagram in the
  migration table has a counterpart in note markdown, total count is 35, and a
  re-run is a no-op.
- `ruff check / format --check explorer/ experiments/ scripts/` clean.
- ADR-002 stays intact — every restored capability is derived at runtime from notes,
  recipes, or frontmatter; no YAML catalogs were reintroduced.

## [explorer-0.5.0] - 2026-05-06

**Parity restoration (R1 → R7) + GitHub Pages mirror catch-up.** Consolidates seven feature
PRs (#40, #41, #43, #44, #45, #46, #47) under one SemVer minor bump and closes the static-site
mirror gap they opened. Release notes: [REL-E050](docs/05_release/release_notes/explorer-v0.5.0.md).

### Added
- **Static-site mirror catch-up** (CLAUDE.md invariant #9): `scripts/build_static_site.py`
  now emits four read-only stub pages (`knowledge-graph.html`, `experiment.html`,
  `troubleshooter.html`, `search.html`) for the interactive surfaces that cannot run as flat
  HTML. The static landing gains a 4-card CTA grid mirroring the Streamlit landing's CTAs;
  Streamlit's landing CTAs were extended from 2 (KG + Lab) to 4 (KG + Lab + Troubleshooter
  + Search) so the surfaces stay parallel. 4 new tests in `test_build_static_site.py`
  including a drift-protection test that fails CI when a Streamlit page is added without a
  matching `INTERACTIVE_PAGES` entry.
- **Release notes — `docs/05_release/release_notes/explorer-v0.5.0.md`** (REL-E050)
  consolidating R1 – R7.

### Phase summary (each phase has its own block below)
- **R1** — Note-detail deep linking, header nav, clickable tags, shared cluster-page router.
- **R2** — Knowledge Graph + 3 cross-reference matrices.
- **R3** — Mermaid diagram rendering on both Streamlit + Pages mirror.
- **R4** — 11-symptom troubleshooter + before/after viewer.
- **R5** — L0/L1/L2/L3 progressive disclosure.
- **R6** — Full-text search + bibliography.
- **R7** — WCAG 2.1 AA accessibility audit + palette darkening.

### Testing
- `pytest explorer/tests/` → **220 passed** on `main`.
- `ruff check / ruff format --check explorer/ experiments/ scripts/` → clean against `ruff==0.11.13`.
- `python scripts/build_static_site.py` → 188 notes, 3 recipes, 4 interactive stubs.

## [Unreleased] — Phase R7: Accessibility audit (WCAG 2.1 AA)

### Added
- **`explorer/lib/a11y.py`** — pure helpers used by both the audit tests and the runtime: WCAG 2.1 contrast-ratio computation (`hex_to_rgb`, `relative_luminance`, `contrast_ratio`, `passes_aa`, `passes_aaa`), `alt_for_before_after(noise_stem)` for screen-reader image text, and `skip_link_html(target_id)` for the WCAG 2.4.1 "Bypass Blocks" requirement.
- **`explorer/tests/test_a11y.py`** — 26 new tests. Hex parsing, contrast-ratio identity / symmetry / black-on-white = 21:1. **Real palette audit**: every design-token pair the explorer uses (body / secondary / heading / banner / cluster accents / severity badges) is exercised against the appropriate AA threshold. Token-consistency tests verify `lib/ia.py::CLUSTER_META` and `lib/troubleshooter.py::severity_color` match the values asserted in the audit suite.

### Fixed
- **Palette tightened to pass WCAG 2.1 AA-large**:
  - Explore-cluster teal `#00A3E0` → **`#0085C0`** (contrast on white 2.87 → 4.10).
  - Build-cluster orange `#F47B20` → **`#D86510`** (2.73 → 3.63).
  - Severity-major orange `#E67E22` → **`#C8550E`** (white-on-color 2.85 → 4.41).
  - Severity-minor blue `#3498DB` → **`#2178B5`** (3.16 → 4.83 AAA-grade).

### Notes
- These were the only three contrast violations the new audit caught. All other tokens (Discover blue, body text, banner pairings) already passed.
- The cluster-color change is visible across `lib/ia.py` consumers — landing-page CTA cards, the cluster H1, the Knowledge-Graph node colors. The shift is a few hundred lumens darker; the visual identity is preserved.
- `pytest explorer/tests/` → with R6 now on `main`, the full stack reaches ≥190 passing tests; on this branch alone (R7 over R6) the count is 196.
- `ruff check / ruff format --check` clean against `ruff==0.11.13`.
- Phase R7 is the **final** phase of the parity restoration plan. Sections R1–R7 collectively bring the new explorer to feature parity with `eberlight-explorer/` (deprecated per ADR-009) while staying ADR-002 compliant — every restored feature reads from the notes at runtime, no hand-curated YAML catalogs were reintroduced.

## [Unreleased] — Phase R6: Search + Bibliography

### Added
- **`explorer/pages/6_Search.py`** — global full-text search + bibliography in one page. Search supports `?q=<query>` deep links; results show title-boosted relevance scores, modality badges, snippet around the first match, and clickable terms-matched chips. Bibliography section filters by title / author / year / entry type.
- **`explorer/lib/search.py`** — in-memory inverted index (~5 KB per note, <10 ms typical query). TF-IDF approximation with title-boost ×2, prefix matching for inflections, and a deterministic tie-breaker on note path. No new dependencies — Whoosh / lunr would have been overkill at our 200-note scale.
- **`explorer/lib/bibliography.py`** — lightweight BibTeX parser that loads both bundled `.bib` files (`08_references/bibliography.bib`, `10_interactive_lab/CITATIONS.bib`). Extracts type, key, title, authors, year, journal/booktitle/venue, DOI, and renders `Author1 et al. (Year). Title. Venue. DOI: …` short-form citations. DOI fields surface as clickable `https://doi.org/…` links.
- **`explorer/tests/test_search.py`** — 12 tests covering tokenizer (lowercase, trailing-punct stripping, dot/dash preservation), index, title boost, prefix matching, snippet, limit, deterministic ordering, and a real-repo smoke test that finds `tomopy`.
- **`explorer/tests/test_bibliography.py`** — 9 tests covering parser (article / inproceedings / no-fields edge cases), DOI URL formatting, APA-short rendering, and real-repo loaders.

### Notes
- ADR-002 stays intact — the index is rebuilt from notes at runtime and never written to disk.
- `pytest explorer/tests/` → after merge with R4 + R5 the total reaches ≥190.
- `ruff check / ruff format --check` clean.
- Phase R7 (Accessibility audit — WCAG 2.1 AA) is the next step.


## [Unreleased] — Phase R5: Detail Level (L0/L1/L2/L3)

### Added
- **`explorer/lib/detail_level.py`** — pure helpers that derive four reading depths from the same markdown body:
  - **L0 Overview** — first paragraph (≤600 chars), top H1 stripped, fenced code blocks ignored.
  - **L1 Sections** — outline of H2/H3 headings + first sentence per heading. Lines starting with `#` inside fenced code blocks (Python comments, shell, etc.) are correctly skipped — closes Codex review P2 on PR #45.
  - **L2 Details** — the full body (default, unchanged behaviour).
  - **L3 Source** — raw markdown in a fenced code block whose outer fence length is **dynamically chosen** (`max(3, longest_inner_run + 1)`) so embedded ``‌```mermaid`` / ``‌```python`` blocks round-trip verbatim with no backslash escapes — closes the second Codex P2 finding.
- **`?level=…` deep linking** — `lib/cluster_page.py` parses `?level=L0|L1|L2|L3` (or the long-form `Overview / Sections / Details / Source` the legacy app used) and renders the chosen level. A pill row above each note shows the four levels with the active one highlighted; clicking switches the param.
- **`explorer/tests/test_detail_level.py`** — 32 tests covering vocabulary, every level's output, dispatcher fallback, the parametrised long-form-label normaliser, plus three new **regression tests** targeting the Codex findings (in-fence Python-comment headings; in-fence shell-comment headings; verbatim L3 round-trip with quadruple-fence containment).

### Changed
- **`.github/workflows/lint.yml` + `.pre-commit-config.yaml`** bump ruff from `0.5.7` to `0.11.13`. The older pin's `ruff format` output disagreed with newer local installs on `assert …, (…)` wrapping (style flipped between 0.8 and 0.9). 0.11.13 is the first stable line where editor-side and CI converge in May 2026 — this resolves the lint-job failure on PR #45.

### Notes
- ADR-002 stays intact — no per-level copies of any note are written to disk; all four levels are derived from the same markdown body.
- `pytest explorer/tests/` → 152 passed on this branch (after merge with R4 the total reaches ≥166: R4 baseline 134 + R5 net 32 - 4 helper duplicates).
- `ruff check / ruff format --check` clean against `ruff==0.11.13`.
- Phase R6 (Search + BibTeX + DOI links) is the next step.

## [Unreleased] — Phase R4: Noise-catalog troubleshooter + before/after viewer

### Added
- **`explorer/pages/5_Troubleshooter.py`** — symptom-based decision-tree page. Pick one of 11 symptom categories (`Circular/Ring Patterns`, `Isolated Bright/Dark Spots`, `Streak/Stripe Patterns`, `Overall Graininess`, `Blurring`, `Intensity Anomalies`, `Spectral Abnormalities`, `Boundary/Stitching`, `Suspicious "Too-Good" Features`, `Phase Map Discontinuities`, `Ghost/Residual`) → see all differential diagnoses as cards with severity badge, conditions list, ▶ Run-experiment link (when a recipe matches), and the bundled before/after image. Sidebar provides modality + severity filters and `?symptom=<id>` deep linking.
- **`09_noise_catalog/troubleshooter.yaml`** — machine-readable companion to the prose `troubleshooter.md`. 11 symptoms × 35 differential cases, each carrying `conditions[]`, `diagnosis.{name,severity,guide,recipe?,image?}`, plus optional Python `quick_checks`. ADR-002 stays intact: prose stays canonical; YAML is a structured view for the page + tests.
- **`explorer/lib/troubleshooter.py`** — typed parser (`Symptom` / `Case` / `Diagnosis` / `QuickCheck`), severity-color helper, and before/after image discovery (maps `<stem>_before_after.png` → `Path` for the 22 bundled comparisons).
- **`explorer/tests/test_troubleshooter.py`** — 14 new tests asserting: 11 symptoms load, every diagnosis has canonical severity + a guide path that resolves to an existing `09_noise_catalog/*` markdown file, every declared `image` filename exists in `09_noise_catalog/images/`, and every declared `recipe` id resolves to a bundled `experiments/**/recipe.yaml`. **Drift protection at CI time** for the cross-references between sections 9, 10, and the Streamlit page.

### Notes
- `pytest explorer/tests/` → 134 passed (was 120 in R3; +14 troubleshooter tests).
- `ruff check / format --check` clean.
- `streamlit run explorer/app.py` → `/_stcore/health` returns `ok`; the new Troubleshooter page is in the sidebar.
- Phase R5 (Detail Level L0/L1/L2/L3 progressive disclosure) is the next step.

## [Unreleased] — Phase R3: Mermaid diagram rendering

### Added
- **Mermaid diagram rendering** across both surfaces of the
  explorer (Streamlit + GitHub Pages mirror). Any note can now
  embed a fenced ```` ```mermaid ```` block in its markdown and
  the diagram renders live — flowcharts, sequence diagrams,
  class diagrams, etc. ADR-002 stays intact: diagrams live inside
  the note markdown, not in page-side dicts (the legacy
  `eberlight-explorer/` carried 40+ inline diagrams in three
  hand-curated `*_DIAGRAMS = {…}` dictionaries).
- **`explorer/components/note_view.py`** — splits the body at
  every ```` ```mermaid ```` block, renders the surrounding
  markdown via `st.markdown(...)` as before, and renders each
  Mermaid block as a `streamlit.components.v1.html` iframe that
  loads `mermaid@10` from the public jsDelivr CDN.
- **`scripts/build_static_site.py`** — adds two helpers,
  `_extract_mermaid_blocks` (lifts each block out of the raw
  markdown body before codehilite mangles it into a syntax-
  highlighted listing) and `_replace_mermaid_blocks` (swaps the
  base64-encoded HTML-comment placeholders back to live
  `<div class="mermaid">` elements after markdown rendering).
  The page head injects the Mermaid runtime once per page that
  actually carries a diagram — no overhead on note pages
  without one.
- **3 demo diagrams** added to actual note markdown:
  - `07_data_pipeline/README.md` — pipeline flowchart
    (Acquisition → Streaming → Processing → Analysis → Storage).
  - `03_ai_ml_methods/denoising/tomogan.md` — TomoGAN
    conditional-GAN architecture (U-Net generator + PatchGAN
    discriminator + VGG-perceptual loss).
  - `09_noise_catalog/tomography/ring_artifact.md` — causal
    flow showing how a defective detector column becomes a
    sinogram stripe and then a reconstructed ring, plus the
    three mitigation algorithms (Vo 2018, Munch 2009, DL).
- **`explorer/tests/test_mermaid.py`** — 13 new tests covering
  the regex pattern (single / multiple / inline-code-not-matched
  / non-mermaid-language-not-matched / trailing-whitespace), the
  static-site round-trip (extract → render → replace preserves
  arrow operators verbatim), and end-to-end build verification
  for each demo note.

### Notes
- `pytest explorer/tests/` → 120 passed (was 107 in R2; +13
  Mermaid tests).
- `ruff check / ruff format --check` clean.
- Static-site build emits `<div class="mermaid">` for every demo
  note plus the Mermaid runtime script in the page head.
- `streamlit run explorer/app.py` → `/_stcore/health` returns
  `ok`; the new diagrams render live in the note-detail view.
- Phase R4 (Noise-catalog 11-symptom troubleshooter +
  before/after image viewer) is the next step.



## [Unreleased] — Phase R2: Knowledge Graph + cross-reference matrices

### Added
- **`explorer/pages/0_Knowledge_Graph.py`** — interactive cross-reference
  network visualising every modality, AI/ML method, paper, tool,
  Interactive-Lab recipe (§10), and noise/artifact in the repository
  on a single page. Plotly + NetworkX spring layout; six entity kinds
  rendered in distinct colours (Recipes highlighted red so the
  Interactive Lab is immediately visible). Layer checkboxes hide /
  show each entity kind. Hover any node for kind, category, doc path.
- **Entity navigator** — searchable selectbox of all 100+ entities
  with a live deep-link to the underlying note (or the Experiment
  page for recipes), reusing the `?note=…` routing introduced in R1.
- **Cross-reference matrices** (3 expandable tables):
  - Modality × noise-type count (which modality has the most
    catalogued artifacts);
  - **Recipe → noise mapping** with deep links — derived from each
    `recipe.yaml`'s `noise_catalog_ref`, so section 10 is treated
    as a first-class graph layer;
  - Tool ↔ paper mention table (best-effort regex against paper
    review markdown).
- **`explorer/lib/cross_refs.py`** — pure data layer that builds the
  graph at runtime from folder structure plus `experiments/**/recipe.yaml`.
  Honours ADR-002 ("notes are the single source of truth") — no
  hand-curated YAML catalogs are reintroduced. Three edge sources:
  folder structure (always reliable), recipe YAML (precise
  recipe→modality / recipe→noise edges), and best-effort regex
  scanning of paper review markdown for tool / method mentions.
- **`explorer/tests/test_cross_refs.py`** — 16 new tests asserting
  the schema (every entity has a unique namespaced id, every edge
  endpoint resolves, no duplicate edges), the canonical six
  modalities are detected, and every section-10 recipe carries both
  a modality and a noise edge.
- **Landing-page CTA grid** — `app.py` now shows a two-column card
  block with Knowledge Graph + Interactive Lab side-by-side, both
  with proper deep links to their pages.
- **Dependencies**: `plotly>=5.18,<7.0`, `networkx>=3.1,<4.0`,
  `pandas>=2.1,<3.0` added to `explorer/requirements.txt`. (NetworkX
  was already a transitive dep through scikit-image; we pin it
  explicitly so future major releases don't surprise the graph.)

### Changed
- `pyproject.toml`: per-file ruff ignores extended for `RUF001` /
  `RUF002` on Streamlit pages (legitimate Unicode use: `×` for
  matrices, `→` for recipe-to-noise, `↔` for typed-graph notation).

### Notes
- `pytest explorer/tests/` → 107 passed (was 91 in R1; +16 new tests
  in `test_cross_refs.py`).
- Static-site mirror still builds: 188 notes from 10 folders, 3 recipes.
- `streamlit run explorer/app.py` → `/_stcore/health` returns `ok`;
  the new Knowledge Graph page is discoverable in the sidebar and
  via the landing-page card.
- Phase R3 (Mermaid diagram library + 40+ architectural diagrams)
  and R4 (Noise-catalog troubleshooter + before/after viewer) are
  scheduled for subsequent PRs.

## [Unreleased] — Phase R1: Critical UX restoration

The deprecation of `eberlight-explorer/` (ADR-009) shipped before the
new `explorer/` had reached feature parity. This first restoration
pass fixes the most painful regressions where core navigation was
silently broken.

### Fixed
- **FR-004 — Note detail deep linking**. Cards on cluster pages
  previously rendered with `href="#"` — clicking did nothing. Cards
  now link to `?note=<path>` and the cluster page (via the new
  `lib/cluster_page.py` router) detects the query parameter and
  switches to the note-detail view (`components/note_view.py`),
  which already existed but had no callsite.
- **FR-011 — Header navigation**. Header cluster links were
  hard-coded to `href="#"` with `pointer-events:none`. They now
  point to `/Discover`, `/Explore`, `/Build`, plus a 🧪 Experiment
  shortcut. Streamlit's auto-discovered page slugs handle the
  routing. The active cluster is highlighted via a translucent
  pill background.
- **FR-005 — Breadcrumb on note view**. The cluster crumb's URL was
  `"#"`. It now points to the actual cluster page so the user can
  navigate up one level.
- **FR-007 — Clickable tags**. Tag pills were inert `<span>`s; they
  are now `<a>` links to `?tag=<value>` and the cluster page filters
  the card grid to matching notes plus a "Filtering by tag: …
  ✕ clear filter" banner.

### Added
- `explorer/lib/cluster_page.py` — single shared router for the
  three cluster landing pages. Three modes: note-detail deep link,
  tag-filtered grid, default folder-grouped grid.
- `Note.url_id(repo_root)` and `find_note_by_url_id(...)` helpers in
  `explorer/lib/notes.py`.
- `render_note_card(note, repo_root)` convenience wrapper in
  `explorer/components/card.py`.
- `render_header(active_cluster=…)` parameter so each page can
  highlight its own cluster.
- `cluster_url` parameter on `render_note_view(...)` so the
  breadcrumb's cluster crumb is a real link instead of `"#"`.

### Changed
- `explorer/pages/1_Discover.py`, `2_Explore.py`, `3_Build.py`
  reduced to thin delegators that call `render_cluster_page(...)`
  with their cluster id. ~50 lines of duplication removed across
  the three.

### Notes
- This is **Phase R1 of a 7-phase parity restoration plan**. Phases
  R2–R7 (Knowledge Graph + Mermaid library + Noise-catalog
  troubleshooter + Detail Level + search + bibliography +
  accessibility) ship in subsequent PRs.
- `pytest explorer/tests/` → 91 passed, 0 failed (unchanged from
  before — R1 doesn't add tests yet; component tests for the new
  router land in R2 along with Knowledge Graph tests).
- Static-site mirror builds cleanly: 188 notes from 10 folders.



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
