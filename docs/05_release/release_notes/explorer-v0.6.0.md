---
doc_id: REL-E060
title: "Release Notes — explorer-v0.6.0"
status: draft
version: 0.6.0
last_updated: 2026-05-07
supersedes: null
related: [ADR-002, ADR-009, REL-E050]
---

# Release Notes — explorer-v0.6.0

**Phase R9 — final feature parity (Mermaid library + legacy URL shapes + power-user UX)**

## Summary

Closes the last five gaps identified in the legacy-vs-new audit (`docs/comparison`-quality
table delivered in conversation). After v0.5.0 the new explorer was already ahead of the
legacy `eberlight-explorer/` (deprecated per ADR-009) on governance, accessibility,
Interactive Lab, search, and static-site mirror — but the legacy app retained five
information-density advantages: a 35-diagram Mermaid library, a sortable `metric` row on
note-detail pages, a hierarchical layout option on the Knowledge Graph, an automatic H2
section-tabs view, and basename-only URL deep links (`?doc=ring_artifact`).

R9 ships all five, all driven by note frontmatter / markdown body so ADR-002 stays intact.

## What's New

### Mermaid library restoration (35 diagrams)

The legacy app stored 5 category, 12 method, and 18 paper Mermaid diagrams in three
page-side `_DIAGRAMS = {...}` Python dictionaries. R9 lifts each one into the matching note
markdown via `scripts/migrate_legacy_mermaid.py` (idempotent — notes that already shipped a
Mermaid block in R3 are left alone). After the migration:

- Every AI/ML method category README in `03_ai_ml_methods/<category>/README.md` opens with
  an **Architecture diagram** section.
- Every method note in `03_ai_ml_methods/<category>/<method>.md` carries its architecture
  diagram inline.
- Every paper review under `04_publications/ai_ml_synchrotron/review_*.md` carries its
  pipeline diagram inline.

Both the Streamlit explorer (`components/note_view.py`'s ` ```mermaid ` iframe split) and
the GitHub Pages mirror (`scripts/build_static_site.py`'s codehilite-bypass) render the
diagrams natively.

`explorer/tests/test_legacy_mermaid_migration.py` adds 5 drift-protection tests: every
declared diagram has a counterpart in note markdown, total count is locked at 35, and the
migration is idempotent (running it twice on the same file does not duplicate the block).

### `?doc=<basename>` deep links

The legacy `8_📡_Noise_Catalog.py` and `7_📊_Data_Structures.py` pages accepted
`?doc=ring_artifact` (basename only). The new `lib/notes.py::find_note_by_basename` resolver
restores this — when `?doc=` is set on a cluster page (`/Discover`, `/Explore`, `/Build`)
the router scans cluster-scoped folders first, then falls back to a global lookup. Folder
hints disambiguate two notes that share a stem (e.g. `x.md` in both
`03_ai_ml_methods/` and `04_publications/`).

The canonical `?note=<full/path>` form continues to work and remains preferred for
permalinks.

### Frontmatter-driven metric row on note-detail

The legacy `2_🔬_Modalities.py` and `5_🛠️_Tools.py` pages opened with a 4-card
`st.metric()` row (resolution / beamlines / AI methods / interaction; or maturity / stage /
language / GPU). R9 generalises that to **any** note: the new optional frontmatter fields
`resolution`, `maturity`, `language`, `gpu`, `year`, `journal`, `authors`, `doi`,
`priority`, `pipeline_stage` are surfaced in `st.metric()` cards above the body whenever
declared.

Notes without rich frontmatter render unchanged — the metric row hides automatically.

### `?view=tabs` — auto-section-tabs view on note-detail

The legacy Publications L2 split each review into 8 hand-coded section tabs (Background /
Method / Key Results / …). R9 generalises to **any** note via `lib/detail_level.py::
split_into_sections`: at L2 + `?view=tabs`, the body is split at every H2 heading and each
chunk lands in its own `st.tabs(...)` panel. The level-selector pill row gains an extra
**📑 Tabs** toggle so users can switch in a single click.

`split_into_sections` is in-fence-aware: a `## comment` inside a Python code fence does
NOT start a new section.

### Knowledge Graph — hierarchical layout toggle

The legacy KG used vis.js with a `hierarchical` flag the user could toggle. R9 brings
matching behaviour to Plotly + NetworkX via `nx.multipartite_layout`: a new pill row above
the graph offers **🕸️ Force-directed (spring)** vs **🪜 Hierarchical (kind columns)**. The
hierarchical mode lays the six entity kinds out in fixed left-to-right columns
(modality → noise → method → recipe → paper → tool), which makes the type taxonomy
immediately visible.

Only meaningful at L2 (the level that renders the spatial graph) — the toggle hides at
L0/L1/L3.

## What's Changed

- `lib/notes.py::Note` gained 10 optional fields plus parsers (`_opt_str`, `_opt_int`,
  `_opt_bool`). All default to `None`/`False`/`[]` so existing notes parse unchanged.
- `lib/cluster_page.py::render_cluster_page` accepts `?doc=` and `?view=tabs` in addition
  to `?note=`, `?tag=`, `?level=`, `?view=cards|table` (R8).
- `components/note_view.py::render_note_view` gained `metrics` + `section_tabs` kwargs.
- `pyproject.toml` per-file ruff ignores extended for `scripts/migrate_legacy_mermaid.py`
  (`T20` — script intentionally prints progress to stdout).

## Testing

- `pytest explorer/tests/` → **251 passed** (was 235, +16: 6 frontmatter / basename
  resolver tests in `test_notes.py`, 5 `split_into_sections` tests in
  `test_detail_level.py`, 5 Mermaid-migration drift tests).
- `ruff check / format --check explorer/ experiments/ scripts/` → clean against
  `ruff==0.11.13`.
- `python scripts/build_static_site.py` → 188 notes, 3 recipes, 4 interactive stubs;
  Mermaid blocks render natively in `*_2020.html`, `*_2024.html`, etc.
- `streamlit run explorer/app.py` → `/_stcore/health` 200 OK.

## Versioning Notes

- This is the explorer-only stream. `notes-v0.10.0` is unchanged; the migrated Mermaid
  blocks are technically content updates to the notes, but the noise-catalog YAML +
  bundled-data inventory is unchanged so the notes SemVer does not bump.
- After R9, the new `explorer/` matches the legacy `eberlight-explorer/` on every
  measured surface (per the comparison table delivered in conversation) and exceeds it on
  governance, accessibility, search, Lab, static mirror, and beyond-legacy UX
  (TOC / prev-next / permalink / recently-viewed / compare-table).
