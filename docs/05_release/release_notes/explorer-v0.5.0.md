---
doc_id: REL-E050
title: "Release Notes — explorer-v0.5.0"
status: draft
version: 0.5.0
last_updated: 2026-05-06
supersedes: null
related: [ADR-002, ADR-007, ADR-009, REL-E040]
---

# Release Notes — explorer-v0.5.0

**Parity Restoration (R1 → R7) + GitHub Pages mirror catch-up**

## Summary

Restores feature parity between the new `explorer/` and the deprecated `eberlight-explorer/`
(ADR-009). Seven sequential phases (R1 – R7) were merged via stacked PRs and consolidated
under `explorer-v0.5.0` once they landed on `main`. ADR-002 ("notes are the single source of
truth") stays intact — every restored feature derives from notes/recipes at runtime, no
hand-curated YAML catalogs were reintroduced.

A follow-up pass closes the CLAUDE.md invariant #9 gap that the R1 – R7 sequence opened: the
static GitHub Pages mirror now ships read-only stub pages for every interactive surface
(Knowledge Graph, Interactive Lab, Troubleshooter, Search) so the Pages URL surface area
matches the Streamlit app.

## What's New

### Phase R1 — Critical UX restoration (PR #40)
- Note-detail deep linking (`?note=<path>`) — cluster cards finally lead somewhere.
- Header navigation — `/Discover`, `/Explore`, `/Build` plus a 🧪 Experiment shortcut.
- Clickable tag pills (`?tag=<value>`) with a "Filtering by tag" banner on the cluster page.
- Single shared cluster-page router (`lib/cluster_page.py`); ~50 lines of duplication removed
  across `1_Discover.py` / `2_Explore.py` / `3_Build.py`.
- FR-004, FR-005, FR-007, FR-011.

### Phase R2 — Knowledge Graph + cross-reference matrices (PR #41)
- New page `pages/0_Knowledge_Graph.py`: Plotly + NetworkX spring layout with six entity
  kinds (modality, AI/ML method, paper, tool, recipe, noise) and 120+ edges.
- Layer toggles, hover tooltips, entity-search navigator, `?note=` deep links.
- Three cross-reference matrices: modality × noise, recipe → noise (FR-022 graph layer),
  tool ↔ paper.
- `lib/cross_refs.py` builds the graph at runtime from folder structure + `recipe.yaml`.
- 16 new tests (`test_cross_refs.py`).

### Phase R3 — Mermaid diagram rendering (PR #43)
- Mermaid blocks render live in both Streamlit (`components/note_view.py`) and the static
  Pages mirror.
- Pages mirror lifts each ` ```mermaid ` block via base64-encoded HTML-comment placeholders
  before codehilite, then re-inserts a `<div class="mermaid">` afterwards — preventing the
  syntax highlighter from mangling diagram source.
- 3 demo diagrams shipped in real notes (pipeline flowchart, TomoGAN architecture, ring
  artifact causal flow).
- 13 new tests (`test_mermaid.py`).

### Phase R4 — Noise-catalog troubleshooter (PR #44)
- New page `pages/5_Troubleshooter.py`: 11 symptom categories × 35 differential cases with
  severity badges, conditions, ▶ Run-experiment links, and bundled before/after images.
- `09_noise_catalog/troubleshooter.yaml` machine-readable companion to the existing prose.
- Sidebar filters: modality, severity. Deep link: `?symptom=<id>`.
- 14 drift-protection tests asserting every diagnosis resolves to a real guide / image / recipe.

### Phase R5 — Detail Level (L0/L1/L2/L3) (PR #45)
- Four reading depths derived from the same markdown body — no per-level copies on disk.
  - L0 Overview · L1 Sections · L2 Details (default) · L3 Source.
- Pill row above each note + `?level=…` deep linking with legacy long-form labels accepted.
- 32 tests, including three regression tests for Codex review findings (in-fence Python /
  shell comments mistaken for headings; verbatim L3 round-trip with quadruple-fence
  containment).

### Phase R6 — Search + Bibliography (PR #46)
- New page `pages/6_Search.py`: global full-text search (`?q=<query>` deep links) +
  filterable BibTeX bibliography in one page.
- `lib/search.py` in-memory inverted index with TF-IDF approximation, title-boost ×2,
  prefix matching for inflections. <10 ms typical query at the 200-note scale.
- `lib/bibliography.py` lightweight BibTeX parser for `08_references/bibliography.bib` +
  `10_interactive_lab/CITATIONS.bib`.
- 12 + 9 tests.

### Phase R7 — Accessibility audit (PR #47)
- `lib/a11y.py` WCAG 2.1 contrast-ratio computation + helpers for alt text and skip links.
- 26 tests including a real palette audit on every design-token pair the explorer uses.
- Palette tightened: 4 tokens darkened to pass WCAG 2.1 AA-large.
  - Explore-cluster teal `#00A3E0` → `#0085C0`, Build-cluster orange `#F47B20` → `#D86510`,
    severity-major `#E67E22` → `#C8550E`, severity-minor `#3498DB` → `#2178B5`.

### Static-site mirror catch-up (this release)
- `scripts/build_static_site.py` now emits four read-only stub pages for the interactive
  surfaces — `knowledge-graph.html`, `experiment.html`, `troubleshooter.html`, `search.html`.
- The static landing page gains a 4-card CTA grid mirroring the Streamlit landing's CTAs;
  the Streamlit landing itself was extended from 2 CTAs (KG + Lab) to 4 (KG + Lab +
  Troubleshooter + Search) so the surfaces stay parallel.
- Each stub renders the page's value proposition and points readers to
  `streamlit run explorer/app.py`. Reuses the FR-022 pattern.
- 4 new tests in `test_build_static_site.py` including a drift-protection test that fails
  CI when a Streamlit page is added without a matching stub entry in `INTERACTIVE_PAGES`.

## What's Changed

- `explorer/app.py` landing CTA grid: 2 → 4 cards.
- `explorer/lib/cluster_page.py` is now the only cluster-page entry point (R1 cleanup).
- `pyproject.toml` per-file ruff ignores extended for `RUF001` / `RUF002` on the new
  pages and lib modules (legitimate Unicode use: `×`, `→`, `↔`, `↗`).
- `.github/workflows/lint.yml` + `.pre-commit-config.yaml` ruff pin bumped from `0.5.7` to
  `0.11.13` (R5 — fixes lint-CI vs editor-side disagreement on `assert ..., (...)` wrapping).
- `requirements.txt`: `plotly>=5.18,<7.0`, `networkx>=3.1,<4.0`, `pandas>=2.1,<3.0` pinned
  (R2). No new dependencies in R3 – R7 or in this catch-up.

## Testing

- `pytest explorer/tests/` → **220 passed** on `main`.
- `ruff check explorer/ experiments/ scripts/` → clean against `ruff==0.11.13`.
- `ruff format --check explorer/ experiments/ scripts/` → clean.
- `mypy explorer/lib explorer/components` → still informational (non-blocking).
- `python scripts/build_static_site.py` → 188 notes from 10 folders, 3 recipes, 4
  interactive stubs.
- CI on PR #47 (the last R-phase): lint + pytest 3.11 + pytest 3.12 all green.

## Versioning Notes

- This release consolidates seven feature PRs (#40, #41, #43, #44, #45, #46, #47) under one
  SemVer minor bump because they landed on `main` over a single sprint and share a common
  thesis (parity restoration). Per CLAUDE.md invariant #6 the explorer stream advances
  independently of the notes stream — `notes-v0.10.0` from the previous release is unchanged.
- Per invariant #3 every shipping feature has a release-note entry — the bullets above
  cite each phase's PR so traceability survives the consolidation.
