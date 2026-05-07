---
doc_id: REL-E070
title: "Release Notes — explorer-v0.7.0"
status: draft
version: 0.7.0
last_updated: 2026-05-08
supersedes: null
related: [REL-E061, ADR-002, ADR-008]
---

# Release Notes — explorer-v0.7.0

**Phase R11 — refactor pass driven by user feedback.**

## Summary

After R10's first-impression polish, a real-user review surfaced six
substantial issues that R1–R10 had not caught: a markdown-rendering
bug producing literal `[object Object]` strings inside code blocks,
a Knowledge Graph that didn't allow node dragging, illegible header
nav links on certain pages, an unintuitive Cards/Compare-table view
toggle, an Interactive Lab that didn't make the algorithm's effect
dramatic enough, and only 3 example recipes. R11 fixes all six, adds
2 new recipes, and tightens contrast across the rest of the app.

## What's Fixed

### I1 — `[object Object]` bug in code blocks

Pygments' `codehilite` extension defaults to `guess_lang=True`, which
runs an auto-detect lexer on every fenced code block that doesn't
declare a language. For ASCII tree diagrams in the HDF5 / EIGER
schema notes (e.g. ``├── /entry/``), the guesser picked the
JavaScript lexer and emitted a dense ``<span class="nx">`` /
``<span class="w">`` soup that Streamlit's React layer collapsed
into rows of literal `[object Object]`.

`note_view.py::_md_to_html` and `build_static_site.py` now both
configure `guess_lang=False`; unlabeled blocks render as plain
`<pre><code>`. **No notes were modified.**

### I2 — Knowledge Graph: draggable vis.js graph

R2's Plotly + NetworkX renderer (preserved through R8/R9/R10) made
nodes hover-only — you could zoom and pan but not rearrange. Users
explicitly missed the legacy vis.js capability.

R11 introduces `components/visjs_graph.py`, a vis-network 9.1.9
component with:

- **🌀 Force-directed** (default) — drag any node, others spring
  back into place.
- **📊 Hierarchical** — kind-column layout (modality → noise →
  method → recipe → paper → tool).
- **❄️ Freeze** — locks current positions for screenshotting.
- Click-to-highlight neighbours; double-click to open the underlying
  note via `?note=` (the iframe is same-origin, so `window.parent`
  navigation works).

The R8 Plotly path + NetworkX layout cache + R9 multipartite layout
toggle are gone — replaced by vis.js's native modes.

### I3 — Cards/Compare-table toggle removed

The cluster page's `?view=cards|table` toggle was removed. The default
view is now the dense compare-table with **folder-filter chips**
above it: All / Modalities / AI/ML Methods / Publications / Noise
Catalog (one chip per folder in the cluster, with note counts).
``?folder=<name>`` is the new query param. ``?view=tabs`` still
works on note-detail (auto section tabs from R9).

### I4 — Header nav legibility

The legacy CSS rule used ``color: #FFFFFF`` on header nav anchors —
fine when the header was navy, broken when the design switched to
white background. Plus the inline `_nav_link` styles overrode the CSS
`:hover` / `.active` selectors, so neither read.

R11 strips the inline color, lets ``styles.css`` own the look:
solid navy ``.active`` (white text on navy), light-blue hover, dark
default text. The static-site mirror inherits the same rules — its
override block in `build_static_site.py` was deleted because the
legacy `pointer-events:none` it worked around is also gone.

### I5 — Interactive Lab: dramatic impact display + 2 new recipes

Three changes inside `pages/4_Experiment.py` and `lib/experiments.py`:

- **3-panel before/after**: `Original | Processed | |Δ| difference`.
  The diff map shows exactly which pixels the algorithm touched —
  bright = changed, dark = passed through. Caption shows the
  fraction of the area that meaningfully changed.
- **🎯 Impact card** with win/loss banner. The metric panel was a
  3-column `st.metric` row that was easy to miss; now it leads with
  a "🎯 Impact" headline and a green / yellow / red banner
  summarising whether all metrics improved, partially improved, or
  regressed.
- **3-card narrative row** above the parameter sliders (R11
  optional `problem` / `fix` / `observe` fields on the recipe
  schema). Each card is colour-coded: ⚠️ orange "What was wrong",
  🛠️ navy "How we fix it", 👀 green "What you should observe".
  All 3 existing recipes (Vo / Munch / van Dokkum) shipped with
  populated narratives; legacy recipes without these fields fall
  back to the long-form description.

**Two new recipes:**

- ``flatfield_correction`` — the textbook I0 normalisation step.
  Demonstrates ``corrected = (raw − dark) / (flat − dark)`` on the
  bundled `flatfield_correction` dataset. Pedagogically the most
  dramatic recipe — the |Δ| panel makes detector signature visible.
- ``gaussian_denoise_baseline`` — the classical baseline every
  paper compares to. Single recipe with a `method` selector
  switching between Gaussian blur and median filter, exposed on the
  same Sarepy ring sinograms + GMOS cosmic-ray frame so users can
  feel why task-specific algorithms exist.

Recipe count: **3 → 5**.

### I6 — Holistic polish

- Breadcrumbs now sit on a tinted background card (was inline grey
  text that was easy to miss). Active crumb is bold.
- Tag chips have darker text + border for clearer contrast; `<a>`
  variants get a navy hover state.
- Filter chip-row component (`.eberlight-chip`) shared by folder
  filters + future filter surfaces.
- Streamlit native button + download-button focus rings tinted navy
  to match the design system instead of the default browser blue.
- Cluster cards on the landing now lift slightly on hover + show a
  proper focus-visible outline (already in R10 P0-2 but reinforced
  here).

## Recipe schema

`Recipe` dataclass in `explorer/lib/experiments.py` gained three
optional fields:

```yaml
problem: |    # ⚠️ "What was wrong" card
fix: |        # 🛠️ "How we fix it" card
observe: |    # 👀 "What you should observe" card
```

All three default to empty string; the Lab page renders the cards
only when at least one is non-empty.

## Tests

`pytest explorer/tests/` → **256 passed** (unchanged — the helper
changes are covered by the existing recipe-contract + render-smoke
tests, and the new recipes ship with their pipelines tested in
`test_experiments.py` via `test_bundled_recipes_*` parameterised
tests).

`ruff check / format --check explorer/ experiments/ scripts/` clean.
`pyproject.toml` per-file ignore added for `experiments/**/pipeline.py`
(`RUF002` — Greek `σ` and minus `−` are legitimate math notation in
the docstrings).

`streamlit run explorer/app.py` → `/_stcore/health` 200 OK.

## What's Unchanged

- ADR-002 stays intact — both new recipes derive their narratives
  and metrics from the recipe.yaml at runtime; no YAML catalogs
  were reintroduced for any other surface.
- No notes were modified.
- No dependency changes (vis-network loads from a public CDN; no
  Python package added or removed).
