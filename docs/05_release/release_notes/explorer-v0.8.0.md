---
doc_id: REL-E080
title: "Release Notes — explorer-v0.8.0"
status: draft
version: 0.8.0
last_updated: 2026-05-11
supersedes: null
related: [REL-E073, ADR-010, DS-001]
---

# Release Notes — explorer-v0.8.0

**Phase R15 — senior-review polish: tone reframing + UX/code-quality lift.**

## Summary

A second external senior-review pass (May 2026) identified a mix of
critical, high, and medium issues. R15 ships the actionable subset —
everything except the public-deployment WCAG audit (out of scope: this
is a personal research project, not a public service) and the
1–2-week custom-data upload to the Lab. Eight focused changes land
together because the senior review framed them as a cluster of
"the project promises a polished portal but the surface is patchy".

This is a **minor release** (0.7.x → 0.8.0) because the design-system
spec moved from v0.1.0 to v0.1.1 (ADR-010) and the project tone
shifted from "ANL-aligned" to "personal-research, unaffiliated."
No note content changed.

## What changed

### 1. Tone reframing — personal research project, unaffiliated

- `README.md` now opens with a personal-research disclaimer block:
  not an APS/ANL/DOE property, original data owners not consulted
  about public deployment, intended for local/private use only.
- `CLAUDE.md` "Project Identity" carries the same disclaimer plus an
  instruction to future agents to never frame the project as official
  or wire up production deploy surfaces.
- `docs/02_design/decisions/ADR-005.md` is amended (status bumped to
  v0.2.0): "Argonne-aligned design tokens" → "Argonne-inspired
  design tokens (personal research)". DOE-stakeholder-demo framing
  removed; alternatives section now says "ANL-inspired
  (unaffiliated)" rather than "aligned."
- `explorer/app.py` landing hero replaces the "Navigate synchrotron
  data analysis knowledge at Argonne's Advanced Photon Source"
  tagline with "Personal study notes on synchrotron data analysis"
  + a one-line "unaffiliated personal research" subtitle.
- `scripts/build_static_site.py` mirrors the same on the GitHub
  Pages output. The static-site footer was reframed: the "This
  research used resources of the Advanced Photon Source… DOE
  Contract DE-AC02-06CH11357" paragraph (which implied institutional
  sponsorship) is replaced with a personal-research disclaimer
  + data-source acknowledgement.

### 2. ADR-010 + design_system.md v0.1.1 — formalise the R7 darkening

The R7 accessibility iteration darkened two cluster accents in code
(`#00A3E0` → `#0085C0`, `#F47B20` → `#D86510`) so they pass WCAG
2.1 AA-large, but `design_system.md` and ADR-005 still listed the
original hex values. ADR-010 formalises the change, lists both
versions in a token change-history table, and explains why the
historic wireframes (`docs/02_design/wireframes/html/*_v0.1.html`)
keep the pre-darkening hex — per invariant #5, versioned wireframes
are immutable; ADR-010 flags them as historical reference.

### 3. `lib/routing.py::query_param` — single source of truth

Before R15, `_query_param(...)` was copy-pasted across five pages
(`cluster_page.py`, `0_Knowledge_Graph.py`, `5_Troubleshooter.py`,
`6_Search.py`, `4_Experiment.py`) with subtly divergent `unquote`
behaviour. The Experiment page's local copy skipped URL decoding to
preserve slug-safe recipe ids. The new `lib.routing.query_param(name,
*, decode=True)` centralises both behaviours: the four pages that
want decoded human text use the default, the Experiment page passes
`decode=False`. New `tests/test_routing.py` covers 7 scenarios
(absent, list value, empty list, percent escapes on/off, non-string
coercion).

### 4. Design tokens as CSS custom properties

`explorer/assets/styles.css` now declares the full palette under
`:root { --color-primary: …; … }`, and the same `:root` block is
embedded in the static site's `SITE_LAYOUT_CSS`. Inline Python
`style="color:#0033A0"` strings across the four highest-density
pages (landing, cluster, Search result card, Lab narrative cards
and impact banner, Troubleshooter h1) were rewritten to
`style="color:var(--color-primary)"`. A future palette tweak no
longer needs nine simultaneous edits.

Three reusable CSS classes also landed so Python pages stop hand-
rolling the same accent-stripe markup:

- `.eberlight-card--accent-{primary,secondary,build,success,warning,error}`
- `.eberlight-banner` + `--info/--success/--warning/--error`
- `.eberlight-onboarding` for the new scenario picker

The Lab impact banner (which previously dictionary-looked-up bg+fg
hex per state) now picks one CSS class. The 3-card narrative row
above the parameter sliders likewise uses the new card variants.

### 5. `prefers-reduced-motion` honoured

`explorer/assets/styles.css` and the static-site CSS now include the
standard `@media (prefers-reduced-motion: reduce) { … }` block that
collapses animations and transitions to ~0ms. Cluster-card lift,
chip hover, smooth-scroll, and the onboarding card hover all respect
the user preference now.

### 6. `render_note_view()` split into three helpers

The 17-parameter god function (`components/note_view.py:126-223`) is
now a thin 30-line orchestrator that delegates to
`_render_note_main_column(...)` (h1, permalink button, metrics row,
body or section tabs, notebooks, prev/next) and
`_render_note_meta_column(...)` (TOC + metadata aside). The
signature is unchanged so all callers keep working; the maintenance
surface for "fix the metadata aside" or "tweak the body pipeline"
is now isolated. Inline `#0033A0`/`#555`/`#888` strings in the file
were replaced with `var(--color-*)` references; the metadata-panel
label markup was lifted into a single shared template.

### 7. Cluster-page orientation

Cluster landing pages (Discover/Explore/Build) had only an h1 +
content-list description, leaving first-time visitors without
"when should I use this cluster" context. The shared
`render_cluster_page` now renders three additional elements:

- A stats line: `35 notes · 2 folders · last reviewed 2026-05-08`.
- A tagline keyed by cluster ("Start here when you have a sample
  and need to choose a modality…").
- A "💡 first steps" line with bold-highlighted folder names so a
  visitor sees a concrete entry point before scrolling to the
  compare table.

The static-site mirror reproduces all three lines via the new
`_cluster_orientation_html(...)` helper.

### 8. Onboarding scenario picker on the landing

Above the three cluster cards, the landing now shows a "New here?
Pick your scenario" panel with three task-oriented cards:

- 🔬 "I have a sample to analyse" → Explore cluster
- 🩺 "I see something weird in my data" → Troubleshooter
- 🧪 "I want to try a noise-mitigation method hands-on" → Lab

The card grid collapses to one column on mobile via the existing
720px breakpoint. The static site mirrors the same widget; the
scenario destinations resolve to the Pages stubs for the
interactive surfaces.

### 9. Glossary auto-link

`08_references/glossary.md` (60+ terms) is now auto-linked into
every note body. After markdown rendering, `lib.glossary.annotate_html`
walks the resulting HTML, wraps the *first* occurrence of each
known glossary term in
`<abbr class="eberlight-glossary" title="…">…</abbr>`, and skips
text inside `<code>`, `<pre>`, `<a>`, `<abbr>`, and `<h1>`–`<h6>`
so headings stay clean and links don't nest. Sorting longest-first
prevents `APS-U` from being shadowed by `APS`; `\b` word boundaries
prevent `APS` from matching inside `GAPS`. The annotator is shared
between the Streamlit body renderer and the static-site note
builder, so both surfaces show the same tooltips.

`tests/test_glossary.py` covers 14 scenarios.

### 10. AppTest smoke tests for the 5 interactive pages

Knowledge Graph (~400 LOC), Interactive Lab (~514 LOC),
Troubleshooter (~301 LOC), Search (~218 LOC), and the landing all
had zero automated coverage. `tests/test_pages_smoke.py` uses
`streamlit.testing.v1.AppTest` to import-and-run each page
in-process and asserts (a) no exception bubbled out, and (b) the
page emitted at least one element to its body — so the
fail-silently-via-early-`st.stop()` mode is also caught. The tests
skip cleanly on Streamlit < 1.28 so the rest of the suite keeps
collecting.

## Files touched

### Created
- `docs/02_design/decisions/ADR-010.md` — token darkening + tone reframing
- `docs/05_release/release_notes/explorer-v0.8.0.md` (this file)
- `explorer/lib/routing.py` — single-source query-param helper
- `explorer/lib/glossary.py` — auto-link annotator
- `explorer/tests/test_routing.py` — 7 unit tests
- `explorer/tests/test_glossary.py` — 14 unit tests
- `explorer/tests/test_pages_smoke.py` — AppTest smokes for 8 page entries

### Modified
- `README.md` — personal-research disclaimer block at the top
- `CLAUDE.md` — project-identity reframing + agent-scope guard
- `docs/02_design/design_system.md` — bumped to v0.1.1
- `docs/02_design/decisions/ADR-005.md` — bumped to v0.2.0 (reframed)
- `explorer/app.py` — hero copy, onboarding widget, tokenised inline styles
- `explorer/assets/styles.css` — `:root` palette, motion preference,
  card variants, banner variants, onboarding, glossary abbr
- `explorer/lib/cluster_page.py` — query-param helper migration,
  cluster orientation section
- `explorer/pages/0_Knowledge_Graph.py` — query-param helper migration
- `explorer/pages/4_Experiment.py` — banner + narrative cards via classes
- `explorer/pages/5_Troubleshooter.py` — query-param helper migration
- `explorer/pages/6_Search.py` — query-param helper migration + tokens
- `explorer/components/note_view.py` — function split, glossary wiring,
  tokenised inline styles
- `explorer/components/header.py` — docstring rename
- `scripts/build_static_site.py` — mirrors all the above: `:root` block,
  disclaimer banner, onboarding picker, cluster orientation, glossary
  annotate-on-render, footer reframing

## Tests

`pytest` results locally (Python 3.13):

- `test_routing.py` — 7 new tests
- `test_glossary.py` — 14 new tests
- `test_pages_smoke.py` — 16 new tests (8 pages × 2 assertions),
  plus 1 import-regression test
- All pre-existing tests continue to pass — no behaviour change for
  cluster/note/troubleshooter/lab loaders.

## Traceability

- **Tone**: ADR-005 v0.2.0, ADR-010, CLAUDE.md project identity, README.md.
- **Tokens**: ADR-010, DS-001 v0.1.1, NFR-001 contrast targets, TST-002.
- **Routing**: senior-review action item #2.
- **Note view split**: senior-review action item #6.
- **Cluster orientation**: senior-review action item #2 (top-priority).
- **Onboarding**: senior-review action item #8, US-006 partial coverage.
- **Glossary**: senior-review action item #9, US-007.
- **Smoke tests**: senior-review action item #7.
- **Static-site mirror**: invariant #9, ADR-007, REL-E073 § Static-site
  mirror.

## Out of scope (deferred)

- **Public WCAG axe/Lighthouse audit** (senior-review #1). The project
  is personal research, not a public service, so a formal audit is
  not warranted. Local accessibility infrastructure (skip-link, ARIA
  labels, contrast tests, motion-reduce media query) is in place.
- **Custom-data upload to the Lab** (senior-review #10). 1–2 weeks of
  work; would need an async job runner, modality auto-detection, and
  storage policy. Out of scope for this release.
