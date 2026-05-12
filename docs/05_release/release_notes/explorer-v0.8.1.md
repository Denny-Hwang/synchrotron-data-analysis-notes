---
doc_id: REL-E081
title: "Release Notes — explorer-v0.8.1"
status: draft
version: 0.8.1
last_updated: 2026-05-11
supersedes: null
related: [REL-E080, ADR-010, DS-001]
---

# Release Notes — explorer-v0.8.1

**Phase R15.1 — re-review follow-ups (bugs + medium issues).**

## Summary

After REL-E080 landed, a second senior-review pass identified 13 small
items (4 bugs introduced by R15's own changes, 4 still-open structure
issues, 5 micro-cleanups). R15.1 ships all of them. Patch release —
no schema changes, no breaking behaviour.

## Bug fixes (introduced by REL-E080)

### B1 — Glossary "first occurrence per note" was actually per-segment

`components/note_view.py::_render_body_with_mermaid` walks the body in
text segments around each ``‌```mermaid`` block, calling `_md_to_html`
per segment. `annotate_html` allocated a fresh `used: set[str]` on
every call, so the same glossary term could be wrapped 2+ times in a
note that contained Mermaid diagrams. Fix: `annotate_html` now
accepts an optional shared `used` set; the body renderer allocates
one and threads it through every segment.

### B2 — Glossary regex recompiled per note in the static-site build

The static-site builder called `annotate_html` once per note (188
notes per build) and the per-call `_build_match_regex(~60 terms)`
alternation compile was hot. Added a thin `_cached_match_regex`
wrapper with `@lru_cache(maxsize=4)` keyed on the term tuple. Build
walltime drops by ~1 second; memory pressure no longer linear in the
note count.

### B3 — `<abbr>` keyboard focus

`<abbr>` elements do not naturally receive focus, so keyboard /
screen-reader users could not reach glossary tooltips at all. Added
`tabindex="0"` to every emitted `<abbr>` plus a `:focus-visible`
outline (primary colour, 2px) and a subtle banner-tinted background
in `styles.css` so the focus state is unmistakable. Full popover
replacement of `title=""` is deferred — this is the minimum
keyboard-accessibility fix.

### B4 — Static-site scenario cards routing to interactive stubs

The REL-E080 onboarding scenario cards on the landing read like
promises of running features. On the static mirror, two of the three
("weird image" → Troubleshooter stub, "hands-on method" → Lab stub)
landed the visitor on a "run Streamlit locally" page — a jarring
mismatch. The static-site versions now carry a small
`(needs local Streamlit)` suffix in the card title so the
expectation is set before the click.

## Structure / UX items

### S1 — "Related views" aside on note detail

Long-form readers deep in a note had no 1-click path to the power
pages (Knowledge Graph, Troubleshooter, Search) — they had to bounce
back to the landing. New `_render_related_views` helper appends a
small aside below the metadata panel with up to four jumps:

- 🧠 Knowledge Graph
- 🩺 Troubleshooter
- 📚 Other `{modality}` notes (only when the note declares a modality)
- 🔎 Search across all notes

`render_note_view(..., related_views=...)` keyword param is optional
so existing callers stay valid. Mirrored on the static site via
`_related_views_html`.

### S2 — Table / Cards layout toggle on cluster pages

The R11 redesign deleted the card grid in favour of a compare-table.
Power users like the dense table, but new visitors wanted to *browse*.
Cluster pages now ship a `📋 Table` / `🃏 Cards` pill row above the
content; the toggle threads through `?layout=cards` on the Streamlit
side and through sibling output files (`discover.html` +
`discover-cards.html`, etc.) on the static side. Folder / tag
filters carry through both layouts.

### S3 — ZoomIndicator marked deferred in the design system

The spec'd-but-never-implemented `ZoomIndicator` component
(`design_system.md:134-153`) has been flagged "Deferred —
superseded by the Detail Level pills on note detail and the REL-E080
cluster orientation header." The spec is preserved as a design
reference. DS-001 bumped to v0.1.2.

### S4 — "Replay-only Lab" banner

The Interactive Lab cannot ingest user-supplied data (deferred to
ADR-008 follow-up, ~1–2 weeks of work). A new banner immediately
under the Lab h1 says so explicitly, points readers at
`experiments/<recipe>/pipeline.py` and `experiments/README.md` for
applying recipes to their own data outside the app, and removes the
"why is there no upload button?" mystery.

### S5 — Tablet breakpoint + OS-level dark mode

- New `@media (max-width: 1024px) and (min-width: 721px)` breakpoint
  collapses the 3-column onboarding grid to 2 columns for iPad-class
  devices that were previously getting the desktop layout.
- New `@media (prefers-color-scheme: dark)` `:root` block re-binds the
  full token palette to a dark variant. Brand hues lift to lighter
  variants so primary/secondary keep ≥4.5:1 contrast against the
  dark surface. Streamlit's native widgets stay on their
  `config.toml` theme (no Streamlit theme migration in this release);
  the `.eberlight-*` components and any inline `var(--color-*)`
  styles follow the OS preference. Both the Streamlit asset CSS and
  the static-site CSS pick up the dark variant.

## Code-quality / hygiene

| # | Change | Where |
|---|---|---|
| M1 | Recipe cache key auto-derived from `dataclasses.fields(Recipe)`; `_CACHE_VERSION` magic string removed | `pages/4_Experiment.py` |
| M2 | GitHub repo URL overridable via `EBERLIGHT_GITHUB_BLOB_PREFIX` / `EBERLIGHT_GITHUB_REPO_URL` env vars | `lib/cluster_page.py`, `scripts/build_static_site.py` |
| M3 | DS-001 status promoted **draft → accepted** at v0.1.2 | `docs/02_design/design_system.md` |
| M4 | `_repo_root_from_explorer` walks ancestors looking for the `10_interactive_lab` marker dir if the default 2-up parent doesn't match — defensive against future file moves | `components/note_view.py` |
| M5 | TOC sidebar now surfaces at **L1** as well as L2 | `lib/cluster_page.py` |
| M6 | Cluster first-steps copy pre-bolded with `<b>` HTML; removed the inline `re.sub(r"\*\*…\*\*")` markdown-to-HTML hack | `lib/cluster_page.py` |
| M7 | Static-site footer softened: "centred on the eBERlight program at the APS" → "uses synchrotron X-ray data analysis as a representative case study, with eBERlight … as one source-material reference" | `scripts/build_static_site.py::_footer_html` |
| M8 | `load_glossary` `lru_cache(maxsize=4)` → `maxsize=1` (only ever called with one repo root) | `lib/glossary.py` |

## Tests

`pytest` results locally:

- `test_glossary.py` — 2 new tests:
  - `test_annotate_emits_tabindex_for_keyboard_focus` (B3 verification)
  - `test_annotate_shared_used_set_dedups_across_calls` (B1 contract)
- Existing 325 tests in REL-E080 continue to pass — no behaviour change.

## Files touched

### Created
- `docs/05_release/release_notes/explorer-v0.8.1.md` (this file)

### Modified
- `README.md` — badge bump to v0.8.1 + REL-E081 entry
- `CHANGELOG.md` — v0.8.1 patch entry
- `docs/02_design/design_system.md` — v0.1.2, ZoomIndicator deferred,
  prefers-color-scheme noted
- `explorer/app.py` — (no functional change in this release)
- `explorer/assets/styles.css` — dark-mode tokens, tablet breakpoint,
  glossary focus styles
- `explorer/components/note_view.py` — B1 shared-set wiring, related-
  views helper, M4 hardening
- `explorer/lib/cluster_page.py` — S1 related-views builder, S2 layout
  toggle, M2 env override, M5 TOC at L1, M6 HTML-bold copy
- `explorer/lib/glossary.py` — B1 `used` parameter, B2 regex cache,
  B3 `tabindex="0"`, M8 maxsize=1
- `explorer/pages/4_Experiment.py` — S4 replay-only banner, M1 cache
  key from Recipe fields
- `explorer/tests/test_glossary.py` — 4 new tests for B1 + B3
- `scripts/build_static_site.py` — full mirror of the above:
  dark-mode + tablet CSS, B4 needs-Streamlit label, S1 related-views
  aside on notes, S2 cards-layout sibling pages + toggle, M2 env
  override, M7 footer tone

## Traceability

- **B1**: senior-review action item #B1.
- **B2**: senior-review action item #B2.
- **B3**: senior-review action item #B3, NFR-001 keyboard access.
- **B4**: senior-review action item #B4.
- **S1**: senior-review action item #S1, addresses long-form-reader IA gap.
- **S2**: senior-review action item #S2, restores the R10-era browsing surface.
- **S3**: DS-001 v0.1.2, ADR-010 follow-up.
- **S4**: senior-review action item #S4, ADR-008 follow-up.
- **S5**: senior-review action item #S5.
- **M1–M8**: senior-review medium-priority list.
