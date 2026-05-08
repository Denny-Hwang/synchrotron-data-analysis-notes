---
doc_id: REL-E072
title: "Release Notes — explorer-v0.7.2"
status: draft
version: 0.7.2
last_updated: 2026-05-08
supersedes: null
related: [REL-E071]
---

# Release Notes — explorer-v0.7.2

**Phase R13 — senior-engineer review acted on (perf + a11y + security).**

## Summary

A holistic review by an external senior application engineer identified
12 high-leverage improvements across performance, accessibility,
security, and code-quality axes. R13 ships the four highest-ROI fixes
plus a refreshed README. The remaining 8 (IA decision, `<main>` element
restructure, e2e harness, etc.) are queued as future phases.

This is a patch release because no user-facing surface changes — same
seven pages, same five recipes, same compare-table layout. What changes
is invisible to the eye but measurable: faster cold start, faster
navigation, no security foot-gun in the recipe loader, and every page
now ships a working skip-to-main-content link for keyboard / screen-
reader users.

## What's Fixed

### Rec #1 — Drop dead `plotly` + `networkx` dependencies

R11 replaced the Knowledge Graph renderer with vis.js (loaded from a
CDN inside an iframe — no Python dependency). The two heavy R2-era
packages stayed in `explorer/requirements.txt` for two more releases:
~30 MB of wheel weight + ~30 s extra cold start on Streamlit Cloud
for code that no longer imports either. R13 drops both. A new
`tests/test_dead_deps.py` is the drift catcher: the CI fails if
either package re-appears in `requirements.txt` or anywhere in
`explorer/**.py`.

### Rec #2 — Cache `load_notes()` and `_get_last_updated()`

Two hot paths the explorer was paying on every render:

- `lib/cluster_page.py` called `load_notes(repo_root)` directly
  on every cluster-page navigation. ADR-002:36 mandated a cache
  but it was never wired up. With 188 markdown files + YAML
  frontmatter parsing, this cost 150–300 ms per click.
- `components/footer.py::_get_last_updated()` shelled out to
  `git log -1 --format=%ci` on every render. Streamlit re-runs
  on every widget interaction, so a slider drag on the Lab page
  forked git 30+ times per second.

R13 wraps `load_notes` in `@st.cache_resource` and resolves the git
date once at module-import time (cached as `_LAST_UPDATED`).

### Rec #3 — Recipe-execution security allow-list

`lib/experiments.py::resolve_function` previously took any dotted
path from `recipe.yaml`'s `function:` field and ran
`importlib.import_module(...)`. A contributor whose PR was approved
without careful review could ship `function: os.system` and the
Lab would execute arbitrary commands.

R13 restricts the path to start with `experiments.` (and added
`experiments/_test_helpers.py` so the test fixtures continue to work
without an escape hatch). Two new tests assert (a) the bundled
recipes still resolve, (b) `os.system` / `subprocess.run` /
`shutil.rmtree` / `numpy.zeros` / `builtins.eval` all raise
`ValueError`.

### Rec #4 — WCAG 2.4.1 "Bypass Blocks" — wire the skip link

`lib/a11y.py::skip_link_html` had been defined since R7 but had
zero callers. Keyboard / screen-reader users had to tab through
the entire header on every page.

R13 makes `components/header.py::render_header` emit:

- The skip-to-main-content link as the very first focusable element.
- A `<a id="main-content" tabindex="-1">` anchor right after the
  header, so the skip link has somewhere to land.

The reveal-on-focus styling was moved out of an inline `style=`
attribute (which can't fake `:focus`) into a proper CSS class
(`.eberlight-skip-link` with `:focus` / `:focus-visible` rules).
A new test in `test_a11y.py` asserts every `render_header()` call
includes both the link and the anchor.

## What's Unchanged

- No notes were modified.
- No recipe behaviour changed.
- No new visible features.

## Tests

`pytest explorer/tests/` → **271 passed** (was 264 in R12).

New tests:

- `test_dead_deps.py::test_dead_deps_not_in_requirements`
- `test_dead_deps.py::test_dead_deps_not_imported_in_source`
- `test_experiments.py::test_resolve_function_rejects_non_experiments_namespace`
- `test_experiments.py::test_resolve_function_accepts_experiments_subpackages`
- `test_a11y.py::test_skip_link_uses_class_not_inline_position`
- `test_a11y.py::test_main_anchor_exists`
- `test_a11y.py::test_render_header_emits_skip_link_and_main_anchor`

Updated:

- `test_resolve_function_dotted_path` now uses an actual bundled
  recipe function instead of `numpy.zeros` (which the new allow-list
  rejects).
- `test_run_pipeline_invokes_function` now points at
  `experiments._test_helpers._add_scalar` (newly created module).

`ruff check / format --check explorer/ experiments/ scripts/` clean.
`streamlit run explorer/app.py` → `/_stcore/health` 200 OK.

## Documentation

The README was already updated in this same branch (commit `d9bdb63`)
to reflect R11 + R12 reality. No further README change required for
R13 since the fixes are invisible to the eye.

## Roadmap — remaining recommendations from the review

These are queued for later phases. None block this release.

| Rec | Description | Effort | Impact |
|---|---|:---:|:---:|
| #5 | SRI hashes on vis.js + Mermaid CDN scripts | S | Medium |
| #6 | Hero copy: lead with verb-first user benefit | S | Medium |
| #7 | Compare-table pagination / server-side filter | M | Medium |
| #8 | Hoist Mermaid extraction into shared `lib/markdown_render.py` | M | Medium |
| #9 | IA decision: drop `🧪 Experiment` from header nav OR hoist KG/Troubleshooter/Search to it | S | Medium |
| #10 | `streamlit.testing.v1.AppTest` e2e harness | M | High |
| #12 | Friendly error UX on Lab pipeline failure | S | Low |
