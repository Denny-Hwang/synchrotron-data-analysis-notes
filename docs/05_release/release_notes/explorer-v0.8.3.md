---
doc_id: REL-E083
title: "Release Notes — explorer-v0.8.3"
status: draft
version: 0.8.3
last_updated: 2026-05-14
supersedes: null
related: [REL-E080, REL-E081, REL-E082, PRD-001, NFR-001, DS-001, VIS-001, RMP-001, ADR-005, ADR-010]
---

# Release Notes — explorer-v0.8.3

**Comprehensive review + framing-leftover cleanup. Patch release.**

## Summary

A four-axis review of the repository (institutional-tone leftovers,
explorer ↔ static-site parity, doc/invariant consistency, code
quality / bugs / tests) surfaced a mix of small issues left behind
by the REL-E080 → REL-E082 reframing waves and a handful of
unrelated code-quality nits. This patch ships the actionable subset:
all the leftover "personal-research disclaimer" gaps and four
concrete bug fixes. No feature changes.

The review explicitly deferred broader UX changes (header search /
skip-link / metadata-panel divergence between Streamlit and static,
landing disclaimer banner port to Streamlit, new test files for
cluster_page / note_view / visjs_graph) — those are clean follow-up
work, not REL-E082 fallout.

## What changed

### 1. Footer parity restored (invariant #9)

REL-E080 reframed the static-site footer and REL-E082 reframed the
Streamlit footer, but the two footers ended up using independently-
drafted variants on the same theme — REL-E082's commit message even
said "parity restored" but the strings diverged. The static-site
`_footer_html` in `scripts/build_static_site.py` now renders the
same disclaimer text the Streamlit footer renders, verbatim:

- Lead line: **"Personal eBERlight archive — not an official site."**
- Explicit unaffiliated framing for **ANL, APS, DOE, and the
  eBERlight program**
- "For the actual research … please refer to the official sites"
  paragraph
- Link labels `APS (official — actual research here)` /
  `eBERlight (official — actual research here)` /
  `Repository`

New tests in `explorer/tests/test_build_static_site.py` pin the
static-site footer to these invariants so it can't regress
independently of the Streamlit copy.

### 2. README — DOE-contract acknowledgment removed + staleness fixed

`README.md` had three classes of staleness:

- **Institutional residue.** The last paragraph still carried the
  DOE Contract No. DE-AC02-06CH11357 acknowledgment that REL-E082
  removed from the Streamlit footer and NFR-001 rewrote to forbid.
  Replaced with the personal-archive disclaimer block that mirrors
  the rendered footer.
- **Stale version mentions.** Badge + body text still said
  `explorer-v0.8.1` / `explorer-v0.8.0` in three places. Bumped to
  `explorer-v0.8.3`.
- **Stale counters.** Recipe count `5 noise-mitigation recipes` →
  `14`. Test count `264 tests` → `300+`. ADR count `9 ADRs` →
  `10 ADRs` (adds ADR-010). Recipe gallery tree and Lab table
  expanded with the 9 R14 recipes (ring_artifact_neutron,
  low_dose_denoise, beam_hardening, phase_unwrap, tv_denoise,
  nlm_denoise, bilateral_denoise, wavelet_denoise,
  inpaint_dead_pixel). Release-note bullet list extended through
  REL-E083.

### 3. DS-001 Footer component spec rewritten

`docs/02_design/design_system.md` Footer anatomy still mandated
"DOE acknowledgment text (Contract No. DE-AC02-06CH11357)" — a
direct contradiction with NFR-001 / FR-010 as rewritten in
REL-E082. Rewrote the Purpose, Anatomy, and Do/Don't blocks to
match the personal-archive disclaimer + reference-pointers spec.
Added a change-history block at the top. Bumped DS-001 to
**v0.1.3**.

### 4. Other framing-leftover docs updated

- **`docs/01_requirements/PRD.md`** — bumped to v0.3.0
  (`last_updated: 2026-05-14`). Scope bullet "ANL/APS-aligned
  visual design" → "ANL/APS-inspired … (unaffiliated personal-
  research framing — see ADR-005 + ADR-010)". FR-008 (ZoomIndicator)
  now annotated `(Deferred per DS-001 v0.1.2 / REL-E081 S3 …)` so
  PRD doesn't list a deferred feature as a hard requirement.
- **`docs/01_requirements/non_functional.md`** — bumped to v0.2.0
  (`last_updated: 2026-05-14`). NFR Compliance block was already
  rewritten in REL-E082; only the frontmatter needed bumping.
- **`docs/00_product/vision.md`** — bumped to v0.2.0 with a leading
  framing disclaimer. Removed "public-facing or interview contexts"
  (contradicted CLAUDE.md "do not deploy publicly"), "DOE users",
  "ANL-branded experience", and the "complements the official BER
  program website" wording. Non-Goals now include "We are NOT
  publishing this app."
- **`docs/00_product/roadmap.md`** — Phase A "ANL-aligned color
  tokens" → "ANL/APS-inspired (unaffiliated)"; exit criterion "DOE
  Contract No. DE-AC02-06CH11357 visible in footer" → "Personal-
  archive disclaimer footer visible on every page".
- **`docs/02_design/decisions/ADR-005.md`** — removed "must convey
  institutional credibility" rejection rationale (the rest of
  ADR-005 already disclaims this; the line was internally
  contradictory).
- **`docs/02_design/decisions/ADR-001.md`** — "aligned with ANL/APS
  standards" → "inspired by ANL/APS materials (unaffiliated)".
  "ANL design system compliance" → "ANL/APS-inspired design
  system".
- **`docs/02_design/decisions/ADR-007.md`** — "ANL-aligned DS-001
  tokens" → "ANL/APS-inspired DS-001 tokens".
- **`docs/README.md`** — bumped to v0.2.0; ADR-010 added to the ADR
  table (was missing); ADR-005 label "Argonne-aligned" →
  "Argonne-inspired"; Release section now lists REL-E082 +
  REL-E083 explicitly instead of "planned".

### 5. Cross-reference repair

`docs/05_release/release_notes/explorer-v0.8.2.md` `related:` was
`[REL-E080, REL-E081, FR-010, NFR-001]`. `FR-010` is not a
standalone doc_id — it lives inside PRD-001. Corrected to
`[REL-E080, REL-E081, PRD-001, NFR-001]`.

### 6. Code bug fixes

- **B1 — `scripts/build_static_site.py:_git_iso_date`** forked
  `git log` once per emitted page (~200 forks per build). Mirrors
  exactly the issue REL-E080 fixed for the Streamlit footer;
  static-side fix was missed. Cached at module import via
  `_GIT_ISO_DATE = _git_iso_date()`; `_page_shell` now passes the
  constant.
- **B2 — `explorer/pages/4_Experiment.py:352-355`** had a dead
  `if sino_input.shape == sino_output.shape: … else: …` where both
  branches called `_difference_map(sino_input, sino_output)`
  identically. Collapsed to one call (`_difference_map` already
  centre-crops on shape mismatch internally).
- **B3 — `explorer/components/note_view.py`** built nbviewer URLs
  with a chained `.replace("/blob/", "/blob/", 1)` no-op
  self-replacement. Removed.
- **B4 — `explorer/pages/5_Troubleshooter.py`** built each card
  across three separate `st.markdown` calls, ending with a third
  call that emitted only `</p></div>` to close tags opened in the
  first call. Streamlit wraps each `st.markdown` in its own
  container, so the closing tags landed in the wrong DOM scope and
  produced malformed HTML (still rendered thanks to browser
  forgiveness, but failed HTML validation). Refactored to a single
  `st.markdown` call that emits the full balanced card.
- **B5 — `scripts/build_static_site.py` top-level import**
  pulled in `lib.experiments` which transitively imports `numpy`,
  so `python scripts/build_static_site.py --help` crashed in
  environments without the scientific stack. Moved the
  `from lib.experiments import load_recipes` into
  `_recipe_gallery_html()` and kept `Recipe` under
  `if TYPE_CHECKING` so type hints still work. `--help` now runs
  without numpy.

## Out of scope (clean follow-up work, not REL-E082 fallout)

- **Header search box / skip-link / metadata-panel divergences**
  between Streamlit and static — Streamlit ships `FR-009` search +
  WCAG skip-link + `last_reviewed` + clickable
  `publication_links`/`tool_links`; the static mirror omits each.
  All MEDIUM/LOW severity per the explorer↔static-site audit; a
  separate UX-parity PR is the cleaner shape.
- **Landing disclaimer banner.** The static-site landing has a
  `.disclaimer-banner` div under the hero; the Streamlit landing
  doesn't. Streamlit relies on the (below-the-fold) footer.
  Whether to port the banner up is a UI-semantics decision worth
  its own pass.
- **CSS token dedup.** `:root` + dark-mode token blocks are
  declared in both `explorer/assets/styles.css` and
  `scripts/build_static_site.py::SITE_LAYOUT_CSS`. They match
  today (concatenation order makes the explorer copy authoritative
  anyway), but the duplication is a drift hazard.
- **New test files** for `cluster_page.py`, `note_view.py`,
  `visjs_graph.py` (currently only smoke-covered). Worth a
  dedicated testing pass.
- **CLAUDE.md REL-N100 doc_id collision** — `notes-v0.10.0`
  currently uses `REL-N100`, but invariant #3 maps `notes-v1.0.0`
  → `REL-N100`. Needs a scheme revision; out of scope for this
  patch.

## Traceability

- Invariant #9 (Streamlit ⇄ static-site mirror) — now verbatim
  parity for the footer; pinned by tests.
- FR-010 / NFR-001 — fully implemented and consistent across
  Streamlit + static + spec docs.
- DS-001 v0.1.3 — Footer component anatomy spec rewritten.
- REL-E082 — this patch finishes the doc-tree fallout the earlier
  release missed.
