---
doc_id: REL-E061
title: "Release Notes — explorer-v0.6.1"
status: draft
version: 0.6.1
last_updated: 2026-05-07
supersedes: null
related: [REL-E060]
---

# Release Notes — explorer-v0.6.1

**Phase R10 — first-impression UX polish.**

## Summary

A user-perspective review after the R1–R9 feature build identified four
critical (P0) bugs that broke the first-time experience even though the
underlying capabilities worked. R10 fixes all four plus eight P1 polish
items, with no new feature surface introduced — the new explorer now
matches the legacy on all features _and_ on first-impression quality.

## What's Fixed

### Critical (P0)

- **P0-1 — Header cluster nav links re-enabled.** A leftover
  ``pointer-events: none; opacity: 0.5`` rule in
  ``explorer/assets/styles.css`` was dimming and disabling the
  Discover / Explore / Build / Experiment links in the Streamlit
  shell (the static-site mirror had already worked around this with
  ``!important``). The disabled rule is replaced with a real hover /
  active state.
- **P0-2 — Landing cluster cards are now clickable.** The 3-card grid
  on the landing page was inert ``<div>``s with "Enter →" text but
  no anchor wrapping. R10 wraps each card in
  ``<a href="/Discover|Explore|Build">`` and adds a hover lift +
  focus-visible outline.
- **P0-3 — Permalink button now actually copies.** Streamlit's
  ``st.markdown(unsafe_allow_html=True)`` sanitises inline ``onclick``
  handlers, so the previous button looked clickable but never wrote
  to the clipboard. R10 routes the button through
  ``streamlit.components.v1.html`` which executes JS in an iframe,
  with a "Show full URL" expander as a copy-paste fallback.
- **P0-4 — Lab + Troubleshooter selectors moved out of the sidebar.**
  Recipe + sample pickers (Lab) and symptom + filter widgets
  (Troubleshooter) lived in ``with st.sidebar:`` blocks, so mobile
  users (sidebar permanently hidden) could not change them. They now
  sit in the main column under numbered stepper headings.

### High priority (P1)

- **P1-1 — Mobile media queries.** The header collapses to a stacked
  layout below 720px, the landing 4-CTA grid drops to a single column,
  and the cluster card grid reflows. Cards shrink padding below 540px.
- **P1-2 — Compare-table title column.** Previously emitted
  ``<a href>...</a>`` into a ``st.dataframe`` cell which renders raw
  HTML. R10 splits into a plain ``Title`` column plus a real
  ``LinkColumn`` rendering ``Open →`` per row.
- **P1-3 — TOC anchor links land on the right heading.** Previously
  our slug used a custom regex that diverged from Python-Markdown's
  ``toc`` extension on punctuation (apostrophes, em-dashes), so the
  TOC sidebar pointed at IDs the rendered HTML did not have. We now
  delegate to the same NFKD-aware slugify routine and add
  ``scroll-margin-top: 72px`` + ``scroll-behavior: smooth`` so the
  jump lands cleanly.
- **P1-4 — Section Tabs render Mermaid blocks.** The tab view called
  ``_md_to_html`` directly, dropping the diagram-iframe split that
  ``_render_body_with_mermaid`` does. Now each tab routes through the
  same renderer the default view uses.
- **P1-5 — Empty search results give the user somewhere to go.**
  ``Index.suggest()`` (new) returns indexed terms close to the user's
  query (prefix-match, ranked by document frequency); the empty-state
  shows "Did you mean: …", a Knowledge Graph fallback link, and 5
  popular queries.
- **P1-6 — Knowledge Graph default view less crowded.** Previously
  all six entity kinds were on (100+ nodes packed tight). Default is
  now the 4-layer "core" view — modality + method + recipe + noise —
  with paper + tool layers opt-in via the existing kind filter.
- **P1-7 — Numbered stepper UI on Lab + Troubleshooter.** Lab now
  shows 1️⃣ Pick recipe → 2️⃣ Pick sample → 3️⃣ Tune parameters →
  4️⃣ Compare before / after. Troubleshooter shows 1️⃣ Pick symptom
  → 2️⃣ Narrow down → 3️⃣ Read differential diagnoses.
- **P1-8 — ``last_reviewed`` surfaced in the metadata panel.** DC-001
  defines the optional ``last_reviewed`` frontmatter field but no
  page rendered it. R10 reads it onto ``Note`` and shows it as a
  "Last reviewed" row in the right-side metadata panel.

## Tests (+5 new)

- ``test_extract_toc_anchor_matches_python_markdown_slugify`` — drift
  protection that verifies our slug equals
  ``markdown.extensions.toc.slugify`` on a battery of real-world
  punctuation cases.
- ``test_suggest_offers_close_terms_on_typo`` /
  ``test_suggest_returns_empty_on_blank_query`` /
  ``test_suggest_does_not_return_exact_query_term`` — three cases for
  the new empty-search suggestions.
- ``test_last_reviewed_frontmatter_parses`` — DC-001 round-trip.

Total: **256 passed** (was 251 in R9). ``ruff check / format`` clean.

## What's Unchanged

- No new dependencies.
- No data shape changes (the new optional ``last_reviewed`` field on
  ``Note`` defaults to ``None``).
- No notes were modified.
- ADR / governance / static-site mirror unchanged.

This is a patch release (0.6.0 → 0.6.1) because the surface area
didn't grow; we just fixed how existing surfaces behave.
