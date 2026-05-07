---
doc_id: REL-E071
title: "Release Notes — explorer-v0.7.1"
status: draft
version: 0.7.1
last_updated: 2026-05-08
supersedes: null
related: [REL-E070]
---

# Release Notes — explorer-v0.7.1

**Phase R12 — four-bug bugfix release.**

## Summary

A real-user review on the v0.7.0 deploy surfaced four issues that
v0.7.0 itself introduced or exposed:

1. The Interactive Lab page crashed with an ``AttributeError`` on
   ``recipe.problem`` when the page first loaded against a cached
   pre-R11 Recipe instance.
2. The cluster compare-table showed many rows with identical titles
   ("Readme", "Data Format", "Ai Ml Methods" repeated for every
   modality / method / paper note).
3. The Troubleshooter "📖 Full guide →" link did nothing visible —
   the click only refreshed the current page.
4. Bibliography author names rendered with raw BibTeX escapes:
   ``J{\\'e}r{\\^o}me``, ``Schr{\\"o}dinger``, ``Stra{\\ss}e``.

R12 fixes all four. No new features; pure bugfix release (0.7.0 →
0.7.1).

## What's Fixed

### B1 — `recipe.problem` AttributeError on cached Recipe

R11 added three optional narrative fields (``problem`` / ``fix`` /
``observe``) to the ``Recipe`` dataclass. Streamlit's
``@st.cache_resource`` keys cached objects by the function's bytecode
hash, **not** by the dataclass schema; on Streamlit Cloud the
deployment loaded the new bytecode but the cache still held pickled
``Recipe`` instances from the previous deploy that lacked the new
fields. The Lab page then crashed at the first
``recipe.problem`` access.

R12:

- Wrapped every new-field access in ``getattr(recipe, "problem", "")``
  so the page renders cleanly regardless of cache state.
- Renamed the cache helpers (``_cached_recipes`` →
  ``_cached_recipes_v3``, etc.) to bust the existing cache key on the
  next deploy. Old aliases are kept so any legacy import still works.

### B2 — Compare-table same-name rows

Notes without YAML frontmatter used to fall back to
``_title_from_filename(stem)``, which produces "Readme" /
"Data Format" / "Ai Ml Methods" — identical for every sibling folder.
The compare-table is then full of indistinguishable rows.

R12 introduces a three-step title-resolution waterfall in
``lib.notes._parse_note``:

1. ``frontmatter.title`` (canonical when set).
2. **First H1 in the body** (almost every note has one, and H1s are
   author-written specific titles like "Crystallography Data Formats"
   or "XRF Microscopy Data Formats").
3. Parent-aware filename fallback — generic basenames
   (``README``, ``data_format``, ``ai_ml_methods``, ``index``) get
   prefixed with their parent directory so even notes without an H1
   end up unique ("Crystallography — Data Format").

Plus a new "Section" column on the cluster compare-table carries the
immediate parent sub-folder, so even if two notes happen to share an
H1 verbatim the rows are still tellable apart.

### B3 — Troubleshooter "Full guide" link does nothing

The link emitted ``?note=09_noise_catalog/...`` but the Troubleshooter
page itself doesn't handle ``?note=`` — only the cluster pages
(`/Discover`, `/Explore`, `/Build`) do. Clicking the link just
appended the query to the current Troubleshooter URL and Streamlit
re-rendered the same page.

R12 routes guide links through ``/Explore?note=09_noise_catalog/...``
(noise-catalog notes belong to the Explore cluster). Relative
``../foo.md`` paths inside the catalog are resolved correctly. As a
companion fix, the "▶ Run experiment" link on diagnoses now actually
selects the matching recipe on the Lab page —
``/Experiment?recipe=<recipe_id>`` is read at page load and pre-fills
the recipe selectbox.

### B4 — BibTeX accent escapes decoded to Unicode

The two bundled `.bib` files use LaTeX accent escapes
(``{\\'e}``, ``{\\^o}``, ``{\\"u}``, ``{\\ss}``, ``\\AA``) for
non-ASCII author names. The bibliography page rendered them
literally, leaving names like ``J{\\'e}r{\\^o}me`` and
``Schr{\\"o}dinger`` in the byline.

R12 adds ``_decode_latex_accents`` in ``lib/bibliography.py``
covering the full standard set:

- Accent + base — `\'`, `` \` ``, `\^`, `\"`, `\~`, `\c`, `\v`
  for all common base letters.
- Bare commands — `\ss` (ß), `\aa` / `\AA` (å / Å), `\o` / `\O`
  (ø / Ø), `\ae` / `\oe` / `\l` and their uppercase variants, with
  the LaTeX-style trailing-space separator (`\AA ngström` →
  `Ångström`).
- Braced bare commands — `{\ss}`, `{\aa}`, `{\oe}`.

The decoder runs on every cleaned field (`_clean`) and on the raw
`author` field before splitting on `` and `` so the splitter
doesn't see escape sequences.

## Tests

`pytest explorer/tests/` → **264 passed** (was 256 in R11; +8 new
tests for B2 H1 fallback / parent-prefix rules and B4 accent
decoding).

`ruff check / format --check explorer/ experiments/ scripts/` clean.

`streamlit run explorer/app.py` → `/_stcore/health` 200 OK.

## What's Unchanged

- Recipe schema: the optional narrative fields from R11 are still
  optional. R10/R9/R8 features unchanged.
- No notes were modified.
- No new dependencies.
- Static-site mirror builds clean (188 notes + 4 stubs + 5 recipes).
