---
doc_id: REL-E073
title: "Release Notes — explorer-v0.7.3"
status: draft
version: 0.7.3
last_updated: 2026-05-08
supersedes: null
related: [REL-E072, REL-N100]
---

# Release Notes — explorer-v0.7.3

**Phase R14 — three rendering hotfixes + Interactive Lab content expansion.**

## Summary

Four-part release:

1. **Hotfix #1 — header HTML leak.** A CommonMark whitespace-only-line
   bug in `components/header.py` was leaking the closing `</div>` and
   the `#main-content` skip-link anchor into the header bar as visible
   raw HTML text on every page. Fixed by collapsing the header markup
   into a single logical line so Streamlit's mistune renderer no
   longer terminates the HTML block early.

2. **Hotfix #2 — `[object Object]` leak in every code block.**
   Streamlit's React frontend detects `<pre><code>` HTML inside
   `st.markdown(unsafe_allow_html=True)` and routes it to its native
   `stCode` component, which expects a *string* child. The legacy
   `_md_to_html` ran `codehilite` on every fenced code block,
   producing a tree of `<span class="kn">…</span>` Pygments tokens;
   React stringified those non-text children as the literal text
   `[object Object]`. **Every Python / shell / YAML code block in
   every note** showed this corruption, plus the entire L3 (Source)
   detail level. Dropping `codehilite` from the Streamlit-side
   `_md_to_html` (the static-site mirror keeps it; no React in the
   middle there) reverted code blocks to plain `<pre><code
   class="language-X">…</code></pre>`, which Streamlit's native
   component highlights via Prism.js and renders correctly. L3 now
   bypasses `_md_to_html` entirely and goes straight to
   `st.code(body, language="markdown")` — the natural fit for a
   "show me the raw source" view.

3. **Hotfix #3 — vis.js tooltips rendered HTML as text.**
   vis-network 9.x renders `node.title` strings via
   `document.createTextNode`, which escapes `<b>…</b>` / `<br>`
   markup so users see the raw tags as text in the Knowledge Graph
   tooltip. The component now converts each title HTML string to an
   actual `HTMLElement` client-side (via `innerHTML` on a wrapper
   div); a `titleHtml` indirection survives the
   `JSON.parse(JSON.stringify(...))` deep-clone the
   layout-mode-switcher uses to reset positions.

4. **Lab content expansion**: nine new recipes + four new bundled
   datasets (~9 MB total). The Lab now ships **14 recipes** (up
   from 5) covering the three case studies the user-feedback round
   asked for — neutron-CT ring artifacts, low-dose / photon-counting
   denoising, and 2-D phase unwrapping — plus six additional
   workhorse denoisers and a beam-hardening correction.

## Hotfix #1 — header HTML leak

### Symptom

Every page displayed `</div>` and `<a id="main-content" tabindex="-1"
aria-hidden="true"></a>` as visible raw text in the right-hand third
of the header bar, beside the search box.

### Root cause

`components/header.py::render_header` used a multi-line f-string with
4-space indentation, and `_search_form_html()` returned a string
starting with `\n`. After interpolation a line consisting of *eight
spaces only* appeared between `</nav>` and `<form …>`. Per the
CommonMark spec, a whitespace-only line ends an HTML block, so
mistune (Streamlit's markdown renderer) closed the outer
`<div class="eberlight-header">` block early and treated the
trailing `</div>` and `#main-content` anchor as **inline HTML inside
markdown**, where they were rendered as raw text.

### Fix

Both the header f-string and `_search_form_html()` are now single
logical lines with no embedded newlines. The header now renders as
one unambiguous HTML block on every page (Discover, Explore, Build,
Experiment, Troubleshooter, Search, Knowledge Graph, landing).

Static-site mirror (`scripts/build_static_site.py`) was unaffected
because it writes literal HTML, not markdown — no fix needed there.

## Hotfix #2 — `[object Object]` in every code block (two layers)

### Symptom

`[object Object],[object Object],[object Object],…` appeared in
**every** Python / shell / YAML code block that contained `#`
comments on a paragraph break, plus the entire L3 (Source) detail
level showed nothing but this text. Reproducible on Streamlit 1.57
(and likely back to 1.32 when the React markdown rewrite landed).

### Root cause — layer 1: codehilite spans

Streamlit's React frontend looks for `<pre><code>` HTML inside
`st.markdown(unsafe_allow_html=True)` and routes that subtree to its
native `stCode` component (visible in the DOM as
`data-testid="stCode"`). `stCode` expects a **string** child — that's
what its Prism.js highlighter operates on.

The legacy `note_view._md_to_html` ran the markdown library with
`codehilite` enabled, which delegates to Pygments. Pygments turns a
Python code block into a tree of `<span class="kn">import</span>
<span class="nn">numpy</span>…` tokens. When React received that
non-string child tree, it did the only fallback it knows: it called
`String(child)` on each node, producing `[object Object]` — the
default string form of a JS object.

### Root cause — layer 2 (R14.1): React re-parses code-block content

After the layer-1 fix dropped codehilite, the rendered HTML for the
Quick Diagnosis block on `09_noise_catalog/tomography/ring_artifact.md`
was clean text:

```html
<pre><code class="language-python">import numpy as np

# Load sinogram (2D: angles x detector_columns)
# Check for vertical stripes by computing column-wise standard deviation
col_std = np.std(sinogram, axis=0)
# Anomalous columns
outlier_cols = np.where(col_std > 3)[0]
</code></pre>
```

Yet playwright showed the rendered DOM *still* contained
`[object Object]`:

```html
<code class="language-python">
  <span class="token keyword">import</span> numpy
  ,[object Object],
  ,[object Object],
  ,[object Object],
  ,[object Object],
  <span>outlier_cols = ...</span>
</code>
```

Streamlit's React markdown renderer (with `unsafe_allow_html=True`)
**re-parses the inner content of `<code>` as markdown**. Lines
starting with `#` after a `\n\n` paragraph break get interpreted as
markdown headings → become `<h1>` React elements → React stringifies
them as `[object Object]` when handing the children to its code
component.

This is fundamental to how rehype-raw + react-markdown work; we cannot
fix it from the Python side except by **never letting code blocks go
through the markdown HTML round-trip**.

### Fix

The R14.1 fix replaces `_render_body_with_mermaid` with a generic
`_render_body_segmented` that walks the raw markdown body and routes
each chunk to the right Streamlit primitive:

| Chunk | Renderer | Why |
|---|---|---|
| ```` ```mermaid ```` block | `components.html(iframe)` | Live diagram via mermaid CDN |
| ```` ```X ```` block (any other lang) | `st.code(text, language=X)` | Bypasses React markdown re-parse entirely; Prism.js highlights raw text |
| Prose between blocks | `st.markdown(_md_to_html(prose), unsafe_allow_html=True)` | Standard rich rendering for tables, links, headings, etc. |

**Other fixes that flowed from this:**

* The L3 (Source) view, added in the layer-1 fix, was already routing
  through `st.code(body, language="markdown")` — same approach scaled
  to the whole body. Unchanged.
* `_normalise_language()` maps unknown language tags to `None` so
  Streamlit doesn't print a console warning per code block.
* The static-site mirror in `scripts/build_static_site.py` keeps
  `codehilite` (no React in the middle there; the Pygments span tree
  renders normally in any browser).

### Verified end-to-end

playwright + headless chromium against a fresh `streamlit run`:

* All 18 routes in the smoke-test sweep render without
  `[object Object]` (was 1 broken, now 0).
* 25 of 95 noise-catalog notes that contain ```` ```python ```` blocks
  with `#` comments — the exact bug pattern — render cleanly.
* Ring Artifact L2 page: was `text=6948 chars + [object Object] x15`,
  now `text=9504 chars` (the missing chars were the comment lines
  that used to render as `[object Object]`).
* Sample of 29 random notes across all 10 folders: all clean.

### Regression tests

| Test | Guards against |
|---|---|
| `test_md_to_html_does_not_emit_pygments_class_spans` | re-introducing codehilite |
| `test_md_to_html_emits_language_class_for_prism` | losing the `language-X` class Prism keys on |
| `test_render_body_with_mermaid_signature_no_highlight_css` | re-introducing the `highlight_css` pipeline |
| `test_render_body_segmented_routes_code_through_st_code` | code blocks ever going through `st.markdown(unsafe_allow_html)` again |
| `test_render_body_segmented_handles_mermaid_plus_code` | the mermaid+code mixed path regressing |
| `test_render_body_segmented_unknown_language_falls_back_to_plain` | unknown language tags spamming the console |
| `test_render_body_segmented_no_code_falls_back_to_markdown` | the fast prose-only path regressing |
| `test_md_to_html_no_longer_emits_pre_code_for_streamlit_path` | a developer accidentally feeding a code block back through `_md_to_html` |

## Hotfix #3 — vis.js Knowledge-Graph tooltips showed raw `<b>…</b>` text

### Symptom

Hovering a node in the Knowledge Graph displayed
`<b>Electron Microscopy</b><br><i>modality</i>` as **literal text**
in the tooltip popup, including the angle brackets, instead of the
intended bold-and-italic two-line layout.

### Root cause

vis-network 9.x renders the `title` field with
`document.createTextNode(title)`, which intentionally escapes HTML.
The library only renders HTML when the caller passes an actual
`HTMLElement` instance, not a string.

### Fix

`components/visjs_graph.py` now ships an `htmlToElement(html)` helper
in the iframe JS that wraps each title's HTML string into a styled
`<div>` (via `innerHTML`, safe because the supply chain is
Python → JSON → this iframe). The HTMLElement is built per-network
inside `createNetwork(...)` so each layout-mode switch (force /
hierarchy / freeze) gets a fresh DOM node owned by that vis instance
— `JSON.parse(JSON.stringify(nodeList))` would otherwise replace any
`HTMLElement` with `{}` and lose the tooltip. The original HTML
string is stashed on a `titleHtml` property that survives the JSON
round-trip.

### Regression tests

- `test_visjs_graph_converts_title_to_html_element` — fires if the
  iframe JS no longer contains `htmlToElement` + `titleHtml`.
- `test_visjs_graph_no_title_means_no_tooltip_object` — ensures the
  conversion guards against undefined `titleHtml`.

## Lab content — three user-requested case studies

### 1. Ring Artifact on Neutron CT (real Sarepy data)

`recipe_id: ring_artifact_neutron_ct` — applies the Vo (2018) sorting
filter to the bundled real neutron sinogram
`sinogram_360_neutron_image.tif` (459 × 503 uint16). The recipe ships
with a smaller default window (15 vs 21 for the X-ray case) calibrated
for the lower angle count. The bundled JPEG reconstructions
(`rec_fbp_before_*`, `rec_after_*`) provide the visual target.

### 2. Low-Dose / Photon-Counting Denoising (TomoGAN-comparable)

`recipe_id: low_dose_wavelet_denoise` — targets the noise regime in
**Liu et al. (2020) TomoGAN** at three exposure levels. The pipeline
uses BayesShrink wavelet shrinkage (Donoho & Johnstone 1994; Chang
2000), which is the **classical baseline** TomoGAN reports beating.
We can't ship the TomoGAN weights (CC BY-NC), but the new dataset
+ recipe lets users see how far the classical lower bound goes.

New dataset (`datasets/tomography/low_dose/`, 4 MB):

- `sinogram_clean.npy` — 512 × 512 centre crop of Sarepy's clean
  reference (Apache-2.0).
- `sinogram_lowdose_high_snr.npy` — 200 photons/pixel.
- `sinogram_lowdose_medium.npy` — 50 photons/pixel.
- `sinogram_lowdose_severe.npy` — 12 photons/pixel (the regime
  TomoGAN targets).

Generated by deterministic `numpy.random.default_rng(seed=1..3)`
applied to a Poisson + Gaussian read-noise model.

### 3. Phase Wrapping → Unwrapping (CDI / phase-contrast)

`recipe_id: phase_unwrap_2d` — wraps `skimage.restoration.unwrap_phase`
(Herraez et al. 2002 reliability-following algorithm). The bundled
test set covers the canonical patterns from the phase-unwrapping
literature:

- **Gaussian bump** (Itoh test surface).
- **Two-bump** (CDI exit-wave proxy).
- **Noisy Gaussian** (speckle-robustness).
- **Vortex / branch-cut** (residue trap — a deliberate
  *false-positive* sample that *cannot* be unwrapped continuously).

New dataset (`datasets/scattering_diffraction/phase_wrapping/`, 2 MB):
8 paired `.npy` arrays — wrapped + clean ground truth for each of
the four test patterns. Synthesised in-tree, **CC0**.

On the Gaussian sample the recipe achieves PSNR 7 → 149 dB and
SSIM 0.345 → 1.000 — a clear pedagogical win.

## Lab content — six additional workhorse recipes

| Recipe | Modality | Algorithm | Reference |
|---|---|---|---|
| `tv_chambolle_denoise`   | cross_cutting | Total Variation | Chambolle 2004 |
| `nlm_denoise`            | cross_cutting | Non-Local Means | Buades 2005 |
| `bilateral_denoise`      | cross_cutting | Bilateral filter | Tomasi 1998 |
| `wavelet_shrinkage_denoise` | cross_cutting | BayesShrink | Donoho 1994 |
| `inpaint_dead_pixel`     | cross_cutting | Biharmonic inpaint | Bertalmio 2000 |
| `beam_hardening_polynomial` | tomography | Polynomial linearisation | Krumm 2008 |

All six are runnable on the bundled low-dose / clean / faulty
sinograms with PSNR/SSIM against the matching clean reference.
On `inpaint_dead_pixel` the algorithm achieves PSNR 14 → 50 dB at
default parameters (recovers two simulated dead columns + 200
random faults).

Two further bundled datasets back these recipes:

- `datasets/tomography/beam_hardening/` (2 MB) — Sarepy clean crop +
  deterministic polynomial cupping (`norm + 0.35·norm² − 0.25·norm³`).
- `datasets/tomography/dead_pixel/` (2.3 MB) — Sarepy clean crop with
  2 dead columns + 200 random faults + uint8 fault mask.

## Recipe count summary

| | Before R14 | After R14 |
|---|---|---|
| Total recipes | 5 | **14** |
| Tomography | 3 | 6 |
| Cross-cutting | 2 | 7 |
| Scattering / Diffraction | 0 | 1 |

## Static-site mirror

`scripts/build_static_site.py` continues to discover recipes via
`load_recipes(experiments/)` — no code change needed for the new
recipes. The `stat_line` for the Build cluster page was bumped from
`3 recipes · 71 real samples` to
`14 recipes · 90+ real samples` and now lists all the new algorithm
families. The matching landing-page CTA card in `explorer/app.py`
was updated identically.

## Tests

- `test_lab_integrity`: every new sample path resolves; every new
  ATTRIBUTION.md ships the required YAML frontmatter and a
  recognised license keyword.
- `test_experiments`: every new `recipe.yaml` parses, every
  `function:` path resolves, every `noise_catalog_ref` exists, and
  every parameter has a valid widget type.
- `test_cross_refs`: every new recipe→modality / recipe→noise edge
  resolves to an existing entity.

Full suite: **279 tests passing** locally on Python 3.13.

## Traceability

- **Hotfix**: ADR-007 (static-site sync), DS-001 (header component spec).
- **Recipes**: ADR-008 (Section 10 Interactive Lab), FR-001 (landing-page CTAs),
  REL-N100 (notes-v1.0.0 — adds the new datasets to section 10).
- **Citations**: 16 new BibTeX entries in
  `10_interactive_lab/CITATIONS.bib` (TomoGAN, Herraez phase-unwrap,
  Donoho wavelet, Chambolle TV, Buades NLM, Tomasi bilateral,
  Bertalmio inpaint, Krumm beam-hardening, scikit-image, Itoh,
  Ghiglia–Pritt, Chang BayesShrink, Rudin TV original).
