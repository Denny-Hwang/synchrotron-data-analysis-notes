---
doc_id: EXP-README-001
title: Experiments — Recipe Schema and Conventions
status: accepted
version: 0.1.0
last_updated: 2026-05-05
supersedes: null
related: [10_interactive_lab/README.md, ADR-008]
---

# Experiments — Recipe Schema and Conventions

This directory holds **noise mitigation recipes** that the Streamlit
Interactive Lab (`explorer/pages/4_Experiment.py`) discovers and runs.
Each recipe is a **pure function** plus a **YAML manifest** describing its
parameters, samples, and citations.

## Layout

```
experiments/
├── README.md
├── __init__.py
└── <modality>/
    └── <noise_case>/
        ├── __init__.py
        ├── recipe.yaml          ← this file is the contract with the UI
        └── pipeline.py          ← pure function(s) referenced by recipe.yaml
```

## Recipe schema

```yaml
schema_version: 1
recipe_id: <unique_snake_case_id>
title: "<Human-readable title>"
modality: tomography | xrf | spectroscopy | scattering | electron_microscopy | ptychography | cross_cutting
noise_catalog_ref: 09_noise_catalog/<modality>/<artifact>.md

description: |
  Multi-paragraph markdown explaining the algorithm and what to learn.

# Dotted import path to the pipeline function.
# Resolved with sys.path == repository root.
function: experiments.<modality>.<noise_case>.pipeline.<function_name>

# One or more bundled samples to choose from.
samples:
  - manifest_path: datasets/<modality>/<noise_case>/<file>
    label: "<Short label>"
    role: noisy_input | clean_reference | false_positive_trap | flat_reference | dark_reference | metadata
    description: "<Optional one-line description>"

# Optional: clean reference for PSNR/SSIM. Same schema as a sample.
clean_reference:
  manifest_path: ...
  label: ...

# Each parameter renders as a Streamlit widget.
parameters:
  - name: <python_kwarg>
    type: int | float | select
    label: "<Human label>"
    default: <value>
    # for int/float:
    min: <number>
    max: <number>
    step: <number>
    # for select:
    options: ["a", "b", "c"]
    help: "<Tooltip text>"

metrics: [psnr, ssim]    # any subset; skipped if no clean_reference

references:
  - title: "<Paper title>"
    authors: "<Last, F. M. et al.>"
    year: <yyyy>
    venue: "<Journal Vol(Issue), pages>"
    doi: "<DOI>"
```

## Pipeline function contract

A pipeline function MUST be:

1. **Pure.** Same input + same params ⇒ same output. No global state, no I/O,
   no logging side effects, no Streamlit calls.
2. **Numpy-typed.** Accept `numpy.ndarray` input; return `numpy.ndarray`.
3. **Validating.** Raise `ValueError` for impossible parameter values
   (the UI may pass anything within the declared range, so still validate).
4. **Dtype-preserving** when reasonable, or document the dtype change.

This makes pipelines trivially unit-testable and `@st.cache_data`-friendly.

## Adding a new recipe

1. Create `experiments/<modality>/<noise_case>/`.
2. Implement `pipeline.py` with a pure function.
3. Write `recipe.yaml` referencing samples in `10_interactive_lab/datasets/`.
4. Add a unit test in `explorer/tests/test_experiments.py` covering:
   - The pipeline runs on at least one bundled sample without raising,
   - Bad parameters raise `ValueError`.
5. Add a row to `docs/05_release/release_notes/explorer-vX.Y.Z.md`.

## Why YAML, not Python?

Recipes are **data**, not code. Decoupling the parameter schema from the
pipeline lets us:

- Auto-generate the Streamlit UI without per-recipe boilerplate.
- Validate recipes in CI without importing matplotlib / scipy / etc.
- Mirror recipe descriptions on the static GitHub Pages site even though
  the pipelines themselves cannot run there (per ADR-007 and ADR-008).
