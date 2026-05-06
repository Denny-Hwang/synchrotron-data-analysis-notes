# DEPRECATED — see `../explorer/`

This directory is the **legacy first-generation Streamlit app**. It is
**superseded by [`../explorer/`](../explorer/)** and will be removed when
`notes-v1.0.0` ships. See [ADR-009](../docs/02_design/decisions/ADR-009.md)
for the rationale and the deletion plan.

## What changed

| Concern | Legacy (`eberlight-explorer/`) | Current (`explorer/`) |
|---|---|---|
| Data layer | 6 YAML catalogs + manual `build_index.py` | Runtime notes loader (ADR-002) |
| Frontmatter | None — heuristic parsing | YAML frontmatter required (ADR-003) |
| Visuals | Streamlit defaults | ANL/APS-aligned design tokens (ADR-005) |
| Pages | Knowledge Graph, Modalities, Methods, Publications, Tools, Pipeline, Data | Discover, Explore, Build, Experiment (ADR-004 IA) |
| Interactive lab | — | `4_Experiment.py` with bundled real data (ADR-008) |
| CI | None | pytest matrix on Py 3.11 + 3.12 |
| Static site mirror | — | GitHub Pages (ADR-007) |

## Removal timeline

This directory is **read-only and frozen**. Bug reports against it will be
closed with a pointer to `explorer/`. Security updates land only in
`explorer/requirements.txt`. The directory will be deleted in the
`notes-v1.0.0` PR; that PR's release notes will record the removal.

## Why is it still here

The before-state baseline for ADR-001 (Streamlit choice), ADR-002 (notes as
SoT), and ADR-005 (design tokens) is this directory. Those ADRs cite it as
"the existing explorer". Removing it before `notes-v1.0.0` would break the
narrative those decisions are anchored on. See ADR-009 for the full
discussion.

## If you really need to look at it

`git log -- eberlight-explorer/` — the history is intact, and the last
working state runs (in principle) under `pip install -r requirements.txt`
and `streamlit run app.py`. No support is provided.
