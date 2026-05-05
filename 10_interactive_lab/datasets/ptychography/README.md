---
doc_id: LAB-PTYCHO-001
title: Ptychography — Sample Data
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/ptychography/]
---

# Ptychography — Sample Data

## Status: External Only

Ptychography raw datasets are typically **multi-GB** per scan (probe positions × diffraction patterns), which exceeds the free-tier limit we set for this repository. Therefore this folder bundles **no raw data**; instead, refer to `../../docs/external_data_sources.md` for the curated list.

## Quickest path to real data

| Source | What you get | Size | License |
|---|---|---|---|
| **PtyPy** sample scripts (https://github.com/ptycho/ptypy) | Synthetic test scenes generated on-the-fly using `flowers.png` / `tree.png` (bundled in PtyPy at `ptypy/resources/`) | n/a (generated) | BSD-3 |
| **PtychoShelves tutorial** | Real beamline data from cSAXS @ PSI | several GB | BSD-3 / academic |
| **CXIDB (Coherent X-ray Imaging Data Bank)** at https://cxidb.org | Curated ptychography & CDI datasets | 100 MB – several GB per entry | varies per entry, mostly CC-BY |

See `../../docs/external_data_sources.md` for download recipes and citation requirements.

## Note for Interactive Lab developers

When the Lab adds a ptychography experiment, prefer the PtyPy synthetic scene generator over redistributing raw beamline data — it gives reproducible, license-clean inputs that demonstrate the same numerics (position error, partial coherence, mixed-state).
