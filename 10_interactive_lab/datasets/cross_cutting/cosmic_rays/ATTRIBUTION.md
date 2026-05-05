---
doc_id: ATTR-COSMIC-001
title: Attribution — Cosmic-Ray Detection Test FITS (astroscrappy)
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/tomography/zinger.md, 09_noise_catalog/cross_cutting/]
---

# Cosmic-Ray / Zinger Detection — Test FITS Image

## Source

- **Repository:** astroscrappy (Astropy implementation of L.A.Cosmic)
- **Upstream URL:** https://github.com/astropy/astroscrappy
- **Mirrored from commit:** `60570d354a6650bc2f8a73ab07858fd6123f443b`
- **Original path:** `astroscrappy/tests/data/gmos.fits`
- **Algorithm origin:** Pieter G. van Dokkum (2001), L.A.Cosmic
- **Maintainer:** Astropy / IfA Hawaii
- **License:** BSD 3-Clause — see `../../../LICENSES/astroscrappy_BSD-3.txt`

## Files

| File | Size | Format | Description |
|---|---|---|---|
| `gmos.fits` | 389 KB | FITS | Real GMOS (Gemini Multi-Object Spectrograph) astronomical CCD frame, used by astroscrappy unit tests. Contains real cosmic-ray hits. |

## Why this matters for synchrotron data

Cosmic-ray hits on a CCD/CMOS detector appear as **single-pixel or tight-cluster outliers** — exactly the same morphology as **zingers** in X-ray tomography (high-energy gamma penetrating shielding), as **outlier spectra** in spectroscopy, and as **dead/hot pixels** in XRF maps. The L.A.Cosmic algorithm and its variants are routinely used in synchrotron pipelines.

## Required Citation

> van Dokkum, P. G. (2001).
> Cosmic-Ray Rejection by Laplacian Edge Detection.
> *Publications of the Astronomical Society of the Pacific, 113*(789), 1420–1427.
> https://doi.org/10.1086/323894

> McCully, C., Crawford, S., Kovacs, G., Turner, J., Streicher, O., Lemon, R., et al. (2018).
> astroscrappy: Speedy Cosmic Ray Annihilation Package in Python.
> Astrophysics Source Code Library, ascl:1907.032

## Disclaimer

This is not an official mirror. Bundled unchanged from the commit cited above for educational purposes.
