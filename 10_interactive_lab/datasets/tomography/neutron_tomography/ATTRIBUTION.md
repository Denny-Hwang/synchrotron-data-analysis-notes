---
doc_id: ATTR-NEUT-001
title: Attribution — 360° Neutron Tomography Sinogram (Sarepy)
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/tomography/ring_artifact.md, 02_xray_modalities/tomography]
---

# 360° Neutron Tomography Sample

## Source

- **Repository:** Sarepy
- **Upstream URL:** https://github.com/nghia-vo/sarepy
- **Mirrored from commit:** `2d00b05eb2c1e0d3bad20b4883b70448d0179818`
- **Original path:** `/data/sinogram_360_neutron_image.tif` and `/data/rec_*.jpg`
- **Authors:** Nghia Vo (NSLS-II / DLS); Daniel S. Hussey (NIST)
- **License:** Apache License 2.0 — see `../../../LICENSES/sarepy_Apache-2.0.txt`

## Files

| File | Size | Type | Description |
|---|---|---|---|
| `sinogram_360_neutron_image.tif` | 452 KB | sinogram | 360° rotation neutron CT, real measurement |
| `rec_fbp_before_sinogram_360_neutron_image.jpg` | 16 KB | reconstruction | FBP reconstruction *before* ring correction |
| `rec_gridrec_before_sinogram_360_neutron_image.jpg` | 48 KB | reconstruction | Gridrec reconstruction *before* ring correction |
| `rec_after_sinogram_360_neutron_image.jpg` | 24 KB | reconstruction | Reconstruction *after* ring correction (target) |

These four files together form a complete before/after demonstration: noisy sinogram → two reconstruction algorithms → corrected reconstruction.

## Required Citation

> Vo, N. T., Atwood, R. C., & Drakopoulos, M. (2018).
> Superior techniques for eliminating ring artifacts in X-ray micro-tomography.
> *Optics Express, 26*(22), 28396–28412.
> https://doi.org/10.1364/OE.26.028396

The neutron CT data were originally acquired in collaboration with NIST. Researchers building on this dataset are encouraged to also acknowledge NIST.

## Disclaimer

This is not an official mirror. Files are bundled unchanged from the upstream commit listed above for educational purposes only.
