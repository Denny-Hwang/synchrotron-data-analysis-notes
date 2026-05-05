---
doc_id: ATTR-RING-001
title: Attribution — Ring Artifact Sample Sinograms (Sarepy)
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/tomography/ring_artifact.md]
---

# Ring Artifact — Sample Sinograms

## Source

- **Repository:** Sarepy (Stripe Artifacts Removal in Python)
- **Upstream URL:** https://github.com/nghia-vo/sarepy
- **Mirrored from commit:** `2d00b05eb2c1e0d3bad20b4883b70448d0179818`
- **Original path:** `/data/` and `/data/challenging/`
- **Author:** Nghia Vo — NSLS-II, Brookhaven National Lab (US) / Diamond Light Source (UK)
- **License:** Apache License 2.0 — see `../../../LICENSES/sarepy_Apache-2.0.txt`

## Files

| File | Size | Type | Description |
|---|---|---|---|
| `sinogram_normal.tif` | 18 MB | sinogram | Clean reference sinogram, no visible stripes (use as ground truth) |
| `sinogram_dead_stripe.tif` | 18 MB | sinogram | Contains "dead stripe" type — sensor failure |
| `sinogram_large_stripe.tif` | 18 MB | sinogram | Large-amplitude vertical stripe |
| `sinogram_partial_stripe.tif` | 25 MB | sinogram | Stripe present only in part of the column range |
| `all_stripe_types_sample1.tif` | 18 MB | sinogram | Multiple stripe types coexisting (challenging) |
| `large_partial_rings.tif` | 18 MB | sinogram | Large partial rings (challenging) |
| `valid_stripes.tif` | 18 MB | sinogram | Looks like stripes but is actual sample feature — false-positive trap |

All files are 16-bit unsigned TIFF, format readable by `tifffile`, `imageio`, or `skimage.io`.

## Required Citation

When using these data in derived work, please cite the originating paper:

> Vo, N. T., Atwood, R. C., & Drakopoulos, M. (2018).
> Superior techniques for eliminating ring artifacts in X-ray micro-tomography.
> *Optics Express, 26*(22), 28396–28412.
> https://doi.org/10.1364/OE.26.028396

BibTeX entry available in `10_interactive_lab/CITATIONS.bib`.

## Disclaimer

This directory is **not** an official mirror of Sarepy. Files are bundled here unchanged from the upstream repository at the commit listed above, solely for the educational purpose of the eBERlight Explorer Interactive Lab. For authoritative copies and the latest version please consult the upstream repository.

## Related Notes

- `09_noise_catalog/tomography/ring_artifact.md` — full taxonomy of ring/stripe artifacts
- `05_tools_and_code/tomopy/reverse_engineering.md` — TomoPy stripe removal internals
