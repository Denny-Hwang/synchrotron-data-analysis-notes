---
doc_id: ATTR-DEADPIX-001
title: Attribution — Dead-Pixel / Hot-Pixel Test Dataset (Sarepy + simulated faults)
status: accepted
version: 1.0.0
last_updated: 2026-05-08
supersedes: null
related: [09_noise_catalog/xrf_microscopy/dead_hot_pixel.md, 09_noise_catalog/cross_cutting/detector_common_issues.md]
---

# Dead / Hot Pixel — Test Dataset

## Source

A 512×512 centre crop of the Sarepy clean reference with deterministic
faults applied:

- two full dead **columns** (all-zero) at x=113 and x=287 — the
  signature of a failed scintillator stripe;
- 100 randomly placed hot pixels (5× the maximum value) and 100
  randomly placed dead pixels (zero), seeded with
  `numpy.random.default_rng(seed=7)`.

The matching recipe in `experiments/cross_cutting/inpaint_dead_pixel/`
demonstrates biharmonic inpainting (`skimage.restoration.inpaint`)
to recover the lost pixels.

| File | Size | Type | Description |
|---|---|---|---|
| `sinogram_clean.npy`             | 1.0 MB | clean reference | 512×512 crop of Sarepy clean sinogram |
| `sinogram_with_dead_pixels.npy`  | 1.0 MB | noisy_input     | Same scene with simulated dead/hot pixels + dead columns |
| `dead_pixel_mask.npy`            | 256 KB | metadata        | uint8 mask, 1 where pixel is faulty (for debugging) |

## License

Sarepy clean reference: **Apache-2.0**, see
`../../../LICENSES/sarepy_Apache-2.0.txt`. The fault simulation is
deterministic, no additional license.

## Required Citation

> Bertero, M., Boccacci, P. (1998). *Introduction to Inverse Problems
> in Imaging.* IOP Publishing — Chapter 5 covers the biharmonic
> inpainting prior used by `skimage.restoration.inpaint`.

> Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018). *Superior
> techniques for eliminating ring artifacts.* Optics Express 26(22),
> 28396. https://doi.org/10.1364/OE.26.028396

## Disclaimer

Hot-pixel masks in production pipelines are typically built from
multi-flat statistics (the `tomopy.misc.corr.remove_outlier` family).
The mask shipped here is a hand-tuned demo, not a real beamline mask.
