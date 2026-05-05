---
doc_id: ATTR-FLAT-001
title: Attribution — TomoPy Flatfield/Sinogram Test Data
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/tomography/flatfield_issues.md, 09_noise_catalog/tomography/ring_artifact.md]
---

# Flatfield Correction & Stripe Mask — TomoPy Test Data

## Source

- **Repository:** TomoPy
- **Upstream URL:** https://github.com/tomopy/tomopy
- **Mirrored from commit:** `3377fed68103959e37a849309a096a9919b0bc97`
- **Original path:** `/test/test_tomopy/test_data/`
- **Maintainer:** Argonne National Laboratory / UChicago Argonne LLC
- **License:** BSD-style (Argonne) — see `../../../LICENSES/tomopy_BSD-Argonne.txt`

## Files

| File | Size | Format | Description |
|---|---|---|---|
| `sinogram.npy` | 16 KB | NumPy | Synthetic sinogram for unit-testing reconstruction |
| `flat.npy` | 3 KB | NumPy | Flat-field reference frame (no sample) |
| `flat_1.npy` | 14 KB | NumPy | Larger flat-field reference frame |
| `dark.npy` | 3 KB | NumPy | Dark-field reference frame (detector offset) |
| `cube.npy` | 4 KB | NumPy | 3D cube projections |
| `distortion_proj.npy` | 1 KB | NumPy | Projection with detector distortion |
| `distortion_sino.npy` | 1 KB | NumPy | Sinogram showing distortion effect |
| `stripes_mask3d.npy` | 9 KB | NumPy | 3D stripe-mask for ring artifact detection |
| `angle.npy` | 0.2 KB | NumPy | Projection angles array |

These were originally produced as **unit-test fixtures** for TomoPy. They are not derived from a single experimental dataset, but reflect canonical detector behaviour (flat/dark frames are real).

## Required Citation

> Gürsoy, D., De Carlo, F., Xiao, X., & Jacobsen, C. (2014).
> TomoPy: a framework for the analysis of synchrotron tomographic data.
> *Journal of Synchrotron Radiation, 21*(5), 1188–1193.
> https://doi.org/10.1107/S1600577514013939

Acknowledgement of the U.S. Department of Energy under contract DE-AC02-06CH11357 is required by the TomoPy license.

## Disclaimer

This is not an official mirror. Bundled unchanged for educational purposes.
