---
doc_id: ATTR-SPEC-001
title: Attribution — EXAFS Reference Spectra & FEFF Calculations
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/spectroscopy/, 06_data_structures/eda/spectroscopy_eda.md]
---

# Spectroscopy — Reference EXAFS Data & FEFF Path Calculations

## Sources

### 1. PyMca EXAFS reference spectra — `exafs_pymca/`
- **Upstream:** https://github.com/vasole/pymca @ `4fbf74ac4edc032b54d9ab05f4ddb75ca19a1aa8`
- **Original path:** `src/PyMca5/PyMcaData/`
- **License:** LGPL-2.1+ — see `../../LICENSES/pymca_LGPL-2.1.txt`

### 2. xraylarch EXAFS sample — `exafs_xraylarch/`
- **Upstream:** https://github.com/xraypy/xraylarch @ `3037c388192794f18faa23be1d503414cb7146c0`
- **Original path:** `tests/larch_scripts/cu2s_xafs.dat`
- **Authors:** Matthew Newville, Mauro Rovezzi, Bruce Ravel, Margaret Koker, Ryuichi Shimogawa
- **License:** MIT — see `../../LICENSES/xraylarch_MIT.txt`

### 3. Athena project files — `athena_projects/`
- **Upstream:** https://github.com/xraypy/xraylarch @ `3037c388192794f18faa23be1d503414cb7146c0`
- **Original path:** `examples/xafsdata/AthenaProjectFiles/`
- **License:** MIT (xraylarch)

### 4. FEFF path calculations — `feff_calculations/`
- **Upstream:** https://github.com/xraypy/xraylarch @ `3037c388192794f18faa23be1d503414cb7146c0`
- **Original path:** `examples/feffit/`
- **License:** MIT (xraylarch wrapping); the underlying FEFF program is by University of Washington / J. J. Rehr et al.

## Files

### `exafs_pymca/` — Real EXAFS measurements
| File | Element | Description |
|---|---|---|
| `EXAFS_Cu.dat` | Cu | Copper K-edge EXAFS spectrum (~21 KB) |
| `EXAFS_Ge.dat` | Ge | Germanium K-edge EXAFS spectrum (~21 KB) |

### `exafs_xraylarch/` — Real Cu-S system
| File | Description |
|---|---|
| `cu2s_xafs.dat` | Cu₂S Cu K-edge XAFS, used as test data in xraylarch unit tests |

### `athena_projects/` — Athena binary projects
| File | Description |
|---|---|
| `abc.prj` | Generic 3-spectrum Athena project — alignment & merging exercise |
| `FeS2_ex.prj` | FeS₂ (pyrite) Athena project |

### `feff_calculations/` — Theoretical scattering paths
| Subfolder | System | Files | Description |
|---|---|---|---|
| `Cu/` | Copper metal | feff0001.dat … feff0013.dat (13 paths) | Single-/multi-scattering paths up to ~5 Å |
| `FeO/` | Iron oxide | feff_feo01.dat … feff_feo06.dat (6 paths) | Wüstite scattering paths |
| `ZnSe/` | Zinc selenide | feff_zn{as,br,ga,ge,kr,rb,se,zn}.dat (8 paths) | Many-element environments |

These FEFF outputs serve as **theoretical reference paths**; combine them with the experimental spectra above to fit χ(k).

## Required Citations

> Newville, M. (2013). Larch: An Analysis Package for XAFS and Related Spectroscopies.
> *Journal of Physics: Conference Series, 430*, 012007.
> https://doi.org/10.1088/1742-6596/430/1/012007

> Solé, V. A. et al. (2007). PyMca multiplatform code (XRF/EXAFS). https://doi.org/10.1016/j.sab.2006.12.002

If you use FEFF outputs:
> Rehr, J. J., Kas, J. J., Vila, F. D., Prange, M. P., & Jorissen, K. (2010).
> Parameter-free calculations of X-ray spectra with FEFF9.
> *Physical Chemistry Chemical Physics, 12*, 5503–5513.
> https://doi.org/10.1039/B926434E

## Disclaimer

Not an official mirror. Bundled unchanged from the commits cited above for educational use.
