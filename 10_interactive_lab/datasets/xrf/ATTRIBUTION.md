---
doc_id: ATTR-XRF-001
title: Attribution — XRF Reference Spectra and PyXRF Configuration Examples
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/xrf_microscopy/, 02_xray_modalities/xrf_microscopy/analysis_pipeline.md]
---

# XRF — Reference Spectra & Configurations

## Sources

This directory bundles real reference materials from two complementary projects:

### 1. PyMca (ESRF) — `xrf_spectra/`
- **Upstream URL:** https://github.com/vasole/pymca
- **Mirrored from commit:** `4fbf74ac4edc032b54d9ab05f4ddb75ca19a1aa8`
- **Original path:** `src/PyMca5/PyMcaData/`
- **Authors:** V. Armando Solé and contributors at the European Synchrotron Radiation Facility (ESRF)
- **License:** LGPL-2.1+ for the framework; bundled data files are reference materials with permissive intent — see `../../LICENSES/pymca_LGPL-2.1.txt`

### 2. PyXRF (NSLS-II) — `pyxrf_configs/`
- **Upstream URL:** https://github.com/NSLS-II/PyXRF
- **Mirrored from commit:** `7b7d8689c667b4653dd1c7e2073f6b0504692c48`
- **Original paths:** `examples/data/` and `configs/`
- **Maintainer:** National Synchrotron Light Source II (NSLS-II), Brookhaven National Laboratory
- **License:** BSD 3-Clause — see `../../LICENSES/PyXRF_BSD-3.txt`

## Files

### `xrf_spectra/`
| File | Size | Format | Description |
|---|---|---|---|
| `XRFSpectrum.mca` | 63 KB | MCA | Reference X-ray fluorescence spectrum (PyMca canonical example) |
| `Steel.spe` | 21 KB | SPE | Steel sample reference spectrum (peak overlap example) |

### `pyxrf_configs/`
| File | Size | Format | Description |
|---|---|---|---|
| `film_11564ev_all.json` | 73 KB | JSON | Fitting parameters for thin-film spectrum at 11.564 keV |
| `film_25000ev_all.json` | 128 KB | JSON | Fitting parameters at 25 keV (high energy) |
| `background_11564ev.json` | 72 KB | JSON | Background-subtraction parameter set |
| `root.json`, `root_all.json` | 6 + 77 KB | JSON | Root-level configurations |
| `xrf_parameter.json` | 7 KB | JSON | Parameter schema example |

## Required Citations

> Solé, V. A., Papillon, E., Cotte, M., Walter, P., & Susini, J. (2007).
> A multiplatform code for the analysis of energy-dispersive X-ray fluorescence spectra.
> *Spectrochimica Acta Part B, 62*(1), 63–68.
> https://doi.org/10.1016/j.sab.2006.12.002

> Li, L., Yan, H., Xu, W., Yu, D., Heroux, A., Lee, W. K., Campbell, S. I., & Chu, Y. S. (2017).
> PyXRF: Python-based X-ray fluorescence analysis package.
> *Proc. SPIE 10389, X-Ray Nanoimaging: Instruments and Methods III.*
> https://doi.org/10.1117/12.2272585

## Disclaimer

This directory is not an official mirror of either PyMca or PyXRF. Files are bundled unchanged from the commits listed above for educational use within the eBERlight Explorer Interactive Lab.
