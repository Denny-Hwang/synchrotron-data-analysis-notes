---
doc_id: ATTR-SCATT-001
title: Attribution — pyFAI Calibrant d-Spacing Reference Files
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/scattering_diffraction/detector_gaps_parallax.md]
---

# Scattering & Diffraction — Calibrant d-Spacing Reference Files

## Source

- **Repository:** pyFAI (Fast Azimuthal Integration)
- **Upstream URL:** https://github.com/silx-kit/pyFAI
- **Mirrored from commit:** `0f97cb433387b40bb3d326e6aff7a60f170c5a73`
- **Original path:** `src/pyFAI/resources/calibration/`
- **Author:** Jérôme Kieffer and contributors (European Synchrotron Radiation Facility — ESRF)
- **License:** MIT — see `../../LICENSES/pyFAI_MIT.txt`

## Files

These `.D` files list theoretical d-spacings for common diffraction calibration standards. They are **reference data**, not measured images: each file contains a list of allowed reflections and their d-spacings (Å), used by pyFAI to convert detector pixels to scattering angle (q or 2θ).

| File | Standard | d-spacings | Common usage |
|---|---|---|---|
| `Si.D` | Silicon | full | NIST SRM 640 reference |
| `Si_SRM640.D` | Si SRM 640 | curated NIST | NIST-traceable Si calibrant |
| `LaB6.D` | Lanthanum hexaboride | full | NIST SRM 660 reference, very sharp peaks |
| `LaB6_SRM660c.D` | LaB6 SRM 660c | curated NIST | NIST 660c-specific |
| `Au.D` | Gold | full | XRD calibrant |
| `Al.D` | Aluminum | full | XRD calibrant |
| `Ni.D` | Nickel | full | Common XRD calibrant |
| `quartz.D` | α-Quartz (SiO₂) | full | Geological & MX |
| `C60.D` | Buckminsterfullerene | full | Molecular crystal example |
| `TiO2.D` | Titanium dioxide | full | Anatase/rutile reference |

Use these as **calibration ground-truth** when investigating:
- detector gap correction
- parallax artifact
- ice rings (compare against d-spacings of ice)
- centre/distance refinement

## Required Citation

> Kieffer, J., Valls, V., Blanc, N., & Hinkle, C. (2020).
> New tools for calibrating diffraction setups.
> *Journal of Synchrotron Radiation, 27*(2), 558–566.
> https://doi.org/10.1107/S1600577520000776

> Ashiotis, G. et al. (2015).
> The fast azimuthal integration Python library: pyFAI.
> *Journal of Applied Crystallography, 48*, 510–519.
> https://doi.org/10.1107/S1600576715004306

## Disclaimer

This directory is **not** an official mirror of pyFAI calibration resources. Files are bundled unchanged from the commit listed above for educational purposes within the eBERlight Explorer Interactive Lab. For authoritative copies, consult the upstream repository.
