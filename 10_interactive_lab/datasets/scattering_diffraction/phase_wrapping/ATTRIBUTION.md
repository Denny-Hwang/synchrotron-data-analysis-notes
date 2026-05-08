---
doc_id: ATTR-PHASEWRAP-001
title: Attribution — Phase Wrapping Test Dataset (synthetic)
status: accepted
version: 1.0.0
last_updated: 2026-05-08
supersedes: null
related: [09_noise_catalog/scattering_diffraction/phase_wrapping.md]
---

# Phase Wrapping — Synthetic Test Dataset

## Source

Synthesised in-tree by `scripts/` from the canonical 2-D phase-unwrapping
test patterns described in:

- Itoh, K. (1982). *Analysis of the phase unwrapping algorithm.*
  Applied Optics 21(14), 2470.
- Goldstein, R. M., Zebker, H. A., Werner, C. L. (1988). *Satellite
  radar interferometry: two-dimensional phase unwrapping.* Radio
  Science 23(4), 713–720.
- Ghiglia, D. C., Pritt, M. D. (1998). *Two-dimensional phase
  unwrapping: theory, algorithms, and software.* Wiley.

The actual unwrapping pipeline is `skimage.restoration.unwrap_phase`
(Herraez et al. 2002), shipped as part of scikit-image.

## Files

| File | Size | Type | Description |
|---|---|---|---|
| `phase_wrapped_gaussian.npy`  | 256 KB | wrapped 2-D phase | Gaussian bump + linear ramp, wrapped to (-π, π] |
| `phase_clean_gaussian.npy`    | 256 KB | clean phase | Ground-truth unwrapped phase (the same Gaussian) |
| `phase_wrapped_twobump.npy`   | 256 KB | wrapped 2-D phase | Two Gaussian bumps (CDI exit-wave proxy) |
| `phase_clean_twobump.npy`     | 256 KB | clean phase | Ground-truth two-bump unwrapped phase |
| `phase_wrapped_noisy.npy`     | 256 KB | wrapped 2-D phase | Gaussian + speckle (tests noise robustness) |
| `phase_clean_noisy.npy`       | 256 KB | clean phase | Ground-truth phase before wrapping |
| `phase_wrapped_vortex.npy`    | 256 KB | wrapped 2-D phase | Vortex / branch-cut field (tests residue handling) |
| `phase_clean_vortex.npy`      | 256 KB | clean phase | Ground-truth vortex phase |

Each pair `(phase_clean_*, phase_wrapped_*)` lets the Lab compute
PSNR/SSIM of the unwrapped output against the ground truth.

## License

These arrays are **CC0 / Public Domain** because they are deterministic
synthesis of well-known mathematical test surfaces. The reproducibility
script (in `Bash` block of the corresponding release note) seeds NumPy
with `20260508` so the bytes are stable across rebuilds.

## Required Citation

When using this dataset, please cite:

> Herráez, M. A., Burton, D. R., Lalor, M. J., & Gdeisat, M. A. (2002).
> Fast two-dimensional phase-unwrapping algorithm based on sorting by
> reliability following a noncontinuous path.
> *Applied Optics, 41*(35), 7437–7444.
> https://doi.org/10.1364/AO.41.007437

> van der Walt, S., Schönberger, J. L., Nunez-Iglesias, J., Boulogne, F.,
> Warner, J. D., Yager, N., Gouillart, E., & Yu, T. (2014).
> scikit-image: image processing in Python.
> *PeerJ, 2*, e453. https://doi.org/10.7717/peerj.453

## Disclaimer

This is a synthetic test dataset for pedagogy. Real CDI exit-wave phase
data are typically subject to download embargoes; see
`docs/external_data_sources.md` for CXIDB and PtychoShelves links.
