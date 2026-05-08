---
doc_id: ATTR-BEAMHARD-001
title: Attribution — Beam Hardening Test Dataset (Sarepy + simulated polynomial)
status: accepted
version: 1.0.0
last_updated: 2026-05-08
supersedes: null
related: [09_noise_catalog/tomography/streak_artifact.md]
---

# Beam Hardening — Test Dataset

## Source

The clean reference is a 512×512 centre crop of Sarepy's
`sinogram_normal.tif`. The "hardened" variant applies a deterministic
2nd-order polynomial non-linearity to the normalised projection
values:

```
hardened = norm + 0.35 * norm^2 - 0.25 * norm^3
```

This reproduces the cupping / streak signature seen with polychromatic
X-ray sources where soft photons are preferentially absorbed by the
high-attenuation rays. The classical correction is a polynomial
linearisation, demonstrated in the matching recipe.

| File | Size | Type | Description |
|---|---|---|---|
| `sinogram_clean.npy`       | 1.0 MB | clean reference | 512×512 crop of Sarepy clean sinogram |
| `sinogram_hardened.npy`    | 1.0 MB | noisy_input     | Polynomial-hardened version of the clean sinogram |

## License

Sarepy clean reference: **Apache-2.0**, see
`../../../LICENSES/sarepy_Apache-2.0.txt`. The polynomial transform is
deterministic and not separately licensed.

## Required Citation

> Krumm, M., Kasperl, S., & Franz, M. (2008).
> *Reducing non-linear artifacts of multi-material objects in
> industrial 3D computed tomography.*
> NDT & E International, 41(4), 242–251.
> https://doi.org/10.1016/j.ndteint.2007.12.001

> Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018). *Superior
> techniques for eliminating ring artifacts.* Optics Express 26(22),
> 28396. https://doi.org/10.1364/OE.26.028396

## Disclaimer

The cupping signature here is a *teaching simulation*, not a measured
beam-hardening artefact. Real beam-hardening corrections rely on
spectrum-aware reconstruction (Herman 1979, Joseph & Spital 1978).
