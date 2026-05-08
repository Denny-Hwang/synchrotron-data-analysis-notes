---
doc_id: ATTR-LOWDOSE-001
title: Attribution — Low-Dose / Photon-Counting Sinograms (Sarepy clean reference + simulated noise)
status: accepted
version: 1.0.0
last_updated: 2026-05-08
supersedes: null
related: [09_noise_catalog/tomography/low_dose_noise.md, 09_noise_catalog/xrf_microscopy/photon_counting_noise.md]
---

# Low-Dose / Photon-Counting — Test Dataset

## Source

The clean reference is a 512×512 centre crop of Sarepy's bundled
`sinogram_normal.tif` (which is itself a real synchrotron tomography
sinogram). The low-dose variants are produced *in-tree* by applying
the standard photon-counting noise model:

```
expected_counts(y, x) = clean(y, x) * photons_per_pixel
noisy_counts          = Poisson(expected_counts) + Normal(0, read_noise)
noisy_transmission    = noisy_counts / photons_per_pixel
```

Three exposure levels are bundled to demonstrate denoising performance
across the regime that Liu et al. (2020) targeted with TomoGAN:

- **`sinogram_lowdose_high_snr.npy`** — 200 photons/pixel, mild dose
  reduction (still well-resolved features, denoising marginal benefit).
- **`sinogram_lowdose_medium.npy`** — 50 photons/pixel, typical
  low-dose regime where classical wavelet shrinkage is competitive
  with learned denoisers.
- **`sinogram_lowdose_severe.npy`** — 12 photons/pixel, the regime
  where TomoGAN beats classical methods.

| File | Size | Type | Description |
|---|---|---|---|
| `sinogram_clean.npy`             | 1.0 MB | clean reference | 512×512 crop of Sarepy clean sinogram (float32) |
| `sinogram_lowdose_high_snr.npy`  | 1.0 MB | noisy_input     | High-SNR Poisson noise (~200 ph/pix) |
| `sinogram_lowdose_medium.npy`    | 1.0 MB | noisy_input     | Typical low-dose (~50 ph/pix) |
| `sinogram_lowdose_severe.npy`    | 1.0 MB | noisy_input     | Severe low-dose (~12 ph/pix) — TomoGAN regime |

## License

The clean reference inherits Sarepy's **Apache-2.0** license — see
`../../../LICENSES/sarepy_Apache-2.0.txt`. The simulated low-dose
variants are derivative works under the same license (the noise
realisation is fixed by `numpy.random.default_rng(seed=1..3)` so
reproductions are exact).

## Required Citations

> Liu, Z., Bicer, T., Kettimuthu, R., Gursoy, D., De Carlo, F.,
> & Foster, I. (2020).
> *TomoGAN: low-dose synchrotron x-ray tomography with generative
> adversarial networks.*
> Journal of the Optical Society of America A, 37(3), 422–434.
> https://doi.org/10.1364/JOSAA.375595

> Vo, N. T., Atwood, R. C., Drakopoulos, M. (2018).
> *Superior techniques for eliminating ring artifacts in X-ray
> micro-tomography.*
> Optics Express 26(22), 28396–28412.
> https://doi.org/10.1364/OE.26.028396

## Disclaimer

The bundled GAN reference (TomoGAN) is **not redistributed** —
trained weights are CC BY-NC and require sign-up at
[anl-tomogan.github.io](https://github.com/ramsesproject/TomoGAN). The
recipe in `experiments/tomography/low_dose_denoise/` provides a
**classical wavelet-shrinkage baseline** (Donoho & Johnstone 1994)
that practitioners typically run *before* paying for a learned model.
