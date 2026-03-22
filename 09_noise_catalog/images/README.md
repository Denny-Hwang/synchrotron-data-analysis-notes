# Image Attribution and Regeneration Guide

## Synthetic Example Images

The following images are generated programmatically by `generate_examples.py` using synthetic data (Shepp-Logan phantom and simulated elemental maps). No external data downloads are required.

| Image File | Noise Type | Generation Method | License |
|-----------|-----------|-------------------|---------|
| `ring_artifact_before_after.png` | Ring artifact | Dead-pixel column in sinogram → FBP reconstruction | MIT (this repo) |
| `zinger_before_after.png` | Zinger | Random high-intensity spots → median filter | MIT (this repo) |
| `low_dose_noise_before_after.png` | Low-dose Poisson noise | Poisson-sampled projections → reconstruction | MIT (this repo) |
| `rotation_center_error_before_after.png` | Rotation center error | Correct vs offset (±5 px) center reconstruction | MIT (this repo) |
| `flatfield_before_after.png` | Flat-field non-uniformity | Non-uniform flat → normalized vs raw projection | MIT (this repo) |
| `sparse_angle_before_after.png` | Sparse-angle artifact | 1800 vs 180 projections → FBP comparison | MIT (this repo) |
| `dead_hot_pixel_before_after.png` | Dead/hot pixel (XRF) | Synthetic elemental map → outlier injection | MIT (this repo) |
| `i0_drop_before_after.png` | I0 drop | Simulated I0 with drops → normalization | MIT (this repo) |

### Regenerating Images

```bash
cd 09_noise_catalog/images/
python generate_examples.py
```

Requirements: `numpy`, `scipy`, `matplotlib`

All images are saved at 300 DPI with white background.

## External Image References

The following external sources are referenced in the noise catalog documents for real-data examples. Images are **not included** in this repository — only linked.

| Source | URL | License | Referenced In |
|--------|-----|---------|--------------|
| Sarepy documentation (Vo et al. 2018) | https://sarepy.readthedocs.io/toc/section3.html | BSD-3 | `tomography/ring_artifact.md` |
| Algotom documentation | https://algotom.readthedocs.io/en/latest/toc/section4/section4_4.html | Apache-2.0 | `tomography/ring_artifact.md` |
| Sarepy stripe categories | https://sarepy.readthedocs.io/toc/section2.html | BSD-3 | `tomography/ring_artifact.md` |
| TomoGAN GitHub | https://github.com/lzhengchun/TomoGAN | BSD-2 | `tomography/low_dose_noise.md` |
| AIScienceTutorial/Denoising | https://github.com/AIScienceTutorial/Denoising | MIT | `tomography/low_dose_noise.md` |
| npj Comp. Mat. (2023), Fig. 3 | https://doi.org/10.1038/s41524-023-00995-9 | CC BY 4.0 | `xrf_microscopy/probe_blurring.md` |
| edgePtychoNN (Babu et al. 2023) | https://doi.org/10.1038/s41467-023-41496-z | CC BY 4.0 | `ptychography/position_error.md` |
| TomoBank (APS) | https://tomobank.readthedocs.io/ | Public domain | `tomography/low_dose_noise.md` |
| TomoPy documentation | https://tomopy.readthedocs.io/ | BSD-3 | `tomography/rotation_center_error.md` |

## Contributing Additional Images

To add example images to this catalog:

1. **Synthetic images**: Add generation code to `generate_examples.py` and re-run
2. **Real data images**: Ensure the image is from a permissively licensed source (CC BY, BSD, MIT, public domain)
3. **Attribution**: Add an entry to the tables above with source URL and license
4. **Format**: PNG, 300 DPI, white background, side-by-side before/after layout preferred
5. **Naming**: `{noise_type}_before_after.png` for comparison images

Submit a PR with the image and updated attribution table.
