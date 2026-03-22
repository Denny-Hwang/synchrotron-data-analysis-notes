# Streak Artifact (Metal Artifact)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Tomography |
| **Noise Type** | Systematic |
| **Severity** | Critical |
| **Frequency** | Common |
| **Detection Difficulty** | Easy |

## Visual Examples

```
  ASCII fallback — streak artifact pattern in a reconstructed slice:

  ............................................
  ..............####..........................
  .........###//####\\###.....................
  ......##///  ####  \\\\##..................
  ....##//     ####     \\##.................
  ...#//       ####       \\#................
  ...#/  sample @@@@  sample \\#..............
  ...#//       @@@@       \\#................
  ....##\\     @@@@     //##.................
  ......##\\\\  @@@@  ///##..................
  .........###\\@@@@//###.....................
  ..............@@@@..........................
  ............................................

  @@@@ = highly absorbing object (metal)
  //// = bright streak radiating outward
```

> **External references:**
> - [Metal Artifact Reduction in CT — review (Gjesteby et al.)](https://doi.org/10.1109/ACCESS.2016.2608813)
> - [NMAR — Normalized Metal Artifact Reduction](https://doi.org/10.1118/1.3484090)

## Description

Streak artifacts appear as bright and dark lines radiating outward from highly absorbing objects such as metal implants, dense mineral inclusions, or other high-Z materials in reconstructed CT slices. The streaks extend across the entire field of view and can obscure surrounding structures. They are most prominent along directions connecting multiple dense objects and between dense objects and sharp boundaries.

## Root Cause

When the X-ray beam traverses a highly absorbing object, very few photons reach the detector (photon starvation), resulting in extremely noisy or near-zero transmission values in those sinogram regions. Additionally, polychromatic beams suffer beam hardening — lower-energy photons are preferentially absorbed, shifting the effective spectrum and violating the monochromatic assumption of standard reconstruction algorithms. The combination of noisy/missing data and inconsistent attenuation values at certain projection angles creates the characteristic streak pattern during filtered back-projection. Incomplete projections through the dense object further contribute to data inconsistency.

## Quick Diagnosis

```python
import numpy as np

# Check sinogram for near-zero transmission regions (photon starvation)
sinogram_transmission = np.exp(-sinogram)  # if sinogram is -log normalized
# Regions with extremely low transmission indicate metal paths
starvation_mask = sinogram_transmission < 1e-4
starvation_fraction = np.sum(starvation_mask) / sinogram.size
print(f"Photon starvation fraction: {starvation_fraction:.4f}")
print(f"Likely streak artifacts: {starvation_fraction > 0.001}")
```

## Detection Methods

### Visual Indicators

- Bright and dark streaks radiating from dense objects in reconstructed slices.
- Streaks extend across the full field of view, often in a star-like pattern.
- Surrounding tissue or material near the dense object appears distorted or obscured.
- In the sinogram, corresponding regions show near-zero or saturated intensity bands.

### Automated Detection

```python
import numpy as np
from scipy import ndimage


def detect_streak_artifacts(reconstruction, threshold_percentile=99.5):
    """
    Detect streak artifacts in a reconstructed CT slice by identifying
    high-intensity linear features radiating from dense objects.

    Parameters
    ----------
    reconstruction : np.ndarray
        2D reconstructed slice.
    threshold_percentile : float
        Percentile to identify highly absorbing objects.

    Returns
    -------
    dict with keys:
        'has_metal_objects' : bool
        'metal_mask' : np.ndarray — binary mask of dense objects
        'starvation_score' : float — fraction of extreme-value pixels
        'streak_likelihood' : str — 'high', 'moderate', 'low'
    """
    # Identify highly absorbing objects
    high_thresh = np.percentile(reconstruction, threshold_percentile)
    metal_mask = reconstruction > high_thresh

    # Label connected metal regions
    labeled, num_objects = ndimage.label(metal_mask)

    # Compute radial variance around metal objects as streak indicator
    # Streaks create high variance along radial directions
    grad_y, grad_x = np.gradient(reconstruction)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Fraction of pixels with extreme gradient (streak edges)
    extreme_grad_thresh = np.percentile(gradient_magnitude, 99)
    extreme_fraction = np.sum(gradient_magnitude > extreme_grad_thresh) / reconstruction.size

    if num_objects > 0 and extreme_fraction > 0.015:
        likelihood = "high"
    elif num_objects > 0 and extreme_fraction > 0.008:
        likelihood = "moderate"
    else:
        likelihood = "low"

    return {
        "has_metal_objects": num_objects > 0,
        "metal_mask": metal_mask,
        "num_dense_objects": num_objects,
        "starvation_score": extreme_fraction,
        "streak_likelihood": likelihood,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Use higher beam energy to improve transmission through dense objects.
- Increase exposure time or photon flux to reduce photon starvation.
- Where possible, remove metal components from the sample before scanning.
- Use monochromatic beam to eliminate beam hardening contribution.
- Consider scanning at multiple energies for spectral decomposition.

### Correction — Traditional Methods

Metal artifact reduction (MAR) techniques segment the metal in the sinogram and replace the corrupted regions with interpolated or inpainted values before reconstruction.

```python
import numpy as np
from scipy import ndimage, interpolate


def simple_mar_correction(sinogram, reconstruction, metal_threshold=None):
    """
    Simple metal artifact reduction by sinogram inpainting.
    Identifies metal regions, forward-projects to create a sinogram mask,
    and interpolates across corrupted sinogram columns.
    """
    if metal_threshold is None:
        metal_threshold = np.percentile(reconstruction, 99.5)

    # Create metal mask from initial reconstruction
    metal_mask = reconstruction > metal_threshold

    # Forward-project metal mask to identify corrupted sinogram regions
    # (simplified — in practice use Radon transform)
    num_angles = sinogram.shape[0]
    sino_mask = np.zeros_like(sinogram, dtype=bool)

    for i in range(num_angles):
        angle_rad = np.pi * i / num_angles
        # Project metal mask at this angle
        rotated = ndimage.rotate(metal_mask.astype(float), np.degrees(angle_rad),
                                 reshape=False, order=1)
        projection = np.sum(rotated, axis=0)
        # Crop or pad to match sinogram width
        proj_len = min(len(projection), sinogram.shape[1])
        sino_mask[i, :proj_len] = projection[:proj_len] > 0.5

    # Interpolate across masked regions in each sinogram row
    sinogram_corrected = sinogram.copy()
    for i in range(num_angles):
        row = sinogram[i].copy()
        mask = sino_mask[i]
        if not np.any(mask):
            continue
        valid = ~mask
        if np.sum(valid) < 2:
            continue
        x_valid = np.where(valid)[0]
        x_masked = np.where(mask)[0]
        interp_func = interpolate.interp1d(x_valid, row[valid],
                                            kind='linear',
                                            fill_value='extrapolate')
        sinogram_corrected[i, mask] = interp_func(x_masked)

    return sinogram_corrected
```

### Correction — AI/ML Methods

Deep learning inpainting approaches have shown strong results for metal artifact reduction. A convolutional neural network (typically U-Net or GAN architecture) is trained to predict clean sinogram regions from metal-corrupted inputs. The network learns to fill in plausible attenuation values in the photon-starved sinogram bands, producing reconstructions with significantly reduced streaks. Supervised training requires paired data with and without metal, often generated synthetically by inserting simulated metal objects into clean datasets. Notable approaches include LI-MAR (learned interpolation MAR) and conditional GAN-based sinogram completion networks.

## Impact If Uncorrected

Streak artifacts can completely obscure the region surrounding dense objects, making it impossible to analyze nearby structures. Quantitative measurements of material density, porosity, or composition in the affected regions are unreliable. Segmentation algorithms will misclassify streak regions as material boundaries or distinct phases. In industrial CT for inspection of metal parts, streaks can hide internal defects such as cracks or voids adjacent to dense features.

## Related Resources

- [AI/ML methods for tomography](../../02_xray_modalities/tomography/ai_ml_methods.md) — DL-based MAR approaches
- Related artifact: [Ring Artifact](ring_artifact.md) — another systematic artifact but with circular rather than radial pattern
- Related artifact: [Low-Dose Poisson Noise](low_dose_noise.md) — photon starvation is an extreme case of low photon count

## Key Takeaway

Streak artifacts arise from photon starvation and beam hardening when the beam traverses highly absorbing objects. Prevention through higher-energy or monochromatic beams is most effective; when metal is unavoidable, sinogram inpainting via MAR or deep learning can substantially reduce streaks before reconstruction.
