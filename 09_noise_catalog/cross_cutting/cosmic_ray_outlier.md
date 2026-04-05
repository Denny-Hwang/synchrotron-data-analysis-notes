# Cosmic Ray / Outlier Spike Detection

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting (All imaging modalities) |
| **Noise Type** | Statistical |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Easy |
| **Origin Domain** | Astronomy / Synchrotron / Electron Microscopy |

## Visual Examples

![Before and after — cosmic ray detection](../images/cosmic_ray_before_after.png)

> **Image source:** Synthetic image with injected cosmic ray points and tracks. Left: bright pixels and short tracks from CR hits. Right: after L.A.Cosmic Laplacian detection and median replacement. MIT license.

## Description

Cosmic ray events (also called zingers in synchrotron context, hot pixels in astronomy) are isolated, anomalously bright pixels or small clusters caused by high-energy particle impacts on the detector. In astronomy, this is one of the most well-characterized noise types with mature detection tools (L.A.Cosmic, astroscrappy). The synchrotron catalog already covers zingers in tomography; this entry focuses on the broader cross-domain perspective and tools from astronomy and electron microscopy.

**Cross-domain value:** Astronomy has decades of experience with cosmic ray rejection algorithms that transfer directly to synchrotron data. The Laplacian edge detection approach (L.A.Cosmic) is applicable to any 2D detector data.

## Root Cause

- High-energy particles (cosmic rays, radioactive decay) deposit energy in detector
- Single event creates: bright pixel, streak/track, or cluster (depending on energy and angle)
- Rate: ~1-5 events/cm²/min for typical silicon detectors at sea level
- Higher at altitude, during solar events, or near radioactive materials
- In counting detectors (Pilatus/Eiger): single pixel > statistical expectation

## Quick Diagnosis

```python
import numpy as np
from scipy.ndimage import median_filter, laplace

def detect_cosmic_rays_laplacian(image, sigma_clip=5.0):
    """L.A.Cosmic-inspired cosmic ray detection using Laplacian."""
    # Laplacian highlights sharp point sources
    lap = laplace(image.astype(float))
    # Normalize by local noise estimate
    med = median_filter(image, size=5)
    noise = np.sqrt(np.abs(med) + 1)  # Poisson noise estimate
    significance = lap / noise
    # Cosmic rays: high negative Laplacian (bright point on smooth background)
    cosmic_mask = significance < -sigma_clip
    n_detected = cosmic_mask.sum()
    print(f"Cosmic ray candidates: {n_detected} pixels ({n_detected/image.size:.4%})")
    return cosmic_mask
```

## Detection Methods

### Visual Indicators

- Isolated extremely bright pixels (much brighter than neighbors)
- Sometimes appear as short tracks/streaks (oblique incidence)
- Do not persist between frames at same position
- Sharp edges (unlike real features which have PSF-limited extent)

### Automated Detection (Astronomical Methods)

```python
import numpy as np
from scipy.ndimage import median_filter

def lacosmic_detect(image, gain=1.0, readnoise=5.0, sigclip=5.0, sigfrac=0.3, objlim=5.0):
    """Simplified L.A.Cosmic algorithm (van Dokkum, 2001)."""
    # Step 1: Subsample and Laplacian
    # Step 2: Build noise model
    med5 = median_filter(image, size=5)
    noise = np.sqrt(np.abs(med5) / gain + readnoise**2)
    # Step 3: Fine-structure image (to distinguish stars from CRs)
    med3 = median_filter(image, size=3)
    fine_structure = med3 - median_filter(med3, size=7)
    # Step 4: Identify CRs
    residual = image - med5
    cr_candidates = residual > sigclip * noise
    # Exclude real structure
    cr_mask = cr_candidates & (fine_structure < objlim * noise)
    return cr_mask

def replace_cosmic_rays(image, cr_mask, method='median'):
    """Replace detected cosmic rays with local median."""
    cleaned = image.copy()
    if method == 'median':
        med = median_filter(image, size=5)
        cleaned[cr_mask] = med[cr_mask]
    return cleaned
```

## Tools from Other Domains

### Astronomy

| Tool | Description | Transferability |
|------|-------------|----------------|
| **L.A.Cosmic** (van Dokkum, 2001) | Laplacian edge detection | Directly applicable to any 2D detector |
| **astroscrappy** | Optimized C implementation of L.A.Cosmic | pip-installable, fast |
| **AstroPy ccdproc** | Complete CCD processing pipeline | Template for synchrotron pipeline |
| **SExtractor** | Source detection with outlier rejection | SAXS/powder diffraction spot finding |

### Electron Microscopy

| Tool | Description |
|------|-------------|
| **SerialEM** | Real-time outlier rejection during tilt series |
| **IMOD** | X-ray/hot pixel removal from tilt series |
| **RELION** — outlier rejection | Per-micrograph hot pixel removal |

### Synchrotron (Existing)

| Tool | Description |
|------|-------------|
| **TomoPy** | `remove_outlier()` for projections |
| **Savu** (Diamond) | Pipeline with outlier removal plugins |
| **DAWN** (Diamond) | Interactive outlier identification |

## Key References

- **van Dokkum (2001)** — "Cosmic-ray rejection by Laplacian edge detection" (L.A.Cosmic) — seminal paper
- **McCully et al. (2018)** — "astroscrappy" — fast L.A.Cosmic implementation
- **Rhoads (2000)** — "Cosmic ray rejection via multiresolution detection" — wavelet approach
- **Offermann et al. (2022)** — "Deep learning for cosmic ray removal in astronomical images"

## Benchmarks & Datasets

| Benchmark | Domain | Description |
|-----------|--------|-------------|
| Hubble ACS darks | Astronomy | Well-characterized CR rate and morphology |
| EMPIAR dark frames | Cryo-EM | Electron detector dark current + cosmic rays |
| TomoBank dark scans | Synchrotron | APS detector darks with zingers |

## Real-World Before/After Examples

The following published sources provide real experimental before/after comparisons:

| Source | Type | Figure | Description | License |
|--------|------|--------|-------------|---------|
| [Astropy CCD Guide — Section 6.3](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-03-Cosmic-ray-removal.html) | Tutorial notebook | Multiple | Before/after cosmic ray removal on real CCD astronomical data | BSD-3 |
| [van Dokkum 2001](https://doi.org/10.1086/323894) | Paper | Fig 2 | Cosmic-Ray Rejection by Laplacian Edge Detection — the seminal L.A.Cosmic paper with real before/after examples | -- |

> **Recommended reference**: [Astropy CCD Guide — Cosmic ray removal notebook with interactive before/after examples](https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/08-03-Cosmic-ray-removal.html)

## Related Resources

- [Zinger](../tomography/zinger.md) — Synchrotron-specific cosmic ray treatment in tomography
- [Dead/hot pixel](../xrf_microscopy/dead_hot_pixel.md) — Persistent pixel defects vs transient cosmic rays
- [Detector common issues](detector_common_issues.md) — General detector noise characterization
