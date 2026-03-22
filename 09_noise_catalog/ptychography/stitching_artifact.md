# Stitching Artifact

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Ptychography |
| **Noise Type** | Computational |
| **Severity** | Minor |
| **Frequency** | Occasional |
| **Detection Difficulty** | Easy |

## Visual Examples

```
Tile A          Tile B                 Stitched result (artifact)
┌──────────┐  ┌──────────┐           ┌──────────┬──────────┐
│  phase=   │  │  phase=   │           │  phase=  ┃  phase=   │
│   1.2 rad │  │  1.2 rad  │    →     │  1.2 rad ┃  1.8 rad  │
│           │  │  + 0.6    │           │          ┃ (offset!) │
│           │  │  (offset) │           │  ────────┃────────── │
└──────────┘  └──────────┘           └──────────┴──────────┘
                                      Visible seam at boundary ↑

After correction:
┌─────────────────────┐
│  phase = 1.2 rad    │  Seamless
│  ────────────────── │
└─────────────────────┘
```

> **External references:**
> - [Dierolf et al. — large-area ptychography](https://doi.org/10.1088/1367-2630/12/3/035017)
> - [Wakonig et al. — PtychoShelves stitching](https://doi.org/10.1107/S1600577520001587)

## Description

Stitching artifacts manifest as visible seams, intensity discontinuities, or phase jumps at the boundaries between separately reconstructed tiles in large-area ptychographic imaging. When a sample exceeds the single-scan field of view, multiple overlapping scans are acquired and independently reconstructed, then stitched together. Differences in convergence level, phase offsets, probe intensity variations, and background drift between tiles create artificial boundaries in the composite image.

## Root Cause

Each ptychographic reconstruction has an inherent phase ambiguity — the absolute phase is undefined, so different tiles converge with arbitrary phase offsets. Additionally, beam intensity may drift between tile acquisitions (minutes to hours), causing amplitude mismatches. Different tiles may converge to different local minima if initial conditions vary, and slight differences in probe state, coherence conditions, or sample drift during scanning create tile-to-tile inconsistencies. Background contributions (e.g., air scatter) may also change between scans if the sample is repositioned.

## Quick Diagnosis

```python
import numpy as np

# Load two adjacent tile reconstructions (complex-valued)
# tile_a, tile_b = ...  # overlapping complex arrays
# Extract the overlap region
overlap_a = tile_a[:, -50:]   # last 50 columns of tile A
overlap_b = tile_b[:, :50]    # first 50 columns of tile B
phase_diff = np.angle(overlap_a * np.conj(overlap_b))
print(f"Mean phase offset: {np.mean(phase_diff):.3f} rad")
print(f"Phase offset std:  {np.std(phase_diff):.3f} rad (>0.1 indicates stitching issue)")
```

## Detection Methods

### Visual Indicators

- Sharp lines or seams visible at tile boundaries in the phase or amplitude image.
- Abrupt intensity or phase step at the overlap region between tiles.
- Moire-like patterns in the overlap if tiles are slightly misaligned.
- Background level differs visibly between adjacent tiles.
- Features crossing tile boundaries appear displaced or duplicated.

### Automated Detection

```python
import numpy as np
from scipy.ndimage import sobel


def detect_stitching_artifacts(composite_image, tile_boundaries,
                                phase_threshold=0.2, intensity_threshold=0.05):
    """
    Detect stitching artifacts by analyzing discontinuities at known
    tile boundaries in a composite reconstruction.

    Parameters
    ----------
    composite_image : np.ndarray
        2D complex-valued stitched reconstruction.
    tile_boundaries : list of dict
        Each dict has 'orientation' ('horizontal' or 'vertical'),
        'position' (int, pixel coordinate of boundary),
        'start' and 'end' (int, extent along boundary).
    phase_threshold : float
        Maximum acceptable phase jump across boundary (radians).
    intensity_threshold : float
        Maximum acceptable relative amplitude jump.

    Returns
    -------
    dict with keys:
        'boundary_results' : list of dict per boundary
        'has_stitching_artifact' : bool
    """
    phase_map = np.angle(composite_image)
    amp_map = np.abs(composite_image)

    results = []
    has_artifact = False

    for boundary in tile_boundaries:
        orient = boundary["orientation"]
        pos = boundary["position"]
        start = boundary.get("start", 0)

        if orient == "vertical":
            end = boundary.get("end", composite_image.shape[0])
            # Extract strips on either side of boundary
            left_phase = phase_map[start:end, max(0, pos - 5):pos]
            right_phase = phase_map[start:end, pos:min(pos + 5, phase_map.shape[1])]
            left_amp = amp_map[start:end, max(0, pos - 5):pos]
            right_amp = amp_map[start:end, pos:min(pos + 5, amp_map.shape[1])]
        else:  # horizontal
            end = boundary.get("end", composite_image.shape[1])
            left_phase = phase_map[max(0, pos - 5):pos, start:end]
            right_phase = phase_map[pos:min(pos + 5, phase_map.shape[0]), start:end]
            left_amp = amp_map[max(0, pos - 5):pos, start:end]
            right_amp = amp_map[pos:min(pos + 5, amp_map.shape[0]), start:end]

        # Phase discontinuity
        mean_phase_left = np.mean(left_phase)
        mean_phase_right = np.mean(right_phase)
        phase_jump = np.abs(mean_phase_right - mean_phase_left)

        # Amplitude discontinuity
        mean_amp_left = np.mean(left_amp)
        mean_amp_right = np.mean(right_amp)
        amp_ratio = abs(mean_amp_right - mean_amp_left) / (
            0.5 * (mean_amp_left + mean_amp_right) + 1e-10
        )

        boundary_has_artifact = (
            phase_jump > phase_threshold or amp_ratio > intensity_threshold
        )
        has_artifact = has_artifact or boundary_has_artifact

        results.append({
            "orientation": orient,
            "position": pos,
            "phase_jump_rad": float(phase_jump),
            "amplitude_ratio": float(amp_ratio),
            "has_artifact": boundary_has_artifact,
        })

    return {
        "boundary_results": results,
        "has_stitching_artifact": has_artifact,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Ensure sufficient overlap between adjacent tiles (at least 20-30% of tile width).
- Maintain stable beam conditions across all tile acquisitions by monitoring I₀.
- Acquire all tiles in rapid succession to minimize drift.
- Use a common reference marker (e.g., gold nanoparticle) visible in multiple tiles for alignment validation.

### Correction — Traditional Methods

Global phase offset correction and overlap-weighted blending eliminate most stitching artifacts. For more complex cases, Laplacian pyramid blending produces seamless composites.

```python
import numpy as np
from scipy.ndimage import gaussian_filter


def stitch_tiles_with_blending(tiles, positions, overlap_pixels=50):
    """
    Stitch ptychographic reconstruction tiles using phase alignment
    and overlap-weighted blending.

    Parameters
    ----------
    tiles : list of np.ndarray
        Complex-valued tile reconstructions.
    positions : list of tuple
        (row, col) top-left corner of each tile in the global canvas.
    overlap_pixels : int
        Expected overlap width in pixels.

    Returns
    -------
    composite : np.ndarray — stitched complex image
    """
    # Determine canvas size
    max_row = max(p[0] + t.shape[0] for p, t in zip(positions, tiles))
    max_col = max(p[1] + t.shape[1] for p, t in zip(positions, tiles))
    composite = np.zeros((max_row, max_col), dtype=complex)
    weight_map = np.zeros((max_row, max_col), dtype=float)

    # Use first tile as phase reference
    ref_tile = tiles[0]

    for i, (tile, (r0, c0)) in enumerate(zip(tiles, positions)):
        # Phase alignment: find global offset relative to composite overlap
        if i > 0:
            # Extract overlap region between current tile and existing composite
            r_end = r0 + tile.shape[0]
            c_end = c0 + tile.shape[1]
            existing_region = composite[r0:r_end, c0:c_end]
            existing_weight = weight_map[r0:r_end, c0:c_end]

            overlap_mask = existing_weight > 0
            if np.sum(overlap_mask) > 100:
                # Compute phase offset from overlap
                phase_offset = np.angle(
                    np.sum(existing_region[overlap_mask]
                           * np.conj(tile[overlap_mask]))
                )
                tile = tile * np.exp(1j * phase_offset)

                # Amplitude scaling
                amp_ratio = (
                    np.mean(np.abs(existing_region[overlap_mask]))
                    / (np.mean(np.abs(tile[overlap_mask])) + 1e-10)
                )
                tile = tile * amp_ratio

        # Create distance-from-edge weight (feathering)
        wy = np.minimum(
            np.arange(tile.shape[0]),
            np.arange(tile.shape[0])[::-1]
        ).astype(float)
        wx = np.minimum(
            np.arange(tile.shape[1]),
            np.arange(tile.shape[1])[::-1]
        ).astype(float)
        weight = np.outer(wy, wx)
        weight = np.clip(weight / (overlap_pixels / 2), 0, 1)

        # Accumulate weighted tile
        composite[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] += tile * weight
        weight_map[r0:r0 + tile.shape[0], c0:c0 + tile.shape[1]] += weight

    # Normalize by total weight
    valid = weight_map > 0
    composite[valid] /= weight_map[valid]

    return composite
```

### Correction — AI/ML Methods

Deep learning inpainting networks can be trained to remove stitching seams by learning the expected continuity of phase and amplitude across boundaries. A U-Net trained on synthetically generated stitching artifacts (random phase offsets applied to sub-regions) can predict and remove residual discontinuities in the final composite. However, these approaches should be used with caution as they may introduce hallucinated features near boundaries.

## Impact If Uncorrected

Stitching artifacts are generally cosmetic for qualitative imaging but can cause quantitative errors if phase measurements are averaged across tile boundaries. Segmentation algorithms may interpret seams as real material boundaries, inflating interface counts. For strain mapping or wavefront sensing applications, phase discontinuities at tile boundaries directly corrupt the gradient field, producing erroneous strain values at those locations. In large-area surveys, many small stitching offsets can accumulate into substantial global phase ramps.

## Related Resources

- [Position Error](position_error.md) — position errors at tile boundaries exacerbate stitching mismatches
- [Partial Coherence Effects](partial_coherence.md) — coherence drift between tiles contributes to intensity mismatches
- [Ptychography overview](../../02_xray_modalities/ptychography/README.md) — scan geometry and overlap requirements
- [Wakonig et al. 2020](https://doi.org/10.1107/S1600577520001587) — PtychoShelves stitching pipeline

## Key Takeaway

Stitching artifacts are the easiest ptychographic artifact to detect and correct — always perform global phase alignment and overlap-weighted blending when compositing tiles, and verify the result by checking that features crossing tile boundaries are continuous and consistent.
