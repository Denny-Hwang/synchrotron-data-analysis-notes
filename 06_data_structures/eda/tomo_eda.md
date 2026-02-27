# Tomography Data Exploratory Data Analysis

## Overview

Tomography datasets consist of projection images acquired at multiple rotation angles,
along with flat-field (open beam) and dark-field (no beam) reference images. EDA for
tomography focuses on projection quality, artifact detection, rotation center
verification, and sinogram analysis -- all of which critically affect reconstruction
quality.

This guide assumes data in the Data Exchange HDF5 format as described in
[tomo_hdf5_schema.md](../hdf5_structure/tomo_hdf5_schema.md).

## EDA Checklist for Tomography

### 1. Data Completeness

- [ ] All expected projections are present (e.g., 1800 for 0.1-degree steps over 180)
- [ ] Flat-field images collected (typically 10--40 frames)
- [ ] Dark-field images collected (typically 10--20 frames)
- [ ] Theta array length matches number of projections
- [ ] Angular range is correct (180 or 360 degrees)
- [ ] Theta values are monotonically increasing

### 2. Reference Frame Quality

- [ ] Flat-field images are uniform (no debris on scintillator)
- [ ] Dark-field images are low and consistent
- [ ] Flat-field variation < 20% across the field of view
- [ ] No saturated pixels in flat fields

### 3. Projection Quality

- [ ] Sample is fully within the field of view at all angles
- [ ] No large intensity jumps between consecutive projections
- [ ] No motion blur (exposure time appropriate for rotation speed)
- [ ] Beam intensity (I0) is stable across the scan

### 4. Artifact Assessment

- [ ] Ring artifact potential assessed via sinogram inspection
- [ ] Zingers (cosmic ray spots) identified
- [ ] Phase contrast fringes characterized (if present)

## Projection Quality Assessment

### Visual Inspection

```python
import h5py
import numpy as np
import matplotlib.pyplot as plt

with h5py.File("tomo_scan.h5", "r") as f:
    proj = f["/exchange/data"]
    flat = f["/exchange/data_white"][:]
    dark = f["/exchange/data_dark"][:]
    theta = f["/exchange/theta"][:]

    nproj, nrow, ncol = proj.shape
    print(f"Projections: {nproj}, Size: {nrow}x{ncol}")
    print(f"Theta range: {np.degrees(theta[0]):.1f} to {np.degrees(theta[-1]):.1f} deg")

    # Show projections at 0, 45, 90, 135 degrees
    angles_deg = np.degrees(theta)
    target_angles = [0, 45, 90, 135]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, target in zip(axes, target_angles):
        idx = np.argmin(np.abs(angles_deg - target))
        ax.imshow(proj[idx], cmap="gray", origin="lower")
        ax.set_title(f"Projection @ {angles_deg[idx]:.1f} deg")
    plt.tight_layout()
```

### Frame Statistics Across Angles

```python
# Compute mean intensity per projection
means = np.zeros(nproj)
for i in range(nproj):
    with h5py.File("tomo_scan.h5", "r") as f:
        means[i] = np.mean(f["/exchange/data"][i])

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(np.degrees(theta), means, lw=0.5)
ax.set_xlabel("Rotation Angle (degrees)")
ax.set_ylabel("Mean Intensity")
ax.set_title("Projection Mean Intensity vs. Angle")

# Flag intensity drops > 10%
median_mean = np.median(means)
drops = np.where(means < 0.9 * median_mean)[0]
if len(drops) > 0:
    ax.scatter(np.degrees(theta[drops]), means[drops], c="red", s=20, zorder=5)
    print(f"WARNING: {len(drops)} projections with >10% intensity drop")
```

## Sinogram Visualization

Sinograms (angle vs. horizontal position at a fixed row) reveal rotation-related
artifacts and are essential for diagnosing reconstruction problems.

```python
with h5py.File("tomo_scan.h5", "r") as f:
    # Extract sinograms at three vertical positions
    rows = [nrow // 4, nrow // 2, 3 * nrow // 4]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, row in zip(axes, rows):
        sino = f["/exchange/data"][:, row, :]
        ax.imshow(sino, cmap="gray", aspect="auto", origin="lower",
                  extent=[0, ncol, np.degrees(theta[0]), np.degrees(theta[-1])])
        ax.set_xlabel("Detector Column")
        ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"Sinogram at row {row}")
    plt.tight_layout()
```

## Ring Artifact Detection

Ring artifacts manifest as vertical stripes in sinograms. They originate from defective
or miscalibrated detector pixels.

```python
def detect_ring_artifacts(sinogram, threshold=3.0):
    """Detect potential ring artifacts by analyzing column-wise statistics."""
    col_means = np.mean(sinogram, axis=0)
    col_median = np.median(col_means)
    col_mad = np.median(np.abs(col_means - col_median))

    # Columns deviating from median by more than threshold * MAD
    ring_cols = np.where(np.abs(col_means - col_median) > threshold * col_mad)[0]
    return ring_cols, col_means

with h5py.File("tomo_scan.h5", "r") as f:
    sino = f["/exchange/data"][:, nrow // 2, :]

ring_cols, col_profile = detect_ring_artifacts(sino)
print(f"Potential ring artifact columns: {len(ring_cols)}")

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(col_profile, lw=0.5)
ax.scatter(ring_cols, col_profile[ring_cols], c="red", s=10, zorder=5)
ax.set_xlabel("Column Index")
ax.set_ylabel("Mean Intensity (across angles)")
ax.set_title("Column-wise Profile (ring artifact detection)")
```

## Rotation Center Estimation

An incorrect rotation center causes characteristic artifacts in reconstruction:
cupping, doubling, or blurring.

```python
import tomopy

with h5py.File("tomo_scan.h5", "r") as f:
    proj = f["/exchange/data"][:]
    flat = f["/exchange/data_white"][:]
    dark = f["/exchange/data_dark"][:]
    theta = f["/exchange/theta"][:]

proj_norm = tomopy.normalize(proj, flat, dark)
proj_norm = tomopy.minus_log(proj_norm)

# Automated center finding
center_vo = tomopy.find_center_vo(proj_norm)
center_pc = tomopy.find_center_pc(proj_norm[0], proj_norm[-1])

print(f"Vo method center:  {center_vo:.2f}")
print(f"Phase corr center: {center_pc:.2f}")
print(f"Image center:      {ncol / 2:.1f}")
print(f"Offset from image center: {center_vo - ncol/2:.2f} pixels")

# Visual verification: reconstruct one slice at multiple centers
test_centers = np.arange(center_vo - 5, center_vo + 5.5, 0.5)
fig, axes = plt.subplots(2, 5, figsize=(20, 8))
for ax, c in zip(axes.ravel(), test_centers[:10]):
    sino_slice = proj_norm[:, nrow//2:nrow//2+1, :]
    rec = tomopy.recon(sino_slice, theta, center=c, algorithm="gridrec")
    ax.imshow(rec[0], cmap="gray")
    ax.set_title(f"c={c:.1f}")
    ax.axis("off")
plt.suptitle("Rotation Center Scan")
plt.tight_layout()
```

## Flat-Field and Dark-Field Analysis

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

# Dark field analysis
dark_mean = np.mean(dark, axis=0)
dark_std = np.std(dark, axis=0)
axes[0, 0].imshow(dark_mean, cmap="gray")
axes[0, 0].set_title(f"Dark Mean (global mean={dark_mean.mean():.1f})")
axes[0, 1].imshow(dark_std, cmap="hot")
axes[0, 1].set_title(f"Dark Std (max={dark_std.max():.1f})")
axes[0, 2].hist(dark_mean.ravel(), bins=100, log=True)
axes[0, 2].set_title("Dark Mean Histogram")

# Flat field analysis
flat_mean = np.mean(flat, axis=0)
flat_std = np.std(flat, axis=0)
flat_norm = flat_std / (flat_mean + 1e-10)  # Coefficient of variation
axes[1, 0].imshow(flat_mean, cmap="gray")
axes[1, 0].set_title(f"Flat Mean (global mean={flat_mean.mean():.1f})")
axes[1, 1].imshow(flat_norm, cmap="hot", vmax=0.1)
axes[1, 1].set_title("Flat Coeff. of Variation")
axes[1, 2].hist(flat_mean.ravel(), bins=100)
axes[1, 2].set_title("Flat Mean Histogram")

plt.tight_layout()
```

## Histogram Analysis

```python
with h5py.File("tomo_scan.h5", "r") as f:
    # Sample random projections for histogram
    indices = np.random.choice(nproj, size=min(50, nproj), replace=False)
    sample_data = np.stack([f["/exchange/data"][i] for i in sorted(indices)])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Raw projection histogram
axes[0].hist(sample_data.ravel(), bins=200, log=True, color="steelblue")
axes[0].set_title("Raw Projection Histogram")
axes[0].set_xlabel("Intensity (counts)")

# After normalization
norm_data = (sample_data - dark_mean) / (flat_mean - dark_mean + 1e-10)
axes[1].hist(norm_data.ravel(), bins=200, log=True, color="darkorange")
axes[1].set_title("Normalized Projection Histogram")
axes[1].set_xlabel("Transmission (I/I0)")

# After -log transform
log_data = -np.log(np.clip(norm_data, 1e-6, None))
axes[2].hist(log_data.ravel(), bins=200, log=True, color="forestgreen")
axes[2].set_title("After -log (Absorption)")
axes[2].set_xlabel("Absorption (a.u.)")

plt.tight_layout()
```

## Zinger (Outlier Spot) Detection

Zingers are isolated bright spots caused by cosmic rays or detector noise bursts:

```python
from scipy.ndimage import median_filter

def detect_zingers(projection, threshold=10.0):
    """Detect zinger pixels in a single projection."""
    filtered = median_filter(projection.astype(float), size=3)
    diff = projection.astype(float) - filtered
    mad = np.median(np.abs(diff))
    zingers = np.abs(diff) > threshold * max(mad, 1.0)
    return zingers

# Check a sample of projections
total_zingers = 0
with h5py.File("tomo_scan.h5", "r") as f:
    for i in range(0, nproj, max(1, nproj // 20)):
        proj_i = f["/exchange/data"][i]
        z = detect_zingers(proj_i)
        count = z.sum()
        total_zingers += count
        if count > 10:
            print(f"  Projection {i}: {count} zingers detected")

print(f"Total zingers found in sampled projections: {total_zingers}")
```

## Related Resources

- [Tomography HDF5 schema](../hdf5_structure/tomo_hdf5_schema.md)
- [Tomography EDA notebook](notebooks/02_tomo_eda.ipynb)
- [Tomography modality overview](../../02_xray_modalities/)
- [TomoPy documentation](https://tomopy.readthedocs.io/)
