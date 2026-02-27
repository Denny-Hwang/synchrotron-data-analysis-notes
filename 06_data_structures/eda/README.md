# Exploratory Data Analysis (EDA) for Synchrotron Data

## Purpose

Exploratory Data Analysis is the essential first step after data acquisition and before
any advanced processing or machine learning. For synchrotron experiments, EDA serves to:

1. **Verify data integrity** -- Confirm that files are complete, uncorrupted, and contain
   the expected number of frames, channels, or energy points.
2. **Assess data quality** -- Identify dead pixels, hot pixels, saturated regions,
   detector artifacts, and noise levels.
3. **Understand distributions** -- Examine intensity histograms, dynamic range utilization,
   and statistical summaries to inform normalization strategies.
4. **Discover structure** -- Find spatial patterns, correlations between channels,
   and unexpected features that may guide analysis.
5. **Inform preprocessing** -- Determine which corrections (background subtraction,
   ring removal, phase retrieval) are needed and parameterize them.
6. **Document baselines** -- Establish quantitative benchmarks (SNR, resolution metrics)
   against which processed results can be compared.

## Universal EDA Checklist

Apply this checklist to any synchrotron dataset regardless of modality:

### File-Level Checks

- [ ] File opens without errors (not corrupted or truncated)
- [ ] Expected groups/datasets are present in the HDF5 hierarchy
- [ ] Dataset shapes match expected dimensions (rows, cols, channels, angles)
- [ ] Data types are correct (uint16, float32, etc.)
- [ ] Metadata attributes are populated (energy, pixel size, timestamps)
- [ ] File size is consistent with expectations (not suspiciously small)

### Array-Level Statistics

- [ ] Compute min, max, mean, median, std for each primary dataset
- [ ] Check for NaN, Inf, or negative values where unexpected
- [ ] Verify dynamic range utilization (not clipped at 0 or saturated)
- [ ] Compare statistics across frames/slices for consistency
- [ ] Identify outlier frames with anomalous statistics

### Spatial Quality

- [ ] Visualize representative frames/slices
- [ ] Check for dead pixel clusters or detector gaps
- [ ] Look for systematic patterns (stripes, rings, moiré)
- [ ] Verify spatial extent matches expected sample dimensions
- [ ] Check alignment between reference frames and data

### Metadata Validation

- [ ] Beam energy matches expected value
- [ ] Pixel size is correct for the configured magnification
- [ ] Scan positions span the expected range
- [ ] Timestamps are sequential and reasonable
- [ ] Motor positions are within expected bounds

## Modality-Specific EDA Guides

Each synchrotron technique has unique data characteristics and quality indicators:

| Modality | Guide | Key Concerns |
|----------|-------|-------------|
| X-ray Fluorescence | [xrf_eda.md](xrf_eda.md) | Channel crosstalk, dead pixels, I0 normalization |
| Tomography | [tomo_eda.md](tomo_eda.md) | Ring artifacts, rotation center, flat/dark quality |
| Spectroscopy | [spectroscopy_eda.md](spectroscopy_eda.md) | Edge alignment, self-absorption, noise |

## Common EDA Tools

| Tool | Purpose | Installation |
|------|---------|-------------|
| **h5py** | HDF5 reading and inspection | `pip install h5py` |
| **matplotlib** | 2D visualization | `pip install matplotlib` |
| **numpy** | Array statistics | `pip install numpy` |
| **scipy** | Signal processing, statistics | `pip install scipy` |
| **silx** | HDF5 GUI browser + plotting | `pip install silx` |
| **pandas** | Tabular metadata analysis | `pip install pandas` |
| **scikit-image** | Image quality metrics | `pip install scikit-image` |
| **seaborn** | Statistical visualization | `pip install seaborn` |

## Standard EDA Code Patterns

### Quick Statistics Summary

```python
import h5py
import numpy as np

def dataset_summary(filepath, dataset_path):
    """Print summary statistics for an HDF5 dataset."""
    with h5py.File(filepath, "r") as f:
        dset = f[dataset_path]
        print(f"Path:   {dataset_path}")
        print(f"Shape:  {dset.shape}")
        print(f"Dtype:  {dset.dtype}")
        print(f"Chunks: {dset.chunks}")
        print(f"Compression: {dset.compression}")

        # Load data (or a subset for large datasets)
        if dset.nbytes > 1e9:
            data = dset[0]   # First frame only
            print("(Statistics for first frame only)")
        else:
            data = dset[:]

        print(f"Min:    {np.nanmin(data):.4g}")
        print(f"Max:    {np.nanmax(data):.4g}")
        print(f"Mean:   {np.nanmean(data):.4g}")
        print(f"Median: {np.nanmedian(data):.4g}")
        print(f"Std:    {np.nanstd(data):.4g}")
        print(f"NaNs:   {np.isnan(data).sum()}")
        print(f"Zeros:  {(data == 0).sum()} ({100*(data==0).mean():.1f}%)")
```

### Frame-by-Frame Consistency

```python
def check_frame_consistency(filepath, dataset_path, axis=0):
    """Check mean/std across frames to find anomalous ones."""
    with h5py.File(filepath, "r") as f:
        dset = f[dataset_path]
        nframes = dset.shape[axis]
        means = np.zeros(nframes)
        stds = np.zeros(nframes)
        for i in range(nframes):
            frame = np.take(dset, i, axis=axis)
            means[i] = np.mean(frame)
            stds[i] = np.std(frame)

    # Flag outlier frames
    mean_of_means = np.mean(means)
    std_of_means = np.std(means)
    outliers = np.where(np.abs(means - mean_of_means) > 3 * std_of_means)[0]
    if len(outliers) > 0:
        print(f"WARNING: {len(outliers)} outlier frames detected: {outliers}")
    return means, stds
```

## Hands-On Notebooks

| Notebook | Description |
|----------|-------------|
| [01_xrf_eda.ipynb](notebooks/01_xrf_eda.ipynb) | XRF elemental map exploration |
| [02_tomo_eda.ipynb](notebooks/02_tomo_eda.ipynb) | Tomography projection & sinogram analysis |
| [03_spectral_eda.ipynb](notebooks/03_spectral_eda.ipynb) | Spectroscopy edge & noise analysis |

## EDA Workflow Summary

```
1. Open file, verify structure
        |
2. Compute global statistics
        |
3. Visualize representative frames
        |
4. Check for artifacts & outliers
        |
5. Examine metadata consistency
        |
6. Document findings
        |
7. Proceed to preprocessing
```

## Related Resources

- [HDF5 structure guides](../hdf5_structure/)
- [Data scale analysis](../data_scale_analysis.md)
- [Processing pipeline](../../07_data_pipeline/)
