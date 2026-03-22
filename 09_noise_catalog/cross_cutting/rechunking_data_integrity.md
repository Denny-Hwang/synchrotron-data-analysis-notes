# Rechunking / Data Integrity Issues

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting |
| **Noise Type** | Computational |
| **Severity** | Major |
| **Frequency** | Occasional |
| **Detection Difficulty** | Moderate |

## Visual Examples

```
Original HDF5 data (correct)           After failed rechunking

Slice 0: ██████████████████           Slice 0: ██████████████████
Slice 1: ██████████████████           Slice 1: ██████████████████
Slice 2: ██████████████████           Slice 2: ░░░░░░░░░░░░░░░░░░  ← zeros (lost)
Slice 3: ██████████████████           Slice 3: ░░░░░░░░░░░░░░░░░░  ← zeros (lost)
Slice 4: ██████████████████           Slice 4: ██████████████████
...                                   Slice 5: ▓▓▓▓▒▒▒▒████░░░░░░  ← corrupted

Dimension check:                      Dimension mismatch:
  (1800, 2048, 2048) ✓                 (1795, 2048, 2048) ✗  ← 5 slices missing!
```

> **External references:**
> - [HDF5 best practices for large datasets](https://docs.hdfgroup.org/hdf5/develop/_l_b_dsetCreate.html)
> - [Dask array rechunking documentation](https://docs.dask.org/en/stable/array-chunks.html)

## Description

Rechunking and data integrity issues encompass a class of data corruption problems that occur during format conversion, rechunking of HDF5 datasets, or data transfer operations. These manifest as missing slices, zero-filled regions, corrupted values, dimension mismatches, or silently truncated datasets. Unlike instrument-related artifacts, these are purely computational failures that destroy information irreversibly if the original data is not preserved.

## Root Cause

The most common cause is interrupted write operations during HDF5 rechunking — if the process is killed (OOM, walltime exceeded, filesystem quota) mid-write, the output file contains a mix of valid and uninitialized chunks. Dask-based rechunking can fail when the task graph exceeds available memory, causing workers to crash and leaving partial output. Network filesystem (NFS, GPFS) issues can corrupt data during parallel writes. Byte-order (endianness) mismatches during format conversion produce scrambled values. Compression codec mismatches between writer and reader silently produce garbage data. Additionally, integer overflow during dtype conversion (e.g., float64 to uint16 without clipping) causes wrap-around corruption.

## Quick Diagnosis

```python
import numpy as np
import h5py

# Quick integrity check on an HDF5 dataset
with h5py.File("reconstructed_volume.h5", "r") as f:
    dset = f["/exchange/data"]
    print(f"Shape: {dset.shape}, Dtype: {dset.dtype}, Chunks: {dset.chunks}")
    # Check for zero-filled slices (a hallmark of incomplete writes)
    for i in range(0, dset.shape[0], dset.shape[0] // 20):
        slice_data = dset[i]
        if np.all(slice_data == 0) or np.any(np.isnan(slice_data)):
            print(f"WARNING: Slice {i} is zero-filled or contains NaN!")
```

## Detection Methods

### Visual Indicators

- Black (zero-valued) slices interspersed with valid data when scrolling through a volume.
- Sudden intensity jumps or discontinuities at chunk boundaries (e.g., every 64 or 128 slices).
- File size significantly smaller than expected for the dataset dimensions and dtype.
- Scrambled or salt-and-pepper noise patterns inconsistent with detector physics.
- NaN or Inf values in data that should be strictly finite.

### Automated Detection

```python
import numpy as np
import h5py
from pathlib import Path


def verify_hdf5_integrity(filepath, dataset_path="/exchange/data",
                           expected_shape=None, expected_dtype=None,
                           sample_fraction=0.1, checksum_file=None):
    """
    Comprehensive integrity check for HDF5 datasets, detecting common
    rechunking failures and data corruption.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    dataset_path : str
        Internal HDF5 path to dataset.
    expected_shape : tuple or None
        Expected dataset shape. If None, only internal consistency is checked.
    expected_dtype : np.dtype or None
        Expected data type.
    sample_fraction : float
        Fraction of slices to check (0.0 to 1.0).
    checksum_file : str or None
        Path to a .npy file containing per-slice checksums from the original data.

    Returns
    -------
    dict with keys:
        'is_valid' : bool
        'issues' : list of str — descriptions of detected problems
        'slice_report' : dict — per-slice status for sampled slices
        'file_metadata' : dict
    """
    issues = []
    slice_report = {}

    if not Path(filepath).exists():
        return {"is_valid": False, "issues": ["File does not exist"],
                "slice_report": {}, "file_metadata": {}}

    try:
        f = h5py.File(filepath, "r")
    except Exception as e:
        return {"is_valid": False, "issues": [f"Cannot open HDF5 file: {e}"],
                "slice_report": {}, "file_metadata": {}}

    try:
        if dataset_path not in f:
            issues.append(f"Dataset '{dataset_path}' not found in file")
            return {"is_valid": False, "issues": issues,
                    "slice_report": {}, "file_metadata": {}}

        dset = f[dataset_path]
        metadata = {
            "shape": dset.shape,
            "dtype": str(dset.dtype),
            "chunks": dset.chunks,
            "compression": dset.compression,
            "file_size_bytes": Path(filepath).stat().st_size,
        }

        # Shape check
        if expected_shape is not None and dset.shape != expected_shape:
            issues.append(
                f"Shape mismatch: expected {expected_shape}, got {dset.shape}"
            )

        # Dtype check
        if expected_dtype is not None and dset.dtype != np.dtype(expected_dtype):
            issues.append(
                f"Dtype mismatch: expected {expected_dtype}, got {dset.dtype}"
            )

        # File size sanity check
        expected_bytes = np.prod(dset.shape) * dset.dtype.itemsize
        actual_bytes = metadata["file_size_bytes"]
        if dset.compression is None and actual_bytes < expected_bytes * 0.9:
            issues.append(
                f"File too small: {actual_bytes} bytes vs "
                f"{expected_bytes} expected (uncompressed)"
            )

        # Sample slices for content checks
        num_slices = dset.shape[0]
        num_samples = max(1, int(num_slices * sample_fraction))
        sample_indices = np.linspace(0, num_slices - 1, num_samples, dtype=int)

        # Load reference checksums if available
        ref_checksums = None
        if checksum_file is not None and Path(checksum_file).exists():
            ref_checksums = np.load(checksum_file)

        zero_slices = []
        nan_slices = []
        inf_slices = []
        checksum_mismatches = []

        for idx in sample_indices:
            idx = int(idx)
            try:
                slice_data = dset[idx]
            except Exception as e:
                slice_report[idx] = f"READ ERROR: {e}"
                issues.append(f"Cannot read slice {idx}: {e}")
                continue

            status = "ok"

            # Check for all-zero slices
            if np.all(slice_data == 0):
                zero_slices.append(idx)
                status = "ZERO-FILLED"

            # Check for NaN
            if np.issubdtype(slice_data.dtype, np.floating):
                nan_count = np.sum(np.isnan(slice_data))
                inf_count = np.sum(np.isinf(slice_data))
                if nan_count > 0:
                    nan_slices.append(idx)
                    status = f"NaN ({nan_count} pixels)"
                if inf_count > 0:
                    inf_slices.append(idx)
                    status = f"Inf ({inf_count} pixels)"

            # Check for value range anomalies
            if slice_data.size > 0 and not np.all(slice_data == 0):
                slice_min, slice_max = float(np.min(slice_data)), float(np.max(slice_data))
                if np.issubdtype(slice_data.dtype, np.integer):
                    info = np.iinfo(slice_data.dtype)
                    if slice_min == info.min or slice_max == info.max:
                        status = "CLIPPED/OVERFLOW"
                        issues.append(f"Slice {idx}: values at dtype limits (overflow?)")

            # Checksum verification
            if ref_checksums is not None and idx < len(ref_checksums):
                current_checksum = np.sum(slice_data.astype(np.float64))
                if not np.isclose(current_checksum, ref_checksums[idx], rtol=1e-6):
                    checksum_mismatches.append(idx)
                    status = "CHECKSUM MISMATCH"

            slice_report[idx] = status

        if zero_slices:
            issues.append(
                f"Zero-filled slices detected: {zero_slices} "
                f"({len(zero_slices)}/{num_samples} sampled)"
            )
        if nan_slices:
            issues.append(f"NaN values in slices: {nan_slices}")
        if inf_slices:
            issues.append(f"Inf values in slices: {inf_slices}")
        if checksum_mismatches:
            issues.append(f"Checksum mismatches at slices: {checksum_mismatches}")

    finally:
        f.close()

    return {
        "is_valid": len(issues) == 0,
        "issues": issues,
        "slice_report": slice_report,
        "file_metadata": metadata,
    }
```

## Solutions and Mitigation

### Prevention (Before Data Collection)

- Always preserve original raw data until the processed output has been fully verified.
- Use atomic write patterns: write to a temporary file, verify, then rename to the final path.
- Set conservative memory limits for Dask workers (50-70% of available RAM) to avoid OOM kills.
- Use checksums (CRC32, SHA256) computed per-slice on the source data for post-conversion verification.
- Pre-allocate the full output file before writing to catch disk space issues early.

### Correction — Traditional Methods

Systematic verification and repair workflow for rechunked HDF5 data.

```python
import numpy as np
import h5py
import hashlib
from pathlib import Path


def safe_rechunk_hdf5(source_path, dest_path, dataset_path="/exchange/data",
                       new_chunks=None, compression="gzip", compression_opts=4):
    """
    Safely rechunk an HDF5 dataset with integrity verification.
    Writes to a temporary file, verifies slice-by-slice, then
    atomically renames to the destination.

    Parameters
    ----------
    source_path : str
        Path to source HDF5 file.
    dest_path : str
        Path for rechunked output.
    dataset_path : str
        Internal HDF5 dataset path.
    new_chunks : tuple or None
        Target chunk shape. If None, auto-determine for slice access.
    compression : str
        HDF5 compression filter.
    compression_opts : int
        Compression level.
    """
    tmp_path = dest_path + ".tmp"

    with h5py.File(source_path, "r") as src:
        dset_src = src[dataset_path]
        shape = dset_src.shape
        dtype = dset_src.dtype

        if new_chunks is None:
            # Optimize for axial slice access: full slices as chunks
            new_chunks = (1, shape[1], shape[2])

        # Pre-compute source checksums
        print("Computing source checksums...")
        src_checksums = np.zeros(shape[0], dtype=np.float64)
        for i in range(shape[0]):
            src_checksums[i] = np.sum(dset_src[i].astype(np.float64))

        # Write rechunked data
        print(f"Rechunking {shape} with chunks={new_chunks}...")
        with h5py.File(tmp_path, "w") as dst:
            dset_dst = dst.create_dataset(
                dataset_path, shape=shape, dtype=dtype,
                chunks=new_chunks, compression=compression,
                compression_opts=compression_opts,
            )

            # Copy slice by slice
            for i in range(shape[0]):
                dset_dst[i] = dset_src[i]
                if i % 100 == 0:
                    print(f"  Written slice {i}/{shape[0]}")

    # Verification pass
    print("Verifying rechunked data...")
    errors = []
    with h5py.File(tmp_path, "r") as dst:
        dset_dst = dst[dataset_path]

        if dset_dst.shape != shape:
            raise RuntimeError(
                f"Shape mismatch: source {shape} vs dest {dset_dst.shape}"
            )

        for i in range(shape[0]):
            dst_checksum = np.sum(dset_dst[i].astype(np.float64))
            if not np.isclose(dst_checksum, src_checksums[i], rtol=1e-6):
                errors.append(i)

    if errors:
        Path(tmp_path).unlink()  # remove corrupt file
        raise RuntimeError(
            f"Verification failed for {len(errors)} slices: {errors[:20]}..."
        )

    # Atomic rename
    Path(tmp_path).rename(dest_path)
    print(f"Rechunking complete and verified: {dest_path}")


def generate_slice_checksums(filepath, dataset_path="/exchange/data",
                              output_path=None):
    """
    Generate per-slice checksums for an HDF5 dataset.
    Save alongside the data for future integrity verification.

    Parameters
    ----------
    filepath : str
        Path to HDF5 file.
    dataset_path : str
        Internal dataset path.
    output_path : str or None
        Where to save checksums. Defaults to filepath + '.checksums.npy'.

    Returns
    -------
    checksums : np.ndarray — float64 array of per-slice sums
    """
    if output_path is None:
        output_path = filepath + ".checksums.npy"

    with h5py.File(filepath, "r") as f:
        dset = f[dataset_path]
        num_slices = dset.shape[0]
        checksums = np.zeros(num_slices, dtype=np.float64)

        for i in range(num_slices):
            checksums[i] = np.sum(dset[i].astype(np.float64))
            if i % 200 == 0:
                print(f"Checksum slice {i}/{num_slices}")

    np.save(output_path, checksums)
    print(f"Checksums saved to {output_path}")
    return checksums
```

### Correction — AI/ML Methods

Data integrity issues are not well-suited to AI/ML correction since the original information is genuinely lost. However, ML-based anomaly detection can identify corrupted slices in large datasets more efficiently than exhaustive checking. A lightweight autoencoder trained on valid slices will produce high reconstruction error on corrupted or zero-filled slices, enabling rapid triage of multi-terabyte volumes.

## Impact If Uncorrected

Corrupted data produces scientifically meaningless results that may not be obviously wrong. Zero-filled slices in a tomographic volume create planar voids in 3D renderings that are misinterpreted as real features. Missing slices shift the z-coordinate of all subsequent data, corrupting spatial registration with other modalities. Byte-swapped values produce physically implausible intensities that can crash downstream processing or, worse, produce plausible-looking but quantitatively incorrect results. In multi-user facilities where data is processed months after acquisition, the original data may have been purged from fast storage, making corruption irreversible.

## Related Resources

- [Tomography EDA notebook](../../06_data_structures/eda/tomo_eda.md) — data inspection and quality checks
- [HDF5 Deep Dive](../../06_data_structures/hdf5_deep_dive.md) — chunking and compression strategies
- [Data Pipeline — Processing](../../07_data_pipeline/processing.md) — safe rechunking patterns
- [HDF5 Group best practices](https://docs.hdfgroup.org/hdf5/develop/_l_b_dsetCreate.html)

## Key Takeaway

Data integrity issues during rechunking and format conversion are silent killers — always compute and verify checksums before and after any data transformation, preserve original raw data until verification is complete, and use atomic write operations to prevent partial-write corruption.
