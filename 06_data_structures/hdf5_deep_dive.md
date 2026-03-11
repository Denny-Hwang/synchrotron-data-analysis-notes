# HDF5 Deep Dive: Internals, Performance, and Limitations

## Overview

This document goes beyond the basics of HDF5 covered in the
[HDF5 README](hdf5_structure/README.md) to examine the internal mechanisms,
performance tuning strategies, and known limitations that are critical for
handling APS-U scale data (5--20 GB/s per beamline). Understanding these
details is essential for designing data pipelines that can keep pace with
next-generation detectors.

## HDF5 Internal Architecture

### File Structure on Disk

An HDF5 file is not simply a flat byte stream. Internally it consists of:

| Component | Role |
|-----------|------|
| **Superblock** | File signature, version, root group address; located at file offset 0 |
| **B-tree** | Indexes group children and chunked dataset storage |
| **Object headers** | Store metadata (dataspace, datatype, filters, attributes) |
| **Data chunks** | The actual array data, stored as independently accessible units |
| **Free-space manager** | Tracks unused regions for reuse after deletions |

The B-tree indexing means that random access to any chunk is O(log N), not
O(N). HDF5 1.10+ introduced additional chunk index types:

| Index Type | Complexity | Use Case |
|-----------|-----------|----------|
| B-tree v2 | O(log N) | Default for extensible datasets |
| Single chunk | O(1) | Dataset fits in one chunk |
| Fixed array | O(1) | Fixed-size dataset, known dimensions |
| Extensible array | O(1) amortized | One unlimited dimension (most common at APS) |

### Virtual Datasets (VDS)

Virtual Datasets allow a single logical dataset to span multiple physical
HDF5 files without copying data. This is critical at APS where detectors
like EIGER write data across multiple files:

```
# EIGER writes multiple data files:
scan_001_master.h5      # VDS mapping + metadata
scan_001_data_000001.h5 # frames 1-1000
scan_001_data_000002.h5 # frames 1001-2000
...
```

```python
import h5py

# Reading is transparent via the master file
with h5py.File("scan_001_master.h5", "r") as f:
    frame_500 = f["/entry/data/data"][500]  # auto-resolves to data_000001.h5
```

VDS benefits:
- No data duplication
- Individual files stay below filesystem size limits
- Parallel write to separate files, unified read via master

## SWMR (Single Writer Multiple Reader)

### What It Is

SWMR mode, introduced in HDF5 1.10, allows **one process to write** data
while **multiple processes read** concurrently -- without closing/reopening
the file. This is the foundation of live data streaming at APS.

### How It Works

```
Writer Process                    Reader Process(es)
─────────────────                ──────────────────────
Open file (SWMR write mode)      Open file (SWMR read mode)
Create datasets (before SWMR)
Enable SWMR mode
  │                              │
  ├─ Append chunk ───────────────┤─ H5Drefresh()
  ├─ H5Dflush()                  ├─ Read new data
  ├─ Append chunk ───────────────┤─ H5Drefresh()
  ├─ H5Dflush()                  ├─ Read new data
  │                              │
Close file                       Close file
```

### Practical Usage at APS

```python
import h5py
import numpy as np

# Writer (areaDetector IOC or acquisition script)
f = h5py.File("live_scan.h5", "w", libver="latest")
dset = f.create_dataset("data", shape=(0, 2048, 2048),
                        maxshape=(None, 2048, 2048),
                        chunks=(1, 2048, 2048),
                        dtype="uint16")
f.swmr_mode = True  # Enable SWMR -- no new datasets after this

for frame in detector_stream():
    dset.resize(dset.shape[0] + 1, axis=0)
    dset[-1] = frame
    dset.flush()

f.close()
```

```python
# Reader (monitoring / real-time analysis)
f = h5py.File("live_scan.h5", "r", libver="latest", swmr=True)
dset = f["data"]

while True:
    dset.refresh()
    n_frames = dset.shape[0]
    if n_frames > last_seen:
        new_data = dset[last_seen:n_frames]
        process(new_data)
        last_seen = n_frames
```

### SWMR Limitations

| Limitation | Impact | Workaround |
|-----------|--------|-----------|
| No new datasets after SWMR enabled | Must pre-create all datasets | Create placeholder datasets with maxshape=None |
| No new attributes after SWMR | Cannot add metadata during scan | Write metadata to separate file or after SWMR close |
| No group creation in SWMR | Cannot add hierarchy | Pre-create full group tree |
| Single writer only | No parallel writes | Use separate files + VDS for multi-writer |
| Flush latency ~1-10 ms | Not truly zero-latency | Acceptable for frame rates < 10 kHz |

## Chunking Strategy Deep Dive

### Why Chunking Matters

At APS-U data rates, chunking is the **single most impactful** HDF5
configuration choice. The wrong chunk shape can cause 10-100x performance
degradation.

### Access Pattern Alignment

For a tomography dataset with shape `(n_proj, n_row, n_col)`:

| Access Pattern | Optimal Chunk Shape | Use Case |
|---------------|-------------------|----------|
| Projection-by-projection | `(1, n_row, n_col)` | Acquisition, flat-field correction |
| Sinogram (row) | `(n_proj, 1, n_col)` | Tomographic reconstruction |
| Sub-volume | `(16, 256, 256)` | 3D rendering, segmentation |

**The fundamental conflict**: Acquisition writes projection-by-projection, but
reconstruction reads sinogram-by-sinogram. No single chunk shape is optimal
for both.

### APS Solution: Two-Pass Strategy

```
Phase 1: Acquisition
  └─ Write with chunks (1, 2048, 2048) → projection-optimized
  └─ ~10 GB/s write throughput

Phase 2: Rechunking (on ALCF)
  └─ Rechunk to (1800, 1, 2048) → sinogram-optimized
  └─ Enables fast TomocuPy reconstruction
  └─ Or use Zarr intermediate (see data_formats_comparison.md)
```

### Chunk Size Guidelines

| Chunk Size | Read Overhead | Cache Efficiency | APS Recommendation |
|-----------|--------------|-----------------|-------------------|
| < 10 KB | Very high (B-tree lookup dominates) | Poor | Never use |
| 10 KB -- 100 KB | High | Moderate | Only for 1D spectra |
| 100 KB -- 1 MB | Low | Good | General purpose |
| 1 MB -- 10 MB | Minimal | Very good | Recommended for images |
| 10 MB -- 100 MB | Minimal | Excellent for sequential | Large frames, bulk reads |
| > 100 MB | Minimal | Wastes memory on partial reads | Avoid |

**APS default**: A single detector frame as one chunk, typically 2048x2048x2 bytes = 8 MB.

## Compression Performance at APS Scale

### Benchmark: Tomography Raw Data (uint16, 2048x2048)

| Filter | Ratio | Compress (MB/s) | Decompress (MB/s) | CPU Cores |
|--------|-------|-----------------|-------------------|-----------|
| None | 1.0x | N/A | N/A | 0 |
| LZ4 | 1.8--2.5x | 3500 | 4200 | 1 |
| bitshuffle+LZ4 | 3.5--6.0x | 2800 | 3200 | 1 |
| Blosc (LZ4, shuffle) | 3.0--5.0x | 3000 | 3800 | multi |
| Blosc (Zstd, shuffle) | 4.0--7.0x | 800 | 2500 | multi |
| gzip level 1 | 2.5--3.5x | 250 | 400 | 1 |
| gzip level 6 | 3.5--5.0x | 80 | 400 | 1 |
| SZ (lossy, 1e-3) | 15--40x | 600 | 900 | 1 |
| ZFP (lossy, rate 8) | 10--20x | 1200 | 1400 | 1 |

### Recommendation by Pipeline Stage

| Stage | Filter | Rationale |
|-------|--------|-----------|
| **Acquisition (real-time)** | bitshuffle+LZ4 | Fastest lossless; keeps up with detector |
| **Transfer (Globus)** | Keep as-is | Re-compressing wastes time |
| **Processing output** | Blosc-Zstd | Best ratio for archival without lossy |
| **Archive (HPSS)** | Blosc-Zstd or SZ | Storage cost dominates; lossy acceptable for raw |
| **Publication dataset** | gzip level 4 | Maximum compatibility |

### Direct Chunk Write

For detectors that compress data internally (e.g., Dectris EIGER with
bitshuffle+LZ4 in firmware), HDF5's `H5DOwrite_chunk` API bypasses the
entire filter pipeline:

```
Standard write path:
  data → HDF5 filter pipeline (compress) → disk
  Throughput: ~2 GB/s

Direct chunk write path:
  pre-compressed data → disk (bypass filters)
  Throughput: ~8 GB/s (disk-limited)
```

The EIGER areaDetector driver uses this path to sustain full detector data
rates.

## Parallel HDF5 (MPI-IO)

### Architecture

Parallel HDF5 uses MPI-IO to allow multiple processes to read/write the
same file simultaneously. This is critical for ALCF processing where
hundreds of GPU nodes process a single tomographic volume.

```
Node 0: read sinograms 0-99     ─┐
Node 1: read sinograms 100-199  ─┤── Single HDF5 file
Node 2: read sinograms 200-299  ─┤   via MPI-IO
Node 3: read sinograms 300-399  ─┘
```

### Configuration at ALCF

```python
from mpi4py import MPI
import h5py

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Collective open
f = h5py.File("volume.h5", "r", driver="mpio", comm=comm)
dset = f["data"]

# Each rank reads its slice
my_start = rank * chunk_size
my_data = dset[my_start:my_start + chunk_size]
```

### Parallel HDF5 Limitations

| Issue | Description |
|-------|-------------|
| **No compression with parallel writes** | Compressed chunks cannot be written in parallel (lock contention). Workaround: write uncompressed, compress in post-processing |
| **Metadata operations are serialized** | Creating datasets/attributes is serialized across all ranks. Solution: one rank creates, others wait at barrier |
| **POSIX lock overhead on Lustre** | File locking can degrade performance. Solution: use `HDF5_USE_FILE_LOCKING=FALSE` on ALCF |
| **Chunk alignment** | Chunks should align with Lustre stripe boundaries (typically 1 MB) |
| **Collective vs. independent I/O** | Collective I/O (coordinated) is much faster but requires synchronization |

### Performance on ALCF (Polaris/Aurora)

| Config | Read Throughput | Write Throughput |
|--------|----------------|-----------------|
| Single process | 2--5 GB/s | 1--3 GB/s |
| 32 MPI ranks, collective | 20--40 GB/s | 10--25 GB/s |
| 256 MPI ranks, collective | 50--100 GB/s | 30--60 GB/s |
| Eagle filesystem peak | 650 GB/s aggregate | 650 GB/s aggregate |

## Known Limitations of HDF5

### 1. Single-File Lock Contention

HDF5 uses POSIX file locks by default. On shared filesystems (GPFS, Lustre),
this causes contention when multiple processes access the same file. While
parallel HDF5 solves this for coordinated access, ad-hoc concurrent reads
can still suffer.

### 2. Cloud / Object Store Access

HDF5 was designed for POSIX filesystems. Accessing HDF5 on cloud object
stores (S3, Azure Blob) is problematic:

| Approach | Latency | Usability |
|----------|---------|-----------|
| Download entire file | Minutes for large files | Impractical |
| HSDS (HDF5 cloud service) | 10--100 ms per request | Requires server deployment |
| kerchunk (virtual Zarr view) | First scan slow, then fast | Read-only, requires index |
| ros3 driver (S3 native) | 50--200 ms per chunk | Limited functionality |

**Why this matters**: As APS data moves toward hybrid cloud/HPC strategies,
HDF5's POSIX dependency becomes a significant barrier compared to
cloud-native formats like Zarr.

### 3. Metadata Scalability

HDF5 stores all metadata (attributes, dataset headers) in the same file as
data. For files with millions of small datasets, metadata operations become
the bottleneck:

| File Complexity | Metadata Open Time | Practical Impact |
|----------------|-------------------|-----------------|
| 10 datasets | < 1 ms | None |
| 1,000 datasets | 10--50 ms | Acceptable |
| 100,000 datasets | 1--10 s | Problematic |
| 1,000,000 datasets | 30--120 s | Unusable for interactive |

This mainly affects XRF mapping files with per-pixel spectral datasets
(rare) and files with extensive processing provenance chains.

### 4. No Native Append Across Files

Unlike Zarr, HDF5 has no built-in concept of a "dataset spanning multiple
files" at the format level. VDS provides this but requires explicit
configuration. If the master file is lost, the relationship between data
files is lost.

### 5. Python GIL Constraint (h5py)

The h5py library is effectively **single-threaded** due to Python's GIL.
Multi-threaded reads from the same HDF5 file do not parallelize. This is
a significant limitation for Python-based analysis pipelines:

| Approach | Throughput | Notes |
|----------|-----------|-------|
| Single-thread h5py | 2--5 GB/s | GIL-limited |
| Multi-thread h5py | 2--5 GB/s (no gain) | GIL prevents parallel reads |
| Multi-process h5py | 10--30 GB/s | Each process opens file independently |
| MPI-parallel (mpi4py + h5py) | 50--100 GB/s | Best performance on ALCF |
| Zarr + Dask (alternative) | 30--80 GB/s | No GIL limitation |

### 6. Cannot Truly Delete Data

Deleting a dataset from an HDF5 file only removes the link -- the file
size does **not shrink**. The on-disk space is marked as free but not
reclaimed. Requires `h5repack` to compact:

```bash
# Repack a 120 GB file to reclaim space -- produces a full copy
h5repack -f GZIP=4 input.h5 output.h5
# For TB-scale files, this is extremely expensive
```

### 7. Schema Evolution

Modifying the structure of an existing HDF5 file (adding new groups,
renaming datasets) requires rewriting the file. There is no in-place
schema migration. This complicates long-running experiments where
requirements change mid-campaign.

### 8. VOL (Virtual Object Layer) -- Potential Future Fix

HDF5 1.12+ introduced the **Virtual Object Layer (VOL)**, which allows
the HDF5 API to interface with arbitrary storage backends via plugins.
In theory, this could enable transparent cloud object store access
without application changes. However, VOL connectors for S3 and other
cloud backends are not yet production-ready.

## Best Practices for APS-U Scale

### Acquisition Phase

1. **Pre-allocate datasets** with known maximum size when possible
2. **Use direct chunk write** for detectors with built-in compression
3. **Enable SWMR** for live monitoring
4. **One frame per chunk** for projection-oriented access
5. **Separate metadata file** from bulk data for EIGER-style multi-file

### Processing Phase

1. **Rechunk for access pattern** using Zarr as intermediate
2. **Use parallel HDF5** on ALCF with collective I/O
3. **Disable file locking** on Lustre (`HDF5_USE_FILE_LOCKING=FALSE`)
4. **Align chunks to stripe size** (1 MB on Lustre)

### Archival Phase

1. **Repack with gzip** for maximum compatibility
2. **Include all metadata** in a self-contained NeXus file
3. **Validate with cnxvalidate** before deposit
4. **Use checksums** (SHA-256) for integrity verification

## Related Documents

- [HDF5 Structure Overview](hdf5_structure/README.md) -- Basic HDF5 concepts
- [Data Formats Comparison](data_formats_comparison.md) -- HDF5 vs. Zarr vs. TIFF
- [Data Scale Analysis](data_scale_analysis.md) -- APS-U data volume projections
- [APS-U Data Challenges](data_challenges_apsu.md) -- Infrastructure and open problems
