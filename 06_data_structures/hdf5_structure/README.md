# HDF5 Data Structure for Synchrotron Science

## What is HDF5?

**Hierarchical Data Format version 5 (HDF5)** is an open-source file format and library
designed for storing and organizing large amounts of scientific data. Developed by The HDF
Group, it has become the universal standard for synchrotron data storage at facilities
worldwide including APS, NSLS-II, ESRF, Diamond Light Source, and SPring-8.

An HDF5 file is a self-contained container that combines:
- **N-dimensional arrays** of arbitrary size and data type
- **Hierarchical organization** using a filesystem-like tree
- **Rich metadata** attached at every level of the hierarchy
- **Efficient compression** with pluggable codec support

## Core Concepts

### Groups

Groups are the HDF5 equivalent of directories. They form the tree structure that organizes
data logically. Every HDF5 file has a root group `/`.

```
/                           # Root group
/entry/                     # Experiment entry
/entry/instrument/          # Instrument configuration
/entry/data/                # Measurement data
/entry/sample/              # Sample metadata
```

### Datasets

Datasets hold the actual numerical arrays. Key properties include:

| Property | Description |
|----------|-------------|
| Shape | N-dimensional array dimensions (e.g., `(1800, 2048, 2048)`) |
| Dtype | Data type (`float32`, `uint16`, `complex64`, etc.) |
| Chunks | Subdivision for efficient partial I/O (e.g., `(1, 2048, 2048)`) |
| Compression | Codec and level (e.g., `gzip level 4`, `lz4`) |
| Fill value | Default for unwritten elements |

### Attributes

Attributes are small metadata items attached to groups or datasets. They store:
- Experimental parameters (energy, exposure time, pixel size)
- Provenance information (software version, processing history)
- Units and calibration factors
- Timestamps and user annotations

```python
# Example: reading attributes
import h5py
with h5py.File("scan.h5", "r") as f:
    energy = f["/entry/instrument/monochromator"].attrs["energy"]  # keV
    pixel_size = f["/entry/instrument/detector"].attrs["x_pixel_size"]  # meters
```

## Compression Options

HDF5 supports multiple compression filters, each with trade-offs:

| Filter | Ratio | Speed | Best For |
|--------|-------|-------|----------|
| gzip (level 4) | 3--5x | Moderate | General purpose, archival |
| LZ4 | 2--3x | Very fast | Real-time streaming |
| Blosc | 3--6x | Fast | In-memory computation |
| bitshuffle + LZ4 | 4--8x | Fast | Detector data with low-bit patterns |
| SZ / ZFP | 10--50x | Fast | Lossy, floating-point arrays |

At APS, the default pipeline uses **bitshuffle + LZ4** for raw detector data and
**gzip level 4** for processed results.

## Why Synchrotrons Use HDF5

1. **Scale** -- A single tomography scan at APS-U can produce 50+ GB of raw data.
   HDF5 handles this natively with chunked I/O and compression.
2. **Self-describing** -- All metadata travels with the data, eliminating the need for
   separate parameter files that can become separated or lost.
3. **Parallel I/O** -- HDF5 supports MPI-parallel reads and writes, critical for
   processing on ALCF (Argonne Leadership Computing Facility) supercomputers.
4. **Streaming** -- Chunked datasets can be appended during acquisition, enabling
   real-time monitoring via tools like Bluesky and databroker.
5. **Ecosystem** -- Virtually every synchrotron analysis package (TomoPy, MAPS, pyFAI,
   Mantid, Dials) reads and writes HDF5.

## NeXus Convention

**NeXus** is an international standard that defines naming conventions and organizational
rules on top of HDF5. It specifies:

- **NXentry** -- Top-level experiment container
- **NXinstrument** -- Source, monochromator, detector descriptions
- **NXsample** -- Sample name, composition, environment
- **NXdata** -- Default plottable signal with axes
- **NXprocess** -- Processing provenance chain

```
/entry:NXentry
  /instrument:NXinstrument
    /source:NXsource
      @name = "APS"
      @energy = 6.0   # GeV
    /detector:NXdetector
      data [2048, 2048] uint16
      @x_pixel_size = 6.5e-6  # meters
  /sample:NXsample
    @name = "soil_core_27B"
  /data:NXdata
    @signal = "data"
    @axes = ["y", "x"]
```

NeXus files use the `.nxs` or `.nx5` extension and are fully valid HDF5 files readable
by any HDF5 tool.

## Modality-Specific Schemas

Each synchrotron technique has established conventions for organizing data within HDF5:

| Modality | Convention | Schema File |
|----------|-----------|-------------|
| X-ray Fluorescence (XRF) | MAPS output | [xrf_hdf5_schema.md](xrf_hdf5_schema.md) |
| Tomography | Data Exchange | [tomo_hdf5_schema.md](tomo_hdf5_schema.md) |
| Ptychography | CXI format | [ptychography_hdf5_schema.md](ptychography_hdf5_schema.md) |

## Python Tools for HDF5

| Package | Purpose |
|---------|---------|
| **h5py** | Low-level Pythonic interface to HDF5 |
| **h5pyd** | h5py-compatible interface for HSDS (cloud HDF5) |
| **nexusformat** | NeXus-aware reading and writing |
| **silx** | HDF5 viewer with GUI tree browser |
| **dxchange** | Read/write Data Exchange tomography files |
| **hdf5plugin** | Additional compression filters (Blosc, bitshuffle, SZ) |

## Hands-On Notebooks

| Notebook | Description |
|----------|-------------|
| [01_hdf5_exploration.ipynb](notebooks/01_hdf5_exploration.ipynb) | Navigate HDF5 files with h5py |
| [02_data_visualization.ipynb](notebooks/02_data_visualization.ipynb) | Visualize multi-channel synchrotron data |

## Best Practices

1. **Use chunking aligned to access patterns** -- For tomography, chunk by projection
   `(1, nrow, ncol)` for reconstruction or `(nproj, 1, ncol)` for sinogram access.
2. **Store units as attributes** -- Always attach `units` attributes to datasets.
3. **Include processing provenance** -- Record software versions, parameters, and timestamps.
4. **Use relative links** -- HDF5 soft links can connect related datasets without duplication.
5. **Validate against NeXus** -- Use `cnxvalidate` or `punx` to check NeXus compliance.
