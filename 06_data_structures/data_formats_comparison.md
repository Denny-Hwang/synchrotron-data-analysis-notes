# Data Formats at ANL: HDF5, Zarr, and Beyond

## Overview

The APS ecosystem uses a diverse set of data formats at different pipeline
stages. This document provides a detailed comparison of formats, explains
when and why each is used at ANL, and covers the critical role of frameworks
like areaDetector, Databroker, and Tiled in managing data flow.

## Format Comparison Matrix

| Feature | HDF5 | Zarr | TIFF Stack | CXI (HDF5) | NeXus (HDF5) | CBF |
|---------|------|------|-----------|-------------|--------------|-----|
| **Self-describing** | Yes | Yes | No | Yes | Yes | Partial |
| **N-dimensional** | Yes | Yes | 2D only | Yes | Yes | 2D |
| **Compression** | Pluggable | Pluggable | LZW/None | Same as HDF5 | Same as HDF5 | CBF packed |
| **Chunking** | Yes | Yes (mandatory) | Per-file | Yes | Yes | No |
| **Parallel write** | MPI-IO | Native (files) | Files | MPI-IO | MPI-IO | No |
| **Cloud-native** | Poor | Excellent | Poor | Poor | Poor | No |
| **Streaming (SWMR)** | Yes | No | N/A | Yes | Yes | No |
| **Metadata** | Attributes | JSON sidecar | None/header | Attributes | Standardized | Mini-header |
| **Max file size** | Exabytes | Unlimited (chunks=files) | 4 GB (classic) | Exabytes | Exabytes | 4 GB |
| **Ecosystem support** | Universal | Growing | Universal | Ptychography | Synchrotron std | Crystallography |

## HDF5 at ANL

### Role

HDF5 is the **primary storage format** at APS for both raw data and
processed results. It is used at every stage from acquisition through
archival.

### Where HDF5 Is Used

| Stage | Format Variant | Example |
|-------|---------------|---------|
| Detector output | EIGER master + data files | `scan_001_master.h5` + `data_000001.h5` |
| areaDetector NDArray | HDF5 with NeXus-like layout | `det_image_0001.h5` |
| Processed volumes | Data Exchange convention | `/exchange/data [float32]` |
| Archive | NeXus/HDF5 | Full experiment metadata + data |
| Publication | NeXus/HDF5 with DOI | Self-contained reproducible dataset |

### Strengths at APS Scale

- **SWMR** enables live monitoring during acquisition
- **Direct chunk write** sustains detector data rates (8+ GB/s)
- **VDS** unifies multi-file EIGER output into single logical dataset
- **Parallel HDF5** on ALCF for distributed reconstruction
- **Universal tool support** (h5py, TomoPy, MAPS, pyFAI, etc.)

### Weaknesses

- **POSIX-dependent**: Poor performance on cloud object stores
- **Single-file bottleneck**: All metadata and data in one file
- **No parallel compressed writes**: Must write uncompressed for parallel I/O
- **Rechunking cost**: Changing access patterns requires full rewrite

## Zarr

### What Is Zarr?

Zarr is a format for chunked, compressed, N-dimensional arrays designed for
cloud and distributed computing. Each chunk is stored as a separate file
(or object store key), with metadata in JSON sidecar files.

### Directory Structure

```
volume.zarr/
├── .zarray          # {"shape": [1800,2048,2048], "chunks": [1,2048,2048], ...}
├── .zattrs          # {"units": "counts", "pixel_size": 6.5e-6}
├── 0.0.0            # chunk (0,0,0) -- binary data
├── 0.0.1            # chunk (0,0,1)
├── 1.0.0            # chunk (1,0,0)
└── ...
```

### Where Zarr Is Used at ANL

| Use Case | Why Zarr? |
|----------|-----------|
| **TomocuPy output** | GPU reconstruction writes Zarr for fast sinogram access |
| **Intermediate processing** | Rechunking from projection to sinogram layout |
| **Dask workflows** | Native Dask array integration for parallel compute |
| **Cloud staging** | When moving data to/from cloud for ML training |

### Zarr vs. HDF5 Performance

#### Sequential Read (Single Process)

| Operation | HDF5 (gzip) | HDF5 (LZ4) | Zarr (Blosc-Zstd) | Zarr (LZ4) |
|-----------|------------|------------|-------------------|------------|
| Read full volume (100 GB) | 45 s | 28 s | 35 s | 25 s |
| Read single slice | 15 ms | 8 ms | 12 ms | 7 ms |
| Random chunk access | 5 ms | 3 ms | 8 ms | 5 ms |

#### Parallel Read (32 Processes)

| Operation | HDF5 (MPI-IO) | Zarr (filesystem) |
|-----------|--------------|-------------------|
| Read full volume | 4 s | 2.5 s |
| Scaling efficiency | 70% | 95% |
| Lock contention | Moderate | None |

Zarr's per-chunk-as-file design eliminates lock contention entirely, giving
near-linear scaling on parallel filesystems.

#### Cloud Access (S3-Compatible)

| Operation | HDF5 (ros3) | Zarr (s3fs) |
|-----------|------------|-------------|
| Open dataset | 2--5 s | 50 ms |
| Read single chunk | 200 ms | 80 ms |
| Random access pattern | Very slow | Fast |
| Partial reads | Requires full chunk | Native |

### Zarr Limitations

| Limitation | Impact at APS |
|-----------|--------------|
| No SWMR equivalent | Cannot stream live data during acquisition |
| No virtual datasets | Cannot unify multi-file detector output natively |
| Metadata in separate JSON | Risk of metadata/data separation |
| No NeXus standard for Zarr | No community schema for synchrotron Zarr files |
| Immature ecosystem | Fewer analysis tools support Zarr natively |
| Many small files | Millions of chunks stress filesystem metadata |

### Zarr v3 (Emerging)

Zarr v3 introduces:
- **Sharding**: Multiple chunks in one file (reduces small-file problem)
- **Codecs pipeline**: Composable compression chain
- **Extension points**: Custom data types and stores
- **Consolidated metadata**: Single metadata read for entire hierarchy

## TIFF Stacks

### Legacy Role

TIFF stacks were the original format for tomographic projections at APS.
Each projection is a separate `.tif` file in a numbered sequence.

```
scan_001/
├── proj_0001.tif
├── proj_0002.tif
├── ...
├── proj_1800.tif
├── flat_001.tif
├── dark_001.tif
└── scan_log.txt   # separate metadata file
```

### Why TIFF Is Being Replaced

| Issue | Impact |
|-------|--------|
| No metadata in file | Parameter files get separated/lost |
| Filesystem stress | 10,000+ files per scan overwhelm metadata operations |
| No compression options | Only LZW/ZIP (slow) or none |
| No chunking | Full image must be loaded |
| 4 GB limit (Classic TIFF) | Exceeded by single EIGER frame set |

### Where TIFF Persists

- Legacy analysis codes that only read TIFF
- Quick visual inspection (any image viewer opens TIFF)
- Interchange format with external collaborators
- Ptychography reconstructed phase images for publication figures

## Vendor-Specific Formats at APS

### Dectris Detectors (EIGER, Pilatus, Mythen)

| Detector | Native Format | APS Output | Notes |
|----------|--------------|-----------|-------|
| EIGER2 X 9M | HDF5 (master + data) | HDF5 via areaDetector | bitshuffle+LZ4 in firmware |
| Pilatus 100K/300K | CBF (crystallography) | HDF5 or TIFF via areaDetector | 32-bit count images |
| Mythen2 1K | Raw binary | HDF5 via EPICS | 1D strip detector |

**EIGER data flow**:
```
EIGER firmware
  ├─ FileWriter → HDF5 master + data files (detector disk)
  ├─ Stream → ZMQ multipart messages (bitshuffle+LZ4)
  └─ Monitor → TIFF images (low-rate preview)
        │
        ▼
areaDetector (ADEiger driver)
  ├─ Decodes ZMQ stream or reads FileWriter output
  ├─ Produces NDArray objects in EPICS pipeline
  └─ Writes HDF5 via NDFileHDF5 plugin
```

### Other Detectors

| Detector | Type | Native Format | APS Conversion |
|----------|------|--------------|---------------|
| PCO.edge 5.5 | CMOS camera | TIFF/Raw | HDF5 via areaDetector |
| Oryx 10GigE | CMOS camera | Raw frames | HDF5 via areaDetector |
| Vortex ME4 | XRF SDD | MCA spectra | HDF5 via MAPS |
| Lambda 2M | GaAs hybrid pixel | Raw binary | HDF5 via areaDetector |

## The areaDetector Framework

### Architecture

areaDetector is an EPICS-based framework that provides a unified interface
to dozens of detector types. It handles acquisition, processing, and file
writing through a plugin chain.

```
Detector Driver (e.g., ADEiger)
    │
    ├─ NDPluginStats     → live statistics (mean, sigma, centroid)
    ├─ NDPluginROI       → region-of-interest extraction
    ├─ NDPluginProcess   → background subtraction, flat-field
    ├─ NDPluginCodec     → compression (Blosc, JPEG, LZ4)
    ├─ NDFileHDF5        → write HDF5 files
    ├─ NDFileTIFF        → write TIFF files (legacy)
    └─ NDPluginPva       → publish via PV Access (streaming)
```

### NDFileHDF5 Plugin

The HDF5 file writer plugin is the primary output path at APS. Key features:

| Feature | Description |
|---------|-------------|
| SWMR support | Enables live reading during acquisition |
| Direct chunk write | Bypasses filter pipeline for pre-compressed data |
| NeXus layout | Configurable XML template for NeXus-compliant output |
| Attribute capture | Records EPICS PV values as HDF5 attributes per frame |
| Compression | LZ4, Blosc, bitshuffle via hdf5plugin filters |
| Multi-dataset | Multiple detectors/channels in one file |

### NeXus XML Template

areaDetector uses an XML template to define the HDF5 group/dataset
hierarchy:

```xml
<group name="entry">
  <attribute name="NX_class" source="constant" value="NXentry" type="string"/>
  <group name="instrument">
    <attribute name="NX_class" source="constant" value="NXinstrument" type="string"/>
    <group name="detector">
      <attribute name="NX_class" source="constant" value="NXdetector" type="string"/>
      <dataset name="data" source="detector" det_default="true">
        <attribute name="units" source="constant" value="counts" type="string"/>
      </dataset>
      <dataset name="x_pixel_size" source="constant" value="75e-6" type="float"/>
    </group>
  </group>
</group>
```

## Databroker and Tiled

### Databroker (Legacy)

Databroker is a Python library from the Bluesky project that catalogs and
retrieves experiment data. It provides a unified API regardless of the
underlying file format.

```python
from databroker import catalog

cat = catalog["aps_8id"]  # Beamline catalog
run = cat[-1]              # Most recent run
data = run.primary.read()  # Returns xarray Dataset
```

**Key features**:
- Catalogs runs by metadata (plan name, sample, date, etc.)
- Lazy loading: data only read when accessed
- Format-agnostic: reads HDF5, TIFF, MongoDB, etc.
- Search: `cat.search({"plan_name": "fly_scan", "sample": "NMC*"})`

### Tiled (Current)

Tiled is the successor to Databroker, designed as a **data access service**
rather than a library. It serves array and tabular data over HTTP.

| Aspect | Databroker | Tiled |
|--------|-----------|-------|
| Architecture | Python library | Client-server (HTTP API) |
| Access | Local process | Any client (Python, web, curl) |
| Format | Reads files directly | Serves tiles/chunks on demand |
| Scaling | Single machine | Horizontally scalable |
| Authentication | File-level | OAuth2, API keys |
| Catalog | MongoDB | Pluggable (PostgreSQL, etc.) |

```python
from tiled.client import from_uri

client = from_uri("https://tiled.aps.anl.gov")
run = client["8id"]["scan_0042"]
# Fetch only the region you need
roi = run["primary"]["data"]["detector_image"][100:200, 500:1500, 500:1500]
```

**Why Tiled matters for APS-U**: With 100+ TB/day, users cannot download
entire datasets. Tiled enables server-side slicing -- users fetch only the
array chunks they need, regardless of where data is stored.

## Format Selection Guide for APS Users

| Scenario | Recommended Format | Rationale |
|----------|-------------------|-----------|
| Detector acquisition | HDF5 (areaDetector) | SWMR, direct chunk write, ecosystem |
| GPU reconstruction | Zarr output | Parallel write, Dask integration |
| Long-term archive | NeXus/HDF5 | Self-describing, standard, validated |
| Cloud/remote analysis | Zarr (or Tiled-served HDF5) | Cloud-native access |
| ML training data | Zarr or HDF5 | Depends on framework (PyTorch DataLoader supports both) |
| Publication | NeXus/HDF5 | DataCite compatibility, DOI registration |
| Quick visualization | TIFF (exported) | Universal viewer support |
| Crystallography | CBF or HDF5 | DIALS/CCP4 compatibility |

## Emerging: kerchunk and VirtualiZarr

**kerchunk** creates a Zarr-compatible index over existing HDF5 files,
enabling Zarr-style access without converting data:

```python
import kerchunk.hdf
import fsspec

# Create virtual Zarr reference for existing HDF5
refs = kerchunk.hdf.SingleHdf5ToZarr("scan_001.h5").translate()

# Access via Zarr API (no data copy)
import zarr
store = fsspec.filesystem("reference", fo=refs).get_mapper("")
z = zarr.open(store, mode="r")
slice_data = z["entry/data/data"][500]
```

This bridges the gap between APS's HDF5 investment and cloud-native
workflows without requiring format conversion.

## Related Documents

- [HDF5 Deep Dive](hdf5_deep_dive.md) -- Internal architecture, SWMR, parallel I/O
- [HDF5 Structure Overview](hdf5_structure/README.md) -- Basic HDF5 concepts
- [APS-U Data Challenges](data_challenges_apsu.md) -- Infrastructure challenges
- [Data Scale Analysis](data_scale_analysis.md) -- Volume projections
