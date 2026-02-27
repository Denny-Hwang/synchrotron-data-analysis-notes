# Data Structures & Exploratory Analysis

## Overview

Synchrotron light sources produce multi-dimensional, multi-modal datasets at extraordinary
rates. The APS-U upgrade amplifies this challenge: 500x brighter beams yield finer spatial
resolution and faster acquisition, driving data volumes from gigabytes to terabytes per
experiment. Effective research depends on well-defined data structures, standardized formats,
and systematic exploratory data analysis (EDA) before any machine learning or advanced
processing is attempted.

This directory provides a comprehensive guide to synchrotron data formats, schemas for
each eBERlight modality, scaling projections for post-APS-U operations, and hands-on
EDA notebooks.

## Why Data Structure Matters

1. **Reproducibility** -- Standardized schemas ensure experiments can be replicated and
   data can be shared across facilities (APS, NSLS-II, ESRF, Diamond).
2. **Interoperability** -- Common formats like HDF5/NeXus allow tools (TomoPy, MAPS,
   Bluesky) to read each other's output without custom parsers.
3. **Scalability** -- Hierarchical, chunked storage enables streaming analysis of
   datasets that exceed available RAM.
4. **Provenance** -- Metadata embedded alongside measurement arrays preserves the full
   experimental context.

## Synchrotron Data Characteristics

| Property | Typical Range | Post-APS-U Range |
|----------|--------------|------------------|
| Dimensionality | 2D--4D | 3D--5D (+ time, energy) |
| Single scan size | 0.5--10 GB | 5--200 GB |
| Acquisition rate | 10--500 MB/s | 1--20 GB/s |
| Detector pixels | 1--4 Mpx | 4--16 Mpx |
| Dynamic range | 16-bit | 16--32-bit |
| Metadata fields | 50--200 | 200--500 |

## Core Data Format: HDF5

The Hierarchical Data Format version 5 (HDF5) is the de facto standard for synchrotron
data storage worldwide. Key advantages:

- **Self-describing**: metadata and data co-located in a single file
- **Hierarchical**: POSIX-like group/dataset tree structure
- **Compressed**: built-in LZ4, gzip, Blosc codecs reduce file size 2--10x
- **Parallel I/O**: supports MPI-based concurrent reads/writes on HPC systems
- **Language-agnostic**: bindings for Python (h5py), C/C++, Fortran, MATLAB, Julia

Synchrotron-specific conventions built on HDF5 include **NeXus** (general), **Data Exchange**
(tomography), and **CXI** (coherent imaging).

## Exploratory Data Analysis (EDA) for Synchrotron Data

Before applying AI/ML models, systematic EDA is essential:

- **Quality assessment** -- Identify dead pixels, saturated channels, ring artifacts
- **Distribution analysis** -- Understand intensity histograms, outlier populations
- **Correlation discovery** -- Find co-localized elements (XRF) or structural features
- **Noise characterization** -- Measure SNR, identify systematic vs. random noise
- **Dimensionality overview** -- Confirm array shapes, check for missing frames

EDA catches data quality problems early and informs preprocessing decisions (normalization,
background subtraction, filtering) that directly impact downstream analysis.

## Directory Contents

| Path | Description |
|------|-------------|
| [hdf5_structure/](hdf5_structure/) | HDF5 format overview, modality-specific schemas |
| [hdf5_structure/xrf_hdf5_schema.md](hdf5_structure/xrf_hdf5_schema.md) | XRF / MAPS HDF5 schema |
| [hdf5_structure/tomo_hdf5_schema.md](hdf5_structure/tomo_hdf5_schema.md) | Tomography Data Exchange schema |
| [hdf5_structure/ptychography_hdf5_schema.md](hdf5_structure/ptychography_hdf5_schema.md) | Ptychography CXI schema |
| [hdf5_structure/notebooks/](hdf5_structure/notebooks/) | HDF5 exploration & visualization notebooks |
| [data_scale_analysis.md](data_scale_analysis.md) | Pre vs. post-APS-U data volume projections |
| [eda/](eda/) | Exploratory data analysis guides and notebooks |
| [eda/xrf_eda.md](eda/xrf_eda.md) | XRF-specific EDA techniques |
| [eda/tomo_eda.md](eda/tomo_eda.md) | Tomography-specific EDA techniques |
| [eda/spectroscopy_eda.md](eda/spectroscopy_eda.md) | Spectroscopy-specific EDA techniques |
| [eda/notebooks/](eda/notebooks/) | Hands-on EDA Jupyter notebooks |
| [sample_data/](sample_data/) | Links to publicly available sample datasets |

## Recommended Reading Order

1. Start with [hdf5_structure/README.md](hdf5_structure/README.md) for format fundamentals
2. Explore the schema file for your modality of interest
3. Run the HDF5 exploration notebook to gain hands-on familiarity
4. Read [data_scale_analysis.md](data_scale_analysis.md) for context on data volumes
5. Follow the EDA guide for your modality before processing data

## Related Directories

- [02_xray_modalities/](../02_xray_modalities/) -- Technique physics and principles
- [05_tools_and_code/](../05_tools_and_code/) -- Software for processing synchrotron data
- [07_data_pipeline/](../07_data_pipeline/) -- End-to-end acquisition-to-analysis pipeline
