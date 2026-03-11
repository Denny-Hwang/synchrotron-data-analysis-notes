# APS-U Data Challenges: Handling 100+ TB/Day

## Overview

The APS Upgrade (APS-U) increases X-ray brightness by up to 500x, enabling
faster scans, higher resolution, and new experimental modalities. The
consequence is a **50--200x increase in data generation** per beamline.
The upgraded facility is projected to produce over **100 petabytes per year**
across all beamlines -- exceeding **270 TB per day** on average, with peak
days reaching 500+ TB.

This document examines how ANL currently handles this data deluge, what
works well, what doesn't, and what open challenges remain.

## The Scale of the Problem

### Data Generation Rates by Modality

| Modality | Detector | Peak Rate | Typical Scan | Daily Volume |
|----------|----------|-----------|-------------|-------------|
| Fast tomography | EIGER2 X 9M | 9.4 GB/s | 120 GB / 2 min | 10--60 TB |
| Nano-XRF | Vortex ME4 + EIGER | 2.5 GB/s | 80 GB / 3 hr | 2--20 TB |
| Ptychography | EIGER2 X 500K | 1.2 GB/s | 5.2 GB / 10 min | 1--5 TB |
| XPCS | EIGER2 X 4M | 4.0 GB/s | 50 GB / 30 min | 5--30 TB |
| XANES spectro-tomo | EIGER + monochromator | 9.4 GB/s | 400 GB / 1 hr | 5--30 TB |

### Cumulative Facility Projections

| Metric | Pre-APS-U (2023) | Post-APS-U (2025+) | Growth |
|--------|-------------------|---------------------|--------|
| Active beamlines | ~40 | ~70 | 1.75x |
| Avg daily per beamline | 300 GB | 15 TB | 50x |
| Facility daily (peak) | 12 TB | 500+ TB | 40x |
| Annual raw | 720 TB | 100+ PB | 140x |
| Annual total (+ processed) | ~1 PB | ~150 PB | 150x |

## Current Infrastructure: How the Data Is Handled

### Data Flow Architecture

```
Detector (EIGER, Pilatus, etc.)
    │
    ├──[areaDetector IOC]──► Beamline SSD Cache (10-50 TB, NVMe)
    │                            │
    │                            ├──[Globus 100 Gbps]──► ALCF Eagle (100 PB, GPFS)
    │                            │                           │
    │                            │                           ├──► GPU Processing (Polaris/Aurora)
    │                            │                           │        │
    │                            │                           │        ├──► TomocuPy reconstruction
    │                            │                           │        ├──► AI/ML inference
    │                            │                           │        └──► Results → Eagle
    │                            │                           │
    │                            │                           ├──[Globus Flow]──► Petrel (warm archive)
    │                            │                           └──[Globus Flow]──► HPSS Tape (cold archive)
    │                            │
    │                            └──► APS Central GPFS (5 PB, 30-day retention)
    │
    ├──[ZMQ Stream]──► Real-time monitors / edge compute
    └──[PV Access]──► Bluesky / live dashboards
```

### Storage Tiers in Detail

#### Tier 0: Beamline SSD Cache

| Parameter | Value |
|-----------|-------|
| Hardware | NVMe SSD arrays, Dell/HPE servers |
| Capacity | 10--50 TB per beamline |
| Bandwidth | 5--10 GB/s (matches detector rate) |
| Retention | Hours to days (buffer only) |
| Purpose | Buffer detector output, enable real-time processing |

**Strengths**: Fast enough for any detector; local to minimize latency.
**Weakness**: Limited capacity; must be drained continuously.

#### Tier 1: APS Central GPFS

| Parameter | Value |
|-----------|-------|
| Hardware | IBM Spectrum Scale (GPFS) on spinning disk + SSD tier |
| Capacity | 5 PB |
| Bandwidth | 50 GB/s aggregate |
| Retention | 30 days |
| Purpose | Short-term shared storage for all APS beamlines |

**Strengths**: Shared filesystem accessible from all beamlines; familiar POSIX interface.
**Weakness**: 30-day retention means data must be migrated or lost; contention during peak usage.

#### Tier 2: ALCF Eagle

| Parameter | Value |
|-----------|-------|
| Hardware | HPE Cray ClusterStor E1000, Lustre filesystem |
| Capacity | 100 PB |
| Bandwidth | 650 GB/s aggregate |
| Retention | 90 days (project allocation) |
| Access | Direct from Polaris/Aurora compute nodes |
| Network | 100 Gbps dedicated link to APS (being upgraded to terabit) |

**Strengths**: Massive bandwidth; directly accessible from supercomputers;
Globus-enabled for external sharing.
**Weakness**: 90-day retention; allocation-based access requires ALCF project.

#### Tier 3: Petrel (Warm Archive)

| Parameter | Value |
|-----------|-------|
| Hardware | Object store at ALCF |
| Capacity | 500 TB per project allocation (expandable) |
| Access | Globus transfer, REST API, web portal |
| Retention | 3 years |
| Sharing | Identity-based ACLs, Globus Auth groups |

**Strengths**: Easy sharing with collaborators; web-browsable; search index
for metadata discovery; DOI landing pages.
**Weakness**: Object store latency (seconds); not suitable for HPC computation.

#### Tier 4: HPSS Tape Archive

| Parameter | Value |
|-----------|-------|
| Hardware | IBM tape library (LTO-9) |
| Capacity | 500+ PB |
| Access | Batch recall via Globus (minutes to hours) |
| Retention | 10+ years (raw), permanent (publication) |
| Cost | ~$20/TB/year (vs. ~$200/TB/year for SSD) |

**Strengths**: Extremely low cost per TB; effectively unlimited capacity;
dual-copy policy for data safety.
**Weakness**: Recall latency (minutes to hours); sequential access only;
not suitable for interactive analysis.

### Data Transfer Infrastructure

#### Globus

Globus is the **unified data management layer** at ANL, handling all
inter-facility transfers:

| Transfer Path | Bandwidth | Latency | Automation |
|--------------|-----------|---------|-----------|
| APS → ALCF (Eagle) | 100 Gbps (12.5 GB/s) | Seconds | Globus Flows |
| ALCF → Petrel | 40 Gbps internal | Seconds | Globus Flows |
| ALCF → HPSS | 20 Gbps | Minutes | Globus Flows |
| APS → NERSC | 100 Gbps via ESnet | Seconds | Manual or Flows |
| APS → collaborator | Variable | Variable | Manual transfer |

**Key capability**: Globus Flows automates the entire post-acquisition
pipeline: transfer → process → archive → notify. A single scan completion
triggers the full chain automatically.

#### Nexus Framework (APS-ALCF Integration)

The Nexus framework, developed at ANL, provides automated near-real-time
streaming from APS beamlines to ALCF supercomputers:

- **XPCS at 8-ID**: Speckle patterns stream from detector → ZMQ → Polaris
  for correlation function computation; results returned in seconds
- **Tomography at 2-BM/32-ID**: Projections transferred via Globus to
  Polaris; TomocuPy reconstruction runs on GPUs; 3D volume available
  within minutes of scan completion
- **XRF at various beamlines**: Spectral data processed on Polaris GPUs
  using MAPS for elemental fitting

## What Works Well

### 1. Automated Pipeline (Globus Flows + Bluesky)

The combination of Bluesky RunEngine for experiment orchestration and
Globus Flows for data management creates a fully automated pipeline:

```
Scan → Write HDF5 → Transfer → Process → Archive → DOI → Notify
  └──────── All automated, no human intervention ──────────┘
```

**Advantage**: Scientists focus on science, not data management.
**Metric**: Median time from scan completion to processed results: 5--15 minutes.

### 2. GPU-Accelerated Reconstruction

TomocuPy on ALCF GPUs delivers reconstruction throughput that matches data
generation:

| Workload | CPU (TomoPy, 128 cores) | GPU (TomocuPy, 4x A100) |
|----------|------------------------|------------------------|
| 2048x2048, 1800 proj | 15 min | 45 s |
| 4096x4096, 3600 proj | 2+ hr | 8 min |

### 3. Live Streaming via SWMR + ZMQ

Real-time data quality assessment during acquisition:
- **SWMR**: Monitoring process reads HDF5 while acquisition writes
- **ZMQ**: Edge compute processes frames in-flight for anomaly detection
- **PV Access**: Live thumbnails and statistics in Bluesky dashboards

### 4. Tiered Storage Model

The hot/warm/cold/archive tiers match data lifecycle to cost:

| Tier | $/TB/year | Access Time | Proportion of Data |
|------|-----------|------------|-------------------|
| SSD cache | $1,000+ | μs | <1% (current scans) |
| GPFS/Eagle | $200 | ms | 5% (active processing) |
| Petrel | $50 | seconds | 15% (recent, shared) |
| HPSS tape | $20 | minutes-hours | 80% (archival) |

## What Doesn't Work Well

### 1. The Rechunking Bottleneck

Detectors write data in projection order (one frame per chunk), but
reconstruction needs sinogram order. Rechunking a 120 GB tomography
dataset takes 10--30 minutes and produces a full copy:

| Problem | Impact |
|---------|--------|
| 2x storage during rechunking | Doubles SSD/GPFS pressure |
| 10--30 min I/O-bound operation | Delays reconstruction start |
| Single-node bottleneck | Cannot parallelize easily in HDF5 |

**Mitigation**: Use Zarr as the intermediate format (parallel rechunking),
or implement streaming reconstruction that processes sinograms on-the-fly.

### 2. Metadata Management at Scale

With millions of scans per year, finding the right dataset becomes
increasingly difficult:

| Challenge | Current State |
|-----------|--------------|
| Search across datasets | Petrel search + manual curation |
| Provenance tracking | Per-file NXprocess (inconsistent) |
| Cross-experiment linking | Manual, via PI knowledge |
| Sample tracking | Spreadsheet-based at most beamlines |

**Need**: A facility-wide metadata catalog with standardized ontology
and automated extraction.

### 3. 30-Day GPFS Retention

The 30-day retention on APS central GPFS means data must be transferred
and verified within that window. Under peak load:

- 15 beamlines × 15 TB/day = 225 TB/day ingest
- 100 Gbps transfer = 12.5 GB/s = 1.08 PB/day theoretical max
- Practical throughput (with contention): 500--800 TB/day
- **Margin is thin during campaigns with many simultaneous beamlines**

### 4. HPSS Recall Latency

When a user needs to reprocess archived data, recall from tape can take
minutes to hours:

| Dataset Size | Recall Time | Impact |
|-------------|------------|--------|
| 1 GB | 2--5 min | Acceptable |
| 100 GB | 15--30 min | Frustrating |
| 10 TB | 2--8 hr | Blocks workflow for a day |

**Mitigation**: Keep frequently reprocessed datasets on Petrel; implement
predictive pre-staging based on user schedules.

### 5. HDF5 in the Cloud

As some workflows move toward cloud-hybrid models (Google Cloud, AWS),
HDF5's POSIX dependency becomes a limitation:

- No efficient partial reads from S3
- No serverless processing of HDF5 files
- Requires either full download or HSDS/Tiled server deployment

## Open Challenges and Research Directions

### 1. Real-Time AI at the Edge

**Goal**: Run AI/ML models at the beamline for instant feedback.

| Challenge | Current | Target |
|-----------|---------|--------|
| Inference latency | 100 ms (GPU server) | <10 ms (edge FPGA/GPU) |
| Model deployment | Manual | Automated CI/CD to beamline |
| Anomaly detection | Post-hoc | Per-frame, real-time |
| Experiment steering | Human decides | AI suggests next measurement |

### 2. Lossy Compression for Raw Data

At 100+ PB/year, even tape is expensive. Lossy compression can reduce
raw data by 10--50x but introduces scientific risk:

| Compressor | Ratio | Error Bound | Status at APS |
|-----------|-------|------------|--------------|
| SZ3 | 15--50x | Absolute or relative | Research stage |
| ZFP | 10--30x | Fixed rate or precision | Pilot at tomography |
| MGARD | 20--100x | Error-controlled multiresolution | Research stage |

**Open question**: What error bounds are acceptable for each modality?
This requires domain-specific validation studies.

### 3. FAIR Data Compliance

DOE mandates compliance with FAIR (Findable, Accessible, Interoperable,
Reusable) principles:

| Principle | Current State | Gap |
|-----------|--------------|-----|
| **Findable** | DOIs for publication datasets; Petrel search | No facility-wide catalog; inconsistent metadata |
| **Accessible** | Globus transfer; Petrel web portal | No standardized API for all data; 12-month embargo |
| **Interoperable** | NeXus/HDF5 standard | Inconsistent NeXus compliance across beamlines |
| **Reusable** | Processing provenance in NXprocess | Incomplete reproducibility chain; no containerized environments |

### 4. Data Reduction at Source

Instead of storing everything and processing later, reduce data at the
detector or beamline:

| Strategy | Reduction Factor | Trade-off |
|----------|-----------------|-----------|
| Region-of-interest cropping | 2--10x | Loss of peripheral data |
| On-detector compression (Dectris) | 3--6x | Lossless, no trade-off |
| Streaming peak detection | 100--1000x | Only for crystallography |
| Real-time sinogram assembly | 1x (reorganize) | Eliminates rechunking step |
| AI-guided sparse sampling | 5--20x | Requires validated model |

### 5. Multi-Facility Data Federation

BER program users often combine data from APS, NSLS-II, SSRL, and LCLS.
Each facility has different:
- File formats and conventions
- Metadata schemas
- Access control systems
- Data catalogs

**Need**: Federated data access layer that spans facilities with unified
authentication (via Globus) and standardized metadata (via NeXus/Tiled).

### 6. Exascale Integration (Aurora)

ALCF's Aurora supercomputer (2+ exaFLOPS) opens new possibilities:

| Capability | Impact for APS |
|-----------|----------------|
| Exascale compute | Full-volume 4D tomography in minutes |
| AI accelerators | Training facility-scale models on all APS data |
| Terabit APS-Aurora link | Near-zero-latency data transfer |
| Unified memory | Process datasets larger than single-node RAM |

**Challenge**: Software must be adapted for Aurora's Intel GPU architecture
(SYCL/oneAPI), requiring porting of CUDA-based codes like TomocuPy.

## Summary: Strengths, Weaknesses, Opportunities

### Strengths
- World-class automated pipeline (Bluesky + Globus Flows)
- Massive HPC resources via ALCF (Polaris, Aurora)
- Proven tiered storage model with clear retention policies
- Strong community tools (TomoPy, TomocuPy, MAPS, Bluesky)

### Weaknesses
- HDF5 rechunking bottleneck between acquisition and processing
- Limited metadata search and discovery across the facility
- HPSS recall latency for reprocessing campaigns
- Inconsistent NeXus compliance across beamlines

### Opportunities
- Edge AI for real-time experiment steering
- Lossy compression to manage storage costs
- Cloud-hybrid with Zarr/kerchunk for remote collaboration
- Aurora exascale for previously intractable analyses

### Threats
- Storage costs growing faster than budgets
- Software porting burden (CUDA → SYCL for Aurora)
- Personnel bottleneck (few data engineers per beamline)
- Increasing complexity of multi-modal, multi-facility experiments

## Related Documents

- [Data Scale Analysis](data_scale_analysis.md) -- Quantitative volume projections
- [HDF5 Deep Dive](hdf5_deep_dive.md) -- Format internals and performance
- [Data Formats Comparison](data_formats_comparison.md) -- HDF5 vs. Zarr vs. TIFF
- [Storage and Management](../07_data_pipeline/storage.md) -- Pipeline storage stage
- [Architecture Diagrams](../07_data_pipeline/architecture_diagram.md) -- System diagrams
