# Data Scale Analysis: Pre- vs. Post-APS-U

## Overview

The Advanced Photon Source Upgrade (APS-U) represents a transformational leap in X-ray
brightness, delivering beams up to 500x brighter than the original APS. This improvement
enables faster acquisitions, higher spatial resolution, and new experimental modalities --
but it also drives a dramatic increase in data generation rates and total volumes that
fundamentally changes how data must be managed, stored, and processed.

This document quantifies the data scale challenge facing eBERlight and the broader APS
user community.

## Detector Technology Evolution

| Generation | Detector | Frame Rate | Pixels | Bit Depth | Raw Rate |
|-----------|----------|-----------|--------|----------|----------|
| Pre-APS-U | PCO.edge 5.5 | 100 fps | 2560x2160 | 16-bit | 1.1 GB/s |
| Pre-APS-U | Pilatus 100K | 300 fps | 487x195 | 32-bit | 0.1 GB/s |
| Post-APS-U | Eiger2 X 9M | 230 fps | 3110x3269 | 32-bit | 9.4 GB/s |
| Post-APS-U | Lambda 2M | 2000 fps | 1536x512 | 24-bit | 4.7 GB/s |
| Post-APS-U | Oryx 10GigE | 162 fps | 4096x3000 | 12-bit | 2.4 GB/s |
| Post-APS-U | Dectris Mythen2 | 1000 fps | 1280x1 | 32-bit | 0.005 GB/s |

## Per-Scan Data Volumes

### Tomography

| Parameter | Pre-APS-U | Post-APS-U |
|-----------|-----------|------------|
| Projections per scan | 900 | 3600 |
| Detector size | 2048 x 2448 | 4096 x 4096 |
| Bit depth | 16-bit | 16-bit |
| Raw scan size | 9 GB | 120 GB |
| Scan duration | 30--60 min | 2--10 min |
| Scans per shift (8 hr) | 8--16 | 48--240 |
| Data per shift | 72--144 GB | 5.7--28.8 TB |

### X-ray Fluorescence Microscopy

| Parameter | Pre-APS-U | Post-APS-U |
|-----------|-----------|------------|
| Scan area | 100x100 um | 500x500 um (or 50x50 um at higher res) |
| Step size | 500 nm | 50 nm |
| Pixels per scan | 40,000 | 10,000,000 |
| Channels per pixel | 2048 | 4096 |
| Dwell time per pixel | 50 ms | 1 ms |
| Scan duration | 33 min | 167 min (or 17 min at 100 nm) |
| Raw scan size | 0.3 GB | 80 GB |

### Ptychography

| Parameter | Pre-APS-U | Post-APS-U |
|-----------|-----------|------------|
| Scan positions | 500 | 5000 |
| Detector size | 256 x 256 | 512 x 512 |
| Bit depth | 32-bit | 32-bit |
| Raw scan size | 0.13 GB | 5.2 GB |
| Scans per shift | 20--40 | 100--500 |
| Data per shift | 2.6--5.2 GB | 520 GB -- 2.6 TB |

### XANES / Spectroscopy

| Parameter | Pre-APS-U | Post-APS-U |
|-----------|-----------|------------|
| Energy points per scan | 200 | 500 |
| Detector frames per point | 10 | 100 |
| Frame size | 512 x 512 | 2048 x 2048 |
| Raw scan size | 1 GB | 400 GB |

## Daily and Annual Projections

### Per-Beamline Daily Volumes

| Modality | Pre-APS-U (GB/day) | Post-APS-U (GB/day) | Growth Factor |
|----------|--------------------|--------------------|---------------|
| Tomography | 200--500 | 10,000--60,000 | 50--120x |
| XRF Microscopy | 50--200 | 2,000--20,000 | 40--100x |
| Ptychography | 20--100 | 1,000--5,000 | 50x |
| Spectroscopy | 50--150 | 5,000--30,000 | 100--200x |
| Diffraction | 100--300 | 5,000--15,000 | 50x |

### Facility-Wide Annual Projections

| Metric | Pre-APS-U (2023) | Post-APS-U (2025+) |
|--------|-------------------|---------------------|
| Operating days per year | 200 | 200 |
| Active beamlines (eBERlight) | 12 | 15 |
| Avg daily volume per beamline | 300 GB | 15 TB |
| **Annual raw data** | **720 TB** | **45 PB** |
| Processed / derived data | 200 TB | 15 PB |
| **Total annual storage need** | **~1 PB** | **~60 PB** |

## Data Rate vs. Processing Rate

A critical challenge is that data generation now outpaces the ability to process and
analyze it in real time:

| Stage | Pre-APS-U Rate | Post-APS-U Rate | Gap |
|-------|---------------|-----------------|-----|
| Acquisition | 0.5--1 GB/s | 5--20 GB/s | -- |
| Network transfer (beamline -> storage) | 10 Gb/s | 100 Gb/s | Sufficient |
| Disk write (GPFS) | 5 GB/s | 20 GB/s | Marginal |
| Tomographic reconstruction (CPU) | 0.3 GB/s | 0.3 GB/s | 17--67x too slow |
| Tomographic reconstruction (GPU) | 5 GB/s | 5 GB/s | Sufficient for most |
| AI/ML inference (per GPU) | 2 GB/s | 2 GB/s | Needs multi-GPU |
| Full pipeline throughput | 0.2 GB/s | 0.5--2 GB/s | 10--40x gap |

## Storage Infrastructure

| System | Capacity | Bandwidth | Purpose |
|--------|----------|-----------|---------|
| Beamline SSD cache | 10--50 TB | 5--10 GB/s | Real-time buffering |
| APS central GPFS | 5 PB | 50 GB/s aggregate | Short-term storage (30 days) |
| Petrel (Globus) | 20+ PB | 100 Gb/s network | Medium-term archive |
| ALCF Eagle | 100 PB | 650 GB/s aggregate | HPC processing |
| Tape archive (HPSS) | 500+ PB | Batch transfer | Long-term archival |

## Implications for eBERlight

### Data Management

- **Automated pipelines** are no longer optional -- manual processing cannot keep pace
- **Edge computing** at the beamline (GPUs, FPGAs) needed for real-time reduction
- **Lossy compression** may be necessary for some raw data (ZFP, SZ compressors)
- **Data lifecycle policies** must define retention tiers (hot / warm / cold / archive)

### AI/ML Opportunities

The data scale challenge is precisely where AI/ML provides the greatest value:

| Challenge | AI/ML Solution |
|-----------|---------------|
| Real-time quality assessment | CNN-based anomaly detection |
| Data reduction at source | Autoencoder compression |
| Faster reconstruction | Neural network priors (PtychoNN) |
| Automated feature extraction | Semantic segmentation |
| Experiment steering | Reinforcement learning for scan optimization |

### Compute Requirements

Post-APS-U processing demands for a single beamline:

| Workload | Pre-APS-U | Post-APS-U |
|----------|-----------|------------|
| CPU cores | 32--128 | 256--1024 |
| GPU cards | 0--1 | 4--16 |
| RAM | 128--512 GB | 1--4 TB |
| Local NVMe | 2--10 TB | 20--100 TB |

## Summary

The APS-U upgrade increases eBERlight data volumes by **50--200x** depending on modality.
Annual facility-wide storage needs grow from approximately 1 PB to 60 PB. This scale
demands automated data pipelines, GPU-accelerated processing, intelligent data reduction
at the edge, and AI/ML-driven analysis to maintain the pace of scientific discovery.

## Related Resources

- [APS-U Data Management Plan](https://www.aps.anl.gov/APS-Upgrade)
- [ALCF computing resources](https://www.alcf.anl.gov/)
- [Data pipeline architecture](../07_data_pipeline/)
