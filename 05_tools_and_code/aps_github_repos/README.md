# APS GitHub Organization -- Analysis Overview

## Purpose

This document catalogues and categorises the open-source repositories
maintained by the Advanced Photon Source (APS) and related groups on GitHub.
The goal is to map the existing software landscape so the APS BER project
can identify reusable components, integration points, and gaps.

## GitHub Organisations Surveyed

| Organisation | Focus |
|-------------|-------|
| [aps-anl](https://github.com/aps-anl) | APS beamline controls and infrastructure |
| [tomography](https://github.com/tomography) | Tomographic reconstruction (TomoPy, TomocuPy) |
| [bluesky](https://github.com/bluesky) | Experiment orchestration framework |
| [BCDA-APS](https://github.com/BCDA-APS) | Beamline Controls & Data Acquisition |
| [xraylib](https://github.com/tschoonj/xraylib) | X-ray physics data (community) |

## Categorisation Scheme

Repositories are grouped into four categories:

1. **Beamline Controls** -- EPICS IOCs, motor drivers, detector plugins.
2. **Data Analysis** -- reconstruction, spectral fitting, image processing.
3. **Simulation** -- X-ray optics, sample modelling, virtual beamlines.
4. **AI / ML** -- machine learning models, training pipelines, inference
   services.

## Key Findings

- Tomographic reconstruction is well-served (TomoPy, TomocuPy, ASTRA).
- XRF analysis depends heavily on the closed-source MAPS package; open-source
  alternatives (PyXRF, XRF-Maps) are available but less mature.
- Bluesky/EPICS adoption is growing across APS-U beamlines but deployment
  patterns vary.
- ML tooling is fragmented; no single APS-wide platform exists (MLExchange
  is a candidate).

## Related Documents

| Document | Description |
|----------|-------------|
| [repo_catalog.md](repo_catalog.md) | Full repository catalogue by category |
