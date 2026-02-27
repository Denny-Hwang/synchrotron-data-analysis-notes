# Open-Source Tools Ecosystem

## Overview

This section analyzes the key open-source software tools used in the BER program and the
broader APS synchrotron science ecosystem. Each tool is examined for architecture,
strengths, limitations, and improvement opportunities.

## Tool Landscape

```
┌─────────────────── Experiment Control ───────────────────┐
│  Bluesky (RunEngine) + EPICS (hardware) + ophyd (devices) │
└──────────────────────┬───────────────────────────────────┘
                       │ Data
                       ▼
┌──────────────── Data Analysis Tools ─────────────────────┐
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │ TomoPy   │  │ TomocuPy │  │  MAPS    │  │ ROI-     │ │
│  │(CPU tomo)│  │(GPU tomo)│  │(XRF fit) │  │ Finder   │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│                                                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│  │  Larch   │  │  PyNX    │  │MLExchange│               │
│  │(XAS)     │  │(ptycho)  │  │(ML plat) │               │
│  └──────────┘  └──────────┘  └──────────┘               │
└──────────────────────────────────────────────────────────┘
```

## Tool Comparison

| Tool | Modality | Language | GPU | Maturity | Active Dev |
|------|----------|---------|-----|----------|-----------|
| **ROI-Finder** | XRF | Python | No | Research | Low |
| **TomocuPy** | Tomography | Python/CuPy | Yes | Production | High |
| **TomoPy** | Tomography | Python/C | No | Production | High |
| **MAPS** | XRF | IDL/C++ | No | Production | Medium |
| **MLExchange** | Multi | Python | Yes | Development | High |
| **Bluesky** | Control | Python | No | Production | High |
| **EPICS** | Control | C/Python | No | Production | High |

## Directory Contents

| Subdirectory | Tool | Focus |
|-------------|------|-------|
| [roi_finder/](roi_finder/) | ROI-Finder | XRF ROI selection (detailed reverse engineering) |
| [tomocupy/](tomocupy/) | TomocuPy | GPU-accelerated tomographic reconstruction |
| [tomopy/](tomopy/) | TomoPy | Standard tomographic reconstruction |
| [maps_software/](maps_software/) | MAPS | XRF spectral analysis |
| [mlexchange/](mlexchange/) | MLExchange | ML platform for light sources |
| [aps_github_repos/](aps_github_repos/) | APS GitHub | Repository catalog |
| [bluesky_epics/](bluesky_epics/) | Bluesky/EPICS | Experiment orchestration |
