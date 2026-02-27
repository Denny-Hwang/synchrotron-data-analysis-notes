# Data Acquisition

## Overview

Data acquisition is the first stage of the eBERlight pipeline. It encompasses
the physical detectors that convert X-ray photons into digital signals, the
EPICS IOC layer that orchestrates hardware, and the Area Detector framework
that packages raw frames with metadata for downstream consumption.

## Detector Types

### EIGER (Dectris)

- **Technology**: Hybrid photon-counting (HPC) silicon sensor
- **Pixel array**: 4M pixels (EIGER2 X 4M) -- 2070 x 2167 pixels
- **Frame rate**: Up to 750 Hz (4M), 2.3 kHz (1M)
- **Bit depth**: 8, 16, or 32 bit per pixel
- **Interface**: 10 GbE / 100 GbE direct connection
- **Use case**: SAXS/WAXS, ptychography, serial crystallography
- **Output format**: HDF5 with bitshuffle-LZ4 compression

### Jungfrau (PSI)

- **Technology**: Charge-integrating hybrid pixel detector
- **Pixel array**: Modular -- 512 x 1024 per module, tiled to 4M+
- **Dynamic range**: Adaptive gain switching (3 gain stages)
- **Frame rate**: Up to 2.2 kHz (full frame)
- **Interface**: 10 GbE per module
- **Use case**: Spectroscopy, XPCS, high-dynamic-range imaging
- **Output format**: Raw binary frames; converted to HDF5 by receiver

### Vortex (Hitachi)

- **Technology**: Silicon drift detector (SDD) for energy-dispersive XRF
- **Channels**: Single-element or 4-element array
- **Energy resolution**: < 130 eV FWHM at 5.9 keV
- **Count rate**: Up to 1 Mcps per element (output count rate)
- **Interface**: Digital pulse processor (xMAP / Mercury)
- **Use case**: X-ray fluorescence mapping, XANES
- **Output format**: MCA spectra per pixel, stored in HDF5

## EPICS IOC and Area Detector Framework

### EPICS IOC Architecture

The Experimental Physics and Industrial Control System (EPICS) provides the
real-time control layer between detector hardware and higher-level software.

```
Detector Hardware
    |
    v
Device Support Layer  (driver-specific C/C++ code)
    |
    v
EPICS IOC             (records, databases, sequencer)
    |
    v
Channel Access / PV Access  (network protocol)
    |
    v
Client Applications   (CSS, caput, pvget, Python scripts)
```

Key IOC databases for acquisition:

| Database | Purpose |
|---|---|
| `cam.db` | Detector mode, exposure, gain, trigger settings |
| `image.db` | Image dimensions, data type, array port |
| `file.db` | File path, name pattern, write plugin config |
| `stats.db` | Real-time statistics (min, max, mean, sigma) |

### Area Detector Plugins

The areaDetector framework provides a plugin-based pipeline within the IOC:

1. **NDPluginStdArrays** -- Publishes frames over Channel Access / PV Access
2. **NDPluginROI** -- Extracts regions of interest before saving
3. **NDPluginProcess** -- Applies background subtraction and flat-field in-IOC
4. **NDPluginHDF5** -- Writes NeXus/HDF5 files with SWMR support
5. **NDPluginCodec** -- Compresses frames (Blosc, JPEG, LZ4) for streaming

## Data Collection Triggers

### Trigger Modes

| Mode | Description | Typical Use |
|---|---|---|
| Internal | Detector free-runs at configured rate | Alignment, testing |
| External (TTL) | Hardware trigger from timing system | Fly scans |
| Software | IOC sends trigger via Channel Access | Step scans |
| Gated | External gate signal defines exposure window | Pump-probe |

### Fly Scan Trigger Chain

```
Motor Controller (Delta Tau / Aerotech)
    |-- position-compare output (TTL pulse)
    v
Timing Fanout (SIS3820 / SoftGlue FPGA)
    |-- trigger to detector
    |-- trigger to scaler
    |-- trigger to MCS
    v
Detector captures frame per trigger
```

Fly scans achieve continuous motion with synchronous data capture, eliminating
overhead from step-and-shoot sequences. Typical angular step: 0.1 deg at
180 deg/s yields 1800 projections in 1 second.

## Metadata Recording

Every acquisition records the following metadata in the HDF5 master file:

- **Beamline parameters**: energy (keV), ring current (mA), undulator gap
- **Detector settings**: exposure time, gain mode, threshold energy
- **Motor positions**: sample x/y/z, rotation angle, detector distance
- **Environment**: temperature, humidity, sample ID, proposal number
- **Timestamps**: ISO-8601 start/end, per-frame EPICS timestamps

Metadata is written according to the **NeXus application definition**
(`NXtomo`, `NXsas`, `NXxas`) to ensure interoperability with community tools
such as DAWN, silx, and Xi-CAM.

## Data Rates

| Detector | Resolution | Frame Rate | Raw Rate |
|---|---|---|---|
| EIGER2 X 4M | 2070 x 2167 x 16 bit | 750 Hz | ~6.7 GB/s |
| Jungfrau 4M | 2048 x 2048 x 16 bit | 2.2 kHz | ~18.4 GB/s |
| Vortex ME-4 | 4 x 4096 channels | 1 kHz | ~32 MB/s |

These rates demand high-bandwidth streaming and processing infrastructure,
described in [streaming.md](streaming.md) and [processing.md](processing.md).
