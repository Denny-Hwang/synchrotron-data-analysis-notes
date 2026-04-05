# PyXRF

## Overview

PyXRF is a Python-based X-ray fluorescence (XRF) analysis package developed at
NSLS-II, Brookhaven National Laboratory. It provides an interactive GUI for
spectral fitting and elemental mapping, along with batch processing capabilities
for large-scale XRF datasets collected at synchrotron beamlines.

## Key Features

- **Interactive GUI** -- graphical interface for XRF spectrum visualization,
  peak fitting, and elemental map generation.
- **Automated spectral fitting** -- non-linear least-squares fitting of XRF
  spectra with support for multiple fluorescence lines, escape peaks, and
  Compton/Rayleigh scattering.
- **Elemental mapping** -- generates quantitative elemental distribution maps
  from scanning XRF datasets.
- **Batch processing** -- supports automated processing of large-scale XRF
  datasets with configurable fitting parameters.
- **Standards-based quantification** -- calibration against reference standards
  for quantitative elemental concentration mapping.
- **HDF5 I/O** -- reads and writes standard synchrotron data formats compatible
  with the Bluesky/Databroker ecosystem.

## Algorithms Supported

| Algorithm | Description |
|-----------|-------------|
| NNLS fitting | Non-negative least-squares spectral decomposition |
| ROI integration | Region-of-interest based elemental mapping |
| Per-pixel fitting | Full spectral fitting at each scan position |
| Background subtraction | Automated background estimation and removal |
| Escape peak correction | Detector escape peak modeling |

## Typical Performance

Batch processing of large XRF maps (1000x1000 pixels, 4096 channels) completes
in minutes on standard workstations. Per-pixel fitting is parallelized across
CPU cores for throughput.

## Repository

- GitHub: <https://github.com/NSLS2/PyXRF>
- License: BSD-3-Clause
- Language: Python
- Primary maintainer: NSLS-II, Brookhaven National Laboratory

## Reference

Li, L. et al. (2017). PyXRF: Python-based X-ray fluorescence analysis package.
*Proc. SPIE 10389, X-Ray Nanoimaging: Instruments and Methods III*, 103890U.
DOI: [10.1117/12.2272585](https://doi.org/10.1117/12.2272585)

## Related Documents

| Document | Description |
|----------|-------------|
| XRF analysis notes | See synchrotron technique documentation |
