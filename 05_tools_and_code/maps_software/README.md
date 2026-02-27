# MAPS -- XRF Spectral Analysis Software

## Overview

MAPS is the standard software package for quantitative X-ray fluorescence
(XRF) microprobe data analysis at the Advanced Photon Source. Developed by
Stefan Vogt and colleagues, it processes energy-dispersive spectra collected at
scanning XRF beamlines (2-ID-D, 2-ID-E, Bionanoprobe) to produce quantitative
elemental concentration maps.

## Key Capabilities

- **Per-pixel spectral fitting** -- fits each XRF spectrum to a combination of
  element emission lines plus background, yielding quantitative
  concentrations (ug/cm2).
- **Batch ROI integration** -- fast element quantification via region-of-interest
  (ROI) windows when full fitting is not needed.
- **Standards calibration** -- fits known reference standards (e.g., NBS thin
  films) to derive detector sensitivity curves.
- **Multi-detector merging** -- combines data from multiple fluorescence
  detectors with appropriate solid-angle weighting.
- **Output formats** -- writes fitted maps to HDF5 files following the MAPS
  schema (`/MAPS/XRF_fits`, `/MAPS/channel_names`, `/MAPS/scalers`).

## Typical Workflow

1. Collect XRF scans and reference standard scans at the beamline.
2. Load raw MDA or HDF5 files into MAPS.
3. Calibrate the detector energy axis and sensitivity using standards.
4. Run per-pixel fitting (Gaussian peak shapes with matrix corrections).
5. Export fitted elemental maps to HDF5 for downstream analysis.

## Software Access

- MAPS is distributed as a compiled binary for Linux and macOS.
- Available to APS users via the beamline computing environment.
- Contact: Stefan Vogt, XSD / APS.

## Integration with eBERlight

The ROI Finder pipeline (see `../roi_finder/`) reads MAPS HDF5 output as its
primary input format. The `/MAPS/XRF_fits` dataset provides the multi-element
concentration maps used for segmentation, feature extraction, and clustering.

## Related Documents

| Document | Description |
|----------|-------------|
| [workflow_analysis.md](workflow_analysis.md) | Data flow and fitting methods |
