# MAPS -- Data Flow and Fitting Methods

## Data Flow

```
Beamline scan
  |
  v
Raw data (MDA / HDF5)
  |
  v
MAPS Import
  |- Energy calibration (keV per channel)
  |- Detector dead-time correction
  |
  v
Standards Fitting
  |- Fit reference standard spectra
  |- Derive detector sensitivity vs energy
  |
  v
Per-Pixel Fitting
  |- Gaussian peak model for each element line
  |- Compton + elastic scatter background
  |- Matrix absorption correction (optional)
  |
  v
Quantification
  |- Convert fitted peak areas to ug/cm2
  |- Apply sensitivity calibration
  |
  v
HDF5 Output
  |- /MAPS/XRF_fits     (n_elements x rows x cols)
  |- /MAPS/channel_names (element labels)
  |- /MAPS/scalers       (I0, live time, dwell, etc.)
```

## Fitting Methods

### 1. ROI Integration (Fast)

For each element, a predefined energy window brackets the primary emission
line. Total counts within the window are summed per pixel. This method is
fast but cannot separate overlapping lines (e.g., Fe-Kbeta and Co-Kalpha).

### 2. Per-Pixel Gaussian Fitting (Standard)

Each pixel spectrum is fit to a model:

```
S(E) = sum_i  A_i * G(E; E_i, sigma_i) + B(E)
```

Where:
- `A_i` is the fitted area for element line `i`
- `G` is a Gaussian centered at the known emission energy `E_i`
- `sigma_i` is the detector resolution at that energy
- `B(E)` is a polynomial or step-function background

MAPS uses non-negative least squares (NNLS) for speed.  For complex spectra,
a Levenberg-Marquardt solver is available.

### 3. NNLS Matrix Fitting

A reference matrix of pure-element spectra (measured or simulated) is
assembled. Each pixel spectrum is expressed as a non-negative linear
combination of the reference spectra. This avoids explicit peak-shape
modeling and handles overlapping lines well.

## Output HDF5 Schema

| HDF5 path | Shape | Description |
|-----------|-------|-------------|
| `/MAPS/XRF_fits` | (N_elem, rows, cols) | Fitted concentration maps |
| `/MAPS/XRF_fits_quant` | (N_elem, rows, cols) | Quantified maps (ug/cm2) |
| `/MAPS/channel_names` | (N_elem,) | Element/line labels (e.g., "Fe", "Zn_K") |
| `/MAPS/scalers` | (N_scaler, rows, cols) | I0, live time, dwell time |
| `/MAPS/scan_metadata` | -- | Scan parameters, motor positions |

## Calibration Notes

- Detector energy calibration: linear fit of channel number to keV using
  known emission lines from the standard.
- Sensitivity curve: derived from standard fits; corrects for detector
  efficiency, filter absorption, and solid angle.
- Dead-time correction: applied per pixel using the input count rate (ICR)
  and output count rate (OCR) from the detector electronics.

## Limitations

- MAPS is closed-source and available only as a compiled binary.
- GUI-driven workflow can be difficult to script for large batch processing.
- Python bindings are limited; most automation relies on the HDF5 output
  and external scripts.
