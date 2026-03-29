# Symptom-Based Troubleshooter

Can't identify what's wrong with your data? Start from what you **see** and follow the decision trees below to diagnose the problem.

## How to Use

1. Find the symptom category that best matches your problem
2. Follow the branching questions in the decision tree
3. Run the quick-check Python snippets to confirm the diagnosis
4. Follow the link to the full guide for detailed solutions

---

## 1. Circular/Ring Patterns

### Symptom: Concentric rings or circular features in reconstructed image

```
Do you see rings in the reconstructed CT image?
│
├── Rings are centered on the rotation axis
│   ├── Rings are sharp and well-defined
│   │   └── → Ring Artifact from defective detector pixels
│   │       Severity: Critical | Full guide: tomography/ring_artifact.md
│   │
│   └── Rings are broad/blurry or partial arcs
│       └── → Flat-Field Contamination (debris on scintillator, non-uniform response)
│           Severity: Major | Full guide: tomography/flatfield_issues.md
│
├── Rings are NOT centered on the rotation axis
│   ├── Object edges appear doubled or "tuning-fork" shaped
│   │   └── → Rotation Center Error
│   │       Severity: Critical | Full guide: tomography/rotation_center_error.md
│   │
│   └── Rings appear around dense inclusions
│       └── → Streak Artifact (beam hardening component)
│           Severity: Critical | Full guide: tomography/streak_artifact.md
│
└── Rings visible in sinogram as vertical stripes
    └── → Ring Artifact (sinogram domain view)
        Severity: Critical | Full guide: tomography/ring_artifact.md
```

**Quick check — sinogram stripe test:**

```python
import numpy as np
sinogram = data[data.shape[0]//2]  # middle slice sinogram
col_std = np.std(sinogram, axis=0)
suspect_cols = np.where(col_std < np.median(col_std) * 0.1)[0]
print(f"Dead columns (ring sources): {suspect_cols}")
```

**Quick check — rotation center:**

```python
# Compare 0-degree and 180-degree projections
proj_0 = data[0]
proj_180 = data[data.shape[0]//2]
flipped = proj_180[:, ::-1]
shift = np.argmax(np.correlate(proj_0.mean(0), flipped.mean(0), mode='full'))
center_offset = shift - proj_0.shape[1] + 1
print(f"Center offset from midpoint: {center_offset/2:.1f} pixels")
```

---

## 2. Isolated Bright/Dark Spots

### Symptom: Random bright or dark spots that don't correspond to real features

```
Where do you see the spots?
│
├── In raw projections (before reconstruction)
│   ├── Spots appear in single frames only (not same position across frames)
│   │   └── → Zinger (cosmic ray / gamma-ray spike)
│   │       Severity: Major | Full guide: tomography/zinger.md
│   │
│   └── Spots appear at the same pixel in every frame
│       └── → Detector Dead/Hot Pixel
│           Severity: Major | Full guide: cross_cutting/detector_common_issues.md
│
├── In elemental maps (XRF)
│   ├── Spots are abnormally bright (10x+ above neighbors)
│   │   └── → Hot Pixel in XRF detector
│   │       Severity: Major | Full guide: xrf_microscopy/dead_hot_pixel.md
│   │
│   └── Spots are zero or near-zero surrounded by signal
│       └── → Dead Pixel in XRF detector
│           Severity: Major | Full guide: xrf_microscopy/dead_hot_pixel.md
│
└── In reconstructed CT slices
    ├── Bright spots correspond to zinger positions in projections
    │   └── → Zinger artifact propagated through reconstruction
    │       Severity: Major | Full guide: tomography/zinger.md
    │
    └── Spots form a ring pattern
        └── → See "Circular/Ring Patterns" above
```

**Quick check — zinger detection:**

```python
import numpy as np
diff = np.abs(np.diff(projections, axis=0))
threshold = np.median(diff) + 10 * np.std(diff)
zingers = np.where(diff > threshold)
print(f"Potential zingers found: {len(zingers[0])}")
```

**Quick check — dead/hot pixel in XRF map:**

```python
import numpy as np
from scipy.ndimage import median_filter
filtered = median_filter(elem_map, size=3)
outliers = np.abs(elem_map - filtered) > 5 * np.std(elem_map)
print(f"Outlier pixels: {np.sum(outliers)} ({100*np.mean(outliers):.1f}%)")
```

---

## 3. Streak/Stripe Patterns

### Symptom: Linear streaks, stripes, or banding in the data

```
What direction and context?
│
├── Bright streaks radiating from dense objects (CT reconstruction)
│   └── → Metal / Streak Artifact
│       Severity: Critical | Full guide: tomography/streak_artifact.md
│
├── Star-like streaks throughout the reconstruction (not from specific objects)
│   └── → Sparse-Angle Artifact (insufficient projections)
│       Severity: Major | Full guide: tomography/sparse_angle_artifact.md
│
├── Horizontal stripes in sinogram
│   ├── Stripe correlates with I0 monitor drops
│   │   └── → Beam Intensity Drop
│   │       Severity: Major | Full guide: tomography/beam_intensity_drop.md
│   │
│   └── Stripes at regular intervals
│       └── → Ring Artifact (sinogram domain)
│           Severity: Critical | Full guide: tomography/ring_artifact.md
│
├── Vertical stripes in sinogram
│   └── → Ring Artifact source (dead/miscalibrated detector column)
│       Severity: Critical | Full guide: tomography/ring_artifact.md
│
├── Horizontal or vertical stripes in XRF elemental maps
│   ├── Stripes align with fast-scan direction
│   │   └── → Scan Stripe (beam drift or I0 fluctuation)
│   │       Severity: Major | Full guide: xrf_microscopy/scan_stripe.md
│   │
│   └── Stripes correlate with I0 variations
│       └── → I0 Normalization Issue
│           Severity: Major | Full guide: xrf_microscopy/i0_normalization.md
│
└── Stripe at tile boundaries (ptychography)
    └── → Stitching Artifact
        Severity: Minor | Full guide: ptychography/stitching_artifact.md
```

**Quick check — streak from dense objects:**

```python
import numpy as np
# Check if sinogram has saturated/zero-transmission pixels
sino_min = np.min(sinogram, axis=0)
saturated_cols = np.sum(sino_min < 0.01 * np.median(sinogram))
print(f"Near-zero transmission columns: {saturated_cols} (metal streak likely if > 0)")
```

**Quick check — sparse angle:**

```python
n_projections = projections.shape[0]
detector_width = projections.shape[-1]
nyquist = int(np.pi/2 * detector_width)
print(f"Projections: {n_projections}, Nyquist minimum: {nyquist}")
print(f"{'UNDERSAMPLED' if n_projections < nyquist else 'OK'}")
```

---

## 4. Overall Graininess/Noise

### Symptom: Image appears grainy, speckled, or noisy throughout

```
What type of data?
│
├── CT reconstruction
│   ├── Noise is spatially uniform (same everywhere)
│   │   └── → Low-Dose Poisson Noise
│   │       Severity: Major | Full guide: tomography/low_dose_noise.md
│   │
│   └── Noise is worse in certain regions
│       ├── Worse inside dense objects
│       │   └── → Photon starvation (low transmission through dense material)
│       │       Severity: Major | Full guide: tomography/streak_artifact.md
│       │
│       └── Worse at edges/periphery
│           └── → Flat-field or beam profile issue
│               Severity: Major | Full guide: tomography/flatfield_issues.md
│
├── XRF elemental map
│   ├── Low-concentration elements are noisier than high-concentration
│   │   └── → Photon Counting Noise (insufficient statistics)
│   │       Severity: Major | Full guide: xrf_microscopy/photon_counting_noise.md
│   │
│   └── All elements equally noisy
│       └── → Check dwell time and detector dead-time
│           Full guide: xrf_microscopy/dead_time_saturation.md
│
└── EXAFS spectrum
    ├── Oscillations noisy at high k (> 10-12 A^-1)
    │   └── → Statistical Noise in EXAFS (normal, needs more scans)
    │       Severity: Major | Full guide: spectroscopy/statistical_noise_exafs.md
    │
    └── Entire spectrum noisy
        └── → Check I0, sample alignment, and detector
            Full guide: spectroscopy/statistical_noise_exafs.md
```

**Quick check — SNR estimate for CT:**

```python
import numpy as np
roi_signal = recon[100:150, 100:150]  # region inside sample
roi_bg = recon[10:30, 10:30]          # background region
snr = np.mean(roi_signal) / np.std(roi_bg)
print(f"SNR estimate: {snr:.1f} (< 5 is very noisy, > 20 is good)")
```

**Quick check — EXAFS noise level:**

```python
import numpy as np
# Estimate noise from high-k region of chi(k)
high_k_region = chi_k[k > 12]
noise_level = np.std(high_k_region)
print(f"High-k noise: {noise_level:.4f} (merge more scans if > 0.01)")
```

---

## 5. Blurring / Loss of Detail

### Symptom: Features appear blurred, smeared, or larger than expected

```
What type of data?
│
├── CT reconstruction
│   ├── Blurring is directional (smeared in one direction)
│   │   └── → Motion Artifact (sample moved during scan)
│   │       Severity: Critical | Full guide: tomography/motion_artifact.md
│   │
│   ├── Blurring is uniform (everything soft/fuzzy)
│   │   ├── Were few projections used?
│   │   │   └── → Sparse-Angle Artifact
│   │   │       Severity: Major | Full guide: tomography/sparse_angle_artifact.md
│   │   │
│   │   └── Full angular sampling was used
│   │       └── → Check rotation center and flat-field
│   │           Full guide: tomography/rotation_center_error.md
│   │
│   └── Object edges appear doubled
│       └── → Rotation Center Error
│           Severity: Critical | Full guide: tomography/rotation_center_error.md
│
├── XRF elemental map
│   ├── Features are larger than expected sample structures
│   │   └── → Probe Blurring (beam spot larger than scan step)
│   │       Severity: Minor | Full guide: xrf_microscopy/probe_blurring.md
│   │
│   └── Sharp features in some elements, blurred in others
│       └── → Check peak overlap / fitting quality
│           Full guide: xrf_microscopy/peak_overlap.md
│
└── Ptychography reconstruction
    ├── Uniform contrast loss
    │   └── → Partial Coherence Effects
    │       Severity: Major | Full guide: ptychography/partial_coherence.md
    │
    └── Blurring varies across field of view
        └── → Position Error (probe positions inaccurate)
            Severity: Critical | Full guide: ptychography/position_error.md
```

**Quick check — motion detection:**

```python
import numpy as np
# Cross-correlate adjacent projections to detect shifts
from scipy.signal import correlate2d
shift_series = []
for i in range(projections.shape[0]-1):
    cc = np.correlate(projections[i].mean(0), projections[i+1].mean(0), 'full')
    shift_series.append(np.argmax(cc) - projections.shape[-1] + 1)
print(f"Max inter-projection shift: {max(np.abs(shift_series))} pixels")
```

---

## 6. Intensity/Value Anomalies

### Symptom: Unexpected intensity variations, plateaus, or jumps

```
What do you observe?
│
├── Sudden intensity jumps or drops in sinogram rows
│   └── → Beam Intensity Drop (beam dump, top-up failure)
│       Severity: Major | Full guide: tomography/beam_intensity_drop.md
│
├── Elemental concentrations plateau at high values (XRF)
│   ├── ICR/OCR ratio deviates significantly from 1.0
│   │   └── → Dead-Time Saturation
│   │       Severity: Critical | Full guide: xrf_microscopy/dead_time_saturation.md
│   │
│   └── ICR/OCR ratio is close to 1.0
│       └── → Self-Absorption (fluorescence re-absorbed by sample)
│           Severity: Major | Full guide: xrf_microscopy/self_absorption.md
│
├── Systematic intensity gradient across scan (XRF)
│   └── → I0 Normalization Issue (beam intensity not corrected)
│       Severity: Major | Full guide: xrf_microscopy/i0_normalization.md
│
├── Fluorescence amplitude appears dampened (XAS)
│   ├── Concentrated/thick sample?
│   │   └── → Self-Absorption in XAS
│   │       Severity: Major | Full guide: spectroscopy/self_absorption_xas.md
│   │
│   └── Dilute/thin sample
│       └── → Check harmonics contamination
│           Full guide: spectroscopy/harmonics_contamination.md
│
└── Negative values or impossible concentrations
    └── → Check normalization, dark-field subtraction, or reconstruction artifacts
        Full guide: tomography/flatfield_issues.md
```

**Quick check — I0 stability:**

```python
import numpy as np
i0 = np.array(i0_values)  # I0 monitor readings
drops = np.where(i0 < 0.8 * np.median(i0))[0]
print(f"I0 drops (>20%): {len(drops)} events at indices {drops[:5]}")
```

**Quick check — dead-time ratio:**

```python
import numpy as np
dt_ratio = ocr / icr  # output count rate / input count rate
print(f"Dead-time ratio range: {dt_ratio.min():.3f} - {dt_ratio.max():.3f}")
print(f"Pixels with >10% loss: {np.sum(dt_ratio < 0.9)} ({100*np.mean(dt_ratio < 0.9):.1f}%)")
```

---

## 7. Spectral Abnormalities

### Symptom: XAS/XANES spectrum looks wrong, features distorted or shifting

```
What's wrong with the spectrum?
│
├── Edge position shifts between successive scans
│   ├── Shift is monotonic (always in same direction)
│   │   └── → Radiation Damage (beam-induced chemical change)
│   │       Severity: Critical | Full guide: spectroscopy/radiation_damage.md
│   │
│   └── Shift is random/oscillatory
│       └── → Energy Calibration Drift (monochromator thermal drift)
│           Severity: Critical | Full guide: spectroscopy/energy_calibration_drift.md
│
├── White-line peak appears flattened or reduced
│   ├── Fluorescence mode with concentrated sample?
│   │   └── → Self-Absorption in XAS
│   │       Severity: Major | Full guide: spectroscopy/self_absorption_xas.md
│   │
│   └── Transmission mode
│       └── → Check for thickness effects or harmonics
│           Full guide: spectroscopy/harmonics_contamination.md
│
├── XANES features are damped or distorted
│   └── → Harmonics Contamination (higher-order reflections)
│       Severity: Major | Full guide: spectroscopy/harmonics_contamination.md
│
├── Individual scans look very different from average
│   └── → Outlier Spectra (glitch, beam dump during scan)
│       Severity: Minor | Full guide: spectroscopy/outlier_spectra.md
│
├── EXAFS amplitudes decrease with successive scans
│   └── → Radiation Damage (bond breaking, reduction)
│       Severity: Critical | Full guide: spectroscopy/radiation_damage.md
│
└── Spectrum has sharp spikes at specific energies
    └── → Monochromator Glitch or Bragg peak from crystal
        Check: spectroscopy/outlier_spectra.md
```

**Quick check — edge drift detection:**

```python
import numpy as np
# Track edge position (E0) across scans
e0_values = []
for scan in scans:
    deriv = np.gradient(scan['mu'], scan['energy'])
    e0_values.append(scan['energy'][np.argmax(deriv)])
drift = max(e0_values) - min(e0_values)
print(f"Edge drift: {drift:.2f} eV across {len(scans)} scans (> 0.5 eV is problematic)")
```

**Quick check — self-absorption test:**

```python
import numpy as np
# Compare white-line height in fluorescence vs expected
wl_height = np.max(normalized_mu) - 1.0  # normalized edge jump
print(f"White-line height: {wl_height:.2f}")
print(f"If < 0.3 for known compound with strong WL → likely self-absorption")
```

---

## 8. Boundary/Stitching Artifacts

### Symptom: Discontinuities at boundaries, missing or shifted features

```
Where is the discontinuity?
│
├── At tile boundaries in ptychography reconstruction
│   ├── Phase jumps between tiles
│   │   └── → Stitching Artifact (phase ambiguity)
│   │       Severity: Minor | Full guide: ptychography/stitching_artifact.md
│   │
│   └── Amplitude discontinuities
│       └── → Stitching Artifact (intensity normalization)
│           Severity: Minor | Full guide: ptychography/stitching_artifact.md
│
├── Features appear shifted or distorted in ptychography
│   └── → Position Error (probe position inaccuracy)
│       Severity: Critical | Full guide: ptychography/position_error.md
│
├── Missing slices or corrupted values after data conversion
│   ├── After HDF5 rechunking operation
│   │   └── → Rechunking Data Integrity Issue
│   │       Severity: Major | Full guide: cross_cutting/rechunking_data_integrity.md
│   │
│   └── After format conversion (TIFF→HDF5, etc.)
│       └── → Data conversion error — verify checksums
│           Full guide: cross_cutting/rechunking_data_integrity.md
│
└── Dimension mismatch or shape errors in arrays
    └── → Rechunking or data pipeline issue
        Full guide: cross_cutting/rechunking_data_integrity.md
```

**Quick check — stitching discontinuity:**

```python
import numpy as np
# Check intensity at tile boundary
boundary_col = tile_width  # column index of boundary
left = recon[:, boundary_col-5:boundary_col]
right = recon[:, boundary_col:boundary_col+5]
jump = np.mean(np.abs(left.mean(1) - right.mean(1)))
print(f"Mean boundary jump: {jump:.4f} (should be close to 0)")
```

**Quick check — data integrity after rechunking:**

```python
import numpy as np, hashlib
original = np.array(f_orig['data'])
rechunked = np.array(f_new['data'])
match = np.allclose(original, rechunked, atol=1e-7)
print(f"Data match after rechunking: {match}")
if not match:
    diff_slices = np.where(~np.all(np.isclose(original, rechunked, atol=1e-7), axis=(1,2)))[0]
    print(f"Mismatched slices: {diff_slices[:10]}")
```

---

## 9. Suspicious "Too-Good" Features

### Symptom: Results from DL/AI processing look too clean or contain unexpected detail

```
Did you apply a neural network (denoising, super-resolution, reconstruction)?
│
├── Yes — and results show features not visible in conventional processing
│   ├── Features are high-frequency details in low-SNR regions
│   │   └── → DL Hallucination (network generating false features)
│   │       Severity: Critical | Full guide: cross_cutting/dl_hallucination.md
│   │
│   ├── Features are periodic/repetitive patterns
│   │   └── → DL Hallucination (learned texture bias)
│   │       Severity: Critical | Full guide: cross_cutting/dl_hallucination.md
│   │
│   └── Features are present in ground truth / conventional recon too
│       └── → Likely genuine — but verify with uncertainty estimate
│           Full guide: cross_cutting/dl_hallucination.md
│
├── Yes — but results look identical to input
│   └── → Network may not be trained for this data distribution
│       Check model applicability and input preprocessing
│
└── No neural network was used
    └── → Re-examine other symptom categories above
```

**Quick check — DL residual analysis:**

```python
import numpy as np
# Compare DL output with conventional reconstruction
residual = dl_output - conventional_recon
residual_std = np.std(residual)
signal_std = np.std(conventional_recon)
print(f"Residual/signal ratio: {residual_std/signal_std:.3f}")
print(f"If > 0.1, DL is adding significant content — verify features carefully")
# Check for structured patterns in residual
from numpy.fft import fft2, fftshift
spectrum = np.abs(fftshift(fft2(residual)))
print(f"Residual spectrum peak/mean: {spectrum.max()/spectrum.mean():.1f} (> 10 suggests hallucination)")
```

---

---

## 10. Phase Map Discontinuities

### Symptom: Abrupt jumps or "cliffs" in retrieved phase images

```
Is the data from phase-contrast imaging, CDI, or interferometry?
│
├── Sharp lines of ±2π discontinuity in phase map
│   └── → Phase Wrapping (phase exceeds [-π, π] range)
│       Severity: Critical | Full guide: scattering_diffraction/phase_wrapping.md
│
├── Oscillating bands parallel to sharp edges
│   └── → Gibbs Ringing (Fourier truncation artifact)
│       Severity: Moderate | Full guide: medical_imaging/gibbs_ringing.md
│
└── Contrast reversals (features flip bright/dark at different defocus)
    └── → CTF Artifact (contrast transfer function zero-crossings)
        Severity: Critical | Full guide: electron_microscopy/ctf_artifact.md
```

---

## 11. Ghost / Residual from Previous Exposure

### Symptom: Faint image of previous sample or bright region persists in subsequent frames

```
Does the residual fade over time?
│
├── Yes — decays over seconds to minutes
│   └── → Afterglow / Persistence (scintillator or charge trapping)
│       Severity: Major | Full guide: cross_cutting/afterglow_persistence.md
│
├── Persists indefinitely at same pixel positions
│   └── → Detector Burn-in or Dead/Hot Pixel
│       Full guide: cross_cutting/detector_common_issues.md
│
└── Affects only low-q region in scattering pattern
    └── → Parasitic Scattering (from slits/windows, not sample)
        Severity: Critical | Full guide: scattering_diffraction/parasitic_scattering.md
```

---

## Cross-Reference: All 47 Noise Types by Symptom

| Symptom Category | Noise Types Covered |
|-----------------|-------------------|
| Circular/ring patterns | [Ring artifact](tomography/ring_artifact.md), [Rotation center error](tomography/rotation_center_error.md), [Flat-field issues](tomography/flatfield_issues.md), [Ice rings](scattering_diffraction/ice_rings.md) |
| Isolated bright/dark spots | [Zinger](tomography/zinger.md), [Dead/hot pixel (XRF)](xrf_microscopy/dead_hot_pixel.md), [Detector issues](cross_cutting/detector_common_issues.md), [Cosmic ray/outlier](cross_cutting/cosmic_ray_outlier.md) |
| Streak/stripe patterns | [Streak artifact](tomography/streak_artifact.md), [Sparse-angle](tomography/sparse_angle_artifact.md), [Scan stripe](xrf_microscopy/scan_stripe.md), [Beam intensity drop](tomography/beam_intensity_drop.md), [Ring artifact](tomography/ring_artifact.md), [I0 normalization](xrf_microscopy/i0_normalization.md), [Stitching artifact](ptychography/stitching_artifact.md), [Metal artifact](medical_imaging/metal_artifact.md), [Detector gaps](scattering_diffraction/detector_gaps_parallax.md) |
| Overall graininess/noise | [Low-dose noise](tomography/low_dose_noise.md), [Photon counting noise](xrf_microscopy/photon_counting_noise.md), [Statistical noise EXAFS](spectroscopy/statistical_noise_exafs.md), [Shot noise low-dose (EM)](electron_microscopy/shot_noise_low_dose.md) |
| Blurring/loss of detail | [Motion artifact](tomography/motion_artifact.md), [Probe blurring](xrf_microscopy/probe_blurring.md), [Partial coherence](ptychography/partial_coherence.md), [Position error](ptychography/position_error.md), [Rotation center error](tomography/rotation_center_error.md), [Drift & vibration](electron_microscopy/drift_vibration.md), [Partial volume effect](medical_imaging/partial_volume_effect.md) |
| Intensity/value anomalies | [Beam intensity drop](tomography/beam_intensity_drop.md), [Dead-time saturation](xrf_microscopy/dead_time_saturation.md), [Self-absorption (XRF)](xrf_microscopy/self_absorption.md), [I0 normalization](xrf_microscopy/i0_normalization.md), [Self-absorption (XAS)](spectroscopy/self_absorption_xas.md), [Harmonics](spectroscopy/harmonics_contamination.md), [Beam hardening](medical_imaging/beam_hardening.md), [Bias field](medical_imaging/bias_field.md), [Charging artifact](electron_microscopy/charging_artifact.md) |
| Spectral abnormalities | [Energy calibration drift](spectroscopy/energy_calibration_drift.md), [Harmonics contamination](spectroscopy/harmonics_contamination.md), [Self-absorption (XAS)](spectroscopy/self_absorption_xas.md), [Radiation damage](spectroscopy/radiation_damage.md), [Outlier spectra](spectroscopy/outlier_spectra.md) |
| Boundary/stitching artifacts | [Stitching artifact](ptychography/stitching_artifact.md), [Position error](ptychography/position_error.md), [Rechunking integrity](cross_cutting/rechunking_data_integrity.md), [Truncation artifact](medical_imaging/truncation_artifact.md), [Detector gaps](scattering_diffraction/detector_gaps_parallax.md) |
| Suspicious "too-good" features | [DL hallucination](cross_cutting/dl_hallucination.md) |
| Phase discontinuities | [Phase wrapping](scattering_diffraction/phase_wrapping.md), [Gibbs ringing](medical_imaging/gibbs_ringing.md), [CTF artifact](electron_microscopy/ctf_artifact.md) |
| Ghost/residual images | [Afterglow/persistence](cross_cutting/afterglow_persistence.md), [Parasitic scattering](scattering_diffraction/parasitic_scattering.md), [Contamination buildup](electron_microscopy/contamination_buildup.md) |
| Sample/beam damage | [Radiation damage (spectroscopy)](spectroscopy/radiation_damage.md), [Radiation damage (MX)](scattering_diffraction/radiation_damage_crystallography.md), [Contamination buildup](electron_microscopy/contamination_buildup.md) |
| Scatter/background | [Scatter artifact](medical_imaging/scatter_artifact.md), [Parasitic scattering](scattering_diffraction/parasitic_scattering.md), [Beam hardening](medical_imaging/beam_hardening.md) |

---

## Still Not Sure?

- Browse all 47 types in the [Summary Table](summary_table.md)
- Check the [Noise Estimation Methods](cross_cutting/noise_estimation_methods.md) for cross-domain characterization tools (NPS, DQE, MAD)
- Check the [Detector Common Issues](cross_cutting/detector_common_issues.md) guide for hardware-related problems
- Ask: **Is the problem in raw data or after processing?** Raw data issues → instrumental/statistical. Post-processing issues → computational/systematic.
