# Detector Afterglow / Persistence (Lag)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | Cross-cutting (Tomography, Diffraction, All modalities) |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |
| **Origin Domain** | Astronomy / Medical Imaging / Synchrotron |

## Description

Afterglow (persistence, lag, image retention) occurs when a detector retains residual signal from previous exposures, contaminating subsequent frames. In scintillator-based detectors, afterglow comes from slow phosphorescence decay; in direct-conversion sensors (CdTe/CZT), it comes from charge trapping. In astronomy, persistence from bright stars can contaminate hours of subsequent exposures.

**Multi-domain relevance:** Extensively studied in astronomy (HST WFC3 IR persistence), medical imaging (flat panel CT), and now increasingly important in synchrotron fast-acquisition experiments.

## Root Cause

- **Scintillator afterglow:** Slow luminescence decay (ms to seconds) in phosphor materials (CsI, GOS, LYSO)
- **Charge trapping:** Carriers trapped in defect states release slowly (CdTe, CdZnTe, Si deep traps)
- **Infrared persistence:** In astronomy, bright sources saturate traps → slow release for hours
- Multi-exponential decay: fast component (~ms) + slow component (seconds to minutes)
- Worse with: higher previous exposure, certain scintillator/sensor materials, low temperature (slower trap release)

## Quick Diagnosis

```python
import numpy as np

def measure_afterglow(dark_frames_after_bright, time_stamps):
    """Measure afterglow decay from dark frames acquired after bright exposure."""
    mean_signals = [frame.mean() for frame in dark_frames_after_bright]
    # Fit bi-exponential decay
    from scipy.optimize import curve_fit
    def biexp(t, a1, tau1, a2, tau2, offset):
        return a1 * np.exp(-t/tau1) + a2 * np.exp(-t/tau2) + offset
    t = np.array(time_stamps) - time_stamps[0]
    try:
        popt, _ = curve_fit(biexp, t, mean_signals,
                           p0=[mean_signals[0]*0.9, 0.1, mean_signals[0]*0.1, 1.0, 0],
                           maxfev=5000)
        print(f"Fast decay: τ₁ = {popt[1]:.3f}s (amplitude {popt[0]:.1f})")
        print(f"Slow decay: τ₂ = {popt[3]:.3f}s (amplitude {popt[2]:.1f})")
    except RuntimeError:
        print("Bi-exponential fit failed — try simple exponential")
    return mean_signals
```

## Detection Methods

### Visual Indicators

- Ghost image of previous bright exposure visible in subsequent dark/dim frames
- Systematic bias in dynamic sequences (CT projections, time-resolved diffraction)
- Intensity that decays over successive frames rather than staying constant
- In tomography: "shadowing" from previous angular positions

### Automated Detection

```python
import numpy as np

def detect_persistence_in_sequence(frames, expected_uniform=True):
    """Detect persistence by checking for decaying residual signal."""
    if expected_uniform:
        means = [f.mean() for f in frames]
        # Check for exponential decay trend
        if len(means) >= 3:
            # Monotonically decreasing = persistence
            diffs = np.diff(means)
            if np.all(diffs < 0) and abs(diffs[0]) > 2 * abs(diffs[-1]):
                print("⚠ Persistence detected: decaying residual signal")
                return True
    return False
```

## Correction Methods

### Prevention

1. **Material selection:** Choose low-afterglow scintillators (GGG, LuAG vs CsI)
2. **Flush frames:** Acquire and discard "dummy" frames after bright exposures
3. **Temporal spacing:** Allow adequate dead time between frames
4. **Bias light:** Constant illumination to fill traps and reduce persistence

### Post-Acquisition

1. **Exponential decay subtraction:** Model afterglow decay, subtract from subsequent frames
2. **Recursive correction:** Each frame corrects based on weighted sum of all previous frames
3. **Dark current monitoring:** Track baseline drift indicative of afterglow

```python
def correct_afterglow_recursive(frames, decay_fraction=0.02):
    """Simple recursive afterglow correction."""
    corrected = [frames[0].copy()]
    for i in range(1, len(frames)):
        afterglow = decay_fraction * corrected[i-1]
        corrected.append(frames[i] - afterglow)
    return corrected
```

### Astronomy Tools (Transferable)

- **HST WFC3 persistence model** — Empirical per-pixel persistence correction
- **AstroPy ccdproc** — Persistence-aware image combination
- **DRAGONS (Gemini)** — Persistence masking in near-IR pipeline

## Key References

- **Long et al. (2015)** — "Persistence in the WFC3 IR Detector" — comprehensive HST model
- **Malik et al. (2020)** — "Afterglow characterization of scintillator detectors for synchrotron CT"
- **Pani et al. (2004)** — "CsI(Tl) afterglow characterization for medical imaging"
- **Prell et al. (2009)** — "Lag correction for flat panel detectors in CT"

## Relevance to Synchrotron Data

| Scenario | Relevance |
|----------|-----------|
| Fast tomography | Short frame intervals don't allow afterglow decay |
| Dynamic experiments | Time-resolved data contaminated by previous states |
| Stroboscopic imaging | Periodic signal corrupted by persistence from bright phases |
| CdTe detectors (Eiger CdTe) | Charge trapping at high flux → ghosting |
| Scintillator micro-CT | GOS, LYSO afterglow at fast rotation speeds |

## Related Resources

- [Detector common issues](detector_common_issues.md) — General detector characterization
- [Flat-field issues](../tomography/flatfield_issues.md) — Persistence corrupts flat-field measurements
- [Beam intensity drop](../tomography/beam_intensity_drop.md) — Both cause frame-to-frame intensity variation
