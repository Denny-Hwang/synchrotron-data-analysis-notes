# Contamination Buildup

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | SEM / TEM |
| **Noise Type** | Systematic |
| **Severity** | Moderate |
| **Frequency** | Common |
| **Detection Difficulty** | Moderate |
| **Origin Domain** | Electron Microscopy |

## Description

Contamination buildup is the deposition of carbonaceous material on the sample surface under electron beam irradiation. Hydrocarbon molecules (from vacuum pump oil, sample preparation, or residual gases) adsorb on the surface and are polymerized/cracked by the electron beam, forming a growing carbon film. This obscures surface details in SEM, adds amorphous background in TEM, and progressively degrades image quality over time.

## Root Cause

- Hydrocarbon molecules present in vacuum chamber (residual gas, pump backstreaming)
- Electron beam cracks/polymerizes hydrocarbons → insoluble carbon deposit
- Deposition rate ∝ beam current × hydrocarbon partial pressure
- Focused beam → contamination spot; scanning → contamination film
- In TEM: contamination creates amorphous layer that scatters electrons and reduces contrast

## Quick Diagnosis

```python
import numpy as np

def detect_contamination_spot(image_sequence):
    """Detect growing contamination from sequential images of same area."""
    # Contamination causes progressive darkening at beam position
    mean_intensities = [img.mean() for img in image_sequence]
    # Monotonic decrease suggests contamination buildup
    diffs = np.diff(mean_intensities)
    if np.all(diffs < 0):
        rate = abs(np.mean(diffs))
        print(f"Progressive signal loss detected: {rate:.2f}/frame — likely contamination")
    return mean_intensities
```

## Correction Methods

### Prevention (Primary)

1. **Plasma cleaning:** Clean sample and holder with Ar/O2 plasma before loading
2. **Beam shower:** Pre-irradiate area with broad beam to remove hydrocarbons
3. **Cryo-shielding (cold finger):** Cold trap near sample condenses hydrocarbons
4. **Improved vacuum:** Better pumping, oil-free pumps, baking
5. **Clean sample preparation:** Minimize organic contamination

### Post-Acquisition

- Compare early vs late frames and discard contaminated frames
- Subtract estimated contamination background (low-frequency component)

## Key References

- **Hren (1979)** — "Barriers to AEM: contamination and etching" — foundational description
- **Egerton et al. (2004)** — "Radiation damage in the TEM and SEM"
- **Mitchell (2015)** — "Contamination mitigation strategies for SEM"

## Relevance to Synchrotron Data

| Scenario | Relevance |
|----------|-----------|
| Carbon K-edge STXM/NEXAFS | Contamination adds spurious carbon signal |
| In-situ experiments | Hydrocarbon contamination from reaction gases |
| Correlative EM + synchrotron | Contamination from EM affects subsequent synchrotron measurements |
| XPS / soft X-ray spectroscopy | Carbon contamination layer attenuates signal |

## Related Resources

- [Radiation damage](../spectroscopy/radiation_damage.md) — Both are beam-induced sample modification
- [Beam intensity drop](../tomography/beam_intensity_drop.md) — Progressive signal changes during experiment
