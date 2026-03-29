# Charging Artifact (SEM)

## Classification

| Attribute | Value |
|-----------|-------|
| **Modality** | SEM |
| **Noise Type** | Instrumental |
| **Severity** | Major |
| **Frequency** | Common |
| **Detection Difficulty** | Easy |
| **Origin Domain** | Scanning Electron Microscopy |

## Description

Charging artifacts occur in SEM when non-conductive or poorly grounded samples accumulate electric charge from the incident electron beam. The built-up charge creates local electric fields that deflect incoming and emitted electrons, producing bright patches, image distortion, image drift, and anomalous contrast. The effect is dynamic — charging builds up during scanning and can cause sudden brightness shifts or image "jumping."

## Root Cause

- Non-conductive sample cannot dissipate incident electron charge
- Charge accumulation → local surface potential builds up (can reach hundreds of volts)
- Electric field deflects primary beam → geometric distortion
- Electric field alters secondary electron emission and collection → anomalous brightness
- Worse with: higher beam current, higher accelerating voltage, poor grounding, insulating samples

## Quick Diagnosis

```python
import numpy as np

def detect_charging(image, direction='horizontal'):
    """Detect charging by brightness shift along scan direction."""
    if direction == 'horizontal':
        # Check for progressive brightening along scan lines
        row_means = image.mean(axis=1)
    else:
        row_means = image.mean(axis=0)
    # Charging causes systematic intensity trend
    trend = np.polyfit(np.arange(len(row_means)), row_means, 1)
    print(f"Intensity slope: {trend[0]:.4f} (non-zero suggests charging)")
    # Check for sudden jumps
    diffs = np.abs(np.diff(row_means))
    jumps = np.where(diffs > 3 * np.std(diffs))[0]
    if len(jumps) > 0:
        print(f"Sudden intensity jumps at lines: {jumps}")
    return trend[0], jumps
```

## Detection Methods

### Visual Indicators

- Bright, saturated patches on insulating regions
- Image "streaking" or brightness shifts between scan lines
- Geometric distortion (features appear shifted or warped)
- Dynamic changes between successive scans (charging builds up over time)
- Anomalously bright edges on insulating particles

### Automated Detection

```python
import numpy as np

def charging_map(image_sequence):
    """Compare sequential frames to detect progressive charging."""
    diffs = []
    for i in range(1, len(image_sequence)):
        diff = image_sequence[i].astype(float) - image_sequence[i-1].astype(float)
        diffs.append(diff)
    cumulative_drift = np.abs(np.array(diffs)).mean(axis=0)
    return cumulative_drift  # High values = charging regions
```

## Correction Methods

### Traditional Approaches (Prevention-Focused)

1. **Conductive coating:** Sputter-coat with Au, Pt, C, or Ir (5-20 nm)
2. **Low-voltage SEM:** Operate at charge-balance voltage (0.5-2 kV) where input = output electrons
3. **Variable-pressure / ESEM:** Gas molecules neutralize surface charge
4. **Charge compensation:** Flood gun or nitrogen gas injection
5. **Frame averaging with charge dissipation:** Allow settling time between frames

### Post-Acquisition Correction

```python
def destripe_charging(image, sigma=50):
    """Remove scan-line charging stripes via row normalization."""
    from scipy.ndimage import gaussian_filter1d
    row_means = image.mean(axis=1)
    smooth_baseline = gaussian_filter1d(row_means, sigma)
    correction = smooth_baseline / row_means
    corrected = image * correction[:, np.newaxis]
    return corrected
```

## Key References

- **Cazaux (2004)** — "Charging in scanning electron microscopy" — comprehensive physics review
- **Joy & Joy (1996)** — "Low voltage SEM" — charge-balance imaging
- **Thiel & Toth (2005)** — "Secondary electron contrast in ESEM" — environmental SEM approach

## Relevance to Synchrotron Data

| Scenario | Relevance |
|----------|-----------|
| STXM / X-PEEM | Photoelectron emission from insulators causes analogous charging |
| XPS at synchrotron | Surface charging shifts binding energies |
| Correlative SEM + synchrotron | Understanding SEM charging aids multimodal registration |
| In-situ electrochemistry | Charge buildup on electrodes affects both SEM and X-ray |

## Related Resources

- [Scan stripe](../xrf_microscopy/scan_stripe.md) — Similar row-by-row intensity variations
- [Beam intensity drop](../tomography/beam_intensity_drop.md) — Time-dependent intensity changes
