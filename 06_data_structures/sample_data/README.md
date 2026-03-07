# Sample Data Links

## Overview

This page provides links to publicly available synchrotron datasets suitable for
testing analysis pipelines, learning data formats, and benchmarking algorithms.
All datasets listed here are freely accessible under open-data licenses.

## Tomography Datasets

### TomoBank

The **TomoBank** repository at Argonne National Laboratory provides curated tomography
datasets with full metadata in Data Exchange HDF5 format.

- **Repository**: [https://tomobank.readthedocs.io/](https://tomobank.readthedocs.io/)
- **Format**: Data Exchange HDF5
- **Includes**: Projections, flat/dark fields, theta arrays, metadata

| Dataset ID | Description | Size | Beamline |
|-----------|-------------|------|----------|
| tomo_00001 | Shale rock core | 12 GB | APS 2-BM |
| tomo_00002 | Dorsal horn (neuroscience) | 8 GB | APS 2-BM |
| tomo_00003 | Volcanicite | 5 GB | APS 2-BM |
| tomo_00004 | Tooth dentin | 6 GB | APS 32-ID |
| tomo_00005 | Fuel cell electrode | 3 GB | APS 32-ID |
| tomo_00006 | Metal foam | 4 GB | APS 2-BM |
| tomo_00007 | Wet cement hydration (time series) | 45 GB | APS 2-BM |

**Download example**:
```bash
# Using Globus CLI
globus transfer <tomobank-endpoint-id>:/tomo_00001/ <local-endpoint>:/data/tomo/
```

### Dynamic Tomography

- **4D Tomography of aluminum alloy solidification**
  - DOI: [10.13139/OLCF/1347542](https://doi.org/10.13139/OLCF/1347542)
  - 600 time steps, 1200 projections each
  - Format: Data Exchange HDF5

## XRF Microscopy Datasets

### ROI-Finder Sample Data

The **ROI-Finder** tool includes sample XRF datasets for testing the automated
region-of-interest detection pipeline.

- **Repository**: [https://github.com/arshadzahangirchowdhury/ROI-Finder](https://github.com/arshadzahangirchowdhury/ROI-Finder)
- **Format**: MAPS HDF5
- **Includes**: Fitted elemental maps, raw spectra, scan coordinates

| Dataset | Description | Elements | Size |
|---------|-------------|----------|------|
| roi_finder_sample_1 | Plant root cross-section | Fe, Zn, Ca, K | 200 MB |
| roi_finder_sample_2 | Soil aggregate | Fe, Mn, Cu, Zn, As | 350 MB |

### IXRF Sample Spectra

- **IXRF Standards**: [https://www.ixrfsystems.com/resources/](https://www.ixrfsystems.com/resources/)
- Reference XRF spectra for calibration and energy verification

## NeXus Example Files

The NeXus international standard provides example HDF5/NeXus files for testing readers:

- **Repository**: [https://github.com/nexusformat/exampledata](https://github.com/nexusformat/exampledata)
- **Format**: NeXus/HDF5
- **Includes**: Examples for various NXclass types

| File | NX Class | Description |
|------|----------|-------------|
| `chopper.nxs` | NXentry | Time-of-flight chopper spectrometer |
| `powder.nxs` | NXentry | Powder diffraction pattern |
| `sans.nxs` | NXentry | Small-angle neutron scattering |
| `tomo.nxs` | NXentry | Basic tomography example |

## Coherent Imaging / Ptychography

### CXIDB (Coherent X-ray Imaging Data Bank)

- **Repository**: [https://www.cxidb.org/](https://www.cxidb.org/)
- **Format**: CXI (HDF5-based)
- **Includes**: Diffraction patterns, scan positions, reconstructions

| Entry | Description | Resolution | Size |
|-------|-------------|-----------|------|
| CXIDB-22 | Gold nanoparticle test pattern | 15 nm | 1.2 GB |
| CXIDB-54 | Integrated circuit (IC) imaging | 20 nm | 3.5 GB |
| CXIDB-88 | Biological cell | 30 nm | 2.1 GB |
| CXIDB-110 | Battery electrode | 25 nm | 4.0 GB |

### PtychoNN Training Data

- **Repository**: [https://github.com/mcherukara/PtychoNN](https://github.com/mcherukara/PtychoNN)
- Training datasets for neural network ptychographic reconstruction

### edgePtychoNN Experimental Data (Babu et al. 2023)

- **Zenodo DOI**: [10.5281/zenodo.8121606](https://zenodo.org/records/8121606)
- **Code**: [https://github.com/vbanakha/edgePtychoNN](https://github.com/vbanakha/edgePtychoNN)
- **Format**: NumPy `.npy` arrays + `positions.csv`
- **License**: CC BY 4.0
- **Includes**: 121 spiral scans × 963 diffraction images (128×128 pixels), collected
  with a Medipix3 detector at 50 nm step size / 0.4 s exposure
- **Paper**: "Deep learning at the edge enables real-time streaming ptychographic imaging"
  ([10.1038/s41467-023-41496-z](https://doi.org/10.1038/s41467-023-41496-z))

**Quick-start example**:
```python
import numpy as np
import matplotlib.pyplot as plt

# Load a single scan (963 diffraction patterns, 128x128 each)
scan = np.load("diff_scan_810.npy")
print(f"Scan shape: {scan.shape}")  # (963, 128, 128)

# Visualize a single diffraction pattern (log scale)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for i, idx in enumerate([0, 480, 962]):
    axes[i].imshow(np.log1p(scan[idx]), cmap="viridis")
    axes[i].set_title(f"Pattern #{idx}")
plt.suptitle("Ptychographic Diffraction Patterns")
plt.tight_layout()
plt.show()

# Load scan positions
import pandas as pd
positions = pd.read_csv("positions.csv")
print(f"Positions: {len(positions)} points")
```

## Spectroscopy Datasets

### XAS Reference Spectra

- **IXAS Spectra Database**: [https://xaslib.xrayabsorption.org/](https://xaslib.xrayabsorption.org/)
  - Community-contributed reference XANES and EXAFS spectra
  - Multiple edges (K, L, M) for most elements
  - Formats: plain text, Athena project files

### XANES Standard Libraries

| Library | Elements | Edges | Format |
|---------|----------|-------|--------|
| APS XANES Library | Fe, Mn, S, P, As | K-edge | ASCII |
| ESRF ID21 Standards | Various | K, L | HDF5/NeXus |
| Diamond I18 Standards | Transition metals | K-edge | Athena (.prj) |

## Scientific Data Papers

Published datasets accompanying peer-reviewed papers:

| Paper | Dataset Type | DOI |
|-------|-------------|-----|
| De Carlo et al. (2014) "Scientific Data Exchange" | Tomography reference | [10.1088/0957-0233/25/11/115501](https://doi.org/10.1088/0957-0233/25/11/115501) |
| Gursoy et al. (2014) "TomoPy benchmarks" | Tomography + phantom | [10.1107/S1600577514013939](https://doi.org/10.1107/S1600577514013939) |
| Vogt (2003) "MAPS paper" | XRF reference | [10.1051/jp4:20030218](https://doi.org/10.1051/jp4:20030218) |
| Cherukara et al. (2020) "PtychoNN" | Ptychography training | [10.1063/5.0013065](https://doi.org/10.1063/5.0013065) |

## Downloading Large Datasets

For datasets larger than a few GB, we recommend:

1. **Globus Transfer** -- Best for multi-GB files from DOE facilities
   ```bash
   pip install globus-cli
   globus login
   globus transfer <src-endpoint>:<src-path> <dst-endpoint>:<dst-path>
   ```

2. **wget/curl** -- For smaller files from web-accessible repositories
   ```bash
   wget -c https://tomobank.readthedocs.io/data/tomo_00001.h5
   ```

3. **Petrel** -- Argonne's data sharing platform via Globus
   - [https://petrel.alcf.anl.gov/](https://petrel.alcf.anl.gov/)

## Related Resources

- [HDF5 structure guides](../hdf5_structure/)
- [EDA notebooks](../eda/notebooks/)
- [Data pipeline architecture](../../07_data_pipeline/)
