# Crystallography Data Formats

## Raw Data Formats

### HDF5 (EIGER Native)

The EIGER detector family produces data in HDF5 format with the following structure:

```
dataset_master.h5
├── /entry/
│   ├── /instrument/
│   │   └── /detector/
│   │       ├── /description = "Eiger 16M"
│   │       ├── /x_pixel_size = 0.000075  # 75 µm
│   │       ├── /y_pixel_size = 0.000075
│   │       ├── /sensor_thickness = 0.00045  # 450 µm Si
│   │       ├── /count_time = 0.01  # seconds
│   │       ├── /frame_time = 0.01
│   │       ├── /beam_center_x = 2073.5  # pixels
│   │       ├── /beam_center_y = 2167.3
│   │       ├── /detector_distance = 0.250  # meters
│   │       └── /detectorSpecific/
│   │           ├── /nimages = 3600
│   │           └── /ntrigger = 1
│   └── /data/
│       ├── data_000001 -> (external link to data file)
│       ├── data_000002 -> ...
│       └── ...
│
dataset_data_000001.h5
├── /entry/data/
│   └── /data [shape: (N_frames, 4362, 4148), dtype: uint32]
│       # Compressed with bitshuffle-LZ4
```

### CBF (Crystallographic Binary File)

Legacy format still used by some detectors (PILATUS, ADSC):

```
# CBF Header (ASCII)
_array_data.header_convention "PILATUS_1.2"
_diffrn_radiation_wavelength 0.97918
_diffrn_scan_axis omega
_diffrn_scan_frame_axis_offset 0.00
...

# Binary section (compressed image data)
--CIF-BINARY-FORMAT-SECTION--
Content-Type: application/octet-stream
X-Binary-Size: 2073600
...
```

## Metadata Structure

### Essential Metadata

| Parameter | Description | Example Value |
|-----------|-------------|---------------|
| `wavelength` | X-ray wavelength (Å) | 0.97918 |
| `detector_distance` | Crystal-detector distance (mm) | 250.0 |
| `beam_center_x/y` | Beam position on detector (pixels) | 2073.5, 2167.3 |
| `oscillation_range` | Rotation per frame (°) | 0.1 |
| `exposure_time` | Integration time per frame (s) | 0.01 |
| `phi/omega` | Goniometer angles (°) | 0.0 – 180.0 |
| `space_group` | Crystal symmetry | P21 |
| `unit_cell` | a, b, c, α, β, γ | 45.2, 67.8, 89.1, 90, 95.3, 90 |

## Python Data Loading

```python
import h5py
import numpy as np

# Load EIGER HDF5 master file
with h5py.File('dataset_master.h5', 'r') as f:
    # Read detector metadata
    det = f['/entry/instrument/detector']
    wavelength = f['/entry/instrument/beam/incident_wavelength'][()]
    distance = det['detector_distance'][()]
    beam_x = det['beam_center_x'][()]
    beam_y = det['beam_center_y'][()]

    # Read diffraction data (from linked data files)
    data = f['/entry/data/data_000001'][:]  # shape: (N, 4362, 4148)
    print(f"Loaded {data.shape[0]} frames, {data.shape[1]}×{data.shape[2]} pixels")

# For CBF files using fabio
import fabio
img = fabio.open('frame_001.cbf')
data = img.data  # numpy array
header = img.header  # dict of metadata
```

## Processed Data Formats

### MTZ (Merged Reflection File)
- Standard format for processed crystallographic data
- Contains: Miller indices (h,k,l), intensities (I), standard deviations (σ)
- Used by: CCP4 suite, PHENIX, SHELX

### PDB/mmCIF (Atomic Coordinates)
- Final refined atomic model
- PDB: legacy fixed-column format (being phased out)
- mmCIF: modern dictionary-based format (now standard)
- Deposited to Protein Data Bank (wwPDB)

## Data Processing Pipeline

```
Raw frames (HDF5/CBF)
    │
    ├─→ Indexing (determine crystal orientation & unit cell)
    │       Tools: DIALS, XDS, MOSFLM
    │
    ├─→ Integration (extract spot intensities)
    │       Tools: DIALS, XDS
    │
    ├─→ Scaling & Merging (combine symmetry-related observations)
    │       Tools: AIMLESS, XSCALE
    │
    ├─→ Phasing (determine phases)
    │       Methods: MR (Molecular Replacement), SAD/MAD, direct methods
    │       Tools: PHASER, SHELXD, AutoSol
    │
    ├─→ Model building (trace polypeptide chain)
    │       Tools: ARP/wARP, Buccaneer, AlphaFold (initial model)
    │
    └─→ Refinement (optimize model against data)
            Tools: REFMAC5, phenix.refine
            Output: PDB/mmCIF + MTZ
```

## Preprocessing Requirements

1. **Bad pixel masking**: Mask dead/hot pixels on detector
2. **Flat-field correction**: Normalize pixel response (usually applied by detector)
3. **Geometry refinement**: Accurate detector distance, beam center, rotation axis
4. **Ice ring removal**: Mask resolution shells contaminated by ice diffraction (if cryo)
5. **Radiation damage assessment**: Monitor intensity decay during data collection
