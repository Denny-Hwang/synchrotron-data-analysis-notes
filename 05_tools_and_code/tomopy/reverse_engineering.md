# TomoPy -- Module Structure and Algorithm Notes

## Top-Level Package Layout

```
tomopy/
  io/           # Data I/O (HDF5, TIFF, SXM, etc.)
  prep/         # Preprocessing (normalize, phase, stripe, alignment)
  recon/        # Reconstruction algorithms
  misc/         # Utilities (morph, corr, phantom generation)
  sim/          # Forward projection / simulation
  util/         # Internal helpers, dtype handling, multiprocessing
```

## Module Details

### `tomopy.io`

- `exchange.py` -- reads/writes the APS Data Exchange HDF5 schema.
- `reader.py` -- generic HDF5 / TIFF stack reader with lazy loading.
- `writer.py` -- TIFF stack and HDF5 writer with optional compression.

Beamline-specific readers (e.g., `read_aps_32id`, `read_aps_2bm`) are thin
wrappers that map beamline HDF5 paths to the standard exchange layout.

### `tomopy.prep`

| Submodule | Key functions |
|-----------|---------------|
| `normalize.py` | `normalize()`, `normalize_bg()` -- flat/dark field correction |
| `phase.py` | `retrieve_phase()` -- Paganin single-distance phase retrieval |
| `stripe.py` | `remove_stripe_fw()`, `remove_stripe_ti()` -- Fourier / Titarenko |
| `alignment.py` | `align_seq()`, `find_center()`, `find_center_vo()` |

### `tomopy.recon`

The `recon()` dispatcher selects the back-end based on the `algorithm`
parameter.

| Algorithm key | Method | Back-end |
|---------------|--------|----------|
| `gridrec` | Fourier gridding FBP | C extension |
| `fbp` | Direct filtered back-projection | C extension |
| `art` | Algebraic Reconstruction Technique | C extension |
| `sirt` | Simultaneous Iterative | C extension |
| `mlem` / `osem` | Maximum-Likelihood EM | C extension |
| `tv` | Total-variation regularised | C extension |
| `astra` | Any ASTRA algorithm | Plugin (GPU) |

Reconstruction is parallelised over sinogram rows via OpenMP threads in the
C layer. Python-side, `multiprocessing` is used for chunk-level parallelism on
very large datasets.

### `tomopy.misc`

- `phantom.py` -- Shepp-Logan and custom phantom generators.
- `corr.py` -- median filtering, Gaussian smoothing.
- `morph.py` -- binary morphological operations on reconstructed volumes.

## C Extensions

Performance-critical loops are implemented in C and exposed via `ctypes`.
Source files live under `src/` and are compiled at install time by
`setup.py` / `meson.build`.

Key C source files:

| File | Purpose |
|------|---------|
| `gridrec.c` | Fourier-space gridding reconstruction |
| `project.c` | Forward / back-projection (ray-driven) |
| `art.c` | ART iteration kernel |
| `sirt.c` | SIRT iteration kernel |
| `utils.c` | Padding, array transposition helpers |

## Data Flow Summary

```
Raw HDF5  -->  io.reader  -->  prep.normalize  -->  prep.phase (optional)
  -->  prep.stripe  -->  recon.recon()  -->  io.writer  -->  TIFF / HDF5
```

## Extension Points

- Custom algorithms can be registered via `tomopy.recon.algorithm.register()`.
- ASTRA integration passes data to `astra.create_sino3d_gpu()` and retrieves
  reconstructed slices, avoiding unnecessary copies when possible.
