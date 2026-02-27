# Reconstruction Methods

## Overview

Reconstruction is the computational process of recovering an image or volume from
indirect measurements. In synchrotron science, this encompasses tomographic
reconstruction (projections → 3D volume) and phase retrieval (diffraction patterns →
complex image).

## Method Landscape

```
Reconstruction Methods
├── Classical Analytical
│   ├── FBP (Filtered Back Projection)
│   └── Gridrec (FFT-based FBP)
│
├── Classical Iterative
│   ├── SIRT, MLEM, CGLS
│   └── Model-based (MBIR)
│
├── GPU-Accelerated Classical
│   └── TomocuPy (20-30× speedup)
│
├── DL Post-Processing
│   └── FBPConvNet (FBP → CNN cleanup)
│
├── Learned Iterative
│   └── Unrolled optimization with learned components
│
└── Neural Representations
    └── INR (continuous coordinate → value mapping)
```

## Directory Contents

| File | Content |
|------|---------|
| [tomocupy.md](tomocupy.md) | GPU-accelerated reconstruction (CuPy-based) |
| [ptychonet.md](ptychonet.md) | CNN-based ptychographic phase retrieval |
| [inr_dynamic.md](inr_dynamic.md) | Implicit Neural Representations for dynamic tomography |
