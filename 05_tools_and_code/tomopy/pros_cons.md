# TomoPy vs TomocuPy -- Comparison

## Strengths of TomoPy

- **Algorithm breadth** -- supports analytical (gridrec, FBP), iterative
  (ART, SIRT, MLEM, OSEM), and regularised (TV) methods, covering a wide range
  of reconstruction problems.
- **Mature ecosystem** -- extensive documentation, published tutorials,
  TomoPy-cli, and integration with TomoBank test datasets.
- **CPU-only operation** -- runs on any machine, from laptops to HPC login
  nodes, with no GPU requirement.
- **Plugin architecture** -- ASTRA Toolbox and UFO back-ends can be enabled
  for GPU acceleration without changing user code.
- **Large community** -- active GitHub issues, mailing list, and regular
  releases since 2014.
- **Beamline integration** -- native readers for APS, ALS, SLS, and other
  synchrotron formats.

## Weaknesses of TomoPy

- **Speed on large datasets** -- CPU-bound gridrec can take several minutes
  on 2048^2 datasets versus ~25 seconds on GPU with TomocuPy.
- **C extension build complexity** -- compiling from source requires a
  working C compiler and can fail on non-standard platforms.
- **Memory pressure** -- large datasets may exceed available RAM when the
  full sinogram stack is loaded at once.
- **Threading overhead** -- OpenMP and Python multiprocessing introduce
  scheduling overhead that limits scaling beyond ~32 cores.

## When to Choose TomoPy

1. You need iterative reconstruction (SIRT, MLEM, TV).
2. No NVIDIA GPU is available.
3. You want the broadest algorithm selection and established community support.
4. You are prototyping and value extensive documentation.

## When to Choose TomocuPy

1. Near-real-time analytical reconstruction is the priority.
2. An NVIDIA GPU (Ampere or newer) is available at the beamline workstation.
3. You are working within the APS 2-BM or 32-ID data management workflow.
4. The dataset fits within single-GPU memory (or can be chunked).

## Combined Workflow

A practical approach at APS is to use TomocuPy for rapid FBP previews during
data collection and then run TomoPy iterative solvers offline for
publication-quality reconstructions.
