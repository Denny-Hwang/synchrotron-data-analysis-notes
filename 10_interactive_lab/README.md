---
doc_id: LAB-README-001
title: Interactive Lab — Real-Data Sandbox for Noise Mitigation
status: accepted
version: 0.1.0
last_updated: 2026-05-05
supersedes: null
related: [09_noise_catalog/README.md, docs/02_design/decisions/ADR-008-interactive-lab.md]
---

# 10_interactive_lab — Real-Data Sandbox for Noise Mitigation Experiments

## Purpose

This is **section 10** of the eBERlight Explorer notes. While `09_noise_catalog/` documents what each noise/artifact looks like and how prior research has mitigated it, this section provides **real input data** so that users can replay those mitigations interactively (in the planned Streamlit `4_Experiment.py` page) and tune parameters.

> Companion ADR: [`docs/02_design/decisions/ADR-008-interactive-lab.md`](../docs/02_design/decisions/ADR-008-interactive-lab.md)
> Companion PRD requirement: FR-XXX (to be added).

## What is here

```
10_interactive_lab/
├── README.md                                  ← this file
├── manifest.yaml                              ← machine-readable inventory
├── CITATIONS.bib                              ← BibTeX for every cited work
├── LICENSES/                                  ← original LICENSE files (verbatim)
│   ├── sarepy_Apache-2.0.txt
│   ├── tomopy_BSD-Argonne.txt
│   ├── algotom_Apache-2.0.txt
│   ├── xraylarch_MIT.txt
│   ├── pymca_LGPL-2.1.txt
│   ├── pyFAI_MIT.txt
│   ├── PyXRF_BSD-3.txt
│   └── astroscrappy_BSD-3.txt
├── docs/
│   └── external_data_sources.md               ← curated list of bigger / lazy-load sources
├── datasets/
│   ├── tomography/
│   │   ├── ring_artifact/                     ← 7 sinograms (~133 MB) [Sarepy]
│   │   ├── neutron_tomography/                ← 1 sino + 3 recons (~540 KB) [Sarepy]
│   │   └── flatfield_correction/              ← 9 NumPy fixtures (~80 KB) [TomoPy]
│   ├── xrf/
│   │   ├── xrf_spectra/                       ← 2 real spectra (~84 KB) [PyMca]
│   │   └── pyxrf_configs/                     ← 6 fitting JSONs (~370 KB) [NSLS-II]
│   ├── spectroscopy/
│   │   ├── exafs_pymca/                       ← 2 EXAFS spectra (~42 KB) [PyMca]
│   │   ├── exafs_xraylarch/                   ← 1 Cu₂S spectrum (~22 KB) [xraylarch]
│   │   ├── athena_projects/                   ← 2 Athena projects (~9 KB) [xraylarch]
│   │   └── feff_calculations/                 ← 27 FEFF paths (~150 KB) [xraylarch]
│   ├── scattering_diffraction/
│   │   └── pyFAI_calibrants/                  ← 10 calibrant d-spacing files (~75 KB) [pyFAI]
│   ├── cross_cutting/
│   │   └── cosmic_rays/                       ← 1 GMOS FITS (~389 KB) [astroscrappy]
│   ├── ptychography/                          ← README only — see external_data_sources.md
│   └── electron_microscopy/                   ← README only — see external_data_sources.md
└── models/
    ├── README.md
    └── lazy_download_recipes.yaml             ← HF + GitHub model registry
```

**Total bundled size:** approximately **135 MB** (dominated by Sarepy ring-artifact sinograms).

## Coverage by noise category

| Modality | Noise/Artifact | Bundled | External link |
|---|---|---|---|
| Tomography | Ring / stripe artifact | ✅ 7 sinograms | TomoBank ring-rich entries |
| Tomography | Flat/dark correction | ✅ 9 fixtures | TomoBank dynamic flats |
| Tomography | Neutron CT before/after | ✅ 4 files | NIST neutron CT |
| Tomography | Low-dose CT denoising | – | TomoGAN demo (CC BY-NC) |
| Tomography | Sparse-angle, motion | – | TomoBank, AFFIRM data |
| XRF | Real reference spectra | ✅ 2 spectra | NSLS-II XFM tutorial |
| XRF | PyXRF fitting configs | ✅ 6 JSONs | NSLS-II PyXRF tutorial |
| XRF | Probe blur / SR | – | Wu et al. 2023 (npj CM) |
| Spectroscopy | EXAFS reference | ✅ 3 spectra (Cu, Ge, Cu₂S) | NIST XAS DB |
| Spectroscopy | FEFF theoretical paths | ✅ 27 path files | Demeter examples |
| Spectroscopy | Athena projects | ✅ 2 projects | xraylarch full examples |
| Scattering | Calibrants | ✅ 10 d-spacing files | NIST SRM database |
| Scattering | SAXS / WAXS images | – | SASBDB, ESRF tutorials |
| Crystallography | Ice rings, MX | – | DIALS regression data |
| Ptychography | All cases | – | CXIDB, PtychoShelves cSAXS |
| Cryo-EM | Shot noise (Topaz) | – | EMPIAR-10025 / -10185 |
| Cryo-EM | CTF / drift | – | EMPIAR various |
| Cross-cutting | Cosmic ray / zinger | ✅ 1 FITS | HST archive |

## How to use

### From a Python script

```python
import tifffile
sino = tifffile.imread(
    "10_interactive_lab/datasets/tomography/ring_artifact/sinogram_dead_stripe.tif"
)
print(sino.shape, sino.dtype)   # → e.g. (1024, 2048) uint16
```

### From the Streamlit Explorer (planned `4_Experiment.py`)

Files are auto-discovered from `manifest.yaml`. Each noise category in the catalog maps to one or more bundled samples; the page will let users:

1. Pick a sample.
2. Choose a mitigation method (traditional or DL).
3. Adjust parameters (slider/select).
4. Compare before/after side-by-side, with PSNR/SSIM metrics.

See [`docs/03_implementation/`](../docs/03_implementation/) (incoming) for the experiment-recipe schema.

### Need more data?

See `docs/external_data_sources.md` — it lists the bigger and more diverse datasets we couldn't bundle, with download recipes, license rules, and citation strings.

## Licensing summary

| Source | License | Redistribution OK? |
|---|---|---|
| Sarepy | Apache-2.0 | ✅ |
| TomoPy | BSD (Argonne) | ✅ with DOE acknowledgement |
| xraylarch | MIT | ✅ |
| PyMca | LGPL-2.1+ | ✅ |
| pyFAI | MIT (MIT/X11) | ✅ |
| PyXRF | BSD-3 | ✅ |
| astroscrappy | BSD-3 | ✅ |

See `LICENSES/` for full text of each.

## Research ethics

Every subfolder contains an `ATTRIBUTION.md` file declaring:
- the upstream repository and exact commit SHA we mirrored,
- the original author(s),
- the license,
- the required citation string (also in `CITATIONS.bib`),
- a disclaimer that we are **not** the official mirror.

If you publish work that uses any of these data, please cite the original paper, not this repository.

## How this fits into the broader project

This section satisfies several CLAUDE.md invariants:

| Invariant | How |
|---|---|
| #1 Notes are SoT | Lab is read-only sample data; the noise taxonomy itself remains in `09_noise_catalog/` |
| #2 PRD/US/ADR traceability | See ADR-008 |
| #3 Release notes + CHANGELOG | See `docs/05_release/release_notes/notes-v0.10.0.md` |
| #4 IA change → ADR | ADR-008 documents the new section-10 directory |
| #7 YAML frontmatter | Every `.md` file here carries it |

## Versioning

This section follows the **notes-vX.Y.Z** SemVer stream (CLAUDE.md invariant #6). Initial release is `notes-v0.10.0`. The lab content is independent of explorer-vX.Y.Z and the static-site generator.
