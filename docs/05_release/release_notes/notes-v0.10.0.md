---
doc_id: REL-N100
title: "Release Notes — notes-v0.10.0"
status: draft
version: 0.10.0
last_updated: 2026-05-05
supersedes: null
related: [ADR-008, ADR-006, ADR-007]
---

# Release Notes — notes-v0.10.0

**Section 10 — Interactive Lab**

## Summary

Adds a tenth note folder, `10_interactive_lab/`, which bundles real
sample data drawn from prior research so that users can replay noise
mitigation techniques (traditional and AI/ML) interactively in the
Streamlit Explorer. ADR-008 captures the rationale and trade-offs.

This is the first note release to add a new top-level folder since the
v0.1.0 content baseline. The 9-folder IA established by ADR-004 grows to
10; the new folder is mapped into the existing **Build** cluster.

## What's New

### `10_interactive_lab/`

```
10_interactive_lab/
├── README.md
├── manifest.yaml                    machine-readable inventory
├── CITATIONS.bib                    BibTeX for every cited work
├── LICENSES/                        verbatim original licences from 8 upstream repos
├── docs/
│   └── external_data_sources.md     curated atlas of bigger / lazy-load datasets
├── datasets/
│   ├── tomography/                  ~133 MB
│   │   ├── ring_artifact/           7 sinograms (Sarepy, Apache-2.0)
│   │   ├── neutron_tomography/      1 sino + 3 recons (Sarepy)
│   │   └── flatfield_correction/    9 NumPy fixtures (TomoPy, BSD-Argonne)
│   ├── xrf/                         ~470 KB
│   │   ├── xrf_spectra/             2 real spectra (PyMca)
│   │   └── pyxrf_configs/           6 fitting JSONs (NSLS-II PyXRF)
│   ├── spectroscopy/                ~340 KB
│   │   ├── exafs_pymca/             Cu, Ge K-edge real EXAFS
│   │   ├── exafs_xraylarch/         Cu₂S K-edge real XAFS
│   │   ├── athena_projects/         abc.prj, FeS2_ex.prj
│   │   └── feff_calculations/       Cu (13), FeO (6), ZnSe (8) FEFF paths
│   ├── scattering_diffraction/      ~75 KB
│   │   └── pyFAI_calibrants/        10 d-spacing reference files (Si, LaB6, Au, …)
│   ├── cross_cutting/               ~390 KB
│   │   └── cosmic_rays/             gmos.fits (astroscrappy)
│   ├── ptychography/                README only, link out
│   └── electron_microscopy/         README only, link out
└── models/
    ├── README.md
    └── lazy_download_recipes.yaml   pooch.retrieve() recipes (TomoGAN, Topaz, NAFNet, SwinIR…)
```

### Data Coverage

| Modality | Bundled samples | Total bundled size |
|---|---:|---:|
| Tomography (ring artifact, neutron, flatfield) | 20 files | ~133 MB |
| XRF (spectra + configs) | 8 files | ~470 KB |
| Spectroscopy (EXAFS + FEFF + Athena) | 32 files | ~340 KB |
| Scattering / Diffraction (calibrants) | 10 files | ~75 KB |
| Cross-cutting (cosmic ray FITS) | 1 file | ~390 KB |
| Ptychography | 0 (external links) | – |
| Electron microscopy | 0 (external links) | – |
| **Total** | **71 files** | **~135 MB** |

### Upstream sources mirrored

All eight upstream repositories are permissively licensed:

| Source | License | Commit pinned |
|---|---|---|
| Sarepy | Apache-2.0 | `2d00b05e…` |
| TomoPy | BSD (Argonne) | `3377fed6…` |
| algotom | Apache-2.0 | `d44f8c94…` |
| xraylarch | MIT | `3037c388…` |
| PyMca | LGPL-2.1+ | `4fbf74ac…` |
| pyFAI | MIT | `0f97cb43…` |
| PyXRF (NSLS-II) | BSD-3 | `7b7d8689…` |
| astroscrappy | BSD-3 | `60570d35…` |

### Research-ethics scaffolding

- Every dataset subfolder carries an `ATTRIBUTION.md` with YAML
  frontmatter, upstream URL, exact pinned commit, authors, license,
  citation string, and an explicit "not the official mirror" disclaimer.
- `LICENSES/` holds the eight upstream LICENSE files verbatim.
- `CITATIONS.bib` consolidates 19 BibTeX entries covering the bundled
  data plus the larger external datasets.
- `docs/external_data_sources.md` includes a "Research Ethics Reminder"
  section with concrete do's/don'ts (acknowledgement, license types,
  metadata preservation, medical-imaging caveats).

### Hugging Face Hub stance

We surveyed the Hub for synchrotron-specific denoising / artifact
mitigation models. **None of TomoGAN, Topaz-Denoise, CryoDRGN, DuDoNet,
ADN, or edgePtychoNN is hosted on the Hub today.** Generic image-restoration
baselines (NAFNet, SwinIR, Swin2SR) **are** on the Hub and are listed in
`models/lazy_download_recipes.yaml` as Tier-2 baselines for comparison,
but no weights are bundled in this release.

## Coverage of `09_noise_catalog/` artifacts

| Artifact | Bundled? | Where |
|---|---|---|
| Ring / stripe artifact (tomography) | ✅ | `datasets/tomography/ring_artifact/` |
| Flat-field issues (tomography) | ✅ | `datasets/tomography/flatfield_correction/` |
| Cosmic ray / zinger | ✅ | `datasets/cross_cutting/cosmic_rays/` |
| Statistical noise (EXAFS) | ✅ | `datasets/spectroscopy/exafs_*/` |
| Outlier spectra | ✅ | `datasets/spectroscopy/athena_projects/` |
| Peak overlap (XRF) | ✅ | `datasets/xrf/xrf_spectra/Steel.spe` |
| Detector gaps / parallax | ✅ (calibration ground truth) | `datasets/scattering_diffraction/pyFAI_calibrants/` |
| Low-dose noise (tomography) | external | `docs/external_data_sources.md` (TomoGAN, TomoBank) |
| Sparse-angle / motion (tomography) | external | TomoBank, AFFIRM data |
| Probe blur / SR (XRF) | external | Wu et al. 2023 |
| Position error / partial coherence (ptychography) | external | CXIDB, PtychoShelves |
| Shot noise / CTF / drift (cryo-EM) | external | EMPIAR |
| MX ice rings / radiation damage | external | DIALS regression data |
| Beam hardening / metal artifact (medical) | external | TCIA, AAPM |

## Known Limitations

- The TomoGAN demo dataset (CC BY-NC, 200 MB) and small EMPIAR/TomoBank
  entries are not yet bundled because the authoring environment lacked
  direct HTTPS egress to Box / EBI / TomoBank. They are documented in
  `external_data_sources.md` and will be added once Streamlit-side
  `pooch` downloads are wired up.
- Pretrained weights are deliberately not bundled; see
  `models/README.md` for the rationale and `models/lazy_download_recipes.yaml`
  for the planned fetch paths.
- The Pages mirror cannot run experiments. The Lab markdown is mirrored
  but interactive parameter tuning is Streamlit-only — consistent with
  ADR-007's existing trade-off.

## Migration / Compatibility

- 9-folder enumerations in documentation must be updated to 10. Notable
  spots: `README.md` directory diagram, `CLAUDE.md` directory map, ADR-004
  cluster mapping (extends the **Build** cluster).
- No notes outside `09_noise_catalog/` are renamed or moved. Existing
  notes-v0.* releases remain valid baselines.

## What's Next

- PRD revision (FR-XXX) to formalise the Lab feature set.
- Implement `explorer/pages/4_Experiment.py` consuming `manifest.yaml`
  with auto-generated parameter UIs from `recipe.yaml` files
  (one per noise mitigation experiment, to be added in `experiments/`).
- Wire `pooch` lazy downloads for TomoBank / EMPIAR / Topaz.
- Add a CI smoke test that validates `manifest.yaml`, `ATTRIBUTION.md`
  frontmatter, and LICENSE presence.
- Mirror the Lab markdown in `scripts/build_static_site.py` and verify
  the GitHub Pages build (per invariant #9).

## References

- ADR-008 — Section 10 — Interactive Lab as a tenth note folder
- CLAUDE.md invariants #1, #3, #4, #6, #7, #9
