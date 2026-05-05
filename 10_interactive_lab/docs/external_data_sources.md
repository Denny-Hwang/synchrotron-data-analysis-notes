---
doc_id: LAB-EXT-001
title: External Data Sources — Extended Atlas for Interactive Lab
status: accepted
version: 1.0.0
last_updated: 2026-05-05
supersedes: null
related: [10_interactive_lab/README.md, 09_noise_catalog/]
---

# External Data Sources — Extended Atlas

> **Purpose.** This document is for users who want to go *beyond* the small samples we bundle in `datasets/` and run the same noise-mitigation experiments on **larger or more diverse real datasets**. We did not redistribute these because of one or more of the following: file size > 100 MB, hosting only via DOI/Box/Globus, license that disallows redistribution, or sheer volume (TB-scale archives).

## How to read this document

For every noise category covered by `09_noise_catalog/`, you will find:

1. **Recommended dataset(s)** — what gives you the cleanest before/after experience.
2. **License & redistribution rules** — what you can and cannot do.
3. **How to fetch** — concrete commands or step-by-step web instructions.
4. **What to cite** — minimum required reference.
5. **Caveats** — gotchas (e.g., needs login, region-restricted, requires academic email).

Sizes given are approximate at time of writing (2026-05).

---

## TOMOGRAPHY

### Ring artifacts — large real datasets

- **TomoBank `tomo_00060` and `tomo_00061`** (Diamond Light Source, ring-rich) — ~5 GB / ~12 GB
  - License: dataset-level, mostly **CC BY 4.0** (check per entry)
  - URL: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.spheres.html
  - Fetch: HTTP from APS Petrel: `wget https://g-a0400a.fd635.8443.data.globus.org/tomobank/...`
  - Cite: De Carlo, F. et al. (2018). *Meas. Sci. Technol. 29, 034004.* https://doi.org/10.1088/1361-6501/aa9c19

- **Sarepy `/data/challenging/`** (already partially bundled) — full set ~150 MB
  - License: Apache-2.0 — fully redistributable
  - URL: https://github.com/nghia-vo/sarepy/tree/master/data/challenging
  - All challenging cases (8 files); we bundled 7 in `datasets/tomography/ring_artifact/`

### Low-dose CT denoising — paired noisy/clean

- **TomoGAN demo dataset** (`demo-dataset-real.h5`) — ~200 MB
  - License: **CC BY-NC** (non-commercial only)
  - URL: https://anl.box.com/s/h6koi0hhwqrj1c9tt82tldzo45tl3x15
  - Fetch: download from ANL Box (browser); 128 paired low-dose / full-dose images
  - Cite: Liu, Z. et al. (2020). *J. Opt. Soc. Am. A 37*, 422-434. https://doi.org/10.1364/JOSAA.375595

- **TomoBank `phantom_00037` (full vs sparse)** — multi-dose APS shale samples
  - License: CC BY 4.0
  - URL: https://tomobank.readthedocs.io/en/latest/source/data/docs.data.dynamic.html

### Motion artifact / alignment

- **Gürsoy et al. nanotomography alignment data** — referenced in TomoPy alignment tests
  - https://doi.org/10.1038/s41598-017-12141-9
  - Sample data via APS contact

---

## XRF MICROSCOPY

### Real synchrotron XRF maps

- **NSLS-II PyXRF tutorial datasets** — 2D Au/Fe-K maps from XFM beamline
  - License: BSD-3 (PyXRF code) — data via NSLS-II open-data agreement
  - URL: https://nsls-ii.github.io/PyXRF/ (tutorial section)
  - Cite: Li, L. et al. (2017). *Proc. SPIE 10389.*

- **APS XFM beamline tutorials** — scan files in HDF5 + auxiliary monitor channels
  - URL: https://www.aps.anl.gov/Imaging/Beamlines/4-ID
  - Mostly CC BY 4.0 or per-experiment terms
  - Useful for: photon counting noise, scan stripe, I0 normalization

### Deep-residual XRF super-resolution

- **Wu et al. paired low/high-res XRF data** (npj Comput. Mater. 2023)
  - https://doi.org/10.1038/s41524-023-00995-9
  - Supplementary data via the journal
  - Cite as the paper.

---

## SPECTROSCOPY (XAS / EXAFS)

### Real beamline EXAFS

- **xraylarch examples** (already partially bundled) — extra: NIST Cu foil, Fe foil, Pb XANES
  - URL: https://github.com/xraypy/xraylarch/tree/main/examples
  - License: MIT — fully redistributable
  - We bundled `cu2s_xafs.dat` + 2 Athena projects + 27 FEFF path files

- **NIST XAS database** — calibration foils
  - URL: https://www.nist.gov/programs-projects/x-ray-absorption-spectroscopy-database
  - License: U.S. Government work, public domain
  - Cite: Hudson, A. C. & Kraft, S. (NIST).

- **APS 12-BM-B Athena examples** — multi-edge XAFS scans
  - URL: https://github.com/bruceravel/demeter/tree/main/examples
  - License: GPL (Demeter framework)

### Radiation damage / dose-dependent XAS

- **Diamond Light Source RADDOSE-3D dataset gallery**
  - URL: https://www.diamond.ac.uk/Instruments/Mx/Common/RADDOSE.html
  - Cite: Zeldin, O. B. et al. (2013). *J. Appl. Cryst. 46*, 1225–1230.

---

## PTYCHOGRAPHY & CDI

### Synthetic-but-physically-realistic (fastest)

- **PtyPy** — `flowers.png`, `tree.png`, `moon.png` bundled at `ptypy/resources/`
  - License: BSD-3
  - URL: https://github.com/ptycho/ptypy
  - Use the built-in scan generators (`MoonFlowerScan`, `BlockFull`, `BlockVanilla`) to create fully reproducible test scenarios

### Real synchrotron ptychography

- **CXIDB (Coherent X-ray Imaging Data Bank)**
  - URL: https://cxidb.org
  - License: per-entry, mostly CC-BY 3.0 / 4.0 (check each)
  - 100+ entries from LCLS, ALS, ESRF, etc.
  - Useful for: position error, partial coherence, low-flux

- **PtychoShelves cSAXS tutorial data** (PSI Swiss Light Source)
  - URL: https://www.psi.ch/en/sls/csaxs/software
  - License: BSD-3 / academic
  - Cite: Wakonig, K. et al. (2020). *J. Synchrotron Rad. 27*, 538–547.

---

## SCATTERING / DIFFRACTION

### SAXS / WAXS calibration & samples

- **pyFAI test images** (calibrated detector frames)
  - URL: https://github.com/silx-kit/pyFAI/tree/main/sandbox  + `silx-kit/pyFAI-calibration-data` 보조 저장소
  - License: MIT
  - Useful for: parasitic scattering, detector gap, parallax

- **SASBDB (Small Angle Scattering Biological Data Bank)**
  - URL: https://www.sasbdb.org
  - License: CC-BY 4.0
  - Real biological SAXS scattering profiles
  - Cite: Kikhney, A. G. et al. (2020). *Protein Science 29*, 66–75.

### Macromolecular crystallography

- **DIALS regression test data**
  - URL: https://dials.github.io/data.html (gz archives via `dials.regression`)
  - License: BSD-3 / academic
  - Useful for: ice rings, MX radiation damage, detector gaps
  - Cite: Winter, G. et al. (2018). *Acta Cryst. D 74*, 85–97.

- **Proteopedia tutorial datasets** — real images for indexing/integration
  - URL: https://www.diamond.ac.uk/Instruments/Mx/I04/Tutorials.html

### Coherent / Phase imaging

- **CXIDB (see Ptychography)** — also covers CDI
- **LCLS open data** — https://lcls.slac.stanford.edu/data
  - Cite: Damiani, D. et al. (2016). *J. Appl. Cryst. 49*, 672–679.

---

## ELECTRON MICROSCOPY (Cryo-EM, SEM, TEM)

### Cryo-EM low-dose denoising — gold standard

- **EMPIAR-10025** (T20S proteasome, Topaz-Denoise training set)
  - License: **CC0**
  - URL: https://www.ebi.ac.uk/empiar/EMPIAR-10025
  - Size: ~50 GB
  - Use case: replicate Topaz-Denoise paper end-to-end
  - Fetch: Aspera, Globus, or HTTPS direct
  - Cite: Bepler, T. et al. (2020). *Nature Communications 11*, 5208. https://doi.org/10.1038/s41467-020-18952-1

- **EMPIAR-10185** (apoferritin) — smaller, faster experiments — ~3 GB
  - License: CC0

- **EMPIAR-10028** (β-galactosidase) — classic benchmark — ~10 GB
  - License: CC0

### CTF estimation (cryo-EM)

- **EMPIAR-10061** (TRPV1, Bartesaghi et al.)
  - License: CC0
  - Use case: CTFFIND4, DeepCTF training

- **CTFFIND4 sample images** (Rohou & Grigorieff)
  - URL: https://grigoriefflab.umassmed.edu/ctffind4
  - Bundled with the binary release (not redistributable in our repo due to academic license)

### Drift / motion correction

- **MotionCor2 sample data** — academic license, not redistributable
  - URL: https://emcore.ucsf.edu/ucsf-software (academic registration)

---

## MEDICAL IMAGING (cross-domain benchmarks)

> Note: use only **anonymized public** datasets with explicit research-use license. Never bundle DICOM data with personal identifiers in your derived work.

### CT — beam hardening, metal artifact, scatter

- **TCIA (The Cancer Imaging Archive)** — many CT collections
  - URL: https://www.cancerimagingarchive.net
  - License: per-collection, mostly CC-BY 3.0 / 4.0
  - Look for "QIN-HEADNECK" (beam hardening), "TCIA-METAL" (metal artifact), "LDCT-and-Projection-data" (low-dose CT)

- **AAPM Low-Dose CT Grand Challenge** — paired full-dose / quarter-dose
  - URL: https://www.aapm.org/GrandChallenge/LowDoseCT/
  - License: research only, registration required
  - Cite: McCollough, C. H. et al. (2020). *Med. Phys. 47*, e911–e926.

### MRI — bias field, Gibbs ringing

- **OASIS** brain MRI — public neuroimaging
  - URL: https://www.oasis-brains.org
  - Cite: Marcus, D. S. et al. (2007). *J. Cogn. Neurosci. 19*, 1498–1507.

---

## CROSS-CUTTING

### Cosmic ray / outlier detection (zinger analog)

- **astroscrappy `gmos.fits`** (already bundled) — 389 KB
- **HST WFC3 images** — large (TB) but small subsets via MAST
  - URL: https://archive.stsci.edu
  - License: U.S. Government work, public domain
  - Useful for: realistic cosmic-ray statistics

### DL hallucination / uncertainty

- **DIDSR/sFRC** sample CT images
  - URL: https://github.com/DIDSR/sfrc
  - License: U.S. Government work, public domain
  - Cite: Bhadra, S. et al. (2021). *IEEE Trans. Med. Imaging 40*, 3249–3260.

---

## Pretrained Model Weights (referenced by the noise catalog)

These are model **weights**, not data, but they are essential to run the AI/ML mitigation experiments. None are redistributed in this repository — fetch on demand.

| Model | Source | Size | License | Notes |
|---|---|---|---|---|
| **TomoGAN** | https://github.com/lzhengchun/TomoGAN | ~80 MB (vgg19 loss net) + small generator | CC BY-NC | Non-commercial only |
| **Topaz-Denoise** | https://github.com/tbepler/topaz/tree/master/topaz/pretrained/denoise | ~30 MB per model | GPL-3.0 | unet, unet-small, fcnet, fcnet2, unet-3d, unet-3d-21x21x21 |
| **Noise2Void / Noise2Noise / CARE** | https://github.com/CAREamics/careamics | varies | BSD-3 | Self-supervised, train on your own data |
| **CryoDRGN** | https://github.com/ml-struct-bio/cryodrgn | varies | GPL | Heterogeneous reconstruction |
| **edgePtychoNN** | https://github.com/AnakhaVB/edgePtychoNN | < 50 MB | MIT | Real-time ptychography |

### Generic image-restoration baselines on Hugging Face Hub (apply to most modalities)

| HF model id | Architecture | License | Notes |
|---|---|---|---|
| `mikestealth/nafnet-models` | NAFNet | MIT | SOTA general denoising |
| Paper page: SwinIR (`huggingface.co/papers/2108.10257`) — KAIR weights | Swin Transformer | Apache-2.0 | Color/grayscale denoise + super-res |
| `caidas/swin2SR-classical-sr-x2-64` (HF transformers) | Swin2SR | Apache-2.0 | Super-resolution baseline |

These HF baselines are **not synchrotron-specific**, but useful as out-of-the-box "transfer-learning" comparisons.

---

## Lazy-download recipe (for the Streamlit Lab)

Use [`pooch`](https://www.fatiando.org/pooch/) for hash-verified, cached fetches:

```python
import pooch

empiar_10185_micrograph = pooch.retrieve(
    url="ftp://ftp.ebi.ac.uk/empiar/world_availability/10185/data/Movies/movie_001.mrc",
    known_hash="sha256:abcdef0123...",   # fill in after first download
    path=pooch.os_cache("eberlight_lab"),
    progressbar=True,
)
```

Recommended pattern:

1. First call: user downloads ~50 MB once.
2. Subsequent calls: served from local cache.
3. Hash mismatch → automatic re-download.
4. Streamlit Cloud: the cache lives on the ephemeral container; the recipe re-fetches on cold start.

A complete `lazy_download.yaml` lives at `10_interactive_lab/models/lazy_download_recipes.yaml` and `datasets/manifest.yaml`.

---

## Research Ethics Reminder

When you use any of the data above in a publication, **always**:

1. **Cite the data paper** in the same way you cite a code paper.
2. **Acknowledge the facility** that produced the data (APS, ESRF, DLS, Diamond, NSLS-II, BNL, Argonne, ALS, etc.) — most facilities have a required acknowledgement string in their user agreements.
3. **Respect the license**:
   - CC0 → no requirement, but cite anyway.
   - CC BY → cite + attribute.
   - CC BY-NC → no commercial use.
   - CC BY-SA → derivative works must be CC BY-SA.
4. **Never redistribute** datasets that require login/registration unless the license explicitly allows it.
5. **Do not strip metadata** (provenance headers, EXIF, FITS headers) when you re-share data.
6. For **medical or biological** data: confirm the dataset is properly anonymized and is permitted for research use under your jurisdiction (HIPAA/GDPR/etc.).

If in doubt, contact the dataset's corresponding author or the hosting facility's data office.

---

*Last updated: 2026-05-05. If you find a broken link, please open an issue or PR.*
