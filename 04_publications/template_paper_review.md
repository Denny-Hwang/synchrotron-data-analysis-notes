# Paper Review Template -- Synchrotron Data Analysis Notes

> Copy this file, rename to `review_<short_descriptor>_<year>.md`, and fill in
> every section. Do not leave sections blank; write "N/A" if truly not
> applicable.

---

## Bibliographic Information

| Field | Value |
|-------|-------|
| **Paper Title** | _Full title of the paper_ |
| **Authors** | _Last, First; Last, First; ..._ |
| **Journal / Conference** | _Full journal name or conference proceedings_ |
| **Year** | _YYYY_ |
| **Volume / Pages** | _e.g., 29, 1024-1030_ |
| **DOI** | _e.g., 10.1107/S1600577522008876_ |
| **URL** | _Direct link to paper_ |
| **Beamline / Facility** | _e.g., 2-ID-E, APS; or "Simulation only"_ |
| **Modality** | _e.g., XRF, XRD, XPCS, Tomography, Ptychography_ |

---

## TL;DR

_One paragraph (3-5 sentences) summarizing the paper's core contribution,
method, and key result. Write this for a busy colleague who has 30 seconds._

---

## Background & Motivation

_Why does this problem matter? What gap in the literature or practical need does
this paper address? Briefly describe the state of the art before this work._

- What scientific or operational problem is being solved?
- What prior approaches existed and why were they insufficient?
- What is the broader impact if this problem is solved?

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | _Beamline, simulation, public dataset, etc._ |
| **Sample type** | _Biological cells, alloy, polymer, etc._ |
| **Data dimensions** | _e.g., 256x256 pixels, 2048 projections, N energy channels_ |
| **Preprocessing** | _Normalization, filtering, alignment, etc._ |

### Model / Algorithm

_Describe the technical approach in detail:_

- Architecture (e.g., U-Net, ResNet, GAN, GMM, PCA + clustering)
- Loss function(s) and training strategy
- Key hyperparameters
- Hardware requirements (GPU, memory, training time)

### Pipeline

_Describe the end-to-end workflow from raw data to final output:_

```
Raw data --> Preprocessing --> Model --> Post-processing --> Output
```

_Include any notable integration points (streaming, edge compute, HPC, etc.)._

---

## Key Results

_Quantitative results with metrics. Use a table where possible._

| Metric | Value | Baseline | Improvement |
|--------|-------|----------|-------------|
| _e.g., SSIM_ | _0.95_ | _0.82_ | _+16%_ |
| _e.g., Processing time_ | _10 ms/frame_ | _2 s/frame_ | _200x_ |

### Key Figures

_Reference and briefly describe the most important figures:_

- **Figure X**: _Description of what it shows and why it matters_
- **Figure Y**: _Description_

---

## Data & Code Availability

| Item | Available? | Link |
|------|-----------|------|
| **Source code** | Yes / No | _URL or "not provided"_ |
| **Trained models** | Yes / No | _URL or "not provided"_ |
| **Training data** | Yes / No | _URL or "not provided"_ |
| **Test data** | Yes / No | _URL or "not provided"_ |

**Reproducibility score**: _X / 5_

| Score | Meaning |
|-------|---------|
| 1 | No code or data; methods description insufficient to reproduce |
| 2 | Methods described but no code; partial data |
| 3 | Code available but incomplete or undocumented; no data |
| 4 | Code available and documented; data partially available |
| 5 | Fully reproducible: code, data, environment, and instructions provided |

---

## Strengths

_Bullet list of what the paper does well:_

- _Strength 1_
- _Strength 2_
- _Strength 3_

---

## Limitations & Gaps

_Bullet list of weaknesses, missing elements, or open questions:_

- _Limitation 1_
- _Limitation 2_
- _Limitation 3_

---

## Relevance to APS BER Program

_How does this work connect to the BER program's mission and beamlines?_

- **Applicable beamlines**: _List specific beamlines or modalities_
- **Integration potential**: _How could this be adopted or adapted?_
- **Priority**: _High / Medium / Low -- with justification_

---

## Actionable Takeaways

_Concrete next steps for the team:_

1. _Action item 1_
2. _Action item 2_
3. _Action item 3_

---

## Notes & Discussion

_Any additional thoughts, questions for the group, or links to related reviews
in this archive._

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | _Name_ |
| **Review date** | _YYYY-MM-DD_ |
| **Last updated** | _YYYY-MM-DD_ |
| **Tags** | _e.g., XRF, clustering, unsupervised, real-time_ |
