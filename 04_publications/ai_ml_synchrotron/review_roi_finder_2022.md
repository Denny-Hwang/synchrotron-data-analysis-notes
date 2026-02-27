# Paper Review: ROI-Finder -- Machine-Learning-Assisted Region-of-Interest Identification for XRF Microscopy

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | ROI-Finder: Machine-Learning-Assisted Region-of-Interest Identification in XRF Maps   |
| **Authors**        | Chowdhury, A. Z.; Wen, H.; Jatla, V.; Huang, X.; Chen, S.; Jacobsen, C.             |
| **Journal**        | Journal of Synchrotron Radiation, 29(4)                                                |
| **Year**           | 2022                                                                                   |
| **DOI**            | [10.1107/S1600577522008876](https://doi.org/10.1107/S1600577522008876)                 |
| **Beamline**       | APS 2-ID-E, Bionanoprobe (9-ID)                                                       |

---

## TL;DR

ROI-Finder automates region-of-interest selection for synchrotron XRF mapping by coupling
PCA-based dimensionality reduction with fuzzy k-means clustering and a decision-support
scoring function, reducing coarse-survey-to-fine-scan turnaround from hours to minutes.

---

## Background & Motivation

High-resolution X-ray fluorescence (XRF) microscopy at third-generation synchrotrons can
map elemental distributions at sub-100 nm resolution, but scanning an entire sample at full
resolution is time-prohibitive. Operators typically acquire a coarse overview map and then
manually select regions of interest for high-resolution scanning. This manual selection is
subjective, slow, and risks missing scientifically important features, especially in
heterogeneous biological or environmental specimens. ROI-Finder was developed to provide
an objective, reproducible, and rapid recommendation system for this workflow bottleneck.

---

## Method

1. **Data ingestion**: Multi-element XRF maps (coarse survey scans) are loaded as 3D
   datacubes (x, y, element channels).
2. **PCA dimensionality reduction**: Principal Component Analysis projects the
   multi-element data into a lower-dimensional feature space, capturing correlated
   elemental signatures while suppressing noise.
3. **Fuzzy k-means clustering**: Soft clustering assigns each pixel a membership degree
   across k clusters, allowing spatial regions with mixed chemistry to be captured more
   faithfully than hard segmentation.
4. **ROI scoring function**: A composite score ranks candidate sub-regions by chemical
   heterogeneity, spatial coherence, and scientific relevance (user-configurable weights).
5. **Visualization dashboard**: An interactive GUI displays ranked ROIs overlaid on the
   coarse map, enabling rapid operator approval.

- No deep learning is used; the pipeline is entirely classical ML.
- Training is unsupervised (no labeled data required).
- Typical runtime: under 60 seconds on a standard workstation for a 256 x 256 pixel map
  with 10+ element channels.

---

## Key Results

| Metric                                   | Value / Finding                                   |
|------------------------------------------|---------------------------------------------------|
| Time to recommend ROIs                   | < 60 s (vs. 10-30 min manual)                     |
| Agreement with expert-selected ROIs      | ~85% overlap (Jaccard index) on test specimens     |
| Number of elements handled simultaneously| Up to 15 channels demonstrated                    |
| Cluster count sensitivity                | Stable recommendations for k = 3 to 8             |
| Demonstrated specimens                   | Biological cells, geological thin sections         |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | [github.com/arshadzahangirchowdhury/ROI-Finder](https://github.com/arshadzahangirchowdhury/ROI-Finder) |
| **Data**       | Example XRF datasets included in the repository                      |
| **License**    | BSD 3-Clause                                                          |
| **Reproducibility Score** | **4 / 5** -- Code runs out-of-the-box with provided data; minor dependency pinning issues noted. |

---

## Strengths

- Fully unsupervised: no labeled training data needed, making it immediately deployable at
  any XRF beamline without annotation effort.
- Lightweight and fast: PCA + fuzzy k-means avoids GPU requirements, enabling execution on
  beamline control workstations in real time.
- Open-source with example data: facilitates independent reproduction and community
  extension.
- Scoring function is configurable: operators can weight chemical diversity vs. spatial
  compactness to suit different scientific goals.
- Directly addresses a genuine operational bottleneck that wastes significant beam time.

---

## Limitations & Gaps

- Evaluated only on XRF elemental maps; applicability to XRF spectra (full spectral
  fitting) or other modalities (XANES, XRD) is not demonstrated.
- Fuzzy k-means requires the user to specify k; the paper does not provide automatic
  model-selection (e.g., BIC, silhouette analysis).
- No quantitative comparison against other clustering approaches (e.g., GMM,
  spectral clustering, HDBSCAN).
- The 85% expert agreement is reported informally without rigorous inter-rater
  reliability statistics or blinded evaluation.
- GUI is tightly coupled to the APS data format; adapting to other facility HDF5
  schemas requires manual configuration.

---

## Relevance to APS BER Program

ROI-Finder directly addresses the scan-strategy optimization problem central to
the BER program's goal of intelligent beamline automation. The PCA + clustering approach
could serve as:

- A baseline comparison for the BER program's more advanced RL/Bayesian-optimization-based
  scan planners.
- A feeder module: ROI-Finder's ranked candidates could initialize the prior for
  the BER program's adaptive sampling algorithms.
- A template for integrating classical ML pipelines into the Bluesky/Tiled data
  infrastructure that the BER program targets.

---

## Actionable Takeaways

1. **Benchmark baseline**: Integrate ROI-Finder as the "classical ML baseline" in
   the BER program's scan-planning evaluation suite.
2. **Auto-k selection**: Extend fuzzy k-means with silhouette-based or gap-statistic-based
   automatic cluster number selection before deploying at BER program beamlines.
3. **Format adapter**: Write a thin HDF5 adapter to ingest NSLS-II/APS-U XRF data
   formats into ROI-Finder's pipeline.
4. **Hybrid approach**: Feed ROI-Finder cluster maps into the BER program's Bayesian
   optimization loop as informative spatial priors.
5. **Latency target**: The < 60 s runtime sets a useful reference point; the BER program's
   DL-based alternatives should aim for comparable or faster inference.

---

*Reviewed for the Synchrotron Data Analysis Notes, 2026-02-27.*
