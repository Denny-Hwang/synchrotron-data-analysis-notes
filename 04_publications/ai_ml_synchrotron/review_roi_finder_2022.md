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
| **Modality**       | X-ray Fluorescence (XRF) Microscopy                                                   |

---

## TL;DR

ROI-Finder automates region-of-interest selection for synchrotron XRF mapping by
coupling PCA-based dimensionality reduction with fuzzy k-means clustering and a
decision-support scoring function. The pipeline reduces coarse-survey-to-fine-scan
turnaround from tens of minutes of manual inspection to under 60 seconds of
automated analysis, achieving approximately 85% agreement with expert-selected
ROIs. The approach is fully unsupervised, requires no labeled training data, and
runs on standard beamline workstations without GPU hardware.

---

## Background & Motivation

High-resolution X-ray fluorescence (XRF) microscopy at third- and fourth-generation
synchrotrons can map elemental distributions at sub-100 nm resolution, but scanning
an entire sample at full resolution is time-prohibitive. The standard workflow at
XRF microprobe beamlines such as APS 2-ID-E follows a two-stage hierarchy:

1. **Coarse survey scan**: A low-resolution, large field-of-view raster scan maps
   elemental distributions across the full sample or a large area of interest. Step
   sizes are typically 1-5 micrometers with short dwell times.

2. **Fine targeted scan**: Based on visual inspection of the survey, the
   experimenter manually selects one or more sub-regions for high-resolution
   scanning (50-300 nm step sizes, longer dwell times per pixel).

The manual ROI selection step is a genuine operational bottleneck:

- **Subjectivity**: Different experimenters select different ROIs from the same
  survey map. There is no standardized, reproducible criterion for what
  constitutes an "interesting" region in a multi-element XRF dataset.
- **Time cost**: For complex, heterogeneous specimens (biological tissues,
  environmental particles, geological sections), manual inspection of 10+
  elemental channels and their spatial correlations can take 10-30 minutes per
  survey scan.
- **Scalability barrier**: Autonomous and high-throughput experiment workflows
  require automated ROI selection to close the loop between survey scanning and
  targeted measurement without human intervention.
- **Prior art limitations**: Previous approaches relied on simple single-element
  thresholding or manual contour drawing, ignoring the multi-element correlations
  that often define the most scientifically interesting regions.

ROI-Finder was developed to provide an objective, reproducible, and rapid
recommendation system that considers the full multi-element information content
of the survey scan.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | Beamline 2-ID-E and Bionanoprobe (9-ID), APS |
| **Sample type** | Biological cells, geological thin sections |
| **Data dimensions** | Multi-element XRF maps, ~256x256 pixels, 10-15 elemental channels after spectral fitting |
| **Preprocessing** | XRF spectral fitting via MAPS software to extract per-pixel elemental concentrations; normalization of elemental maps |

### Model / Algorithm

ROI-Finder is a three-stage unsupervised pipeline built entirely on classical ML:

**Stage 1 -- PCA Dimensionality Reduction**:

The multi-element XRF map stack (pixels x elements) is reshaped into a 2D matrix
and subjected to Principal Component Analysis. The first k principal components
(typically k = 3-5, selected to capture >95% cumulative variance) are retained.
PCA serves to decorrelate the elemental channels and reduce dimensionality while
preserving the dominant spatial patterns that distinguish chemically distinct
regions. The PCA loadings themselves are interpretable: PC1 often corresponds to
total elemental content (cell vs. background), PC2 to intracellular contrast
(nucleus vs. cytoplasm), and higher PCs to trace element hotspots.

**Stage 2 -- Fuzzy K-Means Clustering**:

The PCA-reduced pixel representations are clustered using fuzzy c-means (FCM), a
soft clustering algorithm where each pixel receives a continuous membership degree
(0 to 1) in each cluster, rather than a hard binary assignment. The number of
clusters c is selected via the fuzzy partition coefficient (FPC), with typical
values c = 3-8 for cell XRF data. Fuzzy membership maps provide richer spatial
characterization than hard segmentation, as boundary regions and mixed-chemistry
pixels retain partial membership in multiple clusters.

**Stage 3 -- ROI Scoring and Recommendation**:

Fuzzy membership maps are thresholded to produce binary candidate masks. Connected
component analysis identifies discrete ROI candidates. A composite scoring function
ranks candidates based on configurable criteria: spatial extent (sufficient size for
high-resolution scanning), chemical heterogeneity (presence of interesting
multi-element signatures as measured by PCA variance within the ROI), and
morphological compactness (preference for contiguous regions over fragmented ones).
Operators can adjust the relative weights of these criteria to suit different
experimental goals. The top-ranked ROIs are returned as bounding boxes suitable for
direct input to the beamline scanning software.

### Pipeline

```
Multi-element XRF survey map
  --> Spectral fitting (MAPS)
  --> Per-pixel elemental concentration maps
  --> PCA dimensionality reduction (k = 3-5 components)
  --> Fuzzy c-means clustering (c = 3-8 clusters, FPC selection)
  --> Membership map thresholding + connected component analysis
  --> ROI scoring (heterogeneity, size, compactness)
  --> Ranked ROI bounding boxes --> beamline scan controller
```

Implementation is in Python using NumPy, scikit-learn (PCA), and scikit-fuzzy
(FCM). An interactive GUI built with Matplotlib allows operators to visualize
cluster maps, adjust parameters, and approve/reject recommendations.

---

## Key Results

| Metric                                   | Value / Finding                                   |
|------------------------------------------|---------------------------------------------------|
| Time to recommend ROIs                   | < 60 s (vs. 10-30 min manual)                     |
| Agreement with expert-selected ROIs      | ~85% overlap (Jaccard index) on test specimens     |
| Number of elements handled simultaneously| Up to 15 channels demonstrated                    |
| Cluster count sensitivity                | Stable recommendations for k = 3 to 8             |
| Demonstrated specimens                   | Biological cells, geological thin sections         |
| PCA variance captured (first 3 PCs)      | >90% of total variance in elemental data           |

### Key Figures

- **Figure 2**: PCA decomposition of a multi-element cell XRF map showing that
  PC1-PC3 capture >90% variance, with loadings mapping to biologically meaningful
  contrasts (bulk composition, nucleus-cytoplasm, trace metals).
- **Figure 4**: Fuzzy c-means membership maps compared to expert-drawn ROIs,
  demonstrating strong spatial agreement for primary recommended regions.
- **Figure 5**: Full pipeline output showing ranked ROI bounding boxes overlaid on
  the coarse survey map with composite scores.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | [github.com/arshadzahangirchowdhury/ROI-Finder](https://github.com/arshadzahangirchowdhury/ROI-Finder) |
| **Data**       | Example XRF datasets included in the repository                      |
| **License**    | BSD 3-Clause                                                          |

**Reproducibility Score**: **4 / 5** -- Code runs out-of-the-box with provided
data; minor dependency pinning issues noted. The unsupervised nature means no
pre-trained models are needed. Main limitation is that the full experimental
datasets from the paper are not publicly deposited, though the example data is
sufficient for testing the pipeline.

---

## Strengths

- **Fully unsupervised**: No labeled training data needed, making it immediately
  deployable at any XRF beamline without annotation effort. The pipeline adapts to
  each survey scan independently.
- **Lightweight and fast**: PCA + fuzzy k-means avoids GPU requirements, enabling
  execution on beamline control workstations in real time. The < 60 s processing
  time fits within the dead time between survey and fine scans.
- **Open-source with example data**: Facilitates independent reproduction,
  community extension, and adaptation to other beamline data formats.
- **Configurable scoring function**: Operators can weight chemical diversity vs.
  spatial compactness vs. region size to match different experimental priorities,
  from comprehensive coverage to focused investigation.
- **Directly addresses a genuine operational bottleneck** that wastes significant
  beam time at every XRF microprobe beamline worldwide.
- **Interpretable intermediate outputs**: PCA loadings and fuzzy membership maps
  are physically meaningful and provide additional scientific insight beyond just
  the ROI coordinates.

---

## Limitations & Gaps

- **PCA linearity**: PCA captures only linear correlations between elemental
  channels. Nonlinear relationships (e.g., complex stoichiometric constraints,
  matrix absorption effects) are invisible to PCA. Nonlinear alternatives such as
  UMAP, t-SNE, or autoencoders could provide richer representations.
- **Binary final output**: Despite using fuzzy clustering internally, the final ROI
  recommendation relies on a hard thresholding step, discarding the probabilistic
  membership information. A fully probabilistic ROI ranking would be more
  informative.
- **Requires choosing k and c**: The number of PCA components and clusters must be
  specified or selected heuristically. The FPC provides guidance but no fully
  automated, robust selection method is validated across diverse sample types.
- **No comparison to alternative ML methods**: The paper does not benchmark against
  GMM clustering, HDBSCAN, spectral clustering, or deep learning-based
  segmentation on the same datasets.
- **Limited sample diversity in evaluation**: Demonstrated primarily on biological
  cell maps and geological sections; applicability to battery materials, catalysts,
  or environmental particles is not explicitly validated.
- **Informal accuracy assessment**: The 85% expert agreement is reported without
  rigorous inter-rater reliability statistics, blinded evaluation, or statistical
  significance testing.
- **GUI coupling to APS data format**: The visualization dashboard assumes
  APS-specific HDF5 schemas; adaptation to other facility formats requires manual
  configuration.

---

## Relevance to eBERlight

ROI-Finder directly addresses the scan-strategy optimization problem central to
eBERlight's goal of intelligent beamline automation:

- **Applicable beamlines**: Any XRF microprobe beamline at APS, including upgraded
  sector 2 beamlines (2-ID-E, 2-ID-D) and the Bionanoprobe (9-ID). Also applicable
  to micro-XRD beamlines using survey-then-target workflows.
- **Baseline for autonomous scanning**: ROI-Finder's ranked candidates could
  initialize the prior for eBERlight's adaptive sampling algorithms (Bayesian
  optimization, reinforcement learning-based scan planners).
- **Bluesky integration template**: The pipeline provides a model for integrating
  classical ML analysis into the Bluesky/Tiled data infrastructure. A Bluesky plan
  that runs coarse survey -> ROI-Finder -> fine scan would close the autonomous
  experiment loop.
- **Low deployment barrier**: Unsupervised, no GPU needed, open source, and fast.
  eBERlight can deploy this at partner beamlines within 1-2 weeks.
- **Priority**: **High** -- practical, deployable tool addressing a real operational
  need that aligns directly with eBERlight's core mission.

---

## Actionable Takeaways

1. **Benchmark baseline**: Integrate ROI-Finder as the "classical ML baseline" in
   eBERlight's scan-planning evaluation suite for XRF beamlines.
2. **Auto-k selection**: Extend fuzzy k-means with silhouette-based or gap-statistic
   automatic cluster number selection before deploying at eBERlight beamlines.
3. **Format adapter**: Write a thin HDF5 adapter to ingest APS-U and NSLS-II XRF
   data formats into ROI-Finder's pipeline without modifying the core algorithm.
4. **Hybrid approach**: Feed ROI-Finder cluster maps into eBERlight's Bayesian
   optimization loop as informative spatial priors for adaptive scanning.
5. **Latency target**: The < 60 s runtime sets a useful reference point; eBERlight's
   DL-based alternatives should aim for comparable or faster inference.
6. **Expert feedback collection**: Run ROI-Finder alongside expert selection during
   eBERlight beamtimes and collect accept/reject data for future learning-based
   scoring.

---

## Notes & Discussion

ROI-Finder represents a practical, low-risk entry point for bringing ML into the
XRF experiment loop. The fuzzy c-means component is related to the GMM approach
reviewed in `review_xrf_gmm_2013.md`; both provide soft/probabilistic spatial
segmentation, but FCM does not assume Gaussian-distributed features and is more
computationally efficient. A natural next step would combine ROI-Finder with the
deep residual super-resolution approach (`review_deep_residual_xrf_2023.md`) to
first enhance the coarse survey resolution computationally, then identify ROIs
on the enhanced map.

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | eBERlight AI/ML Team |
| **Review date** | 2025-10-15 |
| **Last updated** | 2025-10-15 |
| **Tags** | XRF, clustering, unsupervised, PCA, fuzzy-c-means, ROI, autonomous, beamline |
