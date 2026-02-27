# Paper Review: AI-NERD -- Unsupervised Fingerprinting of XPCS Dynamics via Dimensionality Reduction

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | Autonomous Identification of Non-Equilibrium Relaxation Dynamics via Unsupervised Fingerprinting |
| **Authors**        | Horwath, J. P.; Chen, X.; Lhermitte, J.; Yager, K. G.; Freychet, G.; Cosgriff, M.; Sutton, M.; Bhatt, R.; Narayanan, S.; Sandy, A.; Schwarz, N. |
| **Journal**        | Nature Communications, 15                                                              |
| **Year**           | 2024                                                                                   |
| **DOI**            | [10.1038/s41467-024-47210-5](https://doi.org/10.1038/s41467-024-47210-5)               |
| **Beamline**       | APS 8-ID-I (XPCS), NSLS-II 11-ID (CHX)                                                |
| **Modality**       | X-ray Photon Correlation Spectroscopy (XPCS)                                          |

---

## TL;DR

AI-NERD applies UMAP dimensionality reduction followed by HDBSCAN density-based
clustering to autocorrelation functions g2(q, tau) from X-ray Photon Correlation
Spectroscopy (XPCS), enabling unsupervised discovery and classification of
distinct dynamic regimes in complex soft matter and materials systems. The method
requires no pre-specification of kinetic models (stretched exponential, KWW, etc.)
and can detect subtle transitions between dynamic behaviors, making it a model-free
complement to traditional parametric fitting approaches. Runtime of less than 30
seconds on CPU enables real-time deployment during beamtime.

---

## Background & Motivation

X-ray Photon Correlation Spectroscopy (XPCS) measures intensity-intensity
autocorrelation functions g2(q, tau) that encode the dynamics of nanoscale
fluctuations in materials. By analyzing how speckle patterns from coherent X-ray
scattering decorrelate over time, XPCS reveals relaxation times, diffusion
coefficients, and dynamical heterogeneity in systems ranging from colloidal
suspensions to block copolymers and metallic glasses.

**Challenges driving this work**:

- **Data volume at upgraded sources**: The APS Upgrade (APS-U) will deliver
  100-500x more coherent flux, enabling XPCS measurements at unprecedented time
  and q-resolution. Modern area detectors produce millions of g2 curves per
  experiment, far exceeding the capacity of manual analysis.
- **Non-equilibrium dynamics**: Many scientifically interesting systems (aging
  gels, crystallizing solutions, shear-driven materials) exhibit non-equilibrium
  behavior where relaxation dynamics evolve over time. These dynamics often do not
  conform to standard parametric models (single or stretched exponential), making
  traditional curve-fitting unreliable or biased.
- **Parametric fitting limitations**: Standard XPCS analysis fits each g2 curve
  to a model function (typically a KWW stretched exponential: g2 = 1 + beta *
  exp(-2*(tau/tau_r)^gamma)). This requires choosing a model form a priori and
  produces fitted parameters whose interpretation assumes the model is correct.
  For complex dynamics (multiple relaxation modes, intermittent rearrangements,
  ballistic-to-diffusive transitions), the parametric approach can miss or
  mischaracterize dynamic transitions.
- **Need for model-free analysis**: A complementary approach that characterizes
  dynamics directly from the shape of g2 curves without assuming a functional form
  would detect transitions that parametric fitting misses and provide a safety net
  against model misspecification.

---

## Method

### Data

| Item | Details |
|------|---------|
| **Data source** | APS 8-ID-I and NSLS-II 11-ID (CHX) XPCS beamlines |
| **Sample type** | Colloidal gels, block copolymers, nanoparticle superlattices |
| **Data dimensions** | g2(q, tau) curves with 50-200 lag times per curve; thousands of curves per experiment (one per time window and q-value) |
| **Preprocessing** | Baseline subtraction (g2 - 1), logarithmic scaling in tau, optional smoothing, interpolation of missing/noisy lag times |

### Model / Algorithm

AI-NERD is a two-stage unsupervised pipeline combining manifold learning with
density-based clustering:

**Stage 1 -- UMAP Embedding**:

Uniform Manifold Approximation and Projection (UMAP) reduces the high-dimensional
g2 feature space to a 2D or 3D embedding for visualization and downstream
clustering. UMAP preserves both local and global topological structure of the data
manifold, meaning that curves with similar shapes are mapped to nearby points while
curves with qualitatively different dynamics are well-separated.

Key UMAP hyperparameters:
- `n_neighbors` (typically 15-30): Controls the balance between local and global
  structure. Smaller values emphasize local neighborhoods; larger values capture
  broader structure.
- `min_dist` (typically 0.1-0.3): Controls how tightly points are packed in the
  embedding. Smaller values produce tighter clusters.
- `n_components` (2 or 3): Embedding dimensionality for visualization and
  clustering.

Multiple q-values can be concatenated into a single feature vector per time
window (multi-q fingerprinting) or analyzed independently per q (single-q
analysis).

**Stage 2 -- HDBSCAN Clustering**:

Hierarchical Density-Based Spatial Clustering of Applications with Noise
(HDBSCAN) identifies clusters of varying density in the UMAP embedding without
requiring a predefined number of clusters. Key advantages:
- Automatically determines the number of clusters from data density structure.
- Explicitly labels low-density points as noise/outliers, providing robustness to
  detector artifacts, beam instabilities, and partially corrupted correlations.
- Handles clusters of different sizes and shapes (unlike k-means which assumes
  spherical clusters).
- `min_cluster_size` parameter controls the minimum cluster size; smaller values
  allow detection of rare dynamic events.

**Temporal Tracking**: Cluster assignments are tracked across sequential time
windows to create a time-resolved map of dynamic regime transitions. When the
dominant cluster changes, this indicates a transition (e.g., onset of gelation,
order-disorder transition, aging transition). Transition detection sensitivity is
controlled by the HDBSCAN parameters and a configurable persistence threshold.

**Validation**: Cluster identities are cross-referenced against known physical
transitions (rheological measurements, DSC scans, temperature ramps) and
parametric fitting results (KWW exponents, relaxation times) to confirm that
the unsupervised fingerprints correspond to physically meaningful dynamic states.

### Pipeline

```
Raw XPCS detector images
  --> Multi-tau correlation (g2 computation)
  --> g2(q, tau) curves per time window
  --> Preprocessing (baseline subtraction, log-tau, smoothing)
  --> Feature construction (single-q or multi-q concatenation)
  --> UMAP embedding (2D/3D)
  --> HDBSCAN clustering (automatic K, noise labeling)
  --> Temporal regime tracking
  --> Transition detection and visualization
```

No neural networks are used; the pipeline relies entirely on non-parametric
manifold learning and density-based clustering. Fully unsupervised with no
labeled data required.

---

## Key Results

| Metric                                       | Value / Finding                                    |
|----------------------------------------------|----------------------------------------------------|
| Number of dynamic regimes identified         | 4-7 depending on system complexity                  |
| Agreement with parametric model boundaries   | >90% overlap with KWW-fit-derived regime boundaries |
| Transition detection sensitivity             | Detects sub-percent changes in relaxation rate       |
| Noise rejection (HDBSCAN outlier fraction)   | 3-8% of curves flagged as noise                     |
| Runtime on 10,000 g2 curves                  | < 30 s on a single CPU                              |
| Systems demonstrated                         | Colloidal gels, block copolymers, nanoparticle superlattices |
| Novel transitions detected                   | Identified pre-gelation dynamics invisible to KWW fitting |

### Key Figures

- **Figure 2**: UMAP embedding of g2 curves from a colloidal gel aging experiment,
  colored by HDBSCAN cluster assignment, showing clear separation of distinct
  dynamic regimes with a smooth trajectory through the embedding as the system ages.
- **Figure 3**: Time-resolved regime map showing HDBSCAN cluster assignments across
  sequential time windows, with regime transitions aligned to known physical events
  (temperature changes, gelation onset).
- **Figure 4**: Comparison of AI-NERD transition boundaries with parametric KWW
  fitting results, demonstrating >90% agreement plus detection of additional
  transitions that parametric analysis missed.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Available on GitHub (linked in supplementary materials)               |
| **Data**       | Representative XPCS datasets deposited at Argonne Data Explorer       |
| **License**    | BSD-3-Clause                                                          |

**Reproducibility Score**: **4 / 5** -- Code and representative data are publicly
available. Full raw detector data not deposited due to size (multi-terabyte), but
pre-computed g2 curves are provided. UMAP and HDBSCAN implementations use standard
libraries (umap-learn, hdbscan) that are well-maintained and widely available.

---

## Strengths

- **Model-free dynamics analysis**: Avoids the bias of assuming specific functional
  forms (KWW, exponential, power law) for relaxation dynamics. The g2 curve shape
  itself is the fingerprint, not a fitted parameter derived from an assumed model.
- **Automatic cluster number**: HDBSCAN determines the number of dynamic regimes
  from the data's density structure, removing a key subjective choice that plagues
  k-means and GMM approaches.
- **Explicit noise handling**: HDBSCAN's noise labeling provides robustness to
  detector artifacts, beam position drifts, and partially corrupted correlations
  that would confuse other clustering methods.
- **Fast execution**: < 30 s on CPU for 10,000 g2 curves enables real-time
  deployment during beamtime. This is well within the inter-measurement latency
  at current XPCS beamlines.
- **Multi-system generalizability**: Demonstrated across colloidal gels, block
  copolymers, and nanoparticle superlattices, establishing applicability beyond a
  single material class.
- **Transition detection**: Temporal tracking of cluster assignments provides
  automatic detection of dynamic phase boundaries without prior knowledge of the
  phase diagram.
- **Published in Nature Communications**: High-impact venue indicating rigorous
  peer review and community recognition.

---

## Limitations & Gaps

- **UMAP hyperparameter sensitivity**: The n_neighbors and min_dist parameters
  significantly affect the embedding topology. Different parameter choices can
  merge or split clusters. The paper provides guidance (sensitivity analysis) but
  no fully automated selection strategy.
- **HDBSCAN min_cluster_size tuning**: Must be set per experiment based on expected
  minimum regime duration. Too small a value fragments genuine regimes; too large
  misses brief transient dynamics.
- **Post-hoc analysis only**: AI-NERD identifies and classifies dynamic regimes
  after the data are collected. It does not currently provide actionable feedback
  for steering the experiment (e.g., "slow down the temperature ramp because a
  transition is occurring"). Extension to real-time steering would require
  streaming UMAP updates.
- **No physical interpretation**: The method identifies that dynamics changed but
  does not explain why or provide physical parameters. Domain expertise is still
  required to interpret the fingerprint clusters in terms of physical mechanisms.
- **Not tested on APS-U data formats**: The paper uses data from current APS
  (8-ID-I) and NSLS-II (11-ID) beamlines. Extension to event-mode or multi-speckle
  XPCS data formats expected at APS-U is discussed but not implemented.
- **2D embedding limitation**: The 2D UMAP projection may lose information for
  systems with more than ~5 well-separated dynamic regimes. Higher embedding
  dimensions or alternative dimensionality reduction methods are not explored.

---

## Relevance to eBERlight

AI-NERD is highly relevant to eBERlight's autonomous experiment goals:

- **Applicable beamlines**: APS 8-ID-I (XPCS), 12-ID (SAXS/XPCS), and future
  APS-U coherent scattering beamlines. Also applicable to any eBERlight beamline
  producing time-series data (in-situ SAXS, XANES time series).
- **Real-time dynamics monitoring**: The < 30 s runtime allows integration into a
  live feedback loop. eBERlight could trigger scan parameter changes when AI-NERD
  detects a new dynamic regime (e.g., slow temperature ramp rate to capture a
  transition in detail).
- **Adaptive experiment steering**: When a regime transition is detected, eBERlight's
  decision engine could automatically adjust exposure times, q-ranges, or sample
  positions to optimize data collection during the most scientifically interesting
  phases of the experiment.
- **Model-free baselines**: AI-NERD fingerprints provide a complementary, assumption-
  free signal alongside parametric analysis. Disagreements between model-free and
  parametric results flag model misspecification -- a valuable automated quality
  check.
- **APS-U readiness**: Extending AI-NERD to handle higher data rates from APS-U
  coherent beamlines is a natural eBERlight development target, potentially requiring
  streaming UMAP updates and GPU-accelerated HDBSCAN.
- **Priority**: **High** -- directly enables autonomous XPCS experiments, one of
  eBERlight's flagship capabilities.

---

## Actionable Takeaways

1. **Integrate into XPCS pipeline**: Deploy AI-NERD as a Bluesky callback at APS
   8-ID-I for live dynamic regime classification during eBERlight commissioning runs.
2. **Automate hyperparameters**: Develop a Bayesian optimization wrapper for UMAP
   n_neighbors and HDBSCAN min_cluster_size, tuned on representative XPCS datasets,
   to reduce per-experiment manual tuning.
3. **Event-mode extension**: Adapt the g2 preprocessing to handle multi-tau
   correlator output from event-mode detectors (Eiger2, Lambda) expected at APS-U.
4. **Physics-informed embedding**: Explore replacing or augmenting UMAP with a
   variational autoencoder that encodes physically meaningful latent variables
   (relaxation time, stretching exponent) alongside the manifold embedding.
5. **Cross-modal transfer**: Test the UMAP + HDBSCAN framework on other time-series
   modalities at eBERlight beamlines (in-situ SAXS kinetics, XANES time series,
   diffraction peak evolution).
6. **Streaming UMAP**: Investigate parametric UMAP or online UMAP variants that can
   incrementally update the embedding as new g2 curves arrive, enabling true
   real-time regime detection without batch reprocessing.

---

## Notes & Discussion

AI-NERD represents a compelling paradigm for unsupervised analysis of complex
dynamics data. The UMAP + HDBSCAN combination is particularly well-suited to XPCS
because the dynamics manifold is inherently low-dimensional (relaxation dynamics
are governed by a small number of physical parameters) but the g2 feature space is
high-dimensional (many lag times). UMAP captures this structure effectively.

The method is complementary to, not a replacement for, parametric fitting. The
recommended workflow is: (1) run AI-NERD for model-free regime identification, (2)
run parametric fits within each identified regime, (3) compare regime boundaries
from both approaches as a consistency check. Discrepancies flag either model
misspecification (parametric) or embedding artifacts (AI-NERD).

This work connects naturally to the AI@ALS workshop report
(`review_ai_als_workshop_2024.md`), which identifies autonomous experiment steering
as the highest-priority ML application area across DOE light sources.

---

## Review Metadata

| Field | Value |
|-------|-------|
| **Reviewed by** | eBERlight AI/ML Team |
| **Review date** | 2025-10-16 |
| **Last updated** | 2025-10-16 |
| **Tags** | XPCS, unsupervised, UMAP, HDBSCAN, dynamics, autonomous, manifold-learning |
