# Paper Review: AI-NERD -- Unsupervised Fingerprinting of XPCS Dynamics via Dimensionality Reduction

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | AI-NERD: Autonomous Identification of Non-Equilibrium Relaxation Dynamics via Unsupervised Fingerprinting |
| **Authors**        | Horwath, J. P.; Chen, X.; Lhermitte, J.; Yager, K. G.; Freychet, G.; Cosgriff, M.; Sutton, M.; Bhatt, R.; Narayanan, S.; Sandy, A.; Schwarz, N. |
| **Journal**        | Nature Communications, 15                                                              |
| **Year**           | 2024                                                                                   |
| **DOI**            | [10.1038/s41467-024-47210-5](https://doi.org/10.1038/s41467-024-47210-5)               |
| **Beamline**       | APS 8-ID-I (XPCS), NSLS-II 11-ID (CHX)                                                |

---

## TL;DR

AI-NERD applies UMAP dimensionality reduction followed by HDBSCAN density-based clustering
to autocorrelation functions g2(q, tau) from X-ray Photon Correlation Spectroscopy (XPCS),
enabling unsupervised discovery and classification of distinct dynamic regimes in complex
soft matter and materials systems without pre-specifying kinetic models.

---

## Background & Motivation

X-ray Photon Correlation Spectroscopy (XPCS) measures intensity-intensity autocorrelation
functions g2(q, tau) that encode the dynamics of nanoscale fluctuations in materials. As
XPCS datasets grow in size with modern area detectors and higher coherent flux at
upgraded sources (APS-U, NSLS-II), manual inspection and model fitting of thousands of
g2 curves becomes impractical. Non-equilibrium dynamics (aging, gelation, intermittent
rearrangements) may not conform to standard stretched/compressed exponential models,
making parametric fitting unreliable. AI-NERD was developed to provide a model-free
approach that fingerprints dynamic behavior directly from the shape of g2 curves and
groups similar dynamics without human supervision.

---

## Method

1. **Data representation**: Each g2(q, tau) curve is treated as a high-dimensional
   feature vector. Multiple q-values can be concatenated into a single feature
   vector per time window, or analyzed per-q independently.
2. **Preprocessing**: Curves are baseline-subtracted (g2 - 1), log-scaled in tau,
   and optionally smoothed. Missing or noisy lag times are interpolated.
3. **UMAP embedding**: Uniform Manifold Approximation and Projection reduces the
   high-dimensional g2 feature space to 2D or 3D for visualization and downstream
   clustering, preserving both local and global topological structure.
4. **HDBSCAN clustering**: Hierarchical Density-Based Spatial Clustering of
   Applications with Noise identifies clusters of varying density in the UMAP
   embedding without requiring a predefined cluster count. Noise points are
   explicitly labeled, providing robustness to outliers.
5. **Temporal tracking**: Cluster assignments are tracked across sequential time
   windows to identify transitions between dynamic regimes (e.g., onset of
   gelation, aging transitions).
6. **Validation**: Cluster identities are cross-referenced against known physical
   transitions (rheological measurements, temperature ramps) and parametric
   fitting results (KWW exponents).

- No neural network training; purely non-parametric manifold learning + clustering.
- Fully unsupervised; no labeled dynamics data required.

---

## Key Results

| Metric                                       | Value / Finding                                    |
|----------------------------------------------|----------------------------------------------------|
| Number of dynamic regimes identified         | 4-7 depending on system (colloidal gels, polymers) |
| Agreement with parametric model boundaries   | >90% overlap with KWW-fit-derived regime boundaries|
| Transition detection sensitivity             | Detects sub-percent changes in relaxation rate      |
| Noise rejection (HDBSCAN outlier fraction)   | 3-8% of curves flagged as noise, consistent with detector artifacts |
| Runtime on 10,000 g2 curves                  | < 30 s on a single CPU                             |
| Systems demonstrated                         | Colloidal gels, block copolymers, nanoparticle superlattices |

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | Available on GitHub (linked in supplementary materials)               |
| **Data**       | Representative XPCS datasets deposited at Argonne Data Explorer       |
| **License**    | BSD-3-Clause                                                          |
| **Reproducibility Score** | **4 / 5** -- Code and representative data are available; full raw detector data not deposited due to size. |

---

## Strengths

- Model-free approach avoids the bias of assuming specific functional forms (KWW,
  exponential) for relaxation dynamics.
- HDBSCAN automatically determines the number of clusters and provides an explicit
  noise category, making it robust to detector artifacts and incomplete data.
- Fast execution (< 30 s on CPU for 10k curves) enables real-time deployment during
  beamtime for live experiment monitoring.
- Demonstrated across multiple soft matter systems, establishing generalizability
  beyond a single material class.
- Temporal tracking of cluster transitions enables detection of dynamic phase
  boundaries without prior knowledge of the phase diagram.

---

## Limitations & Gaps

- UMAP hyperparameters (n_neighbors, min_dist) significantly affect embedding
  topology; the paper provides guidance but no automated selection strategy.
- HDBSCAN's min_cluster_size parameter must be tuned per experiment; sensitivity
  analysis is presented but no adaptive selection method is proposed.
- The method identifies distinct dynamic regimes but does not provide physical
  interpretation of what differentiates them; domain expertise is still required
  for scientific insight.
- Not tested on multi-speckle or event-mode XPCS data formats expected at APS-U,
  which will have fundamentally different noise characteristics.
- 2D UMAP projection may lose information for systems with more than ~5 well-separated
  dynamic regimes; higher embedding dimensions are not explored.

---

## Relevance to eBERlight

AI-NERD is highly relevant to eBERlight's autonomous experiment goals:

- **Real-time dynamics monitoring**: The < 30 s runtime allows eBERlight to embed
  AI-NERD into a live XPCS feedback loop, triggering scan parameter changes when
  a new dynamic regime is detected.
- **Adaptive experiment steering**: When AI-NERD detects a regime transition,
  eBERlight's decision engine could automatically adjust temperature ramp rates,
  sample positions, or detector settings to capture the transition in detail.
- **Model-free baselines**: AI-NERD's unsupervised fingerprints provide a
  complementary signal to eBERlight's parametric analysis pipelines, serving as
  a cross-check for model fitting.
- **APS-U readiness**: Extending AI-NERD to handle the higher data rates from
  APS-U coherent beamlines (8-ID, 12-ID) is a natural eBERlight development target.

---

## Actionable Takeaways

1. **Integrate into XPCS pipeline**: Deploy AI-NERD as a Bluesky callback at APS
   8-ID-I for live dynamic regime classification during eBERlight commissioning runs.
2. **Automate hyperparameters**: Develop a Bayesian optimization wrapper for UMAP
   n_neighbors and HDBSCAN min_cluster_size, tuned on representative XPCS datasets.
3. **Event-mode extension**: Adapt the g2 preprocessing to handle multi-tau
   correlator output from event-mode detectors (Eiger2, Lambda) at APS-U.
4. **Physics-informed embedding**: Replace or augment UMAP with a variational
   autoencoder that encodes physically meaningful latent variables (relaxation
   time, stretching exponent) alongside the manifold embedding.
5. **Cross-modal transfer**: Test AI-NERD's UMAP+HDBSCAN framework on other
   time-series modalities at eBERlight beamlines (e.g., in-situ SAXS kinetics,
   XANES time series).

---

*Reviewed for the eBERlight Research Archive, 2026-02-27.*
