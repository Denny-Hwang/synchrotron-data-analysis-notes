# AI/ML for Synchrotron Science --- Curated Paper Reviews

## Overview

This directory contains in-depth reviews of key publications at the intersection
of artificial intelligence / machine learning and synchrotron X-ray science.
Each review follows the standardized template defined in
`../template_paper_review.md` and includes an explicit assessment of relevance
to the eBERlight program.

The collection spans foundational methods (2013--2019), mature techniques
(2020--2022), and cutting-edge developments (2023--2024), providing a
comprehensive landscape of AI/ML capabilities applicable to beamline science at
the Advanced Photon Source.

---

## Reviews by Category

### Denoising & Dose Reduction

Methods that reduce noise in synchrotron data, enabling lower radiation doses or
faster acquisitions while preserving image quality.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_tomogan_2020.md` | Liu et al., JOSA A (2020) | GAN-based denoising for synchrotron tomography; 4--10x dose reduction |
| `review_realtime_uct_hpc_2020.md` | McClure et al., SMC (2020) | End-to-end AI+HPC workflow including denoising for micro-CT |

### Segmentation & Clustering

Techniques for identifying regions of interest, classifying material phases, or
partitioning hyperspectral data.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_roi_finder_2022.md` | Chowdhury et al., J. Synchrotron Rad. (2022) | PCA + fuzzy k-means for automated XRF ROI recommendation |
| `review_xrf_gmm_2013.md` | Ward et al., Microsc. Microanal. (2013) | GMM soft clustering for subcellular XRF in malaria research |

### Reconstruction & Phase Retrieval

AI-driven approaches to tomographic reconstruction and ptychographic phase
retrieval that improve speed, quality, or both.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_ai_edge_ptychography_2023.md` | Babu et al., Nature Comm. (2023) | Real-time streaming ptychography at 2 kHz on edge GPU/FPGA |
| `review_fullstack_dl_tomo_2023.md` | Fundamental Research (2023) | Full-stack DL vision for synchrotron tomography pipeline |
| `review_ptychonet_2019.md` | Guan et al. (2019) | CNN phase retrieval with 90% speedup over iterative methods |

### Autonomous & Adaptive Experiments

Systems that enable real-time decision-making, adaptive scanning, or autonomous
experimental control during beamtime.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_roi_finder_2022.md` | Chowdhury et al., J. Synchrotron Rad. (2022) | Automated ROI recommendation for beam-time-efficient XRF scanning |
| `review_ai_nerd_2024.md` | Horwath et al., Nature Comm. (2024) | Unsupervised XPCS dynamics fingerprinting via UMAP + HDBSCAN |
| `review_ai_edge_ptychography_2023.md` | Babu et al., Nature Comm. (2023) | Real-time feedback loop for ptychographic imaging |

### Resolution Enhancement

Methods that improve the effective spatial or spectral resolution of synchrotron
measurements beyond the physical limits of the optics or detector.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_deep_residual_xrf_2023.md` | npj Comp. Mater. (2023) | 2--4x effective resolution improvement for XRF via deep residual networks |

### Multimodal, Survey & Infrastructure

Broad surveys, workshop reports, and papers addressing cross-cutting themes
across multiple modalities or facility-wide infrastructure.

| Review | Paper | Key Contribution |
|--------|-------|------------------|
| `review_fullstack_dl_tomo_2023.md` | Fundamental Research (2023) | Comprehensive pipeline vision across all tomography stages |
| `review_realtime_uct_hpc_2020.md` | McClure et al., SMC (2020) | Full pipeline integration for real-time micro-CT analysis |
| `review_ai_als_workshop_2024.md` | Synchrotron Rad. News (2024) | AI@ALS Workshop: facility-wide ML needs survey, applicable to APS |

---

## Complete Review List (Chronological)

| # | File | First Author | Year | Modality |
|---|------|-------------|------|----------|
| 1 | `review_xrf_gmm_2013.md` | Ward | 2013 | XRF |
| 2 | `review_ptychonet_2019.md` | Guan | 2019 | Ptychography |
| 3 | `review_tomogan_2020.md` | Liu | 2020 | Tomography |
| 4 | `review_realtime_uct_hpc_2020.md` | McClure | 2020 | Micro-CT |
| 5 | `review_roi_finder_2022.md` | Chowdhury | 2022 | XRF |
| 6 | `review_ai_edge_ptychography_2023.md` | Babu | 2023 | Ptychography |
| 7 | `review_fullstack_dl_tomo_2023.md` | --- | 2023 | Tomography |
| 8 | `review_deep_residual_xrf_2023.md` | --- | 2023 | XRF |
| 9 | `review_ai_nerd_2024.md` | Horwath | 2024 | XPCS |
| 10 | `review_ai_als_workshop_2024.md` | --- | 2024 | Multi-modal |

---

## Suggested Reading Order

For readers new to AI/ML in synchrotron science, the following reading order
provides a logical progression:

1. **`review_ai_als_workshop_2024.md`** --- Start with the broad survey to
   understand the landscape of ML needs at a modern light source.

2. **`review_fullstack_dl_tomo_2023.md`** --- Read the full-stack pipeline
   vision to understand how individual methods fit into an end-to-end workflow.

3. **`review_tomogan_2020.md`** and **`review_ptychonet_2019.md`** --- Study
   two foundational deep learning approaches (denoising and reconstruction).

4. **`review_roi_finder_2022.md`** and **`review_xrf_gmm_2013.md`** --- Explore
   clustering and segmentation methods for XRF, from classical to modern.

5. **`review_ai_edge_ptychography_2023.md`** and
   **`review_realtime_uct_hpc_2020.md`** --- Examine real-time and edge
   computing approaches.

6. **`review_deep_residual_xrf_2023.md`** --- Resolution enhancement as a
   complementary capability.

7. **`review_ai_nerd_2024.md`** --- Unsupervised methods for dynamics, the
   cutting edge.

---

## Cross-References

- `../template_paper_review.md` --- Review template
- `../eberlight_publications.md` --- eBERlight program publications
- `../../03_ai_ml_methods/` --- Background on AI/ML techniques
- `../../05_tools_and_code/` --- Software tools referenced in reviews
