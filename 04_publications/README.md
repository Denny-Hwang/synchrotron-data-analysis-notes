# 04 Publications -- Synchrotron Data Analysis Notes

## Purpose

This directory houses curated paper reviews, publication tracking, and reference
materials that inform the BER program's AI/ML strategy for synchrotron
science. Every review follows a standardized template so that insights can be
quickly cross-referenced and applied to beamline development at the Advanced
Photon Source (APS).

---

## Reading Guide

| Step | What to read | Why |
|------|-------------|-----|
| 1 | `template_paper_review.md` | Understand the review format before diving in |
| 2 | `ber_program_publications.md` | See what the BER program has published so far |
| 3 | `ai_ml_synchrotron/README.md` | Get the lay of the land for AI/ML papers |
| 4 | Individual reviews (below) | Deep-dive into specific methods and results |

---

## Files in This Directory

| File | Description |
|------|-------------|
| `README.md` | This index (you are here) |
| `template_paper_review.md` | Standardized template for all paper reviews |
| `ber_program_publications.md` | Tracker for BER program-attributed publications (2023-2025) |
| `ai_ml_synchrotron/` | Curated review collection -- AI/ML applied to synchrotron science |

---

## Paper Reviews (10 total)

All reviews live in `ai_ml_synchrotron/` and follow the template. They are
organized here by primary topic for quick navigation.

### Clustering & Segmentation

| # | File | Short Title | Year |
|---|------|-------------|------|
| 1 | `ai_ml_synchrotron/review_roi_finder_2022.md` | ROI-Finder: unsupervised XRF cell segmentation | 2022 |
| 2 | `ai_ml_synchrotron/review_xrf_gmm_2013.md` | GMM soft clustering for subcellular XRF | 2013 |

### Denoising

| # | File | Short Title | Year |
|---|------|-------------|------|
| 3 | `ai_ml_synchrotron/review_tomogan_2020.md` | TomoGAN: GAN-based tomography denoising | 2020 |

### Autonomous / Unsupervised Analysis

| # | File | Short Title | Year |
|---|------|-------------|------|
| 4 | `ai_ml_synchrotron/review_ai_nerd_2024.md` | AI-NERD: unsupervised XPCS fingerprinting | 2024 |

### Ptychography

| # | File | Short Title | Year |
|---|------|-------------|------|
| 5 | `ai_ml_synchrotron/review_ai_edge_ptychography_2023.md` | Real-time edge ptychographic imaging | 2023 |
| 6 | `ai_ml_synchrotron/review_ptychonet_2019.md` | PtychoNet: CNN phase retrieval | 2019 |

### Resolution Enhancement

| # | File | Short Title | Year |
|---|------|-------------|------|
| 7 | `ai_ml_synchrotron/review_deep_residual_xrf_2023.md` | Deep residual networks for XRF resolution | 2023 |

### Full-Pipeline / Infrastructure

| # | File | Short Title | Year |
|---|------|-------------|------|
| 8 | `ai_ml_synchrotron/review_fullstack_dl_tomo_2023.md` | Full-stack DL pipeline for tomography | 2023 |
| 9 | `ai_ml_synchrotron/review_realtime_uct_hpc_2020.md` | Real-time AI+HPC for micro-CT | 2020 |

### Workshop Reports & Surveys

| # | File | Short Title | Year |
|---|------|-------------|------|
| 10 | `ai_ml_synchrotron/review_ai_als_workshop_2024.md` | AI@ALS Workshop Report | 2024 |

---

## Template

The review template (`template_paper_review.md`) standardizes:

- **Bibliographic metadata** -- title, authors, journal, DOI, beamline/facility
- **TL;DR** -- one-paragraph executive summary
- **Background & Motivation** -- why the work matters
- **Method** -- technical approach, model architecture, data pipeline
- **Key Results** -- quantitative outcomes and figures of merit
- **Data & Code Availability** -- links, reproducibility score (1-5)
- **Strengths** -- what the paper does well
- **Limitations & Gaps** -- where it falls short or what is missing
- **Relevance to APS BER Program** -- direct applicability to our program
- **Actionable Takeaways** -- concrete next steps for the team

---

## Contributing

To add a new review:

1. Copy `template_paper_review.md` into the appropriate subdirectory.
2. Rename it `review_<short_descriptor>_<year>.md`.
3. Fill in every section. Do not leave sections blank -- write "N/A" if truly
   not applicable.
4. Update this README and the subdirectory README with the new entry.
5. Commit with a message like `docs(pubs): add review of <Author> <Year>`.

---

## Maintainers

- APS BER program AI/ML team, Argonne National Laboratory
- Last updated: 2025-Q4
