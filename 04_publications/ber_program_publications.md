# BER Program-Attributed Publications (2023-2025)

## Overview

The BER program launched in **October 2023** at the Advanced Photon Source
(APS), Argonne National Laboratory. As a relatively new initiative coinciding
with the APS Upgrade (APS-U) era, the program's earliest publications are
emerging from **commissioning activities**, initial beamline characterization,
and proof-of-concept AI/ML integrations performed during the first operational
cycles.

This document provides a high-level overview of BER program-attributed
publications. A comprehensive and continuously updated list is maintained on the
**official program website** and in the APS publications database.

> **Canonical publication list:**
> Refer to the program website for the authoritative, up-to-date
> publication list. This document is a snapshot intended for offline reference
> and internal planning.

---

## Publication Timeline

### 2023 (Q4) -- Program Launch & Commissioning

The program's first quarter focused on infrastructure setup, beamline
commissioning at upgraded APS-U facilities, and establishing baseline
measurement protocols. Publications from this period are primarily:

- **Technical reports** on beamline commissioning status
- **Internal notes** on AI/ML pipeline architecture design
- **Conference abstracts** presented at synchrotron user meetings

Key activities that seeded later publications:

1. Initial deployment of streaming data pipelines at select beamlines
2. Benchmarking of GPU-accelerated reconstruction codes on APS computing
   infrastructure
3. Proof-of-concept unsupervised clustering applied to XRF mapping data
   collected during commissioning scans

### 2024 -- Early Results & Method Development

As the APS-U beamlines reached operational maturity, the BER program began producing
results from its AI/ML integration efforts:

- **AI-assisted data reduction** -- early demonstrations of real-time denoising
  and segmentation applied to commissioning datasets
- **Workflow automation** -- publications describing the integration of
  autonomous experiment steering with Bluesky/Tiled infrastructure
- **Collaborative papers** -- contributions to multi-institutional studies
  leveraging the BER program's computing and algorithmic capabilities
- **Workshop contributions** -- presentations and proceedings from the AI for
  Synchrotron Science workshops, including contributions to the AI@ALS workshop
  (reviewed separately in this archive)

### 2025 -- Maturing Pipeline & User Science

With beamlines fully operational and the AI/ML pipeline stabilized, the BER program's
publications in 2025 are expected to span:

- **User science papers** where the BER program's AI/ML tools enabled new measurements
  or accelerated data analysis for general user experiments
- **Methods papers** describing novel algorithms developed within the program
  (e.g., self-supervised denoising for low-dose XRF, real-time segmentation for
  tomography)
- **Infrastructure papers** detailing the full-stack data pipeline from detector
  to analysis, including edge computing and HPC integration
- **Review articles** synthesizing lessons learned from the first two years of
  AI/ML integration at a fourth-generation synchrotron

---

## Publication Categories

### Category 1: Core BER Program Methods

Papers where BER program team members are primary authors and the work was
primarily conducted within the program.

| # | Title (abbreviated) | Authors | Status | Target Journal |
|---|---------------------|---------|--------|---------------|
| _1_ | _To be added as publications are finalized_ | -- | -- | -- |

### Category 2: Collaborative / Multi-Beamline

Papers where the BER program contributed AI/ML analysis, computing infrastructure, or
algorithmic support to experiments led by other groups.

| # | Title (abbreviated) | BER Program Contribution | Status |
|---|---------------------|----------------------|--------|
| _1_ | _To be added_ | -- | -- |

### Category 3: Conference Proceedings & Workshop Reports

Shorter contributions to conferences, workshops, and user meetings.

| # | Venue | Title (abbreviated) | Year |
|---|-------|---------------------|------|
| _1_ | _To be added_ | -- | -- |

### Category 4: Preprints & Technical Reports

Works in progress, arXiv preprints, and internal technical reports.

| # | Title (abbreviated) | Repository | Status |
|---|---------------------|-----------|--------|
| _1_ | _To be added_ | -- | -- |

---

## Acknowledgment & Attribution Guidelines

All publications benefiting from the BER program's resources should include the
following acknowledgment (or equivalent):

> This research used resources of the Advanced Photon Source, a U.S. Department
> of Energy (DOE) Office of Science user facility at Argonne National
> Laboratory, and was supported by the eBERlight program under Contract No.
> DE-AC02-06CH11357.

For papers where the BER program's AI/ML tools were used but the primary science is
outside the program, co-authorship of relevant BER program team members should be
discussed and offered where appropriate per APS authorship guidelines.

---

## Metrics & Impact (Tracking)

As the publication portfolio grows, the following metrics will be tracked:

- Total publications per year
- Citations (tracked via Google Scholar and Web of Science)
- Beamline coverage (which beamlines have BER program-enabled publications)
- User vs. staff-led publications ratio
- Code/data release rate (fraction of papers with open code and data)

---

## How to Add a Publication

1. Verify the publication acknowledges the BER program appropriately.
2. Add the entry to the relevant category table above.
3. If the publication involves a novel AI/ML method, consider creating a
   detailed review using `template_paper_review.md` and adding it to the
   `ai_ml_synchrotron/` directory.
4. Update the canonical list on the program website.
5. Commit changes with: `docs(pubs): add <Author> <Year> to publication list`.

---

## Contact

For questions about BER program publications or to report a missing entry:

- BER program coordinators
- APS Scientific Publications Office

---

_Last updated: 2025-Q4_
