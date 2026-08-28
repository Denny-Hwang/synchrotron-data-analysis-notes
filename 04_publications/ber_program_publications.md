# BER Program-Attributed Publications (2023-2026)

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

With beamlines fully operational and the AI/ML pipeline stabilized, 2025
delivered facility-defining results and user-facing program activity:

- **World-record source performance confirmed**: measured 33 pm·rad
  horizontal emittance (announced May 2025); source emittance and coherence
  measurements published in Shi et al., *J. Synchrotron Rad.* 32(5) (2025)
- **Program calls**: FICUS FY2026 proposal call for biological and
  environmental research announced January 2025
- **User science ramp-up**: eBERlight environment and plant research at
  GSECARS 13-BM-D (April 2025)
- **Methods papers** in the wider APS orbit: self-supervised denoising for
  low-dose CT and XRF, diffusion-model ptychography reconstruction (see
  Category 4 below)

### 2026 -- APS-U Completion & Agentic AI

- **January 2026**: DOE grants final approval of the $815M APS Upgrade —
  on budget, ahead of schedule; completion celebrated February 5, 2026
- **Beamline capacity**: 72 beamlines at completion of the beamline
  upgrade program; the PtychoProbe (33-ID) is the final APS-U feature
  beamline, and the In Situ Nanoprobe (sector 19) is producing early
  results with ~20 nm focus
- **LLM agents reach real beamlines**: Argonne demonstrations of agentic
  instrument operation (Vriza et al., *npj Comput. Mater.* 2026); SLAC's
  agentic X-ray scientist (*Nat. Mach. Intell.* 2026)

---

## Publication Categories

> **Scope note (2026-08):** the tables below list *verified APS / APS-U-era
> publications relevant to the program's mission* collected for this personal
> archive. They are **not** an official BER attribution list — consult the
> program website for formal attribution.

### Category 1: Facility & Beamline Performance (APS-U era)

| # | Title (abbreviated) | Authors | Venue | Year |
|---|---------------------|---------|-------|------|
| 1 | Measurements of source emittance and beam coherence of the upgraded APS | Shi, X. et al. | J. Synchrotron Rad. 32(5), DOI 10.1107/S160057752500579X | 2025 |
| 2 | Returning to scientific operations at GM/CA@APS after the APS-Upgrade | Kissick, D. J. et al. | Structural Dynamics, DOI 10.1063/4.0001010 | 2025 |
| 3 | Macromolecular Crystallography at the Upgraded Advanced Photon Source | -- | Synchrotron Radiation News, DOI 10.1080/08940886.2026.2643150 | 2026 |

### Category 2: AI/ML Methods at APS & Partner Facilities

| # | Title (abbreviated) | Contribution | Venue | Year |
|---|---------------------|--------------|-------|------|
| 1 | Operating advanced scientific instruments with AI agents that learn on the job | LLM agents at an APS nanoprobe beamline | npj Comput. Mater. 12, 160, DOI 10.1038/s41524-026-02005-0 | 2026 |
| 2 | An agentic artificially intelligent X-ray scientist | LLM agent autonomously aligns crystals (SLAC) | Nat. Mach. Intell. 8, 1075–1086, DOI 10.1038/s42256-026-01261-5 | 2026 |
| 3 | Self-supervised deep-learning denoising for XRF microscopy with multi-element detectors | First ML denoiser for XRF (ESRF) | Anal. Chem. 98(11), DOI 10.1021/acs.analchem.5c05552 | 2026 |
| 4 | Noise2Inverse on bone SRμCT | Self-supervised low-dose CT denoising (ALS) | J. Synchrotron Rad. 32, 690–699, DOI 10.1107/S1600577525001833 | 2025 |
| 5 | Multi-stage DL artifact reduction for parallel-beam CT | Ring/artifact removal at the right pipeline stage | J. Synchrotron Rad. 32(2), 442–456, DOI 10.1107/S1600577525000359 | 2025 |

### Category 3: Program Activity & Calls

| # | Venue | Item | Year |
|---|-------|------|------|
| 1 | APS News | FICUS FY2026 proposal call for biological and environmental research (announced 2025-01-27) | 2025 |
| 2 | GSECARS | eBERlight environment and plant research at 13-BM-D | 2025 |
| 3 | APS News | DOE final approval + completion celebration of the APS Upgrade | 2026 |

### Category 4: Preprints & Software

| # | Title (abbreviated) | Repository | Year |
|---|---------------------|-----------|------|
| 1 | Pty-Chi: PyTorch-based modern ptychographic data analysis package (APS) | arXiv:2510.20929; github.com/AdvancedPhotonSource/pty-chi | 2025 |
| 2 | Ptychographic reconstruction from limited data via physics-guided diffusion models (Argonne) | arXiv:2502.18767 | 2025 |
| 3 | Fidelity-preserving enhancement of ptychography with foundational text-to-image models (APS) | arXiv:2509.04513 | 2025 |
| 4 | Modular framework for collaborative human-AI multi-beamline experiments (NSLS-II/NIST, Bluesky) | arXiv:2509.22959 | 2025 |
| 5 | PEAR: multi-LLM-agent automation for ptychography (Argonne/Rice) | arXiv:2410.09034 | 2024 |

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

_Last updated: 2026-08-28_
