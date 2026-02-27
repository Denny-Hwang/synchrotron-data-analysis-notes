# Paper Review: AI for Advanced Light Sources -- Workshop Report

## Metadata

| Field              | Value                                                                                  |
|--------------------|----------------------------------------------------------------------------------------|
| **Title**          | AI for Advanced Light Sources: Workshop Report                                         |
| **Authors**        | ALS AI Working Group (multi-institutional collaborative report)                        |
| **Journal**        | Synchrotron Radiation News, 37(4), 14--22                                              |
| **Year**           | 2024                                                                                   |
| **DOI**            | [10.1080/08940886.2024.2391258](https://doi.org/10.1080/08940886.2024.2391258)         |
| **Beamline**       | Facility-wide (Advanced Light Source, LBNL); applicable across DOE light sources       |
| **Modality**       | Cross-cutting (all modalities)                                                          |

---

## TL;DR

The AI for Advanced Light Sources workshop report synthesizes community input from
over 100 beamline scientists, data engineers, and ML researchers across DOE light
source facilities to produce a comprehensive needs assessment and strategic roadmap
for deploying artificial intelligence across synchrotron operations. The report
identifies five priority areas -- autonomous experiments, real-time analysis, data
management, ML model lifecycle, and workforce development -- and provides ranked
recommendations that constitute a community-endorsed blueprint for AI adoption at
fourth-generation synchrotron facilities.

---

## Background & Motivation

The Advanced Light Source (ALS) at Lawrence Berkeley National Laboratory, along with
other DOE light sources undergoing major upgrades (APS-U, NSLS-II, LCLS-II-HE),
faces a critical inflection point: next-generation sources will produce data volumes
and acquisition rates that exceed the capacity of traditional analysis workflows by
one to two orders of magnitude. The APS upgrade alone will increase coherent flux by
100--500x, generating data streams of 10+ GB/s per beamline. Simultaneously, user
demand for complex multi-modal, time-resolved, and operando experiments is growing,
requiring real-time decision-making that manual analysis cannot support.

The AI@ALS workshop was convened to address three strategic questions: (1) Where can
AI/ML have the greatest impact on light source science and operations? (2) What
computing, data, and software infrastructure investments are needed to support AI/ML
deployment? (3) What organizational and workforce changes are required to sustain
AI/ML capabilities over the long term? The workshop brought together participants
from national laboratories (ANL, BNL, SLAC, ORNL, LBNL), universities, and industry
to build cross-facility consensus on priorities and approaches.

The resulting report serves both as a needs assessment grounded in quantitative survey
data and as a strategic planning document with prioritized, actionable recommendations
applicable to any fourth-generation synchrotron facility.

---

## Method

The workshop employed a structured methodology combining quantitative surveying with
expert deliberation:

1. **Pre-workshop survey**: Distributed to approximately 200 ALS staff and active
   users, covering current pain points, existing ML adoption levels, data
   infrastructure gaps, computational resource availability, and desired future
   capabilities. Response rate and demographic breakdown reported.

2. **Plenary sessions**: Invited presentations on AI/ML successes and lessons learned
   at peer facilities (ESRF, Diamond Light Source, DESY, APS, NSLS-II), establishing
   the international state of the art and avoiding redundant investment.

3. **Thematic breakout sessions**: Six parallel working groups organized by strategic
   theme:
   - Autonomous and adaptive experiments
   - Real-time data processing and reconstruction
   - Data management, curation, and FAIR principles
   - ML model development, training, and validation
   - Computing infrastructure (edge compute, on-premises GPU clusters, cloud, HPC)
   - Workforce development, training, and community building

4. **Synthesis and prioritization**: Breakout findings consolidated through a
   modified Delphi process, where participants ranked recommendations by impact,
   feasibility, and urgency, producing a prioritized action list.

5. **Report drafting and community review**: Multi-author collaborative writing
   with two rounds of community feedback before final publication.

---

## Key Results

The report identifies five priority areas with specific findings and recommendations:

| Priority Area                          | Key Findings / Recommendations                        |
|----------------------------------------|-------------------------------------------------------|
| **1. Autonomous experiments**          | Highest user demand. Adaptive scanning, self-driving beamlines, and real-time feedback loops cited as transformative. Most beamlines lack even basic ML-assisted data quality monitoring. Recommendation: develop modular autonomous experiment frameworks compatible with Bluesky. |
| **2. Real-time reconstruction**        | Critical for ptychography, tomography, XPCS. Latency requirements span milliseconds (ptychography) to seconds (tomography). GPU/FPGA edge compute needed at beamlines. Recommendation: standardize edge-compute hardware and software stack across beamlines. |
| **3. Data infrastructure & FAIR**      | FAIR data practices inconsistent across beamlines; metadata schemas vary widely. Unified data platforms (Tiled/Databroker) needed but not yet facility-wide. Recommendation: mandate FAIR-compliant metadata for all new experiments. |
| **4. ML model lifecycle management**   | No standardized pipeline for training, validation, deployment, monitoring, or retraining. Models degrade as beamline configurations evolve. Recommendation: build MLOps infrastructure with version control, automated validation, and drift detection. |
| **5. Workforce & organizational culture** | Shortage of ML-literate beamline scientists. Need for training programs, embedded data scientists, and career paths that value software/ML contributions. Recommendation: create data scientist positions at beamlines and fund synchrotron-ML training programs. |

Quantitative survey findings:

| Survey Finding                                | Statistic                                    |
|-----------------------------------------------|----------------------------------------------|
| Beamlines currently using any form of ML      | 23% (primarily image classification/sorting) |
| Beamlines wanting ML but lacking resources    | 61%                                           |
| Users reporting data analysis as bottleneck   | 78%                                           |
| Staff citing lack of ML training as barrier   | 54%                                           |
| Experiments estimated to benefit from autonomy| 45% (as estimated by beamline scientists)     |
| Median time from data collection to publication | 6--12 months (analysis-dominated)           |

### Key Figures

- **Figure 1**: Radar chart mapping current ML adoption (inner polygon) versus
  desired capability (outer polygon) across six dimensions: autonomous experiments,
  real-time processing, data management, model lifecycle, computing infrastructure,
  and workforce. The gap between current and desired states is largest for autonomous
  experiments and model lifecycle.
- **Figure 3**: Computing infrastructure schematic showing the proposed three-tier
  architecture: edge compute at the beamline for millisecond-latency tasks, on-
  premises GPU cluster for second-scale reconstruction and training, and HPC/cloud
  for large-scale model training and archival reprocessing.
- **Figure 5**: Bar chart of survey responses on analysis bottleneck severity by
  experimental technique, with ptychography and tomography users reporting the
  most severe bottlenecks.

---

## Data & Code Availability

| Resource       | Link / Note                                                           |
|----------------|-----------------------------------------------------------------------|
| **Code**       | N/A (workshop report, not a methods paper)                            |
| **Data**       | Survey results summarized in appendices; raw survey data not released  |
| **License**    | Open access article                                                    |
| **Reproducibility Score** | **N/A** -- Strategic planning document; value lies in community consensus and quantitative needs assessment rather than reproducible methodology. |

---

## Strengths

- Represents genuine community consensus from 100+ practitioners across multiple
  national laboratories, giving the recommendations broad institutional credibility
  and cross-facility applicability.
- Covers the full scope of AI/ML deployment -- from algorithms to infrastructure to
  workforce to organizational culture -- recognizing that technical solutions alone
  are insufficient without supporting ecosystem changes.
- Quantitative survey data grounds recommendations in measured need (78% analysis
  bottleneck, 61% want ML but lack resources) rather than speculation, providing
  concrete justification for investment decisions.
- Cross-facility perspective incorporates lessons learned from ESRF, Diamond, DESY,
  and other international facilities, enabling DOE facilities to avoid reinventing
  solutions to already-solved problems.
- The three-tier computing architecture (edge / on-premises GPU / HPC) provides a
  concrete, implementable infrastructure model that balances latency requirements
  with cost and scalability.

---

## Limitations & Gaps

- Primarily ALS-centric: while the recommendations are broadly applicable, specific
  infrastructure details and cost considerations reflect ALS's particular computing
  environment, beamline portfolio, and user demographics.
- Survey response rate and selection bias are not thoroughly analyzed; respondents
  likely over-represent ML enthusiasts, potentially inflating demand estimates.
- Limited discussion of AI/ML risks: model failure modes, adversarial robustness,
  hallucination in reconstruction, and the danger of over-automation without
  adequate scientific validation receive only brief treatment.
- No cost estimates, staffing projections, or implementation timelines for the
  recommended investments, making budget planning and prioritization difficult
  for facility management.
- The report predates rapid developments in foundation models and large language
  models for scientific applications (2024--2025) that could substantially shift
  the landscape for autonomous experiment design and literature-guided analysis.

---

## Relevance to APS BER Program

This workshop report is a strategic planning document with direct applicability to
the BER program's mission and organizational design:

- **Needs validation**: The survey findings (78% of users cite analysis as their
  primary bottleneck, 61% of beamlines want ML but lack resources) provide
  quantitative external validation for the BER program's core mission and justify the
  investment in AI-driven synchrotron science.
- **Priority alignment**: The BER program's focus areas -- autonomous experiments, real-
  time reconstruction, and adaptive scanning -- align precisely with the workshop's
  top two priority recommendations, confirming strategic direction.
- **Infrastructure blueprint**: The three-tier computing architecture (edge, on-
  premises, HPC) and data platform recommendations (FAIR metadata, Tiled/Databroker)
  should directly inform the BER program's infrastructure procurement and deployment
  decisions.
- **MLOps roadmap**: The BER program should build the standardized ML model lifecycle
  infrastructure (training, validation, deployment, monitoring, retraining with
  drift detection) that the report identifies as absent across all DOE facilities.
- **Workforce strategy**: The BER program's staffing plan should include embedded ML/data
  scientists at target beamlines, consistent with the workshop's workforce
  recommendations, to ensure sustained capability beyond initial deployment.
- **Multi-facility coordination**: The report's emphasis on cross-facility
  benchmarking, shared model repositories, and common software stacks supports
  the BER program's role in coordinating AI/ML efforts across APS, ALS, NSLS-II, and
  other DOE light sources.

---

## Actionable Takeaways

1. **Conduct APS-specific survey**: Replicate the AI@ALS survey at APS to quantify
   ML adoption, barriers, and demand specific to the BER program's target beamlines,
   providing a local baseline for measuring progress.
2. **Adopt FAIR data practices**: Prioritize facility-wide deployment of Tiled/
   Databroker with standardized metadata schemas across all BER program beamlines
   as foundational data infrastructure.
3. **Build MLOps pipeline**: Implement a standardized ML model lifecycle framework
   (MLflow, Weights & Biases, or custom tooling) for training, versioning,
   deploying, monitoring, and retraining models at BER program beamlines, with
   automated drift detection and alerting.
4. **Procure three-tier compute**: Budget for GPU/FPGA edge compute at each
   BER program-affiliated beamline, an on-premises GPU training cluster, and HPC
   allocations for large-scale training and archival reprocessing.
5. **Workforce investment**: Develop a synchrotron-ML training curriculum for
   beamline scientists and hire embedded data scientists at priority beamlines,
   incorporating the workshop's recommendation for hands-on bootcamps and mentored
   project-based learning.
6. **Cross-facility collaboration**: Establish formal collaboration channels with
   ALS, NSLS-II, and LCLS-II AI/ML teams to share models, benchmarks, and
   operational lessons, avoiding duplicated effort.

---

## BibTeX Citation

```bibtex
@article{als_ai_workshop_2024,
  title     = {{AI} for Advanced Light Sources: Workshop Report},
  author    = {{ALS AI Working Group}},
  journal   = {Synchrotron Radiation News},
  volume    = {37},
  number    = {4},
  pages     = {14--22},
  year      = {2024},
  publisher = {Taylor \& Francis},
  doi       = {10.1080/08940886.2024.2391258}
}
```

---

*Reviewed for the Synchrotron Data Analysis Notes, 2026-02-27.*
