---
doc_id: VIS-001
title: "eBERlight Explorer — Vision"
status: draft
version: 0.2.0
last_updated: 2026-05-14
supersedes: null
related: [PRD-001, PER-001, RMP-001, ADR-005, ADR-010]
---

# eBERlight Explorer — Vision

> **Personal research / learning workspace, not an official ANL/APS
> property.** This vision document was originally drafted with the
> framing of a polished public portal. After REL-E080 and REL-E082
> the scope was clarified: this is a personal eBERlight archive for
> one researcher's study, unaffiliated with and unendorsed by ANL,
> APS, DOE, or the eBERlight program. The original goals around
> navigation, progressive disclosure, and IA still apply; the
> "public-facing" / "stakeholder demo" / "DOE users" framing has
> been struck.

## Mission

Enable the author (acting as beamline-scientist / new-BER-user /
computational-researcher in turn) to discover, explore, and reuse
synchrotron data analysis knowledge through an interactive,
well-organized personal portal. The visual style is ANL/APS-inspired
for personal familiarity (see ADR-005 + ADR-010); no institutional
affiliation is claimed.

## Problem Statement

The synchrotron-data-analysis-notes repository is a rich knowledge base covering 6 X-ray modalities, 14 AI/ML methods, 14 publication reviews, 7 open-source tools, HDF5 data schemas, and a complete data pipeline architecture. However:

- **Navigation is file-system-driven.** Browsing the markdown
  directly requires knowing the folder structure. There is no
  task-oriented information architecture that guides reading by
  intent ("I want to learn about ptychography denoising" vs. "I
  want to run TomoPy").
- **No progressive disclosure.** Every reader — first-pass skim or
  deep dive — sees the same flat list of 200+ markdown files with
  no differentiation by expertise or task.
- **Cross-referencing is manual.** Connections between modalities, methods, tools, and publications exist only as inline markdown links, making it hard to see the bigger picture.
- **Visual identity is absent.** The legacy explorer
  (`eberlight-explorer/`, since superseded — see ADR-009) had no
  shared visual language with the notes themselves.

## Target Outcomes

1. **Reduce time-to-answer by 50%.** Looking for AI/ML methods applicable to a given modality should reach relevant content in < 3 clicks from the landing page.
2. **Provide three task-oriented entry points** (Discover the Program, Explore the Science, Build and Compute) aligned with the three primary intents identified during the author's own study workflow.
3. **Achieve WCAG 2.1 AA compliance** on all explorer pages so the personal archive remains usable across the author's own assistive-tech setups.
4. **Deliver a polished personal-archive experience.** The visual style is ANL/APS-inspired for the author's familiarity; no institutional branding is claimed.
5. **Maintain zero content duplication.** The explorer reads note folders at runtime; all content changes flow from the single source of truth.

## Non-Goals

- **We are NOT building a data analysis tool.** The explorer does not process, transform, or visualize raw scientific data. It presents documentation about how to do so.
- **We are NOT replacing or complementing eberlight.aps.anl.gov.** This is a personal archive of the author's study notes. The official eBERlight site is the only authoritative source for the BER program at APS; this repo links to it as a reference.
- **We are NOT hosting raw data.** Sample data links point to external repositories (TomoBank, CXIDB, PDB). No datasets are stored in this repo.
- **We are NOT building user accounts or authentication.** The explorer is a static-content, read-only application with no user data collection.
- **We are NOT publishing this app.** Per CLAUDE.md, the project is intended for local / private use only; original data owners have not been consulted regarding any public hosting.
