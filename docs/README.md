---
doc_id: META-001
title: Documentation Map
status: draft
version: 0.1.0
last_updated: 2026-04-08
supersedes: null
related: []
---

# Documentation Map

This file is the central index of all project documentation for eBERlight Explorer.

## Status Legend

| Status | Meaning |
|--------|---------|
| `draft` | Content written, under review |
| `proposed` | Submitted for team approval |
| `accepted` | Approved and in effect |
| `superseded` | Replaced by a newer document |
| `planned` | Placeholder — not yet written |

## Product Layer (`docs/00_product/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `vision.md` | VIS-001 | planned | Mission, problem statement, target outcomes, non-goals |
| `personas.md` | PER-001 | planned | Three target personas with goals and pain points |
| `roadmap.md` | RMP-001 | planned | 5-phase, 12-week delivery roadmap |

## Requirements (`docs/01_requirements/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `PRD.md` | PRD-001 | planned | Product Requirements Document |
| `user_stories.md` | UST-001 | planned | User stories grouped by persona |
| `non_functional.md` | NFR-001 | planned | Accessibility, performance, security requirements |

## Design (`docs/02_design/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `information_architecture.md` | IA-001 | planned | Folder-to-cluster mapping, 4-zoom model, navigation |
| `design_system.md` | DS-001 | planned | Color tokens, typography, spacing, components |
| `wireframes/landing_v0.1.md` | WF-001 | planned | Landing page wireframe |
| `wireframes/section_v0.1.md` | WF-002 | planned | Section page wireframe |
| `wireframes/tool_v0.1.md` | WF-003 | planned | Tool detail page wireframe |

## Design Decisions (`docs/02_design/decisions/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `ADR-001.md` | ADR-001 | planned | Choose Streamlit over Next.js / Docusaurus / Jekyll |
| `ADR-002.md` | ADR-002 | planned | Notes remain single source of truth |
| `ADR-003.md` | ADR-003 | planned | YAML frontmatter schema for notes |
| `ADR-004.md` | ADR-004 | planned | 8 folders → 3 task clusters IA mapping |
| `ADR-005.md` | ADR-005 | planned | Adopt Argonne-aligned design tokens |
| `ADR-006.md` | ADR-006 | planned | Dual SemVer streams (notes vs explorer) |

## Implementation (`docs/03_implementation/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `setup.md` | IMP-001 | planned | Dev environment setup and run instructions |
| `coding_standards.md` | IMP-002 | planned | Python style, linting, commit conventions |
| `data_contracts.md` | DC-001 | planned | YAML frontmatter schema, controlled vocabularies |

## Testing (`docs/04_testing/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `test_plan.md` | TST-001 | planned | Test pyramid, tooling, coverage targets |
| `accessibility_audit.md` | TST-002 | planned | WCAG 2.1 AA checklist for Streamlit |

## Release (`docs/05_release/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `release_notes/` | — | planned | Per-version release notes directory |

## Meta (`docs/06_meta/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| `glossary.md` | GLO-001 | planned | Domain terminology definitions |
| `contributing.md` | CON-001 | planned | Branch naming, PR flow, ADR process |
