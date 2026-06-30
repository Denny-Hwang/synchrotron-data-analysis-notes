---
doc_id: META-001
title: Documentation Map
status: draft
version: 0.2.0
last_updated: 2026-05-14
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
| [`vision.md`](00_product/vision.md) | VIS-001 | draft | Mission, problem statement, target outcomes, non-goals |
| [`personas.md`](00_product/personas.md) | PER-001 | draft | Three target personas with goals and pain points |
| [`roadmap.md`](00_product/roadmap.md) | RMP-001 | draft | 5-phase, 12-week delivery roadmap |

## Requirements (`docs/01_requirements/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`PRD.md`](01_requirements/PRD.md) | PRD-001 | draft | Product Requirements Document |
| [`user_stories.md`](01_requirements/user_stories.md) | UST-001 | draft | User stories grouped by persona |
| [`non_functional.md`](01_requirements/non_functional.md) | NFR-001 | draft | Accessibility, performance, security requirements |

## Design (`docs/02_design/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`information_architecture.md`](02_design/information_architecture.md) | IA-001 | draft | Folder-to-cluster mapping, 4-zoom model, navigation |
| [`design_system.md`](02_design/design_system.md) | DS-001 | draft | Color tokens, typography, spacing, components |
| [`wireframes/landing_v0.1.md`](02_design/wireframes/landing_v0.1.md) | WF-001 | draft | Landing page wireframe ([HTML](02_design/wireframes/html/landing_v0.1.html)) |
| [`wireframes/section_v0.1.md`](02_design/wireframes/section_v0.1.md) | WF-002 | draft | Section page wireframe ([HTML](02_design/wireframes/html/section_v0.1.html)) |
| [`wireframes/tool_v0.1.md`](02_design/wireframes/tool_v0.1.md) | WF-003 | draft | Tool detail page wireframe ([HTML](02_design/wireframes/html/tool_v0.1.html)) |

## Design Decisions (`docs/02_design/decisions/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`ADR-001.md`](02_design/decisions/ADR-001.md) | ADR-001 | accepted | Choose Streamlit over Next.js / Docusaurus / Jekyll |
| [`ADR-002.md`](02_design/decisions/ADR-002.md) | ADR-002 | accepted | Notes remain single source of truth |
| [`ADR-003.md`](02_design/decisions/ADR-003.md) | ADR-003 | accepted | YAML frontmatter schema for notes |
| [`ADR-004.md`](02_design/decisions/ADR-004.md) | ADR-004 | accepted | 10 folders → 3 task clusters IA mapping |
| [`ADR-005.md`](02_design/decisions/ADR-005.md) | ADR-005 | accepted | Adopt Argonne-inspired design tokens (personal research) |
| [`ADR-006.md`](02_design/decisions/ADR-006.md) | ADR-006 | accepted | Dual SemVer streams (notes vs explorer) |
| [`ADR-007.md`](02_design/decisions/ADR-007.md) | ADR-007 | accepted | Static site mirror of Streamlit explorer for GitHub Pages |
| [`ADR-008.md`](02_design/decisions/ADR-008.md) | ADR-008 | accepted | Section 10 — Interactive Lab as a tenth note folder |
| [`ADR-009.md`](02_design/decisions/ADR-009.md) | ADR-009 | accepted | Deprecate the legacy `eberlight-explorer/` directory |
| [`ADR-010.md`](02_design/decisions/ADR-010.md) | ADR-010 | accepted | R7 secondary-color darkening + design-system tone reframing |

## Implementation (`docs/03_implementation/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`setup.md`](03_implementation/setup.md) | IMP-001 | draft | Dev environment setup and run instructions |
| [`coding_standards.md`](03_implementation/coding_standards.md) | IMP-002 | draft | Python style, linting, commit conventions |
| [`data_contracts.md`](03_implementation/data_contracts.md) | DC-001 | draft | YAML frontmatter schema, controlled vocabularies |
| [`github_pages_sync.md`](03_implementation/github_pages_sync.md) | IMPL-002 | accepted | GitHub Pages ↔ Streamlit explorer sync contract |

## Testing (`docs/04_testing/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`test_plan.md`](04_testing/test_plan.md) | TST-001 | draft | Test pyramid, tooling, coverage targets |
| [`accessibility_audit.md`](04_testing/accessibility_audit.md) | TST-002 | draft | WCAG 2.1 AA checklist for Streamlit |

## Release (`docs/05_release/`)

Per-version release notes live in
[`release_notes/`](05_release/release_notes/). The latest released
versions are **notes-v0.10.0** and **explorer-v0.8.3**; see
[`../CHANGELOG.md`](../CHANGELOG.md) for the full history.

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`release_notes/explorer-v0.8.3.md`](05_release/release_notes/explorer-v0.8.3.md) | REL-E083 | draft | Comprehensive review + framing cleanup |
| [`release_notes/explorer-v0.8.2.md`](05_release/release_notes/explorer-v0.8.2.md) | REL-E082 | draft | Streamlit footer reframed as personal eBERlight archive |
| [`release_notes/explorer-v0.8.1.md`](05_release/release_notes/explorer-v0.8.1.md) | REL-E081 | draft | R15.1 re-review follow-ups |
| [`release_notes/explorer-v0.8.0.md`](05_release/release_notes/explorer-v0.8.0.md) | REL-E080 | draft | Senior-review polish: tone + UX + code quality |
| [`release_notes/explorer-v0.7.x.md`](05_release/release_notes/) | REL-E07x | draft | R11 → R14 phases (see CHANGELOG) |
| [`release_notes/notes-v0.10.0.md`](05_release/release_notes/notes-v0.10.0.md) | REL-N100 | draft | Section 10 Interactive Lab |

## Meta (`docs/06_meta/`)

| Document | Doc ID | Status | Description |
|----------|--------|--------|-------------|
| [`glossary.md`](06_meta/glossary.md) | GLO-001 | draft | Domain terminology definitions |
| [`contributing.md`](06_meta/contributing.md) | CON-001 | draft | Branch naming, PR flow, ADR process |
