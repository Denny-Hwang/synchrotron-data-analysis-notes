---
doc_id: UST-001
title: "User Stories"
status: draft
version: 0.1.0
last_updated: 2026-04-08
supersedes: null
related: [PRD-001, PER-001, VIS-001]
---

# User Stories

## Beamline Scientist (Persona A)

### US-001: Browse AI/ML methods by modality

> As a **Beamline Scientist**, I want to filter AI/ML methods by my X-ray modality, so that I only see methods proven for my data type.

**Acceptance Criteria:**

- Given I am on the "Explore the Science" cluster page, when I click the "XRF Microscopy" modality tag, then only notes tagged with `xrf_microscopy` are displayed.
- Given a method note has no modality tag, when I apply a modality filter, then the untagged note does not appear.
- Given I clear the filter, when the page reloads, then all notes in the cluster are displayed.

### US-002: View related publications from a method page

> As a **Beamline Scientist**, I want to see related publications linked from an AI/ML method page, so that I can read the supporting evidence.

**Acceptance Criteria:**

- Given I am viewing a method note with a `related_publications` field, when the page renders, then publication links appear in the metadata panel.
- Given the method note has no `related_publications`, when the page renders, then the metadata panel shows "No related publications" gracefully.

### US-003: Find tools used at my beamline

> As a **Beamline Scientist**, I want to browse tools filtered by beamline, so that I find software already deployed in my facility.

**Acceptance Criteria:**

- Given I am on the "Build and Compute" cluster page, when I click a beamline tag (e.g., `2-ID-E`), then only tools associated with that beamline are shown.
- Given a tool note lacks beamline metadata, when I apply a beamline filter, then the untagged tool does not appear.

### US-004: Navigate from modality to pipeline stage

> As a **Beamline Scientist**, I want to navigate from a modality overview to the relevant pipeline stage, so that I understand how my data flows from detector to storage.

**Acceptance Criteria:**

- Given I am viewing a modality note, when I click a "Data Pipeline" link in the metadata panel, then I am taken to the relevant pipeline stage note.
- Given the breadcrumb is visible, when I click a parent level, then I return to the cluster landing page.

---

## New BER User (Persona B)

### US-005: Understand the BER program from the landing page

> As a **New BER User**, I want to see a clear overview of the BER program on the landing page, so that I know what the program covers before diving in.

**Acceptance Criteria:**

- Given I land on the home page, when the page loads, then I see a hero section with a one-sentence program description and 3 cluster cards.
- Given I click the "Discover the Program" card, when the cluster page loads, then I see notes about program overview, facilities, and beamlines.

### US-006: Follow a learning path from overview to analysis

> As a **New BER User**, I want to follow a guided path from program overview → modalities → data formats → analysis tools, so that I build understanding progressively.

**Acceptance Criteria:**

- Given I am viewing a program overview note, when I finish reading, then I see "Next" suggestions linking to modality notes in the same cluster.
- Given I am viewing a modality note, when I look at the metadata panel, then I see links to related data format and tool notes.

### US-007: Look up unfamiliar terminology

> As a **New BER User**, I want to hover over or click unfamiliar terms (e.g., "ptychography", "HDF5") and see a brief definition, so that I can learn without leaving the page.

**Acceptance Criteria:**

- Given a glossary exists in `docs/06_meta/glossary.md`, when a known term appears in a note, then it is rendered with a tooltip or link to the glossary entry.
- Given the term is not in the glossary, when it appears in a note, then it renders as normal text with no broken link.

### US-008: See which research domains use which beamlines

> As a **New BER User**, I want to see a mapping of research domains to beamlines, so that I can find the right beamline for my field.

**Acceptance Criteria:**

- Given I am on the "Discover the Program" cluster page, when I navigate to the research domains note, then I see a table mapping domains to beamlines and modalities.

---

## Computational Scientist / Software Developer (Persona C)

### US-009: View the data pipeline end-to-end

> As a **Computational Scientist**, I want to see the full data pipeline (acquisition → streaming → processing → analysis → storage) in a single navigable view, so that I can identify optimization targets.

**Acceptance Criteria:**

- Given I am on the "Build and Compute" cluster page, when I click "Data Pipeline", then I see the pipeline overview with links to each stage.
- Given I click a pipeline stage, when the note loads, then the breadcrumb shows Build > Data Pipeline > [Stage Name].

### US-010: Find tool architecture and code entry points

> As a **Computational Scientist**, I want to view reverse-engineering docs for a tool (e.g., TomocuPy) with architecture diagrams and code pointers, so that I can start contributing quickly.

**Acceptance Criteria:**

- Given I am viewing a tool note, when the page renders, then I see sections for architecture, pros/cons, and reproduction guide.
- Given the tool note has code blocks, when they render, then syntax highlighting uses JetBrains Mono font and the design-system theme.

### US-011: Browse HDF5 data schemas

> As a **Computational Scientist**, I want to browse HDF5 schemas (XRF, tomography, ptychography) with field descriptions, so that I can write correct data loaders.

**Acceptance Criteria:**

- Given I am on the "Build and Compute" cluster page, when I navigate to Data Structures, then I see cards for each HDF5 schema.
- Given I click a schema card, when the note loads, then I see the schema tree with field names, types, and descriptions.

### US-012: Compare tools side by side

> As a **Computational Scientist**, I want to compare two tools (e.g., TomocuPy vs TomoPy) on key dimensions (speed, GPU support, API), so that I can choose the right one for my task.

**Acceptance Criteria:**

- Given I am on the "Build and Compute" cluster page, when I select two tool cards, then I see a comparison view with key attributes side by side.
- Given only one tool is selected, when I look at the page, then the comparison view is not shown.

### US-013: Replay a noise mitigation algorithm with parameter tuning

> As a **Computational Scientist**, I want to pick a noise mitigation recipe, choose a real bundled sample, and tune the algorithm's parameters interactively, so that I can build intuition for how a method behaves before applying it to my own data. (Ref: ADR-008, FR-017–FR-019.)

**Acceptance Criteria:**

- Given I open the Interactive Lab page, when I select a recipe from the sidebar, then I see the recipe title, description, primary citation, and parameter widgets auto-generated from `recipe.yaml`.
- Given I change a parameter slider, when the page re-renders, then the processed image updates within ~2 seconds (the cached pipeline run is reused if I revert).
- Given the chosen sample's shape matches the recipe's `clean_reference`, when the page re-renders, then PSNR and SSIM are shown with delta vs. the raw input.
- Given the chosen sample's shape differs from the `clean_reference`, when the page re-renders, then an info banner explains the metric skip and the visual comparison still works.

### US-014: Compare two algorithms on the same input

> As a **Beamline Scientist**, I want to apply different mitigation algorithms (e.g., sorting-based vs. wavelet-FFT) to the same bundled sinogram, so that I can decide which method is better suited to my detector's specific stripe profile. (Ref: ADR-008.)

**Acceptance Criteria:**

- Given two recipes target the same noise-catalog entry (e.g. `09_noise_catalog/tomography/ring_artifact.md`), when I switch between them in the sidebar, then I keep the same sample selected and the page re-runs only the pipeline.
- Given I keep the same sample across both recipes, when both finish processing, then I can visually inspect that the inputs are identical (sanity check) and compare the outputs.

### US-015: Spot cosmic-ray / zinger artifacts safely

> As a **Beamline Scientist**, I want to run an L.A.Cosmic-style detector on a real CCD frame and see how the threshold trades off detection vs. real-feature preservation, so that I learn to recognise zingers in my own tomography projections. (Ref: ADR-008.)

**Acceptance Criteria:**

- Given the cosmic-ray recipe is selected, when I lower the `sigclip` threshold below ~3.5, then I see real features starting to be flagged (visible as differences between input and output).
- Given the cosmic-ray recipe's primary citation is van Dokkum (2001), when I expand the references panel, then I can click through to the DOI.

### US-016: Reuse models without bundling weights

> As a **Computational Scientist**, I want the Lab to fetch model weights or large datasets only when I ask for them, with the license string visible before any download begins, so that I never accidentally pull a CC-BY-NC or GPL artifact onto my machine without consenting. (Ref: ADR-008, FR-021.)

**Acceptance Criteria:**

- Given a recipe declares an external model in `lazy_download_recipes.yaml` with `license_warning`, when I trigger the fetch, then the warning text is rendered before the download starts.
- Given the same model is requested again, when the file is already in the local cache, then the page reuses the cached copy with no network call.
