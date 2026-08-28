---
doc_id: REL-N110
title: "Release Notes — notes-v0.11.0"
status: draft
version: 0.11.0
last_updated: 2026-08-28
supersedes: null
related: [ADR-006, ADR-008, REL-N100, REL-E090]
---

# Release Notes — notes-v0.11.0

**2025–2026 research refresh + citation hygiene.** Minor release of the
notes stream: new verified publications and method notes, APS-U facility
facts brought current, and two fabricated/duplicate reviews retired.

## New content

### Publication reviews (`04_publications/ai_ml_synchrotron/`) — 13 → 15

- `review_noise2inverse_bone_2025.md` — Obata, Parkinson, Pelt &
  Acevedo, *J. Synchrotron Rad.* 32, 690–699 (2025): self-supervised
  Noise2Inverse on real bone SRμCT, 2–3× dose reduction.
- `review_selfsupervised_xrf_denoise_2026.md` — Shishkov et al.,
  *Anal. Chem.* 98(11) (2026): first ML denoiser purpose-built for XRF,
  using multi-element-detector redundancy as Noise2Noise pairs.
- `review_agentic_xray_scientist_2026.md` — Chen et al., *Nat. Mach.
  Intell.* 8, 1075–1086 (2026): LLM agent autonomously aligns single
  crystals on a real beamline (simulator-first development).

### Method notes (`03_ai_ml_methods/`) — 22 → 24 files

- `denoising/noise2inverse.md` — fills the Noise2X-family gap
  (projection-split self-supervision for tomographic inverse problems).
- `autonomous_experiment/llm_agents_beamline.md` — survey of the
  2024–2026 LLM-agent beamline demonstrations (SLAC agentic scientist,
  Argonne "learn on the job" agents, PEAR, NSLS-II Bluesky framework).

### Program overview (`01_program_overview/aps_facility.md`)

- Measured world-record **33 pm·rad** emittance (May 2025) added
  alongside the 42 pm·rad design figure, with the Shi et al. *JSR*
  32(5) (2025) measurement citation.
- Timeline extended: Jan 2026 DOE final approval of the $815M APS-U
  (on budget, ahead of schedule) and the Feb 2026 completion
  celebration; beamline count updated to **72 at completion of the
  beamline upgrade program**.

### References (`08_references/bibliography.bib`) — 18 → 27 entries

Added: Shi 2025 (emittance), Kissick 2025 (GM/CA return),
Hendriksen 2020 (Noise2Inverse), Obata 2025, Shi 2025 (multi-stage
artifact reduction), Shishkov 2026, Chen 2026, Vriza 2026, Pty-Chi
(arXiv:2510.20929). `useful_links.md` gains Pty-Chi under Ptychography
tools.

### Publications tracker (`04_publications/ber_program_publications.md`)

The four placeholder category tables are populated with verified
2025–2026 items (facility performance, AI/ML methods, program calls,
preprints/software) under an explicit scope note that this is a
personal-archive collection, not an official attribution list. The
"2025 — expected" narrative is replaced with what actually happened,
plus a 2026 section (APS-U completion, agentic AI).

## Retired content (citation hygiene)

- **`review_fullstack_tomo_2023.md` deleted** — its citation was
  fabricated: DOI `10.1016/j.fmre.2023.11.003` resolves to an unrelated
  lipid-nanomedicine review, and the claimed title/author list does not
  exist. The real full-stack paper (Zhang et al., *The Innovation*,
  `10.1016/j.xinn.2023.100539`) was already reviewed in
  `review_fullstack_dl_tomo_2023.md`.
- **`review_aiedge_ptycho_2023.md` deleted** — duplicate of
  `review_ai_edge_ptychography_2023.md` (same DOI
  `10.1038/s41467-023-41496-z`). The kept review's metadata is
  corrected to the real title ("Deep learning at the edge enables
  real-time streaming ptychographic imaging", Nat. Commun. 14, 7059).
- The legacy-Mermaid migration table drops the two matching entries
  (35 → 33; drift-catcher test updated).

## Interactive Lab data corrections (`10_interactive_lab/`)

- `manifest.yaml`: the seven ring-artifact TIFFs are `float32`
  (normalised transmission), not `tiff_uint16`; header rewritten to
  describe the manifest honestly as a CI-validated provenance
  inventory (the shipped Lab page enumerates `recipe.yaml` samples).
- Ring-artifact `ATTRIBUTION.md` and the Lab README dtype/shape
  examples corrected to match the actual bytes.
- Phase-wrapping `ATTRIBUTION.md` no longer claims a bundled synthesis
  script exists (the committed `.npy` files are canonical).
- Beam-hardening attribution: "2nd-order" → cubic polynomial;
  `external_data_sources.md` file-count and manifest-path claims fixed.
- `CITATIONS.bib` 37 → 40: adds Münch 2009 (implemented by the
  wavelet-FFT recipe but previously uncited here), Buades 2010,
  Joseph & Spital 1978.
- Licensing summary gains the previously-orphaned Algotom row;
  `models/README.md` + `SECURITY.md` now state the lazy-download layer
  is staged, not live.
- `09_noise_catalog/troubleshooter.yaml`: five diagnoses that already
  had matching Lab recipes gain `recipe:` links (flat-field, beam
  hardening, dead pixel, low dose, phase wrapping) — Lab reachability
  from the Troubleshooter roughly triples.

Traceability: ADR-008 v0.2.0 (data contract), invariant #1 (notes are
the single source of truth), invariant #7 (frontmatter).
