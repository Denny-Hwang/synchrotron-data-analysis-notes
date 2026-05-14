---
doc_id: REL-E082
title: "Release Notes — explorer-v0.8.2"
status: draft
version: 0.8.2
last_updated: 2026-05-14
supersedes: null
related: [REL-E080, REL-E081, FR-010, NFR-001]
---

# Release Notes — explorer-v0.8.2

**Patch — finish the personal-research footer reframing.**

## Summary

REL-E080 (explorer-v0.8.0) reframed the project tone from
"ANL-aligned" to "personal research, unaffiliated" and updated the
static-site mirror's footer accordingly. The **Streamlit explorer's
own footer** was missed in that pass and was still rendering the
old institutional copy — the DOE funding statement with Contract
No. DE-AC02-06CH11357, plus a definitional "eBERlight is the
integrated BER program at APS…" paragraph and bare `APS` /
`eBERlight` link labels that read like an official portal.

This patch brings the Streamlit footer in line with the
personal-archive framing already shipped in the static site,
and tightens the wording to make the framing unmistakable: this
is a personal eBERlight archive, not an official site.

## What changed

### 1. `explorer/components/footer.py` — copy rewritten

- The first paragraph now leads with a bold
  **"Personal eBERlight archive — not an official site."** and
  explicitly states the project is unaffiliated with and not
  endorsed by ANL, APS, DOE, or the eBERlight program.
- The DOE funding statement and Contract No. DE-AC02-06CH11357
  are **removed**. They implied institutional sponsorship that
  doesn't exist for this repo.
- The second paragraph points readers to the official APS /
  eBERlight sites for the actual research, programs, beamtime
  calls, and authoritative documentation, and notes that any
  opinions or mistakes in the notes are the author's own.
- The link labels are reframed as references: `APS (official —
  actual research here)` and `eBERlight (official — actual
  research here)` instead of the bare names. The `Repository`
  link is unchanged.

### 2. `explorer/tests/test_components.py` — assertions updated

`test_render_footer` previously asserted that `DE-AC02-06CH11357`
appears in the rendered HTML. Inverted to assert the contract
number is **absent**, plus new assertions for the personal-archive
disclaimer text and both reference URLs.

### 3. `docs/01_requirements/PRD.md` — FR-010 rewritten

FR-010 previously read "The DOE acknowledgment footer (Contract
No. DE-AC02-06CH11357) SHALL appear on every page." It now
mandates a personal-archive disclaimer footer that explicitly
disclaims affiliation and points readers to the official APS /
eBERlight sites for the actual research.

### 4. `docs/01_requirements/non_functional.md` — Compliance section rewritten

The "DOE Acknowledgment" requirement (which mandated the DOE
funding sentence with the contract number) and the standalone
"eBERlight Program Acknowledgment" bullet are replaced with two
new bullets:

- **Personal-archive disclaimer** — footer must make the
  unaffiliated framing unmistakable and MUST NOT include the
  contract number or any other institutional-sponsorship cue.
- **Reference pointers to the official sources** — footer must
  direct readers to the official APS and eBERlight sites for the
  actual research.

The MIT licensing bullet is unchanged.

## Out of scope

- The static-site footer (`scripts/build_static_site.py`) — this
  was already reframed in REL-E080 and only needed light wording
  alignment, which has been folded in.
- Historic wireframes
  (`docs/02_design/wireframes/html/*_v0.1.html`) — versioned
  wireframes are immutable per invariant #5; the contract-number
  text in them is preserved as a historical reference, consistent
  with how REL-E080 / ADR-010 handle the pre-darkening hex values.
- Release notes for earlier versions, the upstream `04_publications`
  citation example, and `10_interactive_lab/datasets/tomography/…/ATTRIBUTION.md`
  — these all mention the contract number in contexts where it is
  factually correct (TomoPy license requirement, upstream
  publication metadata) and not as a footer of this app.

## Traceability

- FR-010 (rewritten in this release) — personal-archive
  disclaimer footer on every page.
- NFR-001 (Compliance section rewritten) — personal-archive
  framing + reference pointers, no contract number.
- Invariant #9 (GitHub Pages MUST mirror the Streamlit explorer)
  — this release brings the Streamlit footer back in sync with
  the static site after REL-E080.
