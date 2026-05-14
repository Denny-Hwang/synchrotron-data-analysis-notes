---
doc_id: NFR-001
title: "Non-Functional Requirements"
status: draft
version: 0.1.0
last_updated: 2026-04-08
supersedes: null
related: [PRD-001, VIS-001, DS-001]
---

# Non-Functional Requirements

## Accessibility (WCAG 2.1 AA)

eBERlight Explorer targets WCAG 2.1 Level AA conformance.

### Specific Checks

| Criterion | WCAG Ref | Requirement |
|-----------|----------|-------------|
| Text contrast | 1.4.3 | Normal text ≥ 4.5:1 contrast ratio against background |
| Large text contrast | 1.4.3 | Large text (≥ 18px bold or ≥ 24px) ≥ 3:1 contrast ratio |
| Non-text contrast | 1.4.11 | UI components and graphical objects ≥ 3:1 contrast ratio |
| Keyboard navigable | 2.1.1 | All interactive elements reachable and operable via keyboard |
| Focus visible | 2.4.7 | Visible focus indicator on all interactive elements |
| Heading hierarchy | 1.3.1 | Headings follow logical order (H1 → H2 → H3), no skipped levels |
| Link purpose | 2.4.4 | Link text is descriptive; no bare "click here" links |
| Alt text | 1.1.1 | All images have descriptive alt text |
| Resize text | 1.4.4 | Content readable and functional at 200% zoom |
| Motion | 2.3.1 | No content flashes more than 3 times per second |

### Audit Cadence

- Accessibility audit before every explorer release (see `docs/04_testing/accessibility_audit.md`).
- Automated checks via axe DevTools and Lighthouse on every PR.

## Performance

| Metric | Target | Measurement |
|--------|--------|-------------|
| Landing page TTI (cached) | < 2 seconds | Streamlit app fully interactive after hot reload |
| Note render time | < 500 ms | Time from navigation click to full note display |
| Search response time | < 500 ms | Time from query submission to results display |
| App startup (cold) | < 10 seconds | Time from `streamlit run` to first interactive page |

### Measurement Method

- TTI measured via browser DevTools Performance tab.
- Note render time measured via Python `time.perf_counter()` instrumentation.
- Targets apply to a development machine with note corpus at current scale (~250 files).

## Browser Support

| Browser | Minimum Version | Support Level |
|---------|----------------|---------------|
| Chrome | 90+ | Full |
| Firefox | 90+ | Full |
| Safari | 15+ | Full |
| Edge | 90+ | Full |
| Mobile Safari (iOS) | 15+ | Responsive layout, no Streamlit-specific bugs |
| Chrome Mobile (Android) | 90+ | Responsive layout |

## Responsive Breakpoints

| Breakpoint | Width | Layout |
|------------|-------|--------|
| Mobile | 360px – 767px | Single column, collapsed navigation |
| Tablet | 768px – 1199px | Two columns, side navigation visible |
| Desktop | 1200px+ | Full layout, max container width 1200px |

All layouts use the 8px spacing grid defined in `docs/02_design/design_system.md`.

## Security & Privacy

- **No user data collection.** The explorer does not use cookies, analytics, tracking pixels, or any form of user identification.
- **No authentication.** The application is read-only and publicly accessible when served.
- **Static content only.** No server-side data processing, no database, no file writes.
- **No external API calls.** The explorer reads only local files. No network requests are made at runtime (exception: Streamlit's own WebSocket for hot reload).
- **Dependency security.** All Python dependencies are pinned in `requirements.txt`. Dependabot or equivalent should monitor for CVEs.

## Compliance

- **Personal-archive disclaimer.** The footer on every page MUST make clear that this is a personal study / learning project — a personal eBERlight archive — that is **not an official site** and **not affiliated with or endorsed by** ANL, APS, DOE, or the eBERlight program. The footer MUST NOT include the APS DOE contract number or any other language that could suggest institutional sponsorship.

- **Reference pointers to the official sources.** The footer MUST direct readers to the official APS (`https://www.aps.anl.gov/`) and eBERlight (`https://eberlight.aps.anl.gov/`) sites for the actual research, programs, and authoritative documentation.

- **Open Source License.** The repository is licensed under MIT. All third-party dependencies must have compatible licenses (MIT, BSD, Apache 2.0).
