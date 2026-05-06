# Security Policy

## Reporting a vulnerability

If you discover a vulnerability in this repository — code, bundled data,
lazy-download recipes, or CI workflows — please **do not** open a public
GitHub issue.

Instead, contact the maintainer privately. For Denny-Hwang/synchrotron-data-analysis-notes,
use GitHub's [private vulnerability reporting](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability).

We aim to acknowledge reports within **5 business days** and provide a
remediation plan within **30 days** for confirmed issues.

## Scope

In scope:

- The Streamlit application in `explorer/`.
- The static-site generator in `scripts/build_static_site.py`.
- Pipeline functions in `experiments/`.
- The lazy-download infrastructure in `explorer/lib/model_zoo.py`.
- CI workflows in `.github/workflows/`.

Out of scope:

- The legacy `eberlight-explorer/` directory (deprecated per ADR-009 — receives no security updates; report only if the path is still reachable from GitHub Pages or the live Streamlit app).
- Bundled research data integrity (these are mirrored verbatim from upstream; report integrity issues to the original authors and to us via a GitHub issue rather than a private advisory).

## What counts as a vulnerability

- Remote code execution via crafted recipe.yaml, manifest.yaml, or note frontmatter.
- Path traversal via `manifest_path` or sample loaders.
- Arbitrary file disclosure on the static site.
- Lazy-download fetches that ignore declared SHA-256 hashes.
- License-warning suppression — i.e. a way for the Lab to download a CC-BY-NC or GPL artifact without the user seeing the licence string first (per ADR-008).
- Supply-chain compromise of declared upstream URLs (e.g. an upstream GitHub release that has been replaced).

## What does NOT count

- The fact that bundled samples are CC-BY-NC or GPL (these are documented in `10_interactive_lab/LICENSES/`).
- The fact that some `lazy_download_recipes.yaml` entries have `known_hash: null` — this is a known limitation tracked as a P0 issue and is intended to be tightened progressively as URLs are vetted.
- Streamlit framework-level issues that we cannot fix here — please report to upstream Streamlit.

## After a fix

We will publish a GitHub Security Advisory and a release note once the fix
is deployed. We credit reporters who consent.
