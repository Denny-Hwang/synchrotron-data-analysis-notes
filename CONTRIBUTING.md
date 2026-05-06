# Contributing

Thank you for considering a contribution. This file is the **entry point**;
the canonical guide lives at [`docs/06_meta/contributing.md`](docs/06_meta/contributing.md).

## TL;DR

1. **Open an issue first** for non-trivial changes — describe the problem and the proposed direction.
2. **Branch naming**: `feat/<scope>-<desc>`, `fix/<scope>-<desc>`, `docs/<desc>`, `chore/<desc>`.
3. **Every code change must reference a doc ID** (FR-*, US-*, ADR-*) per CLAUDE.md invariant #2. Cite it in the commit message or PR body.
4. **Every feature PR must update** `docs/05_release/release_notes/` **and** `CHANGELOG.md` per CLAUDE.md invariant #3.
5. **Run the test suite locally** before pushing:
   ```bash
   pip install -r explorer/requirements.txt
   pytest explorer/tests/
   ```
   GitHub Actions runs the same suite on Python 3.11 + 3.12 — see [`.github/workflows/test.yml`](.github/workflows/test.yml).
6. **Squash-and-merge** is the default merge strategy.

## Adding a noise-mitigation recipe

See [`experiments/README.md`](experiments/README.md) for the recipe schema. Each new
recipe must:

- ship a `recipe.yaml` and a pure-function `pipeline.py`,
- reference at least one bundled sample under `10_interactive_lab/datasets/`,
- pass the recipe-contract tests in `explorer/tests/test_experiments.py` and the lab-integrity tests in `explorer/tests/test_lab_integrity.py`,
- carry a citation block (BibTeX entry in `10_interactive_lab/CITATIONS.bib` is appreciated but not required).

## Adding bundled sample data

See [`10_interactive_lab/README.md`](10_interactive_lab/README.md) for the data layout
and licensing rules. Every dataset folder you add must include:

- an `ATTRIBUTION.md` with YAML frontmatter (test_lab_integrity validates this),
- a verbatim copy of the upstream `LICENSE` in `10_interactive_lab/LICENSES/`,
- a citation entry in `CITATIONS.bib`.

If the data is too large to bundle (>100 MB or shows licence-redistribution friction),
add a `lazy_download_recipes.yaml` entry instead and document the fetch path.

## Style and tooling

- Python: follow `docs/03_implementation/coding_standards.md` — PEP 8 + `ruff` + `black` + `mypy` (CI does not enforce these yet — local discipline expected).
- Markdown: every doc carries YAML frontmatter per ADR-003. See `docs/03_implementation/data_contracts.md` for the schema.
- Commit messages: Conventional Commits (`feat(scope): …`, `fix(scope): …`).

## Reporting bugs / requesting features

- Bug reports: open an issue using the **Bug report** template.
- Feature requests: open an issue using the **Feature request** template.
- Security vulnerabilities: **do not** open a public issue — see [SECURITY.md](SECURITY.md).

For anything else (e.g. discussing architecture), open a regular issue and tag with `discussion`.
