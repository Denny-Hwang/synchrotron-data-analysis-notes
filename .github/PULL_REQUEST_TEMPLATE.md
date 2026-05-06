## Summary

<!-- 1-3 sentences: what does this PR change and why? -->

## Doc ID(s)

<!-- Per CLAUDE.md invariant #2 — cite at least one of: FR-*, US-*, ADR-*, NFR-*. -->

## Type of change

- [ ] Feature
- [ ] Bug fix
- [ ] Documentation only
- [ ] Refactor / chore
- [ ] CI / tooling

## Checklist

- [ ] Followed branch naming (`feat/<scope>-…`, `fix/<scope>-…`, `docs/<scope>-…`, `chore/<scope>-…`).
- [ ] Updated `CHANGELOG.md` and `docs/05_release/release_notes/` (per CLAUDE.md invariant #3).
- [ ] If IA / design tokens / data contracts changed: created or updated the corresponding ADR (per CLAUDE.md invariant #4).
- [ ] If a wireframe changed: bumped the filename suffix (`_v0.X` → `_v0.Y`) — never overwrote an existing version (per CLAUDE.md invariant #5).
- [ ] All new `.md` files carry YAML frontmatter (per CLAUDE.md invariant #7).
- [ ] Ran the test suite locally: `pytest explorer/tests/` — all green.
- [ ] If this PR adds a recipe or sample: `pytest explorer/tests/test_lab_integrity.py explorer/tests/test_experiments.py` confirm the contract.
- [ ] No file > 100 MB added (free-tier constraint).
- [ ] No `Git LFS` track added without explicit discussion.

## Test plan

<!-- How did you verify this works? Include commands and expected output. -->

## Screenshots / before-after

<!-- For UI changes. -->
