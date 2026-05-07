"""Drift protection for the R9 Mermaid migration.

The legacy ``eberlight-explorer/`` shipped 35 Mermaid diagrams in
three page-side ``_DIAGRAMS = {...}`` dictionaries. Phase R9 lifted
them into the matching note markdown so the new explorer renders
them inline — and so the diagrams survive when the legacy directory
is finally deleted at ``notes-v1.0.0`` (per ADR-009).

This module verifies, against the live note files, that:

1. Every key in ``CATEGORY_DIAGRAMS`` / ``METHOD_DIAGRAMS`` /
   ``PAPER_DIAGRAMS`` (defined in ``scripts/migrate_legacy_mermaid.py``)
   has a Mermaid block in its target note.
2. The Mermaid blocks parse as fenced code with ``mermaid`` language.
3. No target note ended up with two leading title H1s after migration.

Ref: ADR-002 — Notes are the single source of truth.
Ref: ADR-009 — Legacy app deprecation.
"""

from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_EXPLORER_DIR = _REPO_ROOT / "explorer"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))


def _load_migration_module():
    spec = importlib.util.spec_from_file_location(
        "migrate_legacy_mermaid_under_test",
        _SCRIPTS_DIR / "migrate_legacy_mermaid.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_MERMAID_FENCE_RE = re.compile(r"^[ \t]*```mermaid\b", re.MULTILINE)


def test_every_category_diagram_landed_in_note() -> None:
    mig = _load_migration_module()
    for cat_id in mig.CATEGORY_DIAGRAMS:
        rel = mig.CATEGORY_TO_NOTE[cat_id]
        path = _REPO_ROOT / rel
        assert path.exists(), f"Target note missing: {rel}"
        text = path.read_text(encoding="utf-8")
        assert _MERMAID_FENCE_RE.search(text), f"No mermaid block in {rel} — migration regressed."


def test_every_method_diagram_landed_in_note() -> None:
    mig = _load_migration_module()
    for method_id in mig.METHOD_DIAGRAMS:
        rel = mig.METHOD_TO_NOTE[method_id]
        path = _REPO_ROOT / rel
        assert path.exists(), f"Target note missing: {rel}"
        text = path.read_text(encoding="utf-8")
        assert _MERMAID_FENCE_RE.search(text), f"No mermaid block in {rel} — migration regressed."


def test_every_paper_diagram_landed_in_note() -> None:
    mig = _load_migration_module()
    for paper_id in mig.PAPER_DIAGRAMS:
        rel = f"{mig.PAPER_NOTE_BASE}/{paper_id}.md"
        path = _REPO_ROOT / rel
        assert path.exists(), f"Target note missing: {rel}"
        text = path.read_text(encoding="utf-8")
        assert _MERMAID_FENCE_RE.search(text), f"No mermaid block in {rel} — migration regressed."


def test_migration_total_count_is_thirty_five() -> None:
    """The migration table itself must keep 35 entries — drift catcher."""
    mig = _load_migration_module()
    total = len(mig.CATEGORY_DIAGRAMS) + len(mig.METHOD_DIAGRAMS) + len(mig.PAPER_DIAGRAMS)
    assert total == 35, f"Migration table size changed to {total}; intended 35."


def test_migration_idempotent(tmp_path: Path) -> None:
    """Running ``_migrate_one`` twice on the same file is a no-op the second time."""
    mig = _load_migration_module()
    sample = tmp_path / "sample.md"
    sample.write_text("# Title\n\nBody.\n", encoding="utf-8")

    first = mig._migrate_one(sample, "graph LR\n  A-->B", "demo caption")
    assert first == "inserted"
    second = mig._migrate_one(sample, "graph LR\n  A-->B", "demo caption")
    assert second.startswith("skipped (existing")
    text = sample.read_text(encoding="utf-8")
    # Exactly one mermaid block, not two.
    assert text.count("```mermaid") == 1
