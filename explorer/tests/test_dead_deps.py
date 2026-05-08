"""Drift protection for ``explorer/requirements.txt`` (Phase R13).

R11 replaced the Plotly + NetworkX renderer with vis.js but the deps
stayed in ``explorer/requirements.txt`` for two more releases — about
30 MB of wheel weight + ~30 s of cold start on Streamlit Cloud for
nothing. R13 dropped them; this test asserts they don't sneak back
unless something in ``explorer/`` actually imports them.

Add new dead-dep candidates to ``_EXPECTED_ABSENT`` to extend.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_REQ_PATH = _REPO_ROOT / "explorer" / "requirements.txt"
_EXPLORER_DIR = _REPO_ROOT / "explorer"

# Packages we have explicitly removed and do not want back. Keys are
# the package name as it would appear in requirements.txt; values are
# the import-name regex (Python identifier) used to grep the source.
_EXPECTED_ABSENT: dict[str, str] = {
    "plotly": r"plotly",
    "networkx": r"networkx",
}


def _collect_explorer_imports() -> str:
    """Return the concatenated source text of all ``.py`` files under explorer/."""
    chunks: list[str] = []
    for path in sorted(_EXPLORER_DIR.rglob("*.py")):
        # Skip the test file itself + __pycache__ artefacts.
        if path.name == "__pycache__" or "/__pycache__/" in path.as_posix():
            continue
        try:
            chunks.append(path.read_text(encoding="utf-8"))
        except OSError:
            continue
    return "\n".join(chunks)


def test_dead_deps_not_in_requirements() -> None:
    """``plotly`` and ``networkx`` must NOT be in requirements.txt."""
    if not _REQ_PATH.exists():
        return  # nothing to assert
    text = _REQ_PATH.read_text(encoding="utf-8")
    for pkg in _EXPECTED_ABSENT:
        # Match exactly the start of a line so a comment "# plotly was here"
        # doesn't trip the assertion.
        pattern = rf"^[ \t]*{re.escape(pkg)}\b"
        assert not re.search(pattern, text, flags=re.MULTILINE), (
            f"{pkg} is back in explorer/requirements.txt — was deliberately "
            "dropped in R13 (Rec #1). If you genuinely need it, update "
            "tests/test_dead_deps.py to remove the entry."
        )


def test_dead_deps_not_imported_in_source() -> None:
    """No ``import plotly`` / ``import networkx`` anywhere in explorer/."""
    source = _collect_explorer_imports()
    for pkg, import_re in _EXPECTED_ABSENT.items():
        # Match common import shapes: ``import plotly`` / ``from plotly.x import …``
        # Tolerant of aliases.
        pattern = rf"^\s*(?:from|import)\s+{import_re}(?:\.|\s|$)"
        m = re.search(pattern, source, flags=re.MULTILINE)
        assert m is None, (
            f"{pkg} import found in explorer/ — was supposed to be replaced "
            "by vis.js (R11). Re-add to explorer/requirements.txt and to "
            "tests/test_dead_deps._EXPECTED_ABSENT removal list if intentional."
        )
