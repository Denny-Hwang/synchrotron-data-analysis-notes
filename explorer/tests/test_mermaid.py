"""Tests for Mermaid diagram rendering across the explorer + static site.

The explorer renders ``‌```mermaid`` fenced code blocks via a
Streamlit components iframe (loaded from the public mermaid CDN);
the static site replaces the same blocks with ``<div class="mermaid">``
plus a single script tag in the page head. Both paths must:

1. extract every block in the markdown body (one or many),
2. preserve the exact mermaid source verbatim — no escaping that
   breaks Mermaid's own grammar (``[label]``, ``-->``, ``;``, …),
3. only inject the runtime script once per page (static site),
4. be a no-op for notes that have no Mermaid blocks.

Phase R3 — see CHANGELOG.

Ref: ADR-002 — diagrams live inside the note markdown, not
    page-side dicts. Adding the rendering pipeline doesn't reintroduce
    a separate diagram catalogue.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))


# ---------------------------------------------------------------------------
# Streamlit-side: components.note_view._MERMAID_BLOCK pattern
# ---------------------------------------------------------------------------


def test_mermaid_block_pattern_matches_simple_block() -> None:
    from components.note_view import _MERMAID_BLOCK

    body = "Some text\n\n```mermaid\nflowchart LR\nA-->B\n```\n\nMore text."
    matches = list(_MERMAID_BLOCK.finditer(body))
    assert len(matches) == 1
    assert matches[0].group("code") == "flowchart LR\nA-->B"


def test_mermaid_block_pattern_handles_multiple_blocks() -> None:
    from components.note_view import _MERMAID_BLOCK

    body = (
        "Intro.\n\n"
        "```mermaid\nflowchart LR\nA-->B\n```\n\n"
        "Mid.\n\n"
        "```mermaid\nsequenceDiagram\nA->>B: hi\n```\n\n"
        "End."
    )
    matches = list(_MERMAID_BLOCK.finditer(body))
    assert len(matches) == 2
    assert "flowchart LR" in matches[0].group("code")
    assert "sequenceDiagram" in matches[1].group("code")


def test_mermaid_block_pattern_does_not_match_inline_code() -> None:
    from components.note_view import _MERMAID_BLOCK

    body = "An inline `mermaid` reference, not a code block."
    assert not list(_MERMAID_BLOCK.finditer(body))


def test_mermaid_block_pattern_does_not_match_other_languages() -> None:
    from components.note_view import _MERMAID_BLOCK

    body = "```python\nprint('mermaid')\n```"
    assert not list(_MERMAID_BLOCK.finditer(body))


def test_mermaid_block_pattern_tolerates_trailing_whitespace_after_lang_tag() -> None:
    from components.note_view import _MERMAID_BLOCK

    body = "```mermaid   \nA-->B\n```"
    assert list(_MERMAID_BLOCK.finditer(body))


# ---------------------------------------------------------------------------
# Static-site side: scripts/build_static_site._replace_mermaid_blocks
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bss():
    """Load the static-site generator as an importable module."""
    pytest.importorskip("markdown")
    pytest.importorskip("pygments")
    spec = importlib.util.spec_from_file_location(
        "build_static_site_under_test_mermaid",
        _SCRIPTS_DIR / "build_static_site.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("could not load build_static_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_extract_then_replace_round_trip(bss) -> None:
    """Mermaid-source survives the extract → render-markdown → replace cycle."""
    body = "Intro.\n\n```mermaid\nflowchart LR\nA-->B\n```\n\nEnd."
    body_clean = bss._extract_mermaid_blocks(body)
    # Placeholder must NOT contain the raw arrow (which would close
    # an HTML comment prematurely on certain renderers).
    assert "A-->B" not in body_clean
    # Run a stand-in markdown step so we exercise the same path the
    # generator does — codehilite must NOT see the diagram source.
    import markdown as _md

    html = _md.markdown(body_clean, extensions=["fenced_code", "tables", "toc", "codehilite"])
    out, has_mermaid = bss._replace_mermaid_blocks(html)
    assert has_mermaid is True
    assert '<div class="mermaid">' in out
    # The diagram source — including the literal arrow — round-trips
    # verbatim through base64 encoding.
    assert "flowchart LR" in out
    assert "A-->B" in out


def test_static_no_op_without_mermaid(bss) -> None:
    """If no placeholders are present, the HTML is returned unchanged."""
    html = "<p>nothing here</p><pre><code>not mermaid</code></pre>"
    out, has_mermaid = bss._replace_mermaid_blocks(html)
    assert has_mermaid is False
    assert out == html


def test_static_handles_multiple_blocks(bss) -> None:
    """Two ```mermaid blocks → two <div class='mermaid'> elements."""
    body = "```mermaid\na\n```\n\nmiddle\n\n```mermaid\nb\n```\n"
    body_clean = bss._extract_mermaid_blocks(body)
    import markdown as _md

    html = _md.markdown(body_clean, extensions=["fenced_code", "tables", "toc", "codehilite"])
    out, has_mermaid = bss._replace_mermaid_blocks(html)
    assert has_mermaid is True
    assert out.count('<div class="mermaid">') == 2


def test_static_mermaid_head_contains_cdn_script(bss) -> None:
    """The injected page-head fragment must load mermaid.min.js + initialise once."""
    head = bss._MERMAID_HEAD
    assert "mermaid" in head.lower()
    assert "cdn.jsdelivr.net" in head
    assert "mermaid.initialize" in head


# ---------------------------------------------------------------------------
# Integration — the bundled demo notes actually contain mermaid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel",
    [
        "07_data_pipeline/README.md",
        "03_ai_ml_methods/denoising/tomogan.md",
        "09_noise_catalog/tomography/ring_artifact.md",
    ],
)
def test_demo_note_contains_mermaid_block(rel: str) -> None:
    """The three demo notes shipped in R3 each carry at least one mermaid block."""
    from components.note_view import _MERMAID_BLOCK

    path = _REPO_ROOT / rel
    if not path.exists():
        pytest.skip(f"{rel} not present")
    text = path.read_text(encoding="utf-8")
    matches = list(_MERMAID_BLOCK.finditer(text))
    assert matches, f"{rel} should ship at least one ```mermaid block"


def test_static_site_full_build_includes_mermaid_for_demo_notes(bss, tmp_path) -> None:
    """End-to-end: the rendered HTML for a demo note must contain mermaid markup."""
    out = tmp_path / "site_mermaid"
    bss.build(out)
    pipeline_html = out / "notes" / "07_data_pipeline" / "README.html"
    if not pipeline_html.is_file():
        pytest.skip("static-site build did not produce the pipeline note")
    text = pipeline_html.read_text(encoding="utf-8")
    assert '<div class="mermaid">' in text
    assert "cdn.jsdelivr.net" in text  # mermaid runtime script tag
