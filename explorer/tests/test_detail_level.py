"""Tests for the L0/L1/L2/L3 progressive-disclosure helpers (Phase R5)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.detail_level import (
    LEVEL_HELP,
    LEVEL_LABELS,
    LEVELS,
    normalise_level,
    render,
    render_l0,
    render_l1,
    render_l2,
    render_l3,
)

_SAMPLE = textwrap.dedent(
    """\
    # Ring Artifact

    Concentric rings appear in CT slices when one or more detector
    columns are defective.

    ## Root Cause

    Constant offset on a detector column produces a vertical stripe
    in the sinogram.

    ## Mitigations

    Sorting-based filter (Vo 2018) or wavelet-FFT (Munch 2009).
    """
)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------


def test_levels_canonical_order() -> None:
    assert LEVELS == ("L0", "L1", "L2", "L3")


def test_label_and_help_for_every_level() -> None:
    for lvl in LEVELS:
        assert LEVEL_LABELS.get(lvl)
        assert LEVEL_HELP.get(lvl)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_l0_returns_first_paragraph_only() -> None:
    out = render_l0(_SAMPLE)
    assert "Concentric rings appear" in out
    # Must not include downstream H2 headings or content.
    assert "Root Cause" not in out
    assert "Mitigations" not in out


def test_l0_strips_top_h1() -> None:
    out = render_l0(_SAMPLE)
    assert "# Ring Artifact" not in out
    assert "Ring Artifact" not in out  # title not duplicated as prose


def test_l1_lists_h2_headings_with_first_sentence() -> None:
    out = render_l1(_SAMPLE)
    assert "**Root Cause**" in out
    assert "**Mitigations**" in out
    # H1 (the title) is intentionally dropped — page renders it above.
    assert "**Ring Artifact**" not in out


def test_l1_falls_back_to_l0_for_no_headings() -> None:
    body = "Just one paragraph, no headings at all."
    assert render_l1(body) == render_l0(body)


def test_l2_returns_body_verbatim() -> None:
    assert render_l2(_SAMPLE) == _SAMPLE


def test_l3_wraps_in_markdown_fence() -> None:
    out = render_l3(_SAMPLE)
    assert out.startswith("```markdown\n")
    assert out.endswith("\n```")
    # Original content must be preserved (modulo backtick escaping).
    assert "# Ring Artifact" in out


def test_l3_escapes_inner_fences() -> None:
    body = "Has a fenced block:\n```python\nprint('x')\n```\n"
    out = render_l3(body)
    # Inner triple-backticks must not break the outer fence.
    assert out.startswith("```markdown\n")
    assert out.endswith("\n```")
    # The escape token uses backslashes — the actual ``` should NOT
    # appear unescaped inside the wrapped body.
    inner = out[len("```markdown\n") : -len("\n```")]
    assert "```" not in inner


# ---------------------------------------------------------------------------
# Dispatcher + query-param normalisation
# ---------------------------------------------------------------------------


def test_render_dispatch_uses_correct_function() -> None:
    assert render("L0", _SAMPLE) == render_l0(_SAMPLE)
    assert render("L1", _SAMPLE) == render_l1(_SAMPLE)
    assert render("L2", _SAMPLE) == render_l2(_SAMPLE)
    assert render("L3", _SAMPLE) == render_l3(_SAMPLE)


def test_render_unknown_level_falls_back_to_l2() -> None:
    assert render("L9", _SAMPLE) == render_l2(_SAMPLE)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("L0", "L0"),
        ("l0", "L0"),
        ("Overview", "L0"),
        ("summary", "L0"),
        ("L1", "L1"),
        ("Sections", "L1"),
        ("Outline", "L1"),
        ("L2", "L2"),
        ("Details", "L2"),
        ("Full", "L2"),
        ("L3", "L3"),
        ("Source", "L3"),
        ("raw", "L3"),
        ("nonsense", "L2"),
        (None, "L2"),
        ("", "L2"),
    ],
)
def test_normalise_level(raw: str | None, expected: str) -> None:
    assert normalise_level(raw) == expected


def test_normalise_level_custom_default() -> None:
    assert normalise_level(None, default="L0") == "L0"
