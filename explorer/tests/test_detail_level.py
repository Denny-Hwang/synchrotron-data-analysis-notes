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


def test_l1_ignores_hashes_inside_python_code_fence() -> None:
    """Codex P2: lines starting with `#` inside a ``‌```python`` block
    are Python comments, not markdown headings."""
    body = textwrap.dedent(
        """\
        # Real Title

        ## Real Section

        ```python
        # this is a python comment, NOT a markdown heading
        # neither is this
        x = 1
        ```

        ## Another Real Section
        """
    )
    out = render_l1(body)
    # Real headings are present.
    assert "**Real Section**" in out
    assert "**Another Real Section**" in out
    # Phantom headings from the Python comments must NOT appear.
    assert "this is a python comment" not in out
    assert "neither is this" not in out


def test_l1_ignores_hashes_inside_shell_code_fence() -> None:
    """Same bug class with ``‌```bash`` blocks."""
    body = textwrap.dedent(
        """\
        ## Setup

        ```bash
        # install deps
        pip install foo
        ```

        ## Run
        """
    )
    out = render_l1(body)
    assert "**Setup**" in out
    assert "**Run**" in out
    assert "install deps" not in out


def test_l0_skips_fenced_code_when_finding_first_paragraph() -> None:
    """L0 must not reach into a fenced code block for its excerpt."""
    body = textwrap.dedent(
        """\
        # Title

        ```python
        # leading hash here is a comment, not the answer
        ```

        Real intro paragraph here.
        """
    )
    out = render_l0(body)
    assert "Real intro paragraph" in out
    assert "leading hash" not in out


def test_l2_returns_body_verbatim() -> None:
    assert render_l2(_SAMPLE) == _SAMPLE


def test_l3_wraps_in_markdown_fence() -> None:
    out = render_l3(_SAMPLE)
    # No inner backticks in the simple sample → 3-backtick fence is fine.
    assert out.startswith("```markdown\n")
    assert out.endswith("\n```")
    # Original content must be preserved verbatim.
    assert "# Ring Artifact" in out


def test_l3_preserves_inner_fences_verbatim() -> None:
    """Codex P2: notes with ```mermaid / ```python blocks must round-trip
    through L3 without backslash escapes — copy/paste must yield the
    exact original markdown."""
    body = "Has a fenced block:\n```python\nprint('x')\n```\n"
    out = render_l3(body)
    # Outer fence must be longer (≥4 backticks) so the inner triple
    # doesn't terminate it prematurely.
    assert out.startswith("````markdown\n")
    assert out.endswith("\n````")
    inner = out[len("````markdown\n") : -len("\n````")]
    # The inner ``` MUST appear verbatim — no \`\`\` escape, no replacement.
    assert "```python" in inner
    assert "```\n" in inner
    # The escape token from the buggy implementation must NOT leak in.
    assert "\\`" not in inner


def test_l3_handles_quadruple_fence_inside() -> None:
    """If the body itself contains a 4-backtick fence, outer must be ≥5."""
    body = "Outer 4-tick block:\n````\n```inner\n````\n"
    out = render_l3(body)
    assert out.startswith("`````markdown\n")
    assert out.endswith("\n`````")
    inner = out[len("`````markdown\n") : -len("\n`````")]
    assert "````" in inner


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


# ---------------------------------------------------------------------------
# extract_toc — table-of-contents helper for note-detail (Phase R8)
# ---------------------------------------------------------------------------


def test_extract_toc_returns_depth_anchor_heading() -> None:
    from lib.detail_level import extract_toc

    out = extract_toc(_SAMPLE)
    titles = [t for _, _, t in out]
    assert "Ring Artifact" in titles
    assert "Root Cause" in titles
    assert "Mitigations" in titles


def test_extract_toc_skips_in_fence_python_comments() -> None:
    """Lines starting with `#` inside a ``‌```python`` fence must not be
    surfaced as TOC entries — same regression test as the L1 outline
    helper from PR #45."""
    from lib.detail_level import extract_toc

    body = textwrap.dedent(
        """\
        # Real Title

        ## Real Section

        ```python
        # this is a comment
        x = 1
        ```

        ## Another Real
        """
    )
    out = extract_toc(body)
    titles = [t for _, _, t in out]
    assert "Real Title" in titles
    assert "Real Section" in titles
    assert "Another Real" in titles
    assert "this is a comment" not in titles


def test_extract_toc_anchors_are_lowercase_hyphenated() -> None:
    from lib.detail_level import extract_toc

    body = "# A Long, Punctuated Title!\n\nbody"
    out = extract_toc(body)
    assert out == [(1, "a-long-punctuated-title", "A Long, Punctuated Title!")]


def test_extract_toc_max_depth_filter() -> None:
    from lib.detail_level import extract_toc

    body = "# H1\n\n## H2\n\n### H3\n\n#### H4 should be filtered\n"
    out = extract_toc(body, max_depth=3)
    depths = [d for d, _, _ in out]
    assert depths == [1, 2, 3]
