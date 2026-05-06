"""Accessibility audit tests (Phase R7).

Exercises the design tokens used across the explorer + static-site
mirror against WCAG 2.1 AA contrast requirements. Failures here
mean a CSS / palette change broke contrast somewhere; rerun
``ruff format`` is no help — fix the colors.

Ref: NFR-001 — accessibility targets.
Ref: DS-001 — design tokens.
Ref: TST-002 — WCAG checklist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.a11y import (
    WCAG_AA_LARGE,
    WCAG_AA_NORMAL,
    alt_for_before_after,
    contrast_ratio,
    hex_to_rgb,
    passes_aa,
    passes_aaa,
    relative_luminance,
    skip_link_html,
)

# ---------------------------------------------------------------------------
# Hex parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value, expected",
    [
        ("#000000", (0, 0, 0)),
        ("#FFFFFF", (255, 255, 255)),
        ("#0033A0", (0, 0x33, 0xA0)),
        ("#fff", (255, 255, 255)),
        ("0033A0", (0, 0x33, 0xA0)),
    ],
)
def test_hex_to_rgb(value: str, expected: tuple[int, int, int]) -> None:
    assert hex_to_rgb(value) == expected


def test_hex_to_rgb_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        hex_to_rgb("not a hex")


# ---------------------------------------------------------------------------
# Contrast ratio (WCAG 2.1)
# ---------------------------------------------------------------------------


def test_contrast_ratio_black_on_white_is_max() -> None:
    """Per WCAG, the maximum contrast is 21:1."""
    ratio = contrast_ratio("#000000", "#FFFFFF")
    assert abs(ratio - 21.0) < 0.05


def test_contrast_ratio_identity_is_one() -> None:
    assert contrast_ratio("#0033A0", "#0033A0") == pytest.approx(1.0, abs=1e-6)


def test_contrast_ratio_is_symmetric() -> None:
    assert contrast_ratio("#0033A0", "#FFFFFF") == pytest.approx(
        contrast_ratio("#FFFFFF", "#0033A0"), abs=1e-6
    )


def test_relative_luminance_within_zero_one() -> None:
    for color in ("#000000", "#FFFFFF", "#0033A0", "#888888"):
        L = relative_luminance(hex_to_rgb(color))
        assert 0.0 <= L <= 1.0


# ---------------------------------------------------------------------------
# Design-token contrast — these must hold for the explorer to pass AA.
# ---------------------------------------------------------------------------


# Hard-coded design-system tokens. If you change these, the
# explorer's CSS *and* this test must change in lockstep.
# These values must agree with ``explorer/lib/ia.py``'s CLUSTER_META.
ANL_BLUE = "#0033A0"  # primary CTA / heading
WHITE = "#FFFFFF"
DARK_TEXT = "#1A1A1A"  # body text
SECONDARY_TEXT = "#555555"  # caption / muted prose
BUILD_ORANGE = "#D86510"  # build cluster + recipes (R7-darkened)
DISCOVER_BLUE = "#0033A0"  # discover cluster
EXPLORE_TEAL = "#0085C0"  # explore cluster (R7-darkened)
BANNER_BG = "#E8EEF6"  # filter banner / card backgrounds


@pytest.mark.parametrize(
    "fg, bg, label",
    [
        (DARK_TEXT, WHITE, "body text"),
        (SECONDARY_TEXT, WHITE, "secondary text"),
        (ANL_BLUE, WHITE, "primary heading"),
        (WHITE, ANL_BLUE, "header nav text on header bar"),
        (DARK_TEXT, BANNER_BG, "filter banner text"),
        (ANL_BLUE, BANNER_BG, "filter banner accent"),
    ],
)
def test_palette_passes_aa(fg: str, bg: str, label: str) -> None:
    ratio = contrast_ratio(fg, bg)
    assert passes_aa(fg, bg), (
        f"AA contrast failure for {label}: {fg} on {bg} = {ratio:.2f} (need ≥{WCAG_AA_NORMAL})"
    )


# Cluster accent colors are large-text only (used for H1/H2 + the
# header pill + cluster cards). Use the relaxed AA-large threshold.
@pytest.mark.parametrize(
    "fg",
    [BUILD_ORANGE, EXPLORE_TEAL, DISCOVER_BLUE],
)
def test_cluster_accents_pass_aa_large(fg: str) -> None:
    ratio = contrast_ratio(fg, WHITE)
    assert passes_aa(fg, WHITE, large_text=True), (
        f"AA-large contrast failure: {fg} on white = {ratio:.2f} (need ≥{WCAG_AA_LARGE})"
    )


# ---------------------------------------------------------------------------
# Severity badges (used on the troubleshooter cards) must be
# legible on white as both label and pill background.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fg, label",
    [
        ("#C0392B", "critical"),
        ("#C8550E", "major"),  # R7-darkened from #E67E22
        ("#2178B5", "minor"),  # R7-darkened from #3498DB
    ],
)
def test_severity_badges_have_legible_label(fg: str, label: str) -> None:
    """White text on the severity color must hit AA-large."""
    ratio = contrast_ratio("#FFFFFF", fg)
    assert passes_aa("#FFFFFF", fg, large_text=True), (
        f"severity badge ({label}): white on {fg} = {ratio:.2f}"
    )


# ---------------------------------------------------------------------------
# AAA helper smoke
# ---------------------------------------------------------------------------


def test_passes_aaa_strictly_implies_aa() -> None:
    """If a pair passes AAA it must also pass AA."""
    pair = ("#000000", "#FFFFFF")
    assert passes_aaa(*pair) and passes_aa(*pair)


# ---------------------------------------------------------------------------
# Alt-text helper
# ---------------------------------------------------------------------------


def test_alt_for_before_after_uses_human_label() -> None:
    assert "ring artifact" in alt_for_before_after("ring_artifact")
    assert "before" in alt_for_before_after("anything").lower()


def test_alt_for_before_after_handles_empty() -> None:
    assert alt_for_before_after("") == "Before / after comparison."


# ---------------------------------------------------------------------------
# Skip link
# ---------------------------------------------------------------------------


def test_skip_link_targets_main_content_by_default() -> None:
    out = skip_link_html()
    assert 'href="#main-content"' in out
    assert "Skip to main content" in out


def test_skip_link_target_is_overridable() -> None:
    out = skip_link_html(target_id="custom-target")
    assert 'href="#custom-target"' in out


# ---------------------------------------------------------------------------
# Token consistency — the values declared in this test file must agree
# with the design-system tokens shipped in lib/ia.py and
# lib/troubleshooter.py. If they drift, this test catches it before
# the audit ever runs in production.
# ---------------------------------------------------------------------------


def test_cluster_palette_matches_ia_module() -> None:
    from lib.ia import CLUSTER_META

    assert CLUSTER_META["discover"]["color"] == DISCOVER_BLUE
    assert CLUSTER_META["explore"]["color"] == EXPLORE_TEAL
    assert CLUSTER_META["build"]["color"] == BUILD_ORANGE


def test_severity_palette_matches_troubleshooter_module() -> None:
    from lib.troubleshooter import severity_color

    assert severity_color("critical") == "#C0392B"
    assert severity_color("major") == "#C8550E"
    assert severity_color("minor") == "#2178B5"
