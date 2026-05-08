"""Accessibility helpers (Phase R7).

The eBERlight Explorer commits to **WCAG 2.1 AA**. This module
encapsulates the small bits of logic that benefit from being
unit-tested (color-contrast ratio, design-token validation, alt-text
helpers) so the audit is verifiable in CI and not just a one-shot
manual check.

Pure data layer — no Streamlit, no I/O.

Ref: NFR-001 (non_functional.md) — accessibility targets.
Ref: DS-001 (design_system.md) — color tokens.
Ref: TST-002 (accessibility_audit.md) — WCAG checklist.
Ref: FR-014, FR-015, FR-016 — TTI / keyboard / contrast.
"""

from __future__ import annotations

import re

# WCAG 2.1 AA contrast thresholds.
WCAG_AA_NORMAL = 4.5  # body text
WCAG_AA_LARGE = 3.0  # ≥18pt or ≥14pt bold
WCAG_AAA_NORMAL = 7.0  # AAA-level guarantee for body text


_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    """Parse ``#rgb`` or ``#rrggbb`` into a ``(r, g, b)`` int triple.

    Raises ``ValueError`` for malformed input — callers shouldn't be
    passing untrusted strings here so the strict failure is fine.
    """
    m = _HEX_RE.match(value.strip())
    if not m:
        raise ValueError(f"not a hex color: {value!r}")
    digits = m.group(1)
    if len(digits) == 3:
        digits = "".join(c * 2 for c in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def _channel_to_linear(c: int) -> float:
    """sRGB component → linear-light; per the WCAG formula."""
    cf = c / 255.0
    return cf / 12.92 if cf <= 0.03928 else ((cf + 0.055) / 1.055) ** 2.4


def relative_luminance(rgb: tuple[int, int, int]) -> float:
    """L = 0.2126·R_lin + 0.7152·G_lin + 0.0722·B_lin (WCAG 2.1)."""
    r, g, b = rgb
    return (
        0.2126 * _channel_to_linear(r)
        + 0.7152 * _channel_to_linear(g)
        + 0.0722 * _channel_to_linear(b)
    )


def contrast_ratio(fg: str, bg: str) -> float:
    """Return the WCAG 2.1 contrast ratio between two hex colors.

    Output is always >= 1.0; lighter / darker order is irrelevant.
    """
    l1 = relative_luminance(hex_to_rgb(fg))
    l2 = relative_luminance(hex_to_rgb(bg))
    light, dark = (l1, l2) if l1 >= l2 else (l2, l1)
    return (light + 0.05) / (dark + 0.05)


def passes_aa(fg: str, bg: str, *, large_text: bool = False) -> bool:
    """``True`` iff the pair meets WCAG 2.1 AA for the given text size."""
    threshold = WCAG_AA_LARGE if large_text else WCAG_AA_NORMAL
    return contrast_ratio(fg, bg) >= threshold


def passes_aaa(fg: str, bg: str) -> bool:
    """``True`` iff the pair meets WCAG 2.1 AAA for body text."""
    return contrast_ratio(fg, bg) >= WCAG_AAA_NORMAL


# ---------------------------------------------------------------------------
# Image alt-text helpers
# ---------------------------------------------------------------------------


def alt_for_before_after(noise_stem: str) -> str:
    """Generate readable alt text for ``<noise>_before_after.png``.

    e.g. ``ring_artifact`` → ``"Before / after comparison for ring
    artifact mitigation."``
    """
    label = noise_stem.replace("_", " ").strip()
    if not label:
        return "Before / after comparison."
    return f"Before / after comparison for {label} mitigation."


# ---------------------------------------------------------------------------
# ARIA / keyboard helpers
# ---------------------------------------------------------------------------


def skip_link_html(target_id: str = "main-content") -> str:
    """Skip-to-main-content link rendered as the very first focusable element.

    Hidden until the user tabs into it; this is the WCAG 2.4.1
    "Bypass Blocks" requirement. The reveal-on-focus styling is in
    ``explorer/assets/styles.css`` (``.eberlight-skip-link``) so an
    inline ``style=`` attribute doesn't have to fake a ``:focus``
    pseudo-class.
    """
    return f'<a href="#{target_id}" class="eberlight-skip-link">Skip to main content</a>'


def main_anchor_html(target_id: str = "main-content") -> str:
    """The invisible target the skip-link jumps to.

    Streamlit doesn't easily let us wrap the entire page body in a
    single ``<main>`` tag, so we drop a focusable anchor right after
    the header. Pairs with :func:`skip_link_html`.
    """
    return f'<a id="{target_id}" tabindex="-1" aria-hidden="true"></a>'
