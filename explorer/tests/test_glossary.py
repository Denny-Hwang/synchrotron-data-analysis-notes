"""Unit tests for :mod:`lib.glossary` — the auto-link annotator.

Covers (1) glossary-line parsing, (2) first-occurrence-only wrapping,
(3) the skip-stack (no annotation inside ``<code>``, ``<a>``, headings,
existing ``<abbr>``), (4) the longest-match-first ordering so
``APS-U`` doesn't get shadowed by ``APS``, and (5) word-boundary
behaviour so ``APS`` doesn't match inside ``GAPS``.

Ref: REL-E080 senior-review action item #9 — glossary auto-link.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.glossary import (
    _build_match_regex,
    _parse_glossary_text,
    annotate_html,
    load_glossary,
)

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def test_parse_glossary_extracts_term_and_definition() -> None:
    text = "**APS**: Advanced Photon Source. DOE synchrotron at Argonne.\n"
    glossary = _parse_glossary_text(text)
    assert glossary == {"APS": "Advanced Photon Source. DOE synchrotron at Argonne."}


def test_parse_glossary_skips_two_char_terms() -> None:
    text = "**AB**: too short.\n**ABC**: keep me.\n"
    glossary = _parse_glossary_text(text)
    assert "AB" not in glossary
    assert "ABC" in glossary


def test_parse_glossary_keeps_first_occurrence_of_duplicate() -> None:
    text = "**APS**: definition one.\n**aps**: definition two.\n"
    glossary = _parse_glossary_text(text)
    assert glossary == {"APS": "definition one."}


def test_parse_glossary_ignores_non_glossary_lines() -> None:
    text = "# Glossary\n\n## A\n\n**APS**: Advanced Photon Source.\n\nSome prose about beamlines.\n"
    glossary = _parse_glossary_text(text)
    assert glossary == {"APS": "Advanced Photon Source."}


# ---------------------------------------------------------------------------
# Match regex
# ---------------------------------------------------------------------------


def test_match_regex_longest_first() -> None:
    regex = _build_match_regex(["APS", "APS-U"])
    # The first match in "APS-U" should be the longer alternative.
    found = regex.findall("APS-U is on the way after APS.")
    assert found[0] == "APS-U"
    assert found[1] == "APS"


def test_match_regex_word_boundary_blocks_substring() -> None:
    """``APS`` must not match inside ``GAPS`` or ``LAPSE``."""
    regex = _build_match_regex(["APS"])
    assert regex.findall("GAPS LAPSE APS!") == ["APS"]


def test_match_regex_is_case_insensitive() -> None:
    regex = _build_match_regex(["APS"])
    assert regex.findall("aps Aps APS") == ["aps", "Aps", "APS"]


def test_empty_terms_compiles_to_no_match_pattern() -> None:
    regex = _build_match_regex([])
    assert regex.findall("anything at all") == []


# ---------------------------------------------------------------------------
# annotate_html
# ---------------------------------------------------------------------------


def test_annotate_wraps_first_occurrence_only() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = "<p>APS is at APS, near another APS.</p>"
    out = annotate_html(html, glossary)
    # Exactly one <abbr> opens.
    assert out.count("<abbr") == 1
    assert 'title="Advanced Photon Source."' in out
    # The wrapped term keeps its original casing.
    assert ">APS</abbr>" in out


def test_annotate_skips_inside_code_block() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = "<p>Hello <code>APS variable</code> world.</p>"
    out = annotate_html(html, glossary)
    # No <abbr> inside <code>.
    assert "<abbr" not in out


def test_annotate_skips_inside_pre_block() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = "<pre><code>APS = 7 GeV</code></pre>"
    out = annotate_html(html, glossary)
    assert "<abbr" not in out


def test_annotate_skips_inside_heading() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = "<h2>APS overview</h2><p>APS is at Argonne.</p>"
    out = annotate_html(html, glossary)
    # Heading text stays clean; first body match gets wrapped.
    assert "<h2>APS overview</h2>" in out
    assert "<abbr" in out


def test_annotate_skips_inside_anchor() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = '<p><a href="x">APS link</a> and then APS body.</p>'
    out = annotate_html(html, glossary)
    # No <abbr> inside <a>; one in the body afterwards.
    assert '<a href="x">APS link</a>' in out
    assert out.count("<abbr") == 1


def test_annotate_skips_inside_existing_abbr() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = '<p><abbr title="x">APS</abbr> and APS body.</p>'
    out = annotate_html(html, glossary)
    # The existing <abbr> is left alone; the second occurrence gets one.
    assert out.count("<abbr") == 2


def test_annotate_handles_self_closing_void_elements() -> None:
    glossary = {"APS": "Advanced Photon Source."}
    html = "<p>APS<br/>line two</p>"
    # <br/> must not derail the skip-stack.
    out = annotate_html(html, glossary)
    assert "<abbr" in out


def test_annotate_passthrough_when_no_glossary() -> None:
    assert annotate_html("<p>APS body.</p>", {}) == "<p>APS body.</p>"


def test_annotate_passthrough_when_empty_html() -> None:
    assert annotate_html("", {"APS": "x"}) == ""


def test_annotate_escapes_definition_attribute() -> None:
    """Definition with quotes is HTML-attribute-escaped (defence in depth)."""
    glossary = {"X": '<bad>"q"</bad>'}
    out = annotate_html("<p>X is here.</p>", glossary)
    # &quot; / &lt; / &gt; encoded so the title attribute stays valid.
    assert "&quot;" in out
    assert "&lt;" in out
    assert "&gt;" in out


# ---------------------------------------------------------------------------
# load_glossary (integration)
# ---------------------------------------------------------------------------


def test_load_glossary_from_repo(tmp_path: Path) -> None:
    """End-to-end: write a fake repo with a glossary.md and load it."""
    refs_dir = tmp_path / "08_references"
    refs_dir.mkdir(parents=True)
    (refs_dir / "glossary.md").write_text(
        "## A\n\n**APS**: Advanced Photon Source.\n", encoding="utf-8"
    )
    # ``load_glossary`` is lru_cached; pass a fresh path so we always hit IO.
    glossary = load_glossary(tmp_path)
    assert glossary == {"APS": "Advanced Photon Source."}


def test_load_glossary_returns_empty_when_file_missing(tmp_path: Path) -> None:
    glossary = load_glossary(tmp_path / "nonexistent")
    assert glossary == {}


@pytest.mark.parametrize(
    "input_html,must_contain",
    [
        ("<p>APS-U is here.</p>", ">APS-U</abbr>"),
        ("<p>I love APS!</p>", ">APS</abbr>"),
    ],
)
def test_annotate_real_terms(input_html: str, must_contain: str) -> None:
    glossary = {"APS": "Advanced Photon Source.", "APS-U": "Advanced Photon Source Upgrade."}
    out = annotate_html(input_html, glossary)
    assert must_contain in out
