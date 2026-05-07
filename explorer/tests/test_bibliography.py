"""Tests for the BibTeX parser (Phase R6)."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.bibliography import (
    BibEntry,
    collect_bibliography,
    parse_bib_file,
    parse_bib_text,
)

_SAMPLE = textwrap.dedent(
    """\
    @article{vo2018,
      author  = {Vo, N. T. and Atwood, R. C. and Drakopoulos, M.},
      title   = {Superior techniques for eliminating ring artifacts in X-ray micro-tomography},
      journal = {Optics Express},
      volume  = {26},
      year    = {2018},
      doi     = {10.1364/OE.26.028396}
    }

    @inproceedings{lehtinen2018,
      author    = {Lehtinen, Jaakko and others},
      title     = {Noise2Noise: Learning Image Restoration without Clean Data},
      booktitle = {ICML},
      year      = {2018},
    }

    @misc{just_a_key,
      title = "Untitled stub"
    }
    """
)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_parse_bib_text_returns_three_entries() -> None:
    entries = parse_bib_text(_SAMPLE)
    assert len(entries) == 3


def test_first_entry_fields() -> None:
    entries = parse_bib_text(_SAMPLE)
    e = entries[0]
    assert e.key == "vo2018"
    assert e.entry_type == "article"
    assert "Superior techniques" in e.title
    assert e.year == 2018
    assert e.venue == "Optics Express"
    assert e.doi == "10.1364/OE.26.028396"
    assert e.authors == ("Vo, N. T.", "Atwood, R. C.", "Drakopoulos, M.")


def test_inproceedings_uses_booktitle_as_venue() -> None:
    entries = parse_bib_text(_SAMPLE)
    e = entries[1]
    assert e.entry_type == "inproceedings"
    assert e.venue == "ICML"


def test_entry_without_year_is_none() -> None:
    entries = parse_bib_text(_SAMPLE)
    e = entries[2]
    assert e.year is None
    assert e.doi == ""
    assert e.authors == ()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_doi_url_for_entry_with_doi() -> None:
    e = parse_bib_text(_SAMPLE)[0]
    assert e.doi_url == "https://doi.org/10.1364/OE.26.028396"


def test_doi_url_empty_when_missing() -> None:
    e = parse_bib_text(_SAMPLE)[2]
    assert e.doi_url == ""


def test_render_apa_short_includes_year_title_doi() -> None:
    e = parse_bib_text(_SAMPLE)[0]
    out = e.render_apa_short()
    assert "Vo" in out
    assert "et al." in out
    assert "2018" in out
    assert "Optics Express" in out
    assert "10.1364/OE.26.028396" in out


# ---------------------------------------------------------------------------
# File / repo loaders
# ---------------------------------------------------------------------------


def test_parse_bib_file_missing_returns_empty(tmp_path: Path) -> None:
    assert parse_bib_file(tmp_path / "nope.bib") == []


def test_real_repo_bibliography_loads_entries() -> None:
    entries = collect_bibliography(_REPO_ROOT)
    if not entries:
        pytest.skip("no .bib files in this checkout")
    # Must include at least one bundled bibliography entry.
    assert isinstance(entries[0], BibEntry)
    # Sorted year-descending (None years sink).
    years = [e.year for e in entries if e.year is not None]
    assert years == sorted(years, reverse=True)


def test_real_repo_doi_links_well_formed() -> None:
    for e in collect_bibliography(_REPO_ROOT):
        if e.doi:
            assert e.doi_url.startswith("https://doi.org/")
            assert " " not in e.doi


# ---------------------------------------------------------------------------
# R12 B4 — LaTeX-accent decoding
# ---------------------------------------------------------------------------


def test_decode_latex_accents_braced_forms() -> None:
    """Common BibTeX accent escapes decode to Unicode."""
    from lib.bibliography import _decode_latex_accents

    cases = {
        "J{\\'e}r{\\^o}me": "Jérôme",
        "Val{\\'e}rie": "Valérie",
        'M{\\"u}ller': "Müller",
        'Schr{\\"o}dinger': "Schrödinger",
        "Fran{\\c c}ois": "François",
        "Ja{\\v c}ek": "Jaček",
        "Stra{\\ss}e": "Straße",
        "No accents here": "No accents here",
    }
    for src, want in cases.items():
        got = _decode_latex_accents(src)
        assert got == want, f"{src!r} -> {got!r}, want {want!r}"


def test_decode_latex_accents_lone_commands() -> None:
    """``\\AA``, ``\\ss``, ``\\o`` etc. with their trailing space."""
    from lib.bibliography import _decode_latex_accents

    assert _decode_latex_accents('\\AA ngstr{\\"o}m') == "Ångström"
    assert _decode_latex_accents("Ka\\ss el") == "Kaßel"


def test_parse_authors_decodes_accents() -> None:
    """End-to-end: a BibTeX author field with accents flows through clean."""
    from lib.bibliography import _parse_authors

    raw = "Kieffer, J{\\'e}r{\\^o}me and Valls, Val{\\'e}rie and Schr{\\\"o}dinger, Erwin"
    out = _parse_authors(raw)
    # Expect 3 authors, each containing the decoded characters.
    assert len(out) == 3
    assert any("Jérôme" in a for a in out)
    assert any("Valérie" in a for a in out)
    assert any("Schrödinger" in a for a in out)
    # No backslash-curly residue.
    for a in out:
        assert "\\" not in a, a
        assert "{" not in a, a
        assert "}" not in a, a
