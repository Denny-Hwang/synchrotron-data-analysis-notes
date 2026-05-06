"""Tests for the in-memory search index (Phase R6)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.notes import Note
from lib.search import _tokenize, build_index, search


def _note(folder: str, title: str, body: str, tags: tuple[str, ...] = ()) -> Note:
    return Note(
        path=Path(folder) / f"{title.lower().replace(' ', '_')}.md",
        folder=folder,
        title=title,
        body=body,
        cluster="explore",
        tags=list(tags),
    )


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def test_tokenize_lowercases_and_drops_short_tokens() -> None:
    assert _tokenize("Hello, World!") == ["hello", "world"]


def test_tokenize_keeps_dots_and_dashes() -> None:
    assert "tomopy" in _tokenize("TomoPy v1.14 — see README.")
    assert "1.14" not in _tokenize("v1.14")  # leading digit excluded by [a-z0-9][...]
    assert any("readme" in t for t in _tokenize("README.md"))


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


def test_index_indexes_title_body_and_tags() -> None:
    notes = [
        _note(
            "03_ai_ml_methods",
            "TomoGAN",
            "Conditional GAN for low-dose tomography.",
            ("denoising", "GAN"),
        ),
        _note("05_tools_and_code", "TomoPy", "Reconstruction toolkit."),
    ]
    idx = build_index(notes)
    assert "tomogan" in idx.inverted
    assert "tomography" in idx.inverted  # body
    assert "denoising" in idx.inverted  # tag
    assert "tomopy" in idx.inverted


def test_search_returns_empty_for_empty_query() -> None:
    notes = [_note("x", "X", "y")]
    assert search(build_index(notes), "") == []


def test_search_finds_title_match() -> None:
    notes = [
        _note("03_ai_ml_methods", "TomoGAN", "GAN denoising for tomography."),
        _note("05_tools_and_code", "TomoPy", "Reconstruction toolkit."),
    ]
    idx = build_index(notes)
    hits = search(idx, "tomogan")
    assert hits and hits[0].note.title == "TomoGAN"


def test_search_ranks_title_matches_above_body_matches() -> None:
    notes = [
        _note("a", "Tomography Overview", "An overview."),
        _note("b", "Random Note", "Mentions tomography once."),
    ]
    idx = build_index(notes)
    hits = search(idx, "tomography")
    assert len(hits) == 2
    # The note whose title contains the term must rank first.
    assert hits[0].note.title == "Tomography Overview"


def test_search_prefix_matches_inflections() -> None:
    notes = [_note("x", "Denoising methods", "Survey of image denoiser networks.")]
    idx = build_index(notes)
    hits = search(idx, "denois")
    assert hits and hits[0].note.title == "Denoising methods"
    assert any(t.startswith("denois") for t in hits[0].matched_terms)


def test_snippet_contains_first_match() -> None:
    body = "lorem ipsum " * 50 + "Vo et al. 2018 sorting-based ring removal " + "sit amet " * 50
    notes = [_note("09_noise_catalog", "Ring Artifact", body)]
    hits = search(build_index(notes), "vo 2018")
    assert hits
    assert "vo" in hits[0].snippet.lower()


def test_search_limit_is_honoured() -> None:
    notes = [_note("x", f"Note {i}", "tomography content") for i in range(20)]
    idx = build_index(notes)
    hits = search(idx, "tomography", limit=5)
    assert len(hits) == 5


def test_search_is_deterministic_on_score_ties() -> None:
    """Identical-score notes must come back in the same order across runs."""
    notes = [_note("x", f"Note {chr(65 + i)}", "tomography") for i in range(5)]
    idx = build_index(notes)
    a = [h.note.title for h in search(idx, "tomography")]
    b = [h.note.title for h in search(idx, "tomography")]
    assert a == b


# ---------------------------------------------------------------------------
# Real repo smoke
# ---------------------------------------------------------------------------


def test_real_repo_index_finds_known_notes() -> None:
    """Sanity check on the live notes corpus."""
    pytest.importorskip("yaml")
    from lib.search import index_from_repo

    idx = index_from_repo(_REPO_ROOT)
    if len(idx) == 0:
        pytest.skip("no notes available in this checkout")
    # `tomopy` should be present somewhere in the corpus.
    hits = search(idx, "tomopy")
    assert hits, "expected at least one hit for 'tomopy' in the real repo"
