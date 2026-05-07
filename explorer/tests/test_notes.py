"""Tests for the note loader.

Tests frontmatter parsing, graceful degradation for notes without
frontmatter, and controlled vocabulary validation.

Ref: TST-001 (test_plan.md) — Unit tests for note parser.
Ref: DC-001 (data_contracts.md) — Schema and vocabularies.
"""

import sys
import textwrap
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.notes import _parse_note, _title_from_filename, load_notes


def test_title_from_filename() -> None:
    """Underscores and hyphens become spaces, result is title-cased."""
    assert _title_from_filename("ai_ml_methods") == "Ai Ml Methods"
    assert _title_from_filename("tomography") == "Tomography"
    assert _title_from_filename("xrf-microscopy") == "Xrf Microscopy"


def test_parse_note_with_frontmatter(tmp_path: Path) -> None:
    """Notes with valid YAML frontmatter are parsed correctly."""
    note_file = tmp_path / "test_note.md"
    note_file.write_text(
        textwrap.dedent("""\
        ---
        title: "TomoGAN Denoising"
        cluster: explore
        tags: [denoising, GAN, tomography]
        modality: tomography
        beamline: [2-BM, 32-ID]
        related_publications: [review_tomogan_2020.md]
        ---
        # TomoGAN

        Body content here.
    """)
    )

    note = _parse_note(note_file, "03_ai_ml_methods")

    assert note.title == "TomoGAN Denoising"
    assert note.cluster == "explore"
    assert note.tags == ["denoising", "GAN", "tomography"]
    assert note.modality == "tomography"
    assert note.beamline == ["2-BM", "32-ID"]
    assert note.related_publications == ["review_tomogan_2020.md"]
    assert note.has_frontmatter is True
    assert "Body content here." in note.body


def test_parse_note_without_frontmatter(tmp_path: Path) -> None:
    """Notes without frontmatter get inferred title and cluster."""
    note_file = tmp_path / "xrf_microscopy.md"
    note_file.write_text("# XRF Microscopy\n\nContent about XRF.")

    note = _parse_note(note_file, "02_xray_modalities")

    assert note.title == "Xrf Microscopy"  # inferred from filename
    assert note.cluster == "explore"  # inferred from folder mapping
    assert note.tags == []
    assert note.modality is None
    assert note.beamline == []
    assert note.has_frontmatter is False


def test_parse_note_partial_frontmatter(tmp_path: Path) -> None:
    """Notes with only some frontmatter fields use defaults for the rest."""
    note_file = tmp_path / "partial.md"
    note_file.write_text(
        textwrap.dedent("""\
        ---
        title: "Partial Note"
        tags: [test]
        ---
        Body.
    """)
    )

    note = _parse_note(note_file, "01_program_overview")

    assert note.title == "Partial Note"
    assert note.tags == ["test"]
    assert note.cluster == "discover"  # inferred from folder
    assert note.modality is None
    assert note.has_frontmatter is True


def test_invalid_vocabulary_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Invalid controlled vocabulary values produce warnings."""
    note_file = tmp_path / "bad.md"
    note_file.write_text(
        textwrap.dedent("""\
        ---
        title: "Bad Values"
        cluster: invalid_cluster
        modality: invalid_modality
        beamline: [99-ZZ]
        tags: []
        ---
        Body.
    """)
    )

    import logging

    with caplog.at_level(logging.WARNING):
        note = _parse_note(note_file, "01_program_overview")

    assert "Invalid cluster value" in caplog.text
    assert "Invalid modality value" in caplog.text
    assert "Invalid beamline value" in caplog.text
    # Note still loads despite warnings
    assert note.title == "Bad Values"


def test_load_notes_from_real_repo() -> None:
    """load_notes loads notes from the actual repo."""
    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)

    # Should find notes in at least some folders
    assert len(notes) > 0

    # All notes should have a non-empty title
    for note in notes:
        assert note.title, f"Note at {note.path} has empty title"

    # All notes should have a valid cluster
    valid_clusters = {"discover", "explore", "build"}
    for note in notes:
        assert note.cluster in valid_clusters, (
            f"Note {note.path} has invalid cluster: {note.cluster}"
        )


# ---------------------------------------------------------------------------
# Resolvers + helpers (Phase R8)
# ---------------------------------------------------------------------------


def test_resolve_publication_ref_finds_match() -> None:
    """resolve_publication_ref maps a filename to the matching note."""
    from lib.notes import resolve_publication_ref

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    pub_notes = [n for n in notes if n.folder == "04_publications"]
    if not pub_notes:
        pytest.skip("No publication notes to test against.")
    a_pub = pub_notes[0]
    found = resolve_publication_ref(notes, a_pub.path.name, repo_root)
    assert found is not None
    assert found.path == a_pub.path


def test_resolve_publication_ref_appends_md_extension() -> None:
    """A reference without .md is normalised before lookup."""
    from lib.notes import resolve_publication_ref

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    pub_notes = [n for n in notes if n.folder == "04_publications"]
    if not pub_notes:
        pytest.skip("No publication notes to test against.")
    stem = pub_notes[0].path.stem  # filename without ".md"
    found = resolve_publication_ref(notes, stem, repo_root)
    assert found is not None


def test_resolve_publication_ref_returns_none_for_unknown() -> None:
    from lib.notes import resolve_publication_ref

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    assert resolve_publication_ref(notes, "no_such_file.md", repo_root) is None
    assert resolve_publication_ref(notes, "", repo_root) is None


def test_resolve_tool_ref_finds_readme() -> None:
    """resolve_tool_ref('tomocupy') → the README in 05_tools_and_code/tomocupy/."""
    from lib.notes import resolve_tool_ref

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    found = resolve_tool_ref(notes, "tomocupy", repo_root)
    if found is None:
        pytest.skip("Repo has no 05_tools_and_code/tomocupy/ directory.")
    assert found.folder == "05_tools_and_code"
    assert found.path.parent.name.lower() == "tomocupy"


def test_resolve_tool_ref_returns_none_for_unknown() -> None:
    from lib.notes import resolve_tool_ref

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    assert resolve_tool_ref(notes, "no_such_tool", repo_root) is None
    assert resolve_tool_ref(notes, "", repo_root) is None


def test_neighbor_notes_returns_prev_and_next(tmp_path: Path) -> None:
    """neighbor_notes orders siblings deterministically by path."""
    from lib.notes import Note, neighbor_notes

    a = Note(path=tmp_path / "a.md", folder="x", title="A", body="", cluster="discover")
    b = Note(path=tmp_path / "b.md", folder="x", title="B", body="", cluster="discover")
    c = Note(path=tmp_path / "c.md", folder="x", title="C", body="", cluster="discover")
    notes = [c, a, b]  # unsorted on purpose
    prev_n, next_n = neighbor_notes(notes, b)
    assert prev_n is a
    assert next_n is c
    # First / last edges
    assert neighbor_notes(notes, a) == (None, b)
    assert neighbor_notes(notes, c) == (b, None)


def test_neighbor_notes_only_walks_same_folder(tmp_path: Path) -> None:
    from lib.notes import Note, neighbor_notes

    a = Note(path=tmp_path / "a.md", folder="x", title="A", body="", cluster="discover")
    b = Note(path=tmp_path / "b.md", folder="other", title="B", body="", cluster="discover")
    prev_n, next_n = neighbor_notes([a, b], a)
    assert prev_n is None and next_n is None


def test_discover_sibling_notebooks_real_repo() -> None:
    """At least one tool folder ships .ipynb files; discover_sibling_notebooks
    must surface them on the README note."""
    from lib.notes import discover_sibling_notebooks

    repo_root = _EXPLORER_DIR.parent
    notes = load_notes(repo_root)
    # Find any tool README that has a sibling .ipynb.
    for n in notes:
        if n.folder == "05_tools_and_code" and n.path.name == "README.md":
            books = discover_sibling_notebooks(n, repo_root)
            if books:
                assert all(p.suffix == ".ipynb" for p in books)
                return
    pytest.skip("No tool README has a sibling .ipynb in this repo state.")
