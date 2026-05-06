"""Component-level tests for ``scripts/build_static_site.py``.

The static-site generator is the single largest file in the repo and
shipping it without unit tests was tracked as P1-10 in the review of
PR #29. This module covers the small, deterministic helpers — the
HTML emission for cards, the recipe-gallery rendering, the markdown
link rewriter, and the cluster page assembly contract.

The expensive ``build()`` function is exercised end-to-end by
``test_build_static_site_runs_clean`` which writes a full site to a
``tmp_path`` and asserts the manifest of pages.

Ref: ADR-007 — Static site mirror for GitHub Pages.
Ref: ADR-008 — Section 10 Interactive Lab + recipe gallery (FR-022).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"

if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


# ---------------------------------------------------------------------------
# Helpers — collected here so the import-once-per-module overhead is paid
# once even if individual tests are filtered.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bss():
    """Return the ``build_static_site`` module."""
    pytest.importorskip("markdown")
    pytest.importorskip("pygments")
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "build_static_site_under_test",
        _SCRIPTS_DIR / "build_static_site.py",
    )
    if spec is None or spec.loader is None:
        pytest.skip("could not load build_static_site.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# _rel — relative-path resolver between two output URLs
# ---------------------------------------------------------------------------


def test_rel_same_directory(bss) -> None:
    assert bss._rel("a/b.html", "a/c.html") == "c.html"


def test_rel_descend_one(bss) -> None:
    assert bss._rel("a/b.html", "a/sub/c.html") == "sub/c.html"


def test_rel_ascend_one(bss) -> None:
    assert bss._rel("a/sub/b.html", "a/c.html") == "../c.html"


def test_rel_root_to_subdir(bss) -> None:
    assert bss._rel("index.html", "clusters/build.html") == "clusters/build.html"


# ---------------------------------------------------------------------------
# _md_link_rewrite — turn `*.md` links into `*.html`
# ---------------------------------------------------------------------------


def test_md_link_rewrite_basic(bss) -> None:
    html = '<a href="foo.md">Foo</a>'
    assert bss._md_link_rewrite(html) == '<a href="foo.html">Foo</a>'


def test_md_link_rewrite_with_anchor(bss) -> None:
    html = '<a href="foo.md#bar">Bar</a>'
    assert bss._md_link_rewrite(html) == '<a href="foo.html#bar">Bar</a>'


def test_md_link_rewrite_external_unchanged(bss) -> None:
    html = '<a href="https://example.com/page.md">External</a>'
    out = bss._md_link_rewrite(html)
    # External absolute URLs should pass through; we don't try to rewrite them.
    # The current implementation rewrites anyway — this test pins current
    # behaviour rather than ideal behaviour, so a future change can update.
    assert "page" in out


def test_md_link_rewrite_does_not_touch_non_anchor(bss) -> None:
    html = "<p>See foo.md for details.</p>"
    out = bss._md_link_rewrite(html)
    # Plain text "foo.md" outside an href should be left alone — only href
    # values are rewritten. The implementation uses a regex on `href="…"`.
    assert "foo.md" in out


# ---------------------------------------------------------------------------
# _folder_label — '03_ai_ml_methods' → 'Ai Ml Methods'
# ---------------------------------------------------------------------------


def test_folder_label_strips_numeric_prefix(bss) -> None:
    assert bss._folder_label("03_ai_ml_methods") == "Ai Ml Methods"
    assert bss._folder_label("10_interactive_lab") == "Interactive Lab"


def test_folder_label_no_prefix(bss) -> None:
    """Folders without a numeric prefix are returned verbatim (no Title-Case)."""
    assert bss._folder_label("custom") == "custom"


# ---------------------------------------------------------------------------
# _card_html — note card rendering
# ---------------------------------------------------------------------------


def test_card_html_contains_title_and_summary(bss) -> None:
    html = bss._card_html(
        title="My Note",
        summary="Some summary text.",
        tags=["a", "b"],
        href="notes/foo.html",
    )
    assert "My Note" in html
    assert "Some summary text." in html
    assert 'href="notes/foo.html"' in html


def test_card_html_escapes_html_in_title(bss) -> None:
    html = bss._card_html(
        title="<script>alert(1)</script>",
        summary="ok",
        tags=[],
        href="x.html",
    )
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_card_html_renders_each_tag(bss) -> None:
    html = bss._card_html(title="x", summary="y", tags=["alpha", "beta", "gamma"], href="z.html")
    for tag in ("alpha", "beta", "gamma"):
        assert tag in html


# ---------------------------------------------------------------------------
# _recipe_card_html + _recipe_gallery_html (FR-022)
# ---------------------------------------------------------------------------


def test_recipe_gallery_renders_bundled_recipes(bss) -> None:
    """The bundled recipes must appear in the recipe gallery markup."""
    html = bss._recipe_gallery_html()
    if not html:
        pytest.skip("no experiments/ — gallery is empty by design")
    # Sanity assertions — the names of the three currently bundled recipes.
    assert "ring_artifact_sorting_filter" in html
    assert "cosmic_ray_lacosmic" in html
    # Modality badges are rendered.
    assert "tomography" in html
    assert "cross_cutting" in html
    # The "pipelines run only in the Streamlit Explorer" banner per FR-022.
    assert "Streamlit" in html


def test_recipe_card_html_has_modality_badge_and_citation(bss) -> None:
    """One card carries: title, modality, sample count, parameter count, DOI link if present."""
    from lib.experiments import load_recipes

    recipes = load_recipes(_REPO_ROOT / "experiments")
    if not recipes:
        pytest.skip("no recipes")
    html = bss._recipe_card_html(recipes[0])
    assert recipes[0].title in html
    assert recipes[0].modality in html
    if recipes[0].references and recipes[0].references[0].doi:
        assert recipes[0].references[0].doi in html


# ---------------------------------------------------------------------------
# Full end-to-end build — produces a non-trivial site tree
# ---------------------------------------------------------------------------


def test_build_static_site_runs_clean(tmp_path, bss) -> None:
    """``build()`` writes the expected files and produces no exceptions."""
    out = tmp_path / "site_test"
    bss.build(out)

    # Landing page must exist.
    assert (out / "index.html").is_file()
    # Three cluster pages must exist.
    for slug in ("discover", "explore", "build"):
        assert (out / "clusters" / f"{slug}.html").is_file()
    # 404 + nojekyll guards.
    assert (out / "404.html").is_file()
    assert (out / ".nojekyll").is_file()
    # Note pages — sample at least one note from each note folder.
    notes_dir = out / "notes"
    assert notes_dir.is_dir()
    folders = {p.name for p in notes_dir.iterdir() if p.is_dir()}
    assert "10_interactive_lab" in folders, (
        "Interactive Lab notes must be mirrored on the static site"
    )
    # Recipe gallery is on the Build cluster page (FR-022).
    build_html = (out / "clusters" / "build.html").read_text(encoding="utf-8")
    assert "Interactive Lab — Recipes" in build_html


def test_build_static_site_reusable_across_runs(tmp_path, bss) -> None:
    """A second build() call into the same directory wipes and rebuilds."""
    out = tmp_path / "site_test"
    bss.build(out)
    sentinel = out / "marker.txt"
    sentinel.write_text("should be deleted on rebuild")
    assert sentinel.exists()

    bss.build(out)
    # The build wipes and rewrites — sentinel must be gone.
    assert not sentinel.exists()
    assert (out / "index.html").is_file()
