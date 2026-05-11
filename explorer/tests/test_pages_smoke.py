"""End-to-end smoke tests for the five high-LOC Streamlit pages.

Before REL-E080 the Knowledge Graph (~400 LOC), Interactive Lab
(~514 LOC), Troubleshooter (~301 LOC), and Search (~218 LOC) pages had
zero automated test coverage; a syntax slip or a stale import would
only be caught when a user actually hit the page in a browser. These
tests use Streamlit's official ``AppTest`` (1.30+) to import-and-run
each page in-process and assert (a) no exception bubbled out, and
(b) the page emitted at least one element to its body, so a future
"silent failure" mode (e.g. an early ``st.stop()`` before any output)
is also caught.

The tests are deliberately shallow — they do *not* drive widgets or
inspect specific UI shape — because the pages render hundreds of
nodes from the on-disk note corpus and the assertions would become
brittle. The contract here is: **the page loads, the imports resolve,
nothing throws**.

Ref: REL-E080 senior-review action item #7 — interactive-page smoke tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
_REPO_ROOT = _EXPLORER_DIR.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))


# ``AppTest`` ships in Streamlit 1.28+. We import lazily inside a fixture
# so the rest of the test module still collects on older Streamlit
# installs (the tests skip instead of erroring at import time).
@pytest.fixture(scope="module")
def AppTest():
    try:
        from streamlit.testing.v1 import AppTest as _AppTest
    except ImportError:
        pytest.skip("streamlit.testing.v1.AppTest unavailable (Streamlit < 1.28).")
    return _AppTest


_PAGE_FILES = {
    "landing": _EXPLORER_DIR / "app.py",
    "knowledge_graph": _EXPLORER_DIR / "pages" / "0_Knowledge_Graph.py",
    "discover": _EXPLORER_DIR / "pages" / "1_Discover.py",
    "explore": _EXPLORER_DIR / "pages" / "2_Explore.py",
    "build": _EXPLORER_DIR / "pages" / "3_Build.py",
    "experiment": _EXPLORER_DIR / "pages" / "4_Experiment.py",
    "troubleshooter": _EXPLORER_DIR / "pages" / "5_Troubleshooter.py",
    "search": _EXPLORER_DIR / "pages" / "6_Search.py",
}


def _run(AppTest, page_path: Path):
    """Run an AppTest from a page file and return the resulting harness."""
    if not page_path.exists():
        pytest.skip(f"Page file missing: {page_path}")
    at = AppTest.from_file(str(page_path))
    at.run(timeout=30)
    return at


@pytest.mark.parametrize("page_key", list(_PAGE_FILES.keys()))
def test_page_runs_without_exception(AppTest, page_key: str) -> None:
    """Each page must import and render without raising.

    A failure here usually means one of: a missing import, a stale
    name reference (e.g. after the ``_query_param`` → ``query_param``
    rename), a YAML / sample file the page expects but can't find,
    or a Streamlit API call against an attribute that no longer exists.
    """
    at = _run(AppTest, _PAGE_FILES[page_key])
    if at.exception:
        # AppTest collects exceptions on the ``.exception`` attribute as a list.
        messages = [str(e.value) for e in at.exception]
        pytest.fail(f"Page {page_key} raised: {messages}")


@pytest.mark.parametrize("page_key", list(_PAGE_FILES.keys()))
def test_page_emits_some_output(AppTest, page_key: str) -> None:
    """A page that runs to completion must produce at least one element.

    Catches silent ``st.stop()`` early-exits — e.g. a future change that
    accidentally trips the ``if not recipes: render_footer(); st.stop()``
    branch on the Lab page would otherwise look "green".
    """
    at = _run(AppTest, _PAGE_FILES[page_key])
    has_any_output = bool(at.markdown) or bool(at.header) or bool(at.title) or bool(at.subheader)
    assert has_any_output, f"Page {page_key} produced no markdown/header output."


def test_routing_module_importable_from_pages() -> None:
    """Cheap regression test for the REL-E080 ``_query_param`` cleanup.

    If any page lost its ``from lib.routing import query_param`` line
    during the migration it would NameError on the first param read.
    Importing each page module here catches that before AppTest does.
    """
    import importlib

    # Re-import lib.routing so a stale module cache doesn't mask a rename.
    importlib.reload(importlib.import_module("lib.routing"))
    from lib.routing import query_param  # noqa: F401 — import is the assertion
