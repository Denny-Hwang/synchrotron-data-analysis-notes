"""Unit tests for :mod:`lib.routing` — the single-source query-param helper.

Replaces the five copy-paste ``_query_param`` callsites that were
scattered across pages before REL-E080. Verifies the three Streamlit
``st.query_params`` shapes (string, list, None) all collapse to the
same result and that ``decode=False`` preserves percent-escapes for
slug-safe identifiers like Lab recipe ids.

Ref: TST-001 — Unit test policy.
Ref: REL-E080 senior-review action item #2 — extract ``query_param``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.routing import query_param


class _FakeQueryParams:
    """Mimics ``st.query_params`` mapping access for unit tests."""

    def __init__(self, data: dict[str, object]) -> None:
        self._data = data

    def get(self, key: str, default: object = None) -> object:
        return self._data.get(key, default)


def test_query_param_returns_none_when_absent() -> None:
    with patch("lib.routing.st.query_params", _FakeQueryParams({})):
        assert query_param("note") is None


def test_query_param_returns_string_value() -> None:
    with patch("lib.routing.st.query_params", _FakeQueryParams({"note": "intro"})):
        assert query_param("note") == "intro"


def test_query_param_unquotes_by_default() -> None:
    with patch("lib.routing.st.query_params", _FakeQueryParams({"q": "ring%20artifact"})):
        assert query_param("q") == "ring artifact"


def test_query_param_preserves_raw_when_decode_false() -> None:
    """Slug-safe Lab recipe ids — preserve percent escapes literally."""
    with patch("lib.routing.st.query_params", _FakeQueryParams({"recipe": "ring%20artifact"})):
        assert query_param("recipe", decode=False) == "ring%20artifact"


def test_query_param_handles_list_value_takes_first() -> None:
    """Streamlit returns a list when the param is repeated; take the first."""
    with patch("lib.routing.st.query_params", _FakeQueryParams({"tag": ["denoise", "ring"]})):
        assert query_param("tag") == "denoise"


def test_query_param_handles_empty_list_as_absent() -> None:
    with patch("lib.routing.st.query_params", _FakeQueryParams({"tag": []})):
        assert query_param("tag") is None


def test_query_param_coerces_non_string() -> None:
    """Defensive: Streamlit can hand back ints/bools — coerce to str."""
    with patch("lib.routing.st.query_params", _FakeQueryParams({"level": 2})):
        assert query_param("level") == "2"
