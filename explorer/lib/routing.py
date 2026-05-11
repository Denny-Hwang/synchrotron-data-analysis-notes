"""URL / query-param utilities — single source of truth.

Before R15 this helper was copy-pasted across five Streamlit pages
(``cluster_page.py``, ``0_Knowledge_Graph.py``, ``5_Troubleshooter.py``,
``6_Search.py``, ``4_Experiment.py``) with subtly different
``unquote`` behaviour — the Experiment page skipped URL decoding and
silently dropped percent-escapes on recipe ids that contained them.
This module centralises the logic so a future Streamlit API change
(or a routing bug) only needs to be fixed once.

Ref: senior-review action item #2 (REL-E080).
"""

from __future__ import annotations

from urllib.parse import unquote

import streamlit as st


def query_param(name: str, *, decode: bool = True) -> str | None:
    """Read a single query-param value, robust to Streamlit's API shape.

    Streamlit's ``st.query_params`` returns either a string, a list of
    strings, or ``None`` depending on whether the param was passed once
    or repeated. We collapse the first two cases to a single string;
    multi-valued params keep only the first occurrence (matching the
    legacy behaviour of every prior copy-paste site).

    Args:
        name: The query-string key to read.
        decode: When ``True`` (default) apply ``urllib.parse.unquote``
            to the value before returning so callers get human-readable
            strings (``"ring artifact"`` rather than ``"ring%20artifact"``).
            Pass ``decode=False`` to preserve the raw param — only the
            Experiment page's recipe-id router historically did this.

    Returns:
        The decoded query-param value, or ``None`` when the param is
        absent or empty.
    """
    raw = st.query_params.get(name)
    if raw is None:
        return None
    if isinstance(raw, list):
        if not raw:
            return None
        value = raw[0]
    else:
        value = str(raw)
    return unquote(value) if decode else value
