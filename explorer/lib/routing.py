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

from urllib.parse import quote, unquote

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


def note_url(doc_path: str) -> str:
    """Map a repo-relative note path to its owning cluster page deep-link.

    A bare ``?note=…`` href is relative to the *current* page, and only
    the three cluster pages (Discover / Explore / Build) actually read
    the ``note`` query-param — so links emitted from Search, the
    Knowledge Graph, or the Experiment page must route through the
    cluster that owns the note's top-level folder (same fix as the
    Troubleshooter's ``_guide_url``, R12 B3).

    Args:
        doc_path: Repo-relative note path, e.g.
            ``"09_noise_catalog/tomography/ring_artifact.md"``.

    Returns:
        ``"/<Cluster>?note=<quoted path>"``, or ``"#"`` when the folder
        does not belong to any cluster.
    """
    from lib.ia import FOLDER_TO_CLUSTER

    folder = doc_path.split("/", 1)[0] if doc_path else ""
    cluster = FOLDER_TO_CLUSTER.get(folder)
    if cluster is None:
        return "#"
    return f"/{cluster.title()}?note={quote(doc_path, safe='/')}"
