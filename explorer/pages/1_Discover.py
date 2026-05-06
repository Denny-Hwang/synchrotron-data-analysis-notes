"""Discover the Program — cluster landing page.

Lists notes from ``01_program_overview`` and ``08_references`` as
cards. Renders the note-detail view when ``?note=<url_id>`` is set
and applies a tag filter when ``?tag=<tag>`` is set — see
:mod:`lib.cluster_page` for the shared routing logic.

Ref: IA-001 — Cluster mapping.
Ref: ADR-004 — 3-cluster IA.
Ref: FR-003 — Cluster pages list notes as cards.
Ref: FR-004 — Note-detail deep linking.
Ref: FR-007 — Tag click filtering.
"""

import sys
from pathlib import Path

import streamlit as st

_EXPLORER_DIR = Path(__file__).resolve().parent.parent
if str(_EXPLORER_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPLORER_DIR))

from lib.cluster_page import render_cluster_page

st.set_page_config(page_title="Discover the Program — eBERlight", page_icon="📖", layout="wide")

# Discover holds heterogeneous reference material — group by folder so
# the program overview and the reference index don't visually merge.
render_cluster_page("discover", group_by_folder=True)
