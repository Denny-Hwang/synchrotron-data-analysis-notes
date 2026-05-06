"""Build and Compute — cluster landing page.

Lists notes from ``05_tools_and_code``, ``06_data_structures``,
``07_data_pipeline``, and ``10_interactive_lab`` as cards. Routing
for ``?note=<url_id>`` and ``?tag=<tag>`` is handled by
:mod:`lib.cluster_page`.

Ref: IA-001 — Cluster mapping.
Ref: ADR-004 — 3-cluster IA.
Ref: ADR-008 — Section 10 Interactive Lab joins this cluster.
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

st.set_page_config(page_title="Build and Compute — eBERlight", page_icon="⚙️", layout="wide")

render_cluster_page("build", group_by_folder=True)
